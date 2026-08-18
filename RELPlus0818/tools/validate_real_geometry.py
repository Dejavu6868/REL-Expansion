#!/usr/bin/env python3
"""Validate K, W2C pose and gravity against native global XYZ for a manifest."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rel_plus.camera import (
    backproject_z_depth,
    gravity_in_camera,
    load_stanford_s2d_camera_geometry,
    resize_camera_geometry,
)
from rel_plus.depth import decode_stanford_s2d_depth
from rel_plus.source_helpers import align_points_and_normals_to_gravity


COMPONENT_P95_TOLERANCE_M = 1.0 / 512.0


def decode_global_xyz(path):
    xyz_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if xyz_bgr is None or xyz_bgr.ndim != 3 or xyz_bgr.shape[2] != 3:
        raise OSError("failed to decode global XYZ EXR: {}".format(path))
    return xyz_bgr[:, :, ::-1].astype(np.float64, copy=False)


def evenly_spaced_joint_pixels(mask, count=1024):
    rows, columns = np.nonzero(mask)
    if rows.size < count:
        raise RuntimeError("only {} joint-valid pixels".format(rows.size))
    selected = np.linspace(0, rows.size - 1, count, dtype=np.int64)
    return rows[selected], columns[selected]


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_row(row):
    raw_depth = cv2.imread(row["depth_path"], cv2.IMREAD_UNCHANGED)
    if raw_depth is None or raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise RuntimeError("invalid native Depth16: {}".format(row["depth_path"]))
    camera = load_stanford_s2d_camera_geometry(row["camera_metadata_path"])
    depth_m, depth_valid = decode_stanford_s2d_depth(raw_depth)
    points_from_depth = backproject_z_depth(depth_m, depth_valid, camera.K_json)
    xyz_world = decode_global_xyz(Path(row["global_xyz_path"]))
    if xyz_world.shape != points_from_depth.shape:
        raise RuntimeError(
            "global XYZ/depth shape mismatch: {} versus {}".format(
                xyz_world.shape, points_from_depth.shape
            )
        )
    xyz_valid = np.all(np.isfinite(xyz_world), axis=2) & ~np.all(
        xyz_world == 0.0, axis=2
    )
    sample_rows, sample_columns = evenly_spaced_joint_pixels(
        depth_valid & xyz_valid
    )
    xyz_camera = (
        xyz_world[sample_rows, sample_columns] @ camera.R_world_to_camera.T
        + camera.t_world_to_camera
    )
    depth_points = points_from_depth[sample_rows, sample_columns]
    component_error = np.abs(xyz_camera - depth_points)
    p95 = np.quantile(component_error, 0.95, axis=0)

    projected_u = (
        camera.K_json[0, 0] * xyz_camera[:, 0] / xyz_camera[:, 2]
        + camera.K_json[0, 2]
    )
    projected_v = (
        camera.K_json[1, 1] * xyz_camera[:, 1] / xyz_camera[:, 2]
        + camera.K_json[1, 2]
    )
    reprojection_error = np.hypot(
        projected_u - (sample_columns + 0.5),
        projected_v - (sample_rows + 0.5),
    )

    with Path(row["camera_metadata_path"]).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    declared_center = np.asarray(payload["camera_location"], dtype=np.float64)
    recovered_center = -camera.R_world_to_camera.T @ camera.t_world_to_camera
    camera_center_error = float(np.max(np.abs(recovered_center - declared_center)))

    gravity = gravity_in_camera(camera.R_world_to_camera)
    field = gravity.reshape(1, 1, 3)
    aligned, _, alignment = align_points_and_normals_to_gravity(
        field, field, gravity
    )
    gravity_error = float(np.max(np.abs(aligned[0, 0] - [0.0, 0.0, -1.0])))
    rotation_orthogonality_error = float(
        np.max(
            np.abs(
                camera.R_world_to_camera.T @ camera.R_world_to_camera - np.eye(3)
            )
        )
    )
    alignment_orthogonality_error = float(
        np.max(np.abs(alignment.T @ alignment - np.eye(3)))
    )

    canonical = resize_camera_geometry(camera, raw_depth.shape, (480, 480))
    expected_k = camera.K_json.copy()
    expected_k[0, :] *= 480.0 / raw_depth.shape[1]
    expected_k[1, :] *= 480.0 / raw_depth.shape[0]
    canonical_k_error = float(np.max(np.abs(canonical.K_json - expected_k)))

    geometry_pass = bool(
        np.all(p95 <= COMPONENT_P95_TOLERANCE_M)
        and camera_center_error <= 1e-4
        and gravity_error <= 1e-6
        and rotation_orthogonality_error <= 1e-5
        and alignment_orthogonality_error <= 1e-5
        and canonical_k_error == 0.0
    )
    return {
        "sample_id": row["sample_id"],
        "area": row["area"],
        "room": row["room"],
        "camera": row["camera"],
        "probe_count": len(sample_rows),
        "p95_abs_x_m": float(p95[0]),
        "p95_abs_y_m": float(p95[1]),
        "p95_abs_z_m": float(p95[2]),
        "max_euclidean_m": float(
            np.max(np.linalg.norm(component_error, axis=1))
        ),
        "reprojection_p95_pixels": float(np.quantile(reprojection_error, 0.95)),
        "camera_center_max_abs_error_m": camera_center_error,
        "rotation_orthogonality_max_abs_error": rotation_orthogonality_error,
        "rotation_determinant": float(np.linalg.det(camera.R_world_to_camera)),
        "gravity_alignment_max_abs_error": gravity_error,
        "alignment_orthogonality_max_abs_error": alignment_orthogonality_error,
        "canonical_k_max_abs_error": canonical_k_error,
        "component_p95_tolerance_m": COMPONENT_P95_TOLERANCE_M,
        "geometry_status": "PASS" if geometry_pass else "FAIL",
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    if len(manifest) != 10:
        raise RuntimeError("expected exactly 10 review samples")
    rows = [validate_row(row) for row in manifest]
    write_csv(output_root / "real_geometry_validation.csv", rows)
    passed = all(row["geometry_status"] == "PASS" for row in rows)
    summary = {
        "status": "PASS_REAL_GEOMETRY_VALIDATION" if passed else "FAIL_REAL_GEOMETRY_VALIDATION",
        "sample_count": len(rows),
        "pass_count": sum(row["geometry_status"] == "PASS" for row in rows),
        "areas": sorted({row["area"] for row in rows}),
        "component_p95_tolerance_m": COMPONENT_P95_TOLERANCE_M,
        "max_component_p95_m": max(
            max(row["p95_abs_x_m"], row["p95_abs_y_m"], row["p95_abs_z_m"])
            for row in rows
        ),
        "max_camera_center_error_m": max(
            row["camera_center_max_abs_error_m"] for row in rows
        ),
        "max_gravity_alignment_error": max(
            row["gravity_alignment_max_abs_error"] for row in rows
        ),
        "k_convention": "JSON half-pixel centres; u+0.5/v+0.5 backprojection",
        "pose_convention": "X_camera = R_world_to_camera @ X_world + t",
    }
    (output_root / "real_geometry_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
