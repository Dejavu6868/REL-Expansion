#!/usr/bin/env python3
"""Validate native/canonical global XYZ and physical pose on 12 real frames."""

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
    load_stanford_s2d_camera_geometry,
)
from rel_plus.depth import decode_stanford_s2d_depth
from rel_plus.generator import generate_rel_plus_v2
from rel_plus.stanford_s2d import load_canonical_frame
from rel_plus.validation.canonical_geometry import validate_canonical_geometry
from rel_plus.validation.geometry_oracle import evenly_spaced_joint_pixels
from rel_plus.validation.pose_physics import validate_pose_physics


def _read_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _decode_global_xyz(path):
    xyz_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if xyz_bgr is None or xyz_bgr.ndim != 3 or xyz_bgr.shape[2] != 3:
        raise OSError("failed to decode global XYZ EXR: {}".format(path))
    return xyz_bgr[:, :, ::-1].astype(np.float64, copy=False)


def _semantic_ids(class_mapping_path):
    payload = json.loads(Path(class_mapping_path).read_text(encoding="utf-8"))
    by_name = {name: int(index) for index, name in payload["stored_ids"].items()}
    return {name: by_name[name] for name in ("floor", "ceiling", "wall")}


def _strong_pose_arrays(raw, xyz_world, camera):
    depth_m, depth_valid = decode_stanford_s2d_depth(raw)
    points = backproject_z_depth(depth_m, depth_valid, camera.K_json)
    xyz_valid = np.all(np.isfinite(xyz_world), axis=2) & ~np.all(xyz_world == 0.0, axis=2)
    rows, columns = evenly_spaced_joint_pixels(depth_valid & xyz_valid, maximum_count=4096)
    pixels = np.column_stack([columns + 0.5, rows + 0.5])
    return xyz_world[rows, columns], points[rows, columns], pixels


def _metric(result, stage, group, key):
    if result is None:
        return None
    return result[stage][group][key]


def validate_row(row, semantic_ids):
    raw = cv2.imread(row["depth_path"], cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("invalid native Depth16")
    declared_shape = (int(row["intrinsics_height"]), int(row["intrinsics_width"]))
    camera = load_stanford_s2d_camera_geometry(row["camera_metadata_path"], declared_shape)
    camera.assert_matches_image_shape(raw.shape)
    canonical_raw, canonical_camera, _ = load_canonical_frame(
        row["depth_path"], row["camera_metadata_path"]
    )
    _, debug = generate_rel_plus_v2(canonical_raw, canonical_camera, return_debug=True)
    labels = cv2.imread(row["label_path"], cv2.IMREAD_UNCHANGED)
    if labels is None or labels.shape != canonical_raw.shape:
        raise ValueError("canonical semantic label is missing or misaligned")

    xyz_path = Path(row.get("global_xyz_path", ""))
    canonical_result = None
    if xyz_path.is_file():
        xyz_world = _decode_global_xyz(xyz_path)
        canonical_result = validate_canonical_geometry(raw, xyz_world, camera)
        world_points, camera_points, pixels = _strong_pose_arrays(raw, xyz_world, camera)
        pose_result = validate_pose_physics(
            camera,
            world_points=world_points,
            camera_points=camera_points,
            pixel_coordinates=pixels,
            labels=labels,
            normals_aligned=debug["normals_aligned"],
            points_aligned_m=debug["points_aligned_m"],
            semantic_ids=semantic_ids,
        )
        geometry_oracle = "available"
        validation_level = "strong"
        geometry_status = canonical_result["status"]
    else:
        pose_result = validate_pose_physics(
            camera,
            labels=labels,
            normals_aligned=debug["normals_aligned"],
            points_aligned_m=debug["points_aligned_m"],
            semantic_ids=semantic_ids,
        )
        geometry_oracle = "unavailable"
        validation_level = "weak"
        geometry_status = "NOT_APPLICABLE"

    native_component_p95 = (
        max(
            canonical_result["native"]["component_error_m"][axis]["p95"]
            for axis in ("x", "y", "z")
        )
        if canonical_result else None
    )
    canonical_component_p95 = (
        max(
            canonical_result["canonical"]["component_error_m"][axis]["p95"]
            for axis in ("x", "y", "z")
        )
        if canonical_result else None
    )
    return {
        "sample_id": row["sample_id"],
        "area": row["area"],
        "room": row["room"],
        "camera": row["camera"],
        "geometry_oracle": geometry_oracle,
        "validation_level": validation_level,
        "native_geometry_status": geometry_status if canonical_result else "NOT_APPLICABLE",
        "canonical_geometry_status": geometry_status,
        "native_component_p95_m": native_component_p95,
        "canonical_component_p95_m": canonical_component_p95,
        "native_euclidean_p95_m": _metric(canonical_result, "native", "euclidean_error_m", "p95"),
        "canonical_euclidean_p95_m": _metric(canonical_result, "canonical", "euclidean_error_m", "p95"),
        "native_reprojection_p95_pixels": _metric(canonical_result, "native", "reprojection_pixels", "p95"),
        "canonical_reprojection_p95_pixels": _metric(canonical_result, "canonical", "reprojection_pixels", "p95"),
        "canonical_reprojection_max_pixels": _metric(canonical_result, "canonical", "reprojection_pixels", "max"),
        "pose_physical_status": pose_result.status,
        "pose_evidence_level": pose_result.evidence_level,
        "pose_metrics": json.dumps(pose_result.metrics, ensure_ascii=False, sort_keys=True),
        "pose_warnings": ";".join(pose_result.warnings),
        "invalid_depth_ratio": float(1.0 - np.mean(debug["depth_valid"])),
        "nonfinite_normal_ratio": debug["normal_invalid_ratio"],
        "zero_normal_ratio": debug["zero_normal_ratio"],
        "low_support_ratio": debug["low_support_ratio"],
        "gravity_vector": json.dumps(debug["gravity_camera"].tolist()),
        "sample_status": (
            "PASS"
            if geometry_status in ("PASS", "NOT_APPLICABLE") and pose_result.status != "FAIL"
            else "FAIL"
        ),
        "canonical_detail": canonical_result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--class-mapping",
        default="/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480/class_mapping.json",
    )
    args = parser.parse_args()
    rows = _read_manifest(args.manifest)
    if len(rows) != 12:
        raise ValueError("formal real validation requires exactly 12 samples")
    semantic_ids = _semantic_ids(args.class_mapping)
    results = [validate_row(row, semantic_ids) for row in rows]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    detail = {
        row["sample_id"]: row.pop("canonical_detail") for row in results
    }
    with (output_root / "real_geometry_validation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output_root / "canonical_geometry_details.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    passed = all(row["sample_status"] == "PASS" for row in results)
    strong = [row for row in results if row["geometry_oracle"] == "available"]
    summary = {
        "status": "PASS" if passed else "FAIL",
        "sample_count": len(results),
        "pass_count": sum(row["sample_status"] == "PASS" for row in results),
        "areas": sorted({row["area"] for row in results}),
        "strong_geometry_count": len(strong),
        "weak_area1_count": sum(row["area"] == "area_1" for row in results),
        "area1_oracle_limitation": "global XYZ unavailable; weak pose/normal review only",
        "max_native_component_p95_m": max(row["native_component_p95_m"] for row in strong),
        "max_canonical_component_p95_m": max(row["canonical_component_p95_m"] for row in strong),
        "max_canonical_reprojection_p95_pixels": max(row["canonical_reprojection_p95_pixels"] for row in strong),
        "max_canonical_reprojection_max_pixels": max(row["canonical_reprojection_max_pixels"] for row in strong),
        "pose_status_counts": {
            status: sum(row["pose_physical_status"] == status for row in results)
            for status in ("PASS", "FAIL", "NOT_APPLICABLE")
        },
        "full_cache_status": "NOT_GENERATED_REVIEW_ONLY",
    }
    (output_root / "real_geometry_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
