#!/usr/bin/env python3
"""Lightweight manifest preflight; it never generates REL+ images."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rel_plus.camera import gravity_in_camera, load_stanford_s2d_camera_geometry
from rel_plus.geometry import GravityAlignmentSingularity, align_points_and_normals_to_gravity


CORE_FAILURES = {
    "K_RESOLUTION_MISMATCH",
    "K_REFERENCE_SHAPE_MISSING",
    "POSE_CONVENTION_FAILURE",
    "GRAVITY_SINGULARITY",
}


def _declared_intrinsics_shape(row):
    try:
        height = int(row["intrinsics_height"])
        width = int(row["intrinsics_width"])
    except (KeyError, TypeError, ValueError):
        return None
    return (height, width) if min(height, width) > 0 else None


def scan_row(row):
    sample_id = row.get("sample_id", "<missing-sample-id>")
    reasons = []
    depth_path = Path(row.get("depth_path", ""))
    pose_path = Path(row.get("camera_metadata_path", ""))
    for field in ("rgb_path", "label_path", "depth_path", "camera_metadata_path"):
        if not row.get(field) or not Path(row[field]).is_file():
            reasons.append("FILE_PAIR_MISSING:{}".format(field))
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.is_file() else None
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
        reasons.append("DEPTH_DTYPE_OR_SHAPE_FAILURE")
    intrinsics_shape = _declared_intrinsics_shape(row)
    if intrinsics_shape is None:
        reasons.append("K_REFERENCE_SHAPE_MISSING")

    camera_center = [None, None, None]
    gravity = [None, None, None]
    if pose_path.is_file() and intrinsics_shape is not None:
        try:
            camera = load_stanford_s2d_camera_geometry(pose_path, intrinsics_shape)
            camera_center = (
                -camera.R_world_to_camera.T @ camera.t_world_to_camera
            ).tolist()
            if raw is not None:
                try:
                    camera.assert_matches_image_shape(raw.shape)
                except ValueError:
                    reasons.append("K_RESOLUTION_MISMATCH")
            gravity_array = gravity_in_camera(camera.R_world_to_camera)
            gravity = gravity_array.tolist()
            probe = gravity_array.reshape(1, 1, 3)
            try:
                align_points_and_normals_to_gravity(
                    probe, probe, gravity_array, sample_id=sample_id
                )
            except GravityAlignmentSingularity:
                reasons.append("GRAVITY_SINGULARITY")
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append("POSE_CONVENTION_FAILURE")
    elif not pose_path.is_file():
        reasons.append("POSE_CONVENTION_FAILURE")

    reasons = sorted(set(reasons))
    return {
        "sample_id": sample_id,
        "depth_dtype": str(raw.dtype) if raw is not None else "UNAVAILABLE",
        "depth_shape": "{}x{}".format(*raw.shape) if raw is not None else "UNAVAILABLE",
        "intrinsics_shape": (
            "{}x{}".format(*intrinsics_shape) if intrinsics_shape else "UNAVAILABLE"
        ),
        "camera_center": json.dumps(camera_center),
        "gravity_camera": json.dumps(gravity),
        "status": "PASS" if not reasons else "FAIL",
        "reasons": ";".join(reasons),
    }


def scan_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest is empty")
    return [scan_row(row) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = scan_manifest(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    core_failures = sorted(
        {
            reason
            for row in results
            for reason in row["reasons"].split(";")
            if reason in CORE_FAILURES
        }
    )
    summary = {
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "sample_count": len(results),
        "pass_count": sum(row["status"] == "PASS" for row in results),
        "core_failures": core_failures,
        "full_cache_status": (
            "NOT_READY_FOR_FULL_CACHE" if core_failures else "READY_FOR_REVIEW_ONLY"
        ),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
