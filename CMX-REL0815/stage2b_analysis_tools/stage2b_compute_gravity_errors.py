#!/usr/bin/env python3
"""Compute Area5 EstGravity-versus-pose disagreement on the exact eval geometry."""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relplus.geometry import backproject_z_depth, load_camera_metadata
from relplus.pipeline import (
    SpatialTransformParameters,
    estimate_gravity_down_camera,
    generate_relplus_from_depth_local,
    transform_depth_geometry,
)
from relplus.representation import estimate_rel_normals


_DATASET_ROOT = None
_PARAMETERS = None
_GRAVITY_DOWN_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _initialize_worker(dataset_root, height, width):
    global _DATASET_ROOT, _PARAMETERS
    _DATASET_ROOT = Path(dataset_root)
    _PARAMETERS = SpatialTransformParameters(
        resize_height=height,
        resize_width=width,
        crop_y=0,
        crop_x=0,
        crop_height=height,
        crop_width=width,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        flip=False,
    )
    cv2.setNumThreads(1)


def _load_transformed_geometry(name):
    depth_path = _DATASET_ROOT / "Depth16" / (name + ".png")
    pose_path = _DATASET_ROOT / "Pose" / (name + ".json")
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError("invalid uint16 depth: {}".format(depth_path))
    depth = raw.astype(np.float64) / 512.0
    valid = (raw != 65535) & (raw > 0) & np.isfinite(depth)
    camera = load_camera_metadata(pose_path)
    transformed_depth, transformed_valid, transformed_k = transform_depth_geometry(
        depth, valid, camera.k, _PARAMETERS
    )
    return transformed_depth, transformed_valid, transformed_k, camera


def _compute_one(name):
    depth, valid, intrinsics, camera = _load_transformed_geometry(name)
    points = backproject_z_depth(depth, intrinsics, pixel_origin=0.5)
    normals, normal_valid = estimate_rel_normals(points, valid, radius=3)
    rel_valid = valid & normal_valid
    estimated = estimate_gravity_down_camera(normals, rel_valid)
    pose = camera.r_world_to_camera @ _GRAVITY_DOWN_WORLD
    pose /= np.linalg.norm(pose)
    angle = float(
        np.degrees(np.arccos(np.clip(np.dot(estimated, pose), -1.0, 1.0)))
    )
    if not np.isfinite(angle):
        raise ValueError("non-finite angular disagreement: {}".format(name))
    return {
        "name": name,
        "area": name.split("/", 1)[0],
        "estimated_gx": float(estimated[0]),
        "estimated_gy": float(estimated[1]),
        "estimated_gz": float(estimated[2]),
        "pose_gx": float(pose[0]),
        "pose_gy": float(pose[1]),
        "pose_gz": float(pose[2]),
        "angular_error_deg": angle,
        "estimated_norm": float(np.linalg.norm(estimated)),
        "normal_valid_count": int(np.count_nonzero(rel_valid)),
    }


def _verify_against_production(name, row):
    depth, valid, intrinsics, _ = _load_transformed_geometry(name)
    _, _, auxiliary = generate_relplus_from_depth_local(
        depth, valid, intrinsics, normal_radius=3
    )
    production = np.asarray(auxiliary["gravity_down_camera"], dtype=np.float64)
    direct = np.array(
        [row["estimated_gx"], row["estimated_gy"], row["estimated_gz"]],
        dtype=np.float64,
    )
    maximum_error = float(np.max(np.abs(production - direct)))
    if maximum_error > 1.0e-12:
        raise ValueError(
            "gravity fast path differs from production for {}: {}".format(
                name, maximum_error
            )
        )
    return maximum_error


def _write_csv_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--eval-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--expected-count", type=int, default=17593)
    parser.add_argument("--verify-count", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    samples = [
        line.strip()
        for line in args.eval_source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(samples) != len(set(samples)):
        raise ValueError("eval source contains duplicate sample names")
    if args.limit is None and len(samples) != args.expected_count:
        raise ValueError(
            "expected {} samples, found {}".format(args.expected_count, len(samples))
        )
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("eval source is empty")

    rgb = cv2.imread(
        str(args.dataset_root / "RGB" / (samples[0] + ".png")),
        cv2.IMREAD_COLOR,
    )
    if rgb is None or rgb.shape[:2] != (480, 480):
        raise ValueError("evaluation RGB geometry is not 480x480")

    _initialize_worker(str(args.dataset_root), 480, 480)
    verification = []
    for name in samples[: args.verify_count]:
        row = _compute_one(name)
        verification.append(
            {
                "name": name,
                "maximum_component_error": _verify_against_production(name, row),
            }
        )

    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(str(args.dataset_root), 480, 480),
    ) as executor:
        for index, row in enumerate(
            executor.map(_compute_one, samples, chunksize=8), start=1
        ):
            rows.append(row)
            if index % 500 == 0 or index == len(samples):
                print("gravity_progress={}/{}".format(index, len(samples)), flush=True)

    if [row["name"] for row in rows] != samples:
        raise ValueError("gravity output order differs from eval source")
    norms = np.asarray([row["estimated_norm"] for row in rows], dtype=np.float64)
    angles = np.asarray([row["angular_error_deg"] for row in rows], dtype=np.float64)
    if not np.isfinite(norms).all() or np.max(np.abs(norms - 1.0)) > 1.0e-10:
        raise ValueError("EstGravity contains a non-finite or non-unit vector")

    summary = {
        "status": "PASS_AREA5_GRAVITY_ERRORS",
        "count": len(rows),
        "mean_deg": float(np.mean(angles)),
        "std_deg": float(np.std(angles, ddof=1)),
        "median_deg": float(np.median(angles)),
        "p33_333_deg": float(np.quantile(angles, 1.0 / 3.0)),
        "p66_667_deg": float(np.quantile(angles, 2.0 / 3.0)),
        "p95_deg": float(np.quantile(angles, 0.95)),
        "max_deg": float(np.max(angles)),
        "minimum_normal_valid_count": int(
            min(row["normal_valid_count"] for row in rows)
        ),
        "production_crosscheck": verification,
        "geometry_contract": {
            "depth_decode": "raw_uint16 / 512.0; 0 and 65535 invalid",
            "resize": "nearest-neighbor from native Depth16 to 480x480",
            "intrinsics": "same 0.5-pixel-center update used by ValPre",
            "normal_radius": 3,
            "estimator": "initial camera +Y; thresholds 45/15 degrees; iterations 5/5",
            "pose_gravity": "R_world_to_camera @ [0,0,-1]",
        },
    }
    _write_csv_atomic(args.output, rows)
    _write_json_atomic(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
