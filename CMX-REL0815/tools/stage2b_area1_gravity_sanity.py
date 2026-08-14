#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from relplus.geometry import backproject_z_depth, camera_to_world, load_camera_metadata
from relplus.representation import decode_stanford_depth


TARGET_FRAMES = 24
MIN_FRAMES = 20
MEDIAN_LIMIT_DEGREES = 15.0
P95_LIMIT_DEGREES = 30.0


def fit_floor_normal(points_world, camera_center, seed):
    rng = np.random.RandomState(seed)
    if len(points_world) > 6000:
        points_world = points_world[rng.choice(len(points_world), 6000, replace=False)]
    best = None
    for _ in range(400):
        sample = points_world[rng.choice(len(points_world), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length <= 1e-10:
            continue
        normal /= length
        distance = np.abs((points_world - sample[0]) @ normal)
        inliers = distance < 0.03
        score = int(np.count_nonzero(inliers))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0] < 200:
        raise ValueError("no stable floor plane")
    inlier_points = points_world[best[1]]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if np.dot(normal, camera_center - centroid) < 0.0:
        normal = -normal
    residual = np.abs((inlier_points - centroid) @ normal)
    return normal, len(points_world), len(inlier_points), float(np.median(residual))


def candidates(source):
    seen_rooms = set()
    for line in source.read_text().splitlines():
        sample_id = line.strip()
        if not sample_id.startswith("area_1/"):
            continue
        yield sample_id, seen_rooms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_mapping = json.loads((args.dataset_root / "class_mapping.json").read_text())
    floor_ids = [int(key) for key, name in class_mapping["stored_ids"].items() if name == "floor"]
    if len(floor_ids) != 1:
        raise ValueError("dataset must declare exactly one stored floor class")
    floor_class = floor_ids[0]
    rows = []
    rooms = set()
    failures = []
    for index, (sample_id, _) in enumerate(candidates(args.train_source)):
        pose_path = args.dataset_root / "Pose" / (sample_id + ".json")
        label_path = args.dataset_root / "Label" / (sample_id + ".png")
        depth_path = args.dataset_root / "Depth16" / (sample_id + ".png")
        try:
            payload = json.loads(pose_path.read_text())
            room = str(payload.get("room", ""))
            if not room or room in rooms:
                continue
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if label is None or raw_depth is None:
                raise ValueError("missing label or depth")
            floor = cv2.resize(
                (label == floor_class).astype(np.uint8),
                (raw_depth.shape[1], raw_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            depth, valid = decode_stanford_depth(raw_depth)
            usable = floor & valid
            if np.count_nonzero(usable) < 2000:
                continue
            camera = load_camera_metadata(pose_path)
            camera_points = backproject_z_depth(depth, camera.k, pixel_origin=1.0)
            world_points = camera_to_world(
                camera_points[usable], camera.r_world_to_camera, camera.camera_center_world
            )
            normal, sampled, inliers, residual = fit_floor_normal(
                world_points, camera.camera_center_world, seed=7319 + index
            )
            angle = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
            rows.append({
                "sample_id": sample_id,
                "room": room,
                "floor_pixels": int(np.count_nonzero(usable)),
                "sampled_points": sampled,
                "plane_inliers": inliers,
                "median_plane_residual_m": residual,
                "normal_world_x": float(normal[0]),
                "normal_world_y": float(normal[1]),
                "normal_world_z": float(normal[2]),
                "angle_to_world_plus_z_degrees": angle,
            })
            rooms.add(room)
            if len(rows) >= TARGET_FRAMES:
                break
        except Exception as error:
            failures.append({"sample_id": sample_id, "error": str(error)})
    angles = np.asarray([row["angle_to_world_plus_z_degrees"] for row in rows], dtype=np.float64)
    median = float(np.median(angles)) if len(angles) else None
    p95 = float(np.percentile(angles, 95)) if len(angles) else None
    maximum = float(np.max(angles)) if len(angles) else None
    passed = len(rows) >= MIN_FRAMES and median <= MEDIAN_LIMIT_DEGREES and p95 <= P95_LIMIT_DEGREES
    report = {
        "status": "PASS_AREA1_WORLD_UP_SANITY" if passed else "BLOCKED_AREA1_WORLD_UP_SANITY",
        "semantic_floor_stored_id": floor_class,
        "sampled_unique_rooms": len(rows),
        "required_minimum_rooms": MIN_FRAMES,
        "median_angle_degrees": median,
        "p95_angle_degrees": p95,
        "max_angle_degrees": maximum,
        "gate_limits_degrees": {"median": MEDIAN_LIMIT_DEGREES, "p95": P95_LIMIT_DEGREES},
        "method": "Depth16 backprojection; nearest-neighbor 480-to-1080 floor mask; deterministic RANSAC plus SVD; floor normal oriented toward camera",
        "failures": failures,
    }
    with (args.output_dir / "area1_floor_normals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "area1_world_up_sanity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
