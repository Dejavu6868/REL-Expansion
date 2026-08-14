#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relplus.geometry import load_camera_metadata
from relplus.pipeline import generate_relplus_from_depth_local


def vector_text(vector):
    return "[{:.12g},{:.12g},{:.12g}]".format(*vector)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    rows = []
    for sample in samples:
        raw = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise ValueError("invalid uint16 depth: {}".format(sample["depth_path"]))
        depth = raw.astype(np.float64) / 512.0
        valid = (raw != 65535) & (raw > 0) & np.isfinite(depth)
        camera = load_camera_metadata(sample["pose_path"])
        _, _, auxiliary = generate_relplus_from_depth_local(depth, valid, camera.k, normal_radius=3)
        estimated = auxiliary["gravity_down_camera"]
        pose = camera.r_world_to_camera @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        pose /= np.linalg.norm(pose)
        angle = float(np.degrees(np.arccos(np.clip(np.dot(estimated, pose), -1.0, 1.0))))
        rows.append({
            "area": sample["area"], "frame_id": sample["frame_id"],
            "estimated_gravity": vector_text(estimated), "pose_gravity": vector_text(pose),
            "angular_error_deg": angle, "estimated_finite": bool(np.isfinite(estimated).all()),
            "estimated_norm": float(np.linalg.norm(estimated)),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    angles = np.array([row["angular_error_deg"] for row in rows], dtype=np.float64)
    summary = {
        "status": "PASS_ESTGRAVITY_FINITE_24" if all(row["estimated_finite"] for row in rows) else "FAIL_ESTGRAVITY_NONFINITE",
        "count": len(rows), "mean_deg": float(angles.mean()), "median_deg": float(np.median(angles)),
        "p95_deg": float(np.quantile(angles, 0.95)), "max_deg": float(angles.max()),
        "estimator": "REL-default floor/wall normals; thresholds 45/15 deg; iterations 5/5; initial camera +Y",
        "purpose": "diagnostic only; no tuning",
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["status"].startswith("PASS_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
