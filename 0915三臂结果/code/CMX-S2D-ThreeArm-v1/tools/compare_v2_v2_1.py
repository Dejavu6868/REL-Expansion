#!/usr/bin/env python3
"""Prove v2.1 core bytes equal the independent deployed v2 baseline."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rel_plus.generator import generate_rel_plus_v2_1
from rel_plus.profiles import STANFORD_S2D_PROFILE
from rel_plus.stanford_s2d import load_canonical_frame


def _read_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_v2(v2_root, python, depth, pose, output):
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            python,
            str(Path(v2_root) / "tools/generate_rel_plus.py"),
            "--depth",
            str(depth),
            "--camera-json",
            str(pose),
            "--output",
            str(output),
        ],
        cwd=str(v2_root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("v2 baseline failed: {}".format(completed.stdout))


def _compare_case(case_name, depth, pose, v2_root, python, output_root):
    baseline_path = output_root / "baseline_v2" / (case_name + ".png")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    _run_v2(v2_root, python, depth, pose, baseline_path)
    baseline = cv2.imread(str(baseline_path), cv2.IMREAD_UNCHANGED)
    raw, camera, _ = load_canonical_frame(
        depth, pose, dataset_profile=STANFORD_S2D_PROFILE
    )
    current = generate_rel_plus_v2_1(raw, camera)
    changed = np.any(baseline != current, axis=2)
    locations = np.argwhere(baseline != current)
    result = {
        "case": case_name,
        "status": "PASS" if locations.size == 0 else "FAIL",
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_channel_count": int(locations.shape[0]),
        "first_changed_row": "",
        "first_changed_column": "",
        "first_changed_channel": "",
        "v2_value": "",
        "v2_1_value": "",
    }
    if locations.size:
        row, column, channel = locations[0]
        result.update(
            {
                "first_changed_row": int(row),
                "first_changed_column": int(column),
                "first_changed_channel": int(channel),
                "v2_value": int(baseline[row, column, channel]),
                "v2_1_value": int(current[row, column, channel]),
            }
        )
    return result


def _write_synthetic(root):
    root.mkdir(parents=True, exist_ok=True)
    rows, columns = np.indices((1080, 1080))
    raw = np.rint((2.0 + rows * 0.001 + columns * 0.0005) * 512.0).astype(
        np.uint16
    )
    raw[:100, :80] = 0
    raw[700:760, 500:600] = 65535
    depth = root / "depth.png"
    pose = root / "pose.json"
    if not cv2.imwrite(str(depth), raw):
        raise OSError("failed to write synthetic depth")
    pose.write_text(
        json.dumps(
            {
                "camera_k_matrix": [
                    [800.0, 0.0, 540.0],
                    [0.0, 810.0, 540.0],
                    [0.0, 0.0, 1.0],
                ],
                "camera_rt_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                "camera_location": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    return depth, pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--real-limit", type=int, default=12)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest(args.manifest)[: args.real_limit]
    if len(rows) != args.real_limit:
        raise ValueError("real regression manifest has too few rows")
    results = []
    synthetic_depth, synthetic_pose = _write_synthetic(
        output_root / "synthetic_fixture"
    )
    results.append(
        _compare_case(
            "synthetic",
            synthetic_depth,
            synthetic_pose,
            args.v2_root,
            args.python,
            output_root,
        )
    )
    for index, row in enumerate(rows):
        results.append(
            _compare_case(
                "real_{:02d}".format(index),
                row["depth_path"],
                row["camera_metadata_path"],
                args.v2_root,
                args.python,
                output_root,
            )
        )
    with (output_root / "v2_v2_1_byte_regression.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "protocol_id": "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT",
        "case_count": len(results),
        "synthetic_case_count": 1,
        "real_case_count": len(rows),
        "changed_pixel_count": sum(row["changed_pixel_count"] for row in results),
        "changed_channel_count": sum(row["changed_channel_count"] for row in results),
    }
    (output_root / "v2_v2_1_byte_regression.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
