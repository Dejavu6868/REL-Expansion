#!/usr/bin/env python3
"""Read-only RawDepth/HHA contract audit for future three-arm comparison."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_full_relplus_cache import read_manifest


def classify_raw_depth_contract(stats):
    dtype_counts = stats.get("dtype_counts", {})
    uint16_count = int(dtype_counts.get("uint16", 0))
    uint8_count = int(dtype_counts.get("uint8", 0))
    if uint16_count:
        return {
            "status": "RGBD_INPUT_CONTRACT_REQUIRES_DECISION",
            "reason": (
                "RawDepth contains uint16 inputs; the current source loader uses "
                "cv2.IMREAD_GRAYSCALE and may silently compress them to uint8."
            ),
        }
    if uint8_count == int(stats.get("file_count", 0)) and uint8_count > 0:
        return {
            "status": "RGBD_INPUT_READY",
            "reason": "all decoded RawDepth inputs are uint8",
        }
    return {
        "status": "RGBD_INPUT_CONTRACT_REQUIRES_DECISION",
        "reason": "RawDepth files are missing, undecodable, or have mixed dtypes",
    }


def audit_raw_depth(rows, root):
    dtype_counts = Counter()
    shape_counts = Counter()
    failures = []
    minimum = None
    maximum = None
    for row in rows:
        path = Path(root) / (row["sample_id"] + ".png")
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            failures.append({"sample_id": row["sample_id"], "reason": "decode"})
            continue
        dtype_counts[str(value.dtype)] += 1
        shape_counts["x".join(str(item) for item in value.shape)] += 1
        current_min = float(np.min(value))
        current_max = float(np.max(value))
        minimum = current_min if minimum is None else min(minimum, current_min)
        maximum = current_max if maximum is None else max(maximum, current_max)
    stats = {
        "file_count": len(rows) - len(failures),
        "expected_count": len(rows),
        "failure_count": len(failures),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "min": minimum,
        "max": maximum,
    }
    denominator = max(1, stats["file_count"])
    stats["dtype_ratios"] = {
        name: count / denominator for name, count in sorted(dtype_counts.items())
    }
    stats.update(classify_raw_depth_contract(stats))
    return stats, failures


def audit_hha(rows, root):
    failures = []
    bgr_read_count = 0
    decoded_count = 0
    dtype_counts = Counter()
    shape_counts = Counter()
    for row in rows:
        path = Path(root) / (row["sample_id"] + ".png")
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            failures.append({"sample_id": row["sample_id"], "reason": "decode"})
            continue
        decoded_count += 1
        dtype_counts[str(value.dtype)] += 1
        shape_counts["x".join(str(item) for item in value.shape)] += 1
        reasons = []
        if value.shape != (480, 480, 3):
            reasons.append("shape_channels")
        if value.dtype != np.uint8:
            reasons.append("dtype")
        if reasons:
            failures.append(
                {"sample_id": row["sample_id"], "reason": "|".join(reasons)}
            )
        else:
            bgr_read_count += 1
    return {
        "status": "HHA_INPUT_READY" if not failures else "HHA_INPUT_CONTRACT_REQUIRES_DECISION",
        "file_count": decoded_count,
        "expected_count": len(rows),
        "failure_count": len(failures),
        "required_dtype": "uint8",
        "required_shape": [480, 480, 3],
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "opencv_channel_behavior": "IMREAD_UNCHANGED returns stored BGR byte order",
        "valid_bgr_decode_count": bgr_read_count,
    }, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-depth-root", required=True, type=Path)
    parser.add_argument("--hha-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    raw_stats, raw_failures = audit_raw_depth(rows, args.raw_depth_root)
    hha_stats, hha_failures = audit_hha(rows, args.hha_root)
    report = {
        "status": "PASS",
        "scope": "future_three_arm_data_contract_only",
        "blocks_current_relplus_single_arm": False,
        "blocks_future_three_arm_comparison": bool(
            raw_stats["status"] != "RGBD_INPUT_READY"
            or hha_stats["status"] != "HHA_INPUT_READY"
        ),
        "manifest_path": str(args.manifest.resolve()),
        "raw_depth_root": str(args.raw_depth_root.resolve()),
        "hha_root": str(args.hha_root.resolve()),
        "raw_depth": raw_stats,
        "hha": hha_stats,
        "raw_depth_failures_preview": raw_failures[:100],
        "hha_failures_preview": hha_failures[:100],
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
