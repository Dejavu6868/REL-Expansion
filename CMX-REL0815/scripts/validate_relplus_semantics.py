#!/usr/bin/env python3
"""Recompute audited samples and require exact REL-default cache equality."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from relplus.geometry import load_camera_metadata
from relplus.io import read_relplus_png, resolve_sample_paths
from relplus.representation import compute_relplus, decode_stanford_depth
from relplus.spec import RELPLUS_SPEC, RELPLUS_SPEC_SHA256


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def validate(run_dir):
    run = Path(run_dir).resolve()
    audit = json.loads((run / "data_reports" / "data_audit.json").read_text())
    dataset_root = audit["dataset_root"]
    selected = audit.get("selected_samples")
    if not isinstance(selected, list) or not selected:
        raise ValueError("data audit has no selected samples for semantic validation")

    rows = []
    matched = 0
    for selected_row in selected:
        sample_id = selected_row.get("sample_id")
        row = {"sample_id": sample_id, "exact_match": False}
        try:
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("invalid sample identifier")
            paths = resolve_sample_paths(dataset_root, sample_id)
            raw = cv2.imread(paths["depth"], cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint16:
                raise ValueError("expected uint16 depth")
            camera = load_camera_metadata(paths["pose"])
            depth, valid_depth = decode_stanford_depth(raw)
            rel_native, auxiliary = compute_relplus(depth, valid_depth, camera)
            expected = cv2.resize(rel_native, (480, 480), interpolation=cv2.INTER_LINEAR)
            valid_output = cv2.resize(
                auxiliary["valid"].astype(np.uint8),
                (480, 480),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            expected[~valid_output] = 255
            expected = np.ascontiguousarray(expected, dtype=np.uint8)
            cached = read_relplus_png(str(run / "relplus_cache" / (sample_id + ".png")))
            if cached.shape != expected.shape or cached.dtype != expected.dtype:
                raise ValueError(
                    "cache shape/dtype mismatch: {} {}".format(cached.shape, cached.dtype)
                )
            mismatch = cached != expected
            row.update(
                {
                    "exact_match": bool(not np.any(mismatch)),
                    "mismatched_values_by_channel": [
                        int(np.count_nonzero(mismatch[..., channel])) for channel in range(3)
                    ],
                    "max_abs_difference_by_channel": [
                        int(
                            np.max(
                                np.abs(
                                    cached[..., channel].astype(np.int16)
                                    - expected[..., channel].astype(np.int16)
                                )
                            )
                        )
                        for channel in range(3)
                    ],
                    "valid_mask_exact": bool(
                        np.array_equal(
                            ~np.all(cached == 255, axis=-1),
                            ~np.all(expected == 255, axis=-1),
                        )
                    ),
                }
            )
            matched += int(row["exact_match"])
        except Exception as error:
            row["error"] = "{}: {}".format(type(error).__name__, error)
        rows.append(row)

    passed = matched == len(rows)
    return {
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run),
        "representation_semantics": RELPLUS_SPEC["representation_semantics"],
        "representation_version": RELPLUS_SPEC["representation_version"],
        "representation_spec_sha256": RELPLUS_SPEC_SHA256,
        "sample_count": len(rows),
        "matched_samples": matched,
        "all_exact": passed,
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    report_path = args.run_dir.resolve() / "data_reports" / "semantics_validation.json"
    exit_path = args.run_dir.resolve() / "status" / "semantics_validation.exitcode"
    try:
        report = validate(args.run_dir)
    except Exception as error:
        report = {
            "status": "failed",
            "exit_code": 1,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(args.run_dir.resolve()),
            "all_exact": False,
            "sample_count": 0,
            "matched_samples": 0,
            "error": "{}: {}".format(type(error).__name__, error),
        }
    atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write(exit_path, "{}\n".format(report["exit_code"]))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
