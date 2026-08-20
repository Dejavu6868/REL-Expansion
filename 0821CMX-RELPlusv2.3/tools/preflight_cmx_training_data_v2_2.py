#!/usr/bin/env python3
"""Decode-audit RGB, labels and REL+ cache using the dataset mapping."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_full_relplus_cache import read_manifest, write_csv


RESUME_FIELDS = (
    "protocol_id",
    "sample_id",
    "rgb_path",
    "depth_path",
    "camera_metadata_path",
    "label_path",
    "dataset_profile",
)


def _mapping(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = {int(value) for value in payload["stored_ids"]}
    if 0 not in stored:
        raise ValueError("class mapping must explicitly contain unknown stored ID 0")
    return payload, stored


def _resume_matches(row, previous):
    return all(str(row.get(field, "")) == str(previous.get(field, "")) for field in RESUME_FIELDS)


def audit_row(row, cache_root, valid_stored_ids):
    sample_id = row["sample_id"]
    reasons = []
    rgb = cv2.imread(row["rgb_path"], cv2.IMREAD_UNCHANGED)
    label = cv2.imread(row["label_path"], cv2.IMREAD_UNCHANGED)
    rel = cv2.imread(
        str(Path(cache_root) / "RELPlus" / (sample_id + ".png")),
        cv2.IMREAD_UNCHANGED,
    )
    mask = cv2.imread(
        str(Path(cache_root) / "ValidMask" / (sample_id + ".png")),
        cv2.IMREAD_UNCHANGED,
    )
    if rgb is None or rgb.shape != (480, 480, 3) or rgb.dtype != np.uint8:
        reasons.append("rgb_decode_shape_dtype")
    if label is None or label.shape != (480, 480):
        reasons.append("label_decode_shape")
        stored_values = set()
        model_values = set()
    else:
        stored_values = {int(value) for value in np.unique(label)}
        if not stored_values.issubset(valid_stored_ids):
            reasons.append("label_stored_ids")
        mapped = label.astype(np.int16) - 1
        mapped[label == 0] = 255
        mapped[label == 255] = 255
        model_values = {int(value) for value in np.unique(mapped)}
        if not model_values.issubset(set(range(13)) | {255}):
            reasons.append("label_model_ids")
    if rel is None or rel.shape != (480, 480, 3) or rel.dtype != np.uint8:
        reasons.append("rel_plus_decode_shape_dtype")
    if mask is None or mask.shape != (480, 480) or mask.dtype != np.uint8:
        reasons.append("valid_mask_decode_shape_dtype")
    if (
        rgb is not None
        and label is not None
        and rel is not None
        and not (rgb.shape[:2] == label.shape == rel.shape[:2])
    ):
        reasons.append("spatial_mismatch")
    result = dict(row)
    result.update(
        {
            "status": "PASS" if not reasons else "FAIL",
            "reasons": "|".join(reasons),
            "stored_label_ids": json.dumps(sorted(stored_values)),
            "model_label_ids": json.dumps(sorted(model_values)),
            "resume_action": "SCANNED",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--class-mapping", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    mapping, valid_ids = _mapping(args.class_mapping)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "cmx_training_data_preflight.csv"
    previous = {}
    if args.resume and output_csv.is_file():
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            previous = {
                row["sample_id"]: row for row in csv.DictReader(handle)
            }
    results = []
    reused = 0
    for row in rows:
        old = previous.get(row["sample_id"])
        if old and old.get("status") == "PASS" and _resume_matches(row, old):
            value = dict(old)
            value["resume_action"] = "REUSED_MATCHING_CONTRACT"
            results.append(value)
            reused += 1
        else:
            results.append(audit_row(row, args.cache_root, valid_ids))
    fieldnames = list(results[0])
    write_csv(output_csv, results, fieldnames)
    failure_count = sum(row["status"] != "PASS" for row in results)
    summary = {
        "status": "PASS" if failure_count == 0 else "FAIL",
        "sample_count": len(results),
        "failure_count": failure_count,
        "resume_reused_count": reused,
        "resume_rescanned_count": len(results) - reused,
        "resume_contract_fields": list(RESUME_FIELDS),
        "class_mapping": str(args.class_mapping),
        "class_mapping_loader_transform": mapping.get("loader_transform"),
        "file_hash_written": False,
    }
    (args.output_dir / "cmx_training_data_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
