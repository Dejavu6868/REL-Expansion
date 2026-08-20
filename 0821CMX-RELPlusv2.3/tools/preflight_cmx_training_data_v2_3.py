#!/usr/bin/env python3
"""Full CMX RGB/label/REL+/mask preflight bound to one audited cache."""

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_full_relplus_cache import validate_ordered_split_lists
from tools.generate_full_relplus_cache import read_manifest, write_csv


REPRESENTATION_PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
INTEGRATION_PROTOCOL_ID = "CMX_RELPLUS_V2_3"
RESUME_FIELDS = (
    "protocol_id",
    "sample_id",
    "split",
    "rgb_path",
    "depth_path",
    "camera_metadata_path",
    "label_path",
    "dataset_profile",
    "rel_plus_path",
    "valid_mask_path",
)


def _mapping(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = {int(value) for value in payload["stored_ids"]}
    if 0 not in stored:
        raise ValueError("class mapping must explicitly contain unknown stored ID 0")
    if stored != set(range(14)):
        raise ValueError("class mapping stored IDs must be exactly 0..13")
    return payload, stored


def _resume_matches(row, previous):
    return all(
        str(row.get(field, "")) == str(previous.get(field, ""))
        for field in RESUME_FIELDS
    )


def _expected_cache_paths(cache_root, sample_id):
    relative = Path(str(sample_id) + ".png")
    root = Path(cache_root)
    return root / "RELPlus" / relative, root / "ValidMask" / relative


def _path_matches_sample(path, sample_id):
    expected = str(Path(str(sample_id) + ".png"))
    return str(path).replace("\\", "/").endswith(expected.replace("\\", "/"))


def audit_row(row, cache_root, valid_stored_ids):
    sample_id = row["sample_id"]
    reasons = []
    rel_path, mask_path = _expected_cache_paths(cache_root, sample_id)
    rgb = cv2.imread(row["rgb_path"], cv2.IMREAD_UNCHANGED)
    label = cv2.imread(row["label_path"], cv2.IMREAD_UNCHANGED)
    rel = cv2.imread(str(rel_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if row.get("protocol_id") != REPRESENTATION_PROTOCOL_ID:
        reasons.append("representation_protocol_id")
    for source_path in (row["rgb_path"], row["label_path"], rel_path, mask_path):
        if not _path_matches_sample(source_path, sample_id):
            reasons.append("sample_path_identity")
            break

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
        mapped[(label == 0) | (label == 255)] = 255
        model_values = {int(value) for value in np.unique(mapped)}
        if not model_values.issubset(set(range(13)) | {255}):
            reasons.append("label_model_ids")
    if rel is None or rel.shape != (480, 480, 3) or rel.dtype != np.uint8:
        reasons.append("rel_plus_decode_shape_dtype")
    if mask is None or mask.shape != (480, 480) or mask.dtype != np.uint8:
        reasons.append("valid_mask_decode_shape_dtype")
    elif not set(np.unique(mask).tolist()).issubset({0, 255}):
        reasons.append("valid_mask_binary")

    shapes = []
    for value in (rgb, label, rel, mask):
        if value is not None:
            shapes.append(value.shape[:2])
    if shapes and any(shape != shapes[0] for shape in shapes[1:]):
        reasons.append("spatial_mismatch")

    result = dict(row)
    result.update(
        {
            "rel_plus_path": str(rel_path.resolve()),
            "valid_mask_path": str(mask_path.resolve()),
            "status": "PASS" if not reasons else "FAIL",
            "reasons": "|".join(sorted(set(reasons))),
            "stored_label_ids": json.dumps(sorted(stored_values)),
            "model_label_ids": json.dumps(sorted(model_values)),
            "resume_action": "SCANNED",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--resolved-manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--class-mapping", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-source", type=Path)
    parser.add_argument("--eval-source", type=Path)
    parser.add_argument(
        "--config-module",
        default=(
            "configs.stanford2d3d_s2d."
            "cmx_mit_b2_rel_plus_v2_3_formal"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = importlib.import_module(args.config_module).config
    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    mapping, valid_ids = _mapping(args.class_mapping)
    cache_root = args.cache_root.resolve()
    train_source = (args.train_source or cache_root / "train.txt").resolve()
    eval_source = (args.eval_source or cache_root / "test.txt").resolve()
    split_failures = validate_ordered_split_lists(rows, train_source, eval_source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "cmx_training_data_preflight.csv"
    previous = {}
    if args.resume and output_csv.is_file():
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            previous = {row["sample_id"]: row for row in csv.DictReader(handle)}

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
            results.append(audit_row(row, cache_root, valid_ids))
    if results:
        write_csv(output_csv, results, list(results[0]))
    failure_count = sum(row["status"] != "PASS" for row in results)
    failure_count += len(split_failures)
    summary = {
        "status": "PASS" if failure_count == 0 else "FAIL",
        "integration_protocol_id": config.integration_protocol_id,
        "representation_protocol_id": config.representation_protocol_id,
        "cache_root": str(cache_root),
        "rel_plus_root": str((cache_root / "RELPlus").resolve()),
        "valid_mask_root": str((cache_root / "ValidMask").resolve()),
        "manifest_path": str(args.manifest.resolve()),
        "resolved_manifest_path": str(args.resolved_manifest.resolve()),
        "train_source": str(train_source),
        "eval_source": str(eval_source),
        "manifest_count": len(rows),
        "sample_count": len(results),
        "train_count": sum(row.get("split") == "train" for row in rows),
        "test_count": sum(row.get("split") == "test" for row in rows),
        "failure_count": failure_count,
        "split_identity_failure_count": len(split_failures),
        "resume_reused_count": reused,
        "resume_rescanned_count": len(results) - reused,
        "all_samples_decoded_this_run": reused == 0,
        "resume_contract_fields": list(RESUME_FIELDS),
        "class_mapping": str(args.class_mapping.resolve()),
        "class_mapping_loader_transform": mapping.get("loader_transform"),
        "file_hash_written": False,
    }
    (args.output_dir / "cmx_training_data_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
