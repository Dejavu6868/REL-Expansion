#!/usr/bin/env python3
"""Audit REL+ cache structure and regenerate selected samples byte-for-byte."""

import argparse
import csv
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rel_plus.generator import generate_rel_plus_v2_1
from rel_plus.profiles import STANFORD_S2D_PROFILE
from rel_plus.stanford_s2d import load_canonical_frame
from tools.generate_full_relplus_cache import (
    REPRESENTATION_PROTOCOL_ID,
    read_manifest,
    write_csv,
)


def _failure(sample_id, reason, detail=""):
    return {
        "sample_id": str(sample_id),
        "status": "FAIL",
        "reason": reason,
        "detail": detail,
    }


def _expected_path(cache_root, kind, sample_id):
    return Path(cache_root) / kind / (str(sample_id) + ".png")


def find_duplicate_sample_ids(sample_ids):
    """Return duplicate IDs in O(N) counting time."""
    counts = Counter(sample_ids)
    return sorted(sample_id for sample_id, count in counts.items() if count > 1)


def audit_cache_rows(
    rows,
    cache_root,
    *,
    expected_shape=(480, 480),
    integration_protocol_id="CMX_RELPLUS_V2_3"
):
    rows = list(rows)
    cache_root = Path(cache_root)
    failures = []
    sample_ids = [row.get("sample_id", "") for row in rows]
    duplicates = find_duplicate_sample_ids(sample_ids)
    for sample_id in duplicates:
        failures.append(_failure(sample_id, "duplicate_sample_id"))
    train_ids = {row["sample_id"] for row in rows if row.get("split") == "train"}
    test_ids = {row["sample_id"] for row in rows if row.get("split") == "test"}
    for sample_id in sorted(train_ids & test_ids):
        failures.append(_failure(sample_id, "train_test_sample_overlap"))
    for row in rows:
        sample_id = row.get("sample_id", "")
        if row.get("protocol_id") != REPRESENTATION_PROTOCOL_ID:
            failures.append(_failure(sample_id, "protocol_id"))
        rel_path = _expected_path(cache_root, "RELPlus", sample_id)
        mask_path = _expected_path(cache_root, "ValidMask", sample_id)
        if not rel_path.is_file():
            failures.append(_failure(sample_id, "rel_plus_missing", str(rel_path)))
            rel = None
        else:
            rel = cv2.imread(str(rel_path), cv2.IMREAD_UNCHANGED)
            if rel is None:
                failures.append(_failure(sample_id, "rel_plus_decode", str(rel_path)))
            else:
                if rel.shape != tuple(expected_shape) + (3,):
                    failures.append(
                        _failure(sample_id, "rel_plus_shape_or_channels", str(rel.shape))
                    )
                if rel.dtype != np.uint8:
                    failures.append(_failure(sample_id, "rel_plus_dtype", str(rel.dtype)))
        if not mask_path.is_file():
            failures.append(_failure(sample_id, "valid_mask_missing", str(mask_path)))
            mask = None
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                failures.append(_failure(sample_id, "valid_mask_decode", str(mask_path)))
            else:
                if mask.shape != tuple(expected_shape):
                    failures.append(_failure(sample_id, "valid_mask_shape", str(mask.shape)))
                if mask.dtype != np.uint8:
                    failures.append(_failure(sample_id, "valid_mask_dtype", str(mask.dtype)))
                elif not set(np.unique(mask).tolist()).issubset({0, 255}):
                    failures.append(_failure(sample_id, "valid_mask_binary"))
        if (
            rel is not None
            and mask is not None
            and rel.shape == tuple(expected_shape) + (3,)
            and mask.shape == tuple(expected_shape)
            and rel.dtype == np.uint8
            and mask.dtype == np.uint8
        ):
            invalid = mask == 0
            if np.any(rel[invalid] != 255):
                failures.append(_failure(sample_id, "invalid_storage_relation"))

    expected_rel = {
        str(Path(sample_id + ".png")) for sample_id in sample_ids
    }
    actual_counts = {}
    for kind, reason in (("RELPlus", "extra_rel_plus"), ("ValidMask", "extra_valid_mask")):
        directory = cache_root / kind
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*.png")
        } if directory.is_dir() else set()
        actual_counts[kind] = len(actual)
        expected = expected_rel
        for relative in sorted(actual - expected):
            failures.append(_failure(relative[:-4], reason, relative))

    split_paths = {}
    for split in ("train", "test"):
        for field in ("rgb_path", "label_path", "depth_path", "camera_metadata_path"):
            split_paths[(split, field)] = {
                row.get(field) for row in rows if row.get("split") == split and row.get(field)
            }
    for field in ("rgb_path", "label_path", "depth_path", "camera_metadata_path"):
        overlap = split_paths[("train", field)] & split_paths[("test", field)]
        for value in sorted(overlap):
            failures.append(_failure(value, "train_test_path_overlap", field))

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "integration_protocol_id": integration_protocol_id,
        "representation_protocol_id": REPRESENTATION_PROTOCOL_ID,
        "manifest_count": len(rows),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "unique_sample_count": len(set(sample_ids)),
        "expected_rel_plus_count": len(sample_ids),
        "expected_valid_mask_count": len(sample_ids),
        "rel_plus_file_count": actual_counts["RELPlus"],
        "valid_mask_file_count": actual_counts["ValidMask"],
        "failure_count": len(failures),
        "file_hash_written": False,
    }
    return summary, failures


RISK_FIELDS = (
    "depth_invalid_ratio",
    "normal_quality_ratio",
    "gravity_alignment_angle_deg",
)
FORMAL_AREAS = (
    "area_1",
    "area_2",
    "area_3",
    "area_4",
    "area_5a",
    "area_5b",
    "area_6",
)


def merge_risk_metrics(rows, risk_rows):
    """Attach full-preflight risk fields by exact sample ID."""
    risk_ids = [row.get("sample_id", "") for row in risk_rows]
    duplicates = find_duplicate_sample_ids(risk_ids)
    if duplicates:
        raise ValueError(
            "risk metrics contain duplicate sample IDs: {}".format(
                duplicates[:10]
            )
        )
    by_id = {row.get("sample_id", ""): row for row in risk_rows}
    merged = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        source = by_id.get(sample_id)
        if source is None:
            raise ValueError("risk metrics missing sample_id {}".format(sample_id))
        value = dict(row)
        for field in RISK_FIELDS:
            if source.get(field) in (None, ""):
                raise ValueError(
                    "risk metrics missing {} for {}".format(field, sample_id)
                )
            value[field] = source[field]
        merged.append(value)
    return merged


def _metric(row, field):
    value = row.get(field)
    if value in (None, ""):
        raise ValueError("{} is required for {}".format(field, row.get("sample_id")))
    return float(value)


def _quantile_row(values, field, quantile):
    ordered = sorted(values, key=lambda row: (_metric(row, field), row["sample_id"]))
    index = int(round((len(ordered) - 1) * float(quantile)))
    return ordered[index]


def select_regeneration_rows(rows, count, *, seed=2303):
    """Select ten deterministic high/median/low-risk rows per formal area."""
    rows = list(rows)
    if count < 0 or count > len(rows):
        raise ValueError("regeneration count is outside manifest range")
    if count != len(FORMAL_AREAS) * 10:
        raise ValueError("formal risk regeneration requires 70 rows")
    for row in rows:
        for field in RISK_FIELDS:
            _metric(row, field)

    groups = {area: [] for area in FORMAL_AREAS}
    for row in rows:
        area = row.get("area") or row.get("area_group")
        if area in groups:
            groups[area].append(row)
    selected = []
    for area_index, area in enumerate(FORMAL_AREAS):
        values = groups[area]
        if len(values) < 10:
            raise ValueError("{} has fewer than ten risk rows".format(area))
        invalid = sorted(
            values,
            key=lambda row: (_metric(row, "depth_invalid_ratio"), row["sample_id"]),
        )
        normal = sorted(
            values,
            key=lambda row: (_metric(row, "normal_quality_ratio"), row["sample_id"]),
        )
        invalid_median = _metric(
            _quantile_row(values, "depth_invalid_ratio", 0.5),
            "depth_invalid_ratio",
        )
        invalid_p90 = _metric(
            _quantile_row(values, "depth_invalid_ratio", 0.9),
            "depth_invalid_ratio",
        )
        normal_median = _metric(
            _quantile_row(values, "normal_quality_ratio", 0.5),
            "normal_quality_ratio",
        )
        candidate_orders = [
            ("invalid_low", invalid),
            (
                "invalid_median",
                sorted(
                    values,
                    key=lambda row: (
                        abs(_metric(row, "depth_invalid_ratio") - invalid_median),
                        row["sample_id"],
                    ),
                ),
            ),
            (
                "invalid_p90",
                sorted(
                    values,
                    key=lambda row: (
                        abs(_metric(row, "depth_invalid_ratio") - invalid_p90),
                        row["sample_id"],
                    ),
                ),
            ),
            ("invalid_high", list(reversed(invalid))),
            ("normal_quality_low", normal),
            (
                "normal_quality_median",
                sorted(
                    values,
                    key=lambda row: (
                        abs(_metric(row, "normal_quality_ratio") - normal_median),
                        row["sample_id"],
                    ),
                ),
            ),
            ("normal_quality_high", list(reversed(normal))),
            (
                "gravity_tilt_large",
                sorted(
                    values,
                    key=lambda row: (
                        -abs(_metric(row, "gravity_alignment_angle_deg") - 90.0),
                        row["sample_id"],
                    ),
                ),
            ),
        ]
        chosen = []
        chosen_ids = set()
        for reason, ordered_candidates in candidate_orders:
            row = next(
                item
                for item in ordered_candidates
                if item["sample_id"] not in chosen_ids
            )
            value = dict(row)
            value["selection_reason"] = reason
            chosen.append(value)
            chosen_ids.add(row["sample_id"])

        used_room_cameras = {
            (row.get("room", ""), row.get("camera", "")) for row in chosen
        }
        diverse = sorted(
            (row for row in values if row["sample_id"] not in chosen_ids),
            key=lambda row: (
                (row.get("room", ""), row.get("camera", ""))
                in used_room_cameras,
                row.get("room", ""),
                row.get("camera", ""),
                row["sample_id"],
            ),
        )
        if diverse:
            value = dict(diverse[0])
            value["selection_reason"] = "room_camera_diversity"
            chosen.append(value)
            chosen_ids.add(value["sample_id"])

        remaining = [row for row in values if row["sample_id"] not in chosen_ids]
        rng = random.Random(int(seed) + area_index)
        if remaining and len(chosen) < 10:
            row = rng.choice(sorted(remaining, key=lambda item: item["sample_id"]))
            value = dict(row)
            value["selection_reason"] = "fixed_random"
            chosen.append(value)
            chosen_ids.add(value["sample_id"])

        fill = sorted(
            (row for row in values if row["sample_id"] not in chosen_ids),
            key=lambda row: (
                -_metric(row, "depth_invalid_ratio"),
                _metric(row, "normal_quality_ratio"),
                -abs(_metric(row, "gravity_alignment_angle_deg") - 90.0),
                row["sample_id"],
            ),
        )
        for row in fill:
            if len(chosen) == 10:
                break
            value = dict(row)
            value["selection_reason"] = "risk_fill"
            chosen.append(value)
            chosen_ids.add(value["sample_id"])
        if len(chosen) != 10:
            raise RuntimeError("failed to select ten unique rows for {}".format(area))
        selected.extend(chosen)
    return selected


def _read_sample_ids(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def validate_ordered_split_lists(rows, train_source, eval_source):
    """Compare ordered split-list identities to the frozen manifest."""
    expected_train = [row["sample_id"] for row in rows if row.get("split") == "train"]
    expected_test = [row["sample_id"] for row in rows if row.get("split") == "test"]
    actual_train = _read_sample_ids(train_source)
    actual_test = _read_sample_ids(eval_source)
    failures = []
    if actual_train != expected_train:
        failures.append(
            _failure("__train_source__", "train_ordered_sample_ids_mismatch")
        )
    if actual_test != expected_test:
        failures.append(
            _failure("__eval_source__", "test_ordered_sample_ids_mismatch")
        )
    if len(actual_train) != len(set(actual_train)):
        failures.append(_failure("__train_source__", "train_duplicate_sample_id"))
    if len(actual_test) != len(set(actual_test)):
        failures.append(_failure("__eval_source__", "test_duplicate_sample_id"))
    if set(actual_train) & set(actual_test):
        failures.append(_failure("__splits__", "train_test_sample_overlap"))
    if set(actual_train) | set(actual_test) != {
        row["sample_id"] for row in rows
    }:
        failures.append(_failure("__splits__", "split_manifest_union_mismatch"))
    return failures


def regenerate_and_compare(rows, cache_root):
    results = []
    for row in rows:
        sample_id = row["sample_id"]
        result = {
            "sample_id": sample_id,
            "area": row.get("area", ""),
            "area_group": row.get("area_group", ""),
            "room": row.get("room", ""),
            "camera": row.get("camera", ""),
            "selection_reason": row.get("selection_reason", ""),
            "depth_invalid_ratio": row.get("depth_invalid_ratio", ""),
            "normal_quality_ratio": row.get("normal_quality_ratio", ""),
            "gravity_alignment_angle_deg": row.get(
                "gravity_alignment_angle_deg", ""
            ),
        }
        try:
            raw_depth, camera, _ = load_canonical_frame(
                row["depth_path"],
                row["camera_metadata_path"],
                dataset_profile=STANFORD_S2D_PROFILE,
            )
            generated = generate_rel_plus_v2_1(raw_depth, camera)
            cached = cv2.imread(
                str(_expected_path(cache_root, "RELPlus", sample_id)),
                cv2.IMREAD_UNCHANGED,
            )
            difference = np.abs(
                generated.astype(np.int16) - cached.astype(np.int16)
            )
            result.update(
                {
                    "status": "PASS" if not np.any(difference) else "FAIL",
                    "changed_pixels": int(
                        np.count_nonzero(np.any(difference != 0, axis=2))
                    ),
                    "changed_channels": int(np.count_nonzero(difference)),
                    "max_difference": int(difference.max()),
                }
            )
        except Exception as error:
            result.update(
                {
                    "status": "FAIL",
                    "changed_pixels": "",
                    "changed_channels": "",
                    "max_difference": "",
                    "error": "{}: {}".format(type(error).__name__, error),
                }
            )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config-module",
        default=(
            "configs.stanford2d3d_s2d."
            "cmx_mit_b2_rel_plus_v2_3_formal"
        ),
    )
    parser.add_argument("--regeneration-count", type=int, default=70)
    parser.add_argument(
        "--risk-metrics",
        type=Path,
        default=Path(
            "/data/zhuzhaoziao/RELPlus/outputs/"
            "REL_plus_v2_1_implementation/preflight/full_preflight.csv"
        ),
    )
    parser.add_argument("--selection-seed", type=int, default=2303)
    args = parser.parse_args()
    config = importlib.import_module(args.config_module).config
    rows = read_manifest(args.manifest)
    summary, failures = audit_cache_rows(
        rows,
        args.cache_root,
        integration_protocol_id=config.integration_protocol_id,
    )
    cache_root = args.cache_root.resolve()
    train_source = (cache_root / "train.txt").resolve()
    eval_source = (cache_root / "test.txt").resolve()
    try:
        failures.extend(
            validate_ordered_split_lists(rows, train_source, eval_source)
        )
    except OSError as error:
        failures.append(_failure("__splits__", "split_list_unreadable", str(error)))
    if len(rows) > 36:
        expected_counts = {
            "manifest_count": config.num_train_imgs + config.num_eval_imgs,
            "train_count": config.num_train_imgs,
            "test_count": config.num_eval_imgs,
        }
        for field, expected in expected_counts.items():
            if summary[field] != expected:
                failures.append(
                    _failure(
                        "__manifest__",
                        "{}_mismatch".format(field),
                        "expected {}, found {}".format(expected, summary[field]),
                    )
                )
    risk_rows = read_manifest(args.risk_metrics)
    risk_bound_rows = merge_risk_metrics(rows, risk_rows)
    selected = select_regeneration_rows(
        risk_bound_rows,
        args.regeneration_count,
        seed=args.selection_seed,
    )
    regeneration = regenerate_and_compare(selected, args.cache_root)
    regeneration_failures = [row for row in regeneration if row["status"] != "PASS"]
    for row in regeneration_failures:
        failures.append(
            _failure(row["sample_id"], "sample_regeneration", row.get("error", "byte mismatch"))
        )
    summary["regeneration_count"] = len(regeneration)
    summary["regeneration_failure_count"] = len(regeneration_failures)
    summary["failure_count"] = len(failures)
    summary["status"] = "PASS" if not failures else "FAIL"
    summary["review_notes"] = (
        "No separate manual review state; optional review notes do not alter PASS/FAIL."
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path = (
        args.output_dir / "cache_manifest_resolved.csv"
    ).resolve()
    summary.update(
        {
            "cache_root": str(cache_root),
            "rel_plus_root": str((cache_root / "RELPlus").resolve()),
            "valid_mask_root": str((cache_root / "ValidMask").resolve()),
            "manifest_path": str(args.manifest.resolve()),
            "resolved_manifest_path": str(resolved_manifest_path),
            "train_source": str(train_source),
            "eval_source": str(eval_source),
            "risk_metrics_path": str(args.risk_metrics.resolve()),
            "selection_seed": args.selection_seed,
        }
    )
    write_csv(
        args.output_dir / "cache_audit_failures.csv",
        failures,
        ["sample_id", "status", "reason", "detail"],
    )
    regeneration_fields = [
        "sample_id",
        "area",
        "depth_invalid_ratio",
        "normal_quality_ratio",
        "gravity_alignment_angle_deg",
        "area_group",
        "room",
        "camera",
        "selection_reason",
        "status",
        "changed_pixels",
        "changed_channels",
        "max_difference",
        "error",
    ]
    write_csv(
        args.output_dir / "cache_audit_sample_regeneration.csv",
        regeneration,
        regeneration_fields,
    )
    resolved = []
    failed_ids = {row["sample_id"] for row in failures}
    for row in rows:
        value = dict(row)
        value["rel_plus_path"] = str(
            _expected_path(args.cache_root, "RELPlus", row["sample_id"])
        )
        value["valid_mask_path"] = str(
            _expected_path(args.cache_root, "ValidMask", row["sample_id"])
        )
        value["audit_status"] = (
            "FAIL" if row["sample_id"] in failed_ids else "PASS"
        )
        resolved.append(value)
    resolved_fields = list(dict.fromkeys(list(rows[0]) + [
        "rel_plus_path", "valid_mask_path", "audit_status",
    ]))
    write_csv(
        resolved_manifest_path,
        resolved,
        resolved_fields,
    )
    (args.output_dir / "cache_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
