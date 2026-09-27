#!/usr/bin/env python3
"""Compare two full epoch-200 evaluator artifact directories exactly."""

import argparse
import csv
import filecmp
import json
from pathlib import Path


EXPECTED_METRICS = {
    "mIoU_percent": 59.58536567805432,
    "pixel_accuracy_percent": 81.85393152664864,
    "mean_accuracy_percent": 67.64220937148674,
}
METRIC_FIELDS = (
    "mIoU",
    "pixel_accuracy",
    "mean_accuracy",
    "mIoU_percent",
    "pixel_accuracy_percent",
    "mean_accuracy_percent",
    "per_class_iou",
    "per_class_iou_percent",
    "valid_pixel_count",
)
EXACT_SOURCE_PATHS = (
    "tools/eval_rel_plus_v2_3_full.py",
    "engine/relplus_evaluator.py",
    "utils/metric.py",
    "models",
    "dataloader/dataloader.py",
    "dataloader/samplers.py",
)


def _csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_rows(old_root, new_root):
    rows = []
    for relative in EXACT_SOURCE_PATHS:
        old = old_root / relative
        new = new_root / relative
        if old.is_dir() and new.is_dir():
            comparison = filecmp.dircmp(str(old), str(new))
            stack = [comparison]
            different = []
            while stack:
                current = stack.pop()
                different.extend(current.left_only)
                different.extend(current.right_only)
                different.extend(current.diff_files)
                different.extend(current.funny_files)
                stack.extend(current.subdirs.values())
            exact = not different
        else:
            exact = old.is_file() and new.is_file() and filecmp.cmp(
                str(old), str(new), shallow=False
            )
            different = [] if exact else [relative]
        rows.append(
            {"path": relative, "exact": exact, "difference_count": len(different)}
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--old-code-root", required=True, type=Path)
    parser.add_argument("--new-code-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-reference-world-size", type=int, default=8)
    parser.add_argument("--expected-candidate-world-size", type=int, default=8)
    args = parser.parse_args()
    reference_dir = args.reference_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    reference = json.loads(
        (reference_dir / "metrics.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (candidate_dir / "metrics.json").read_text(encoding="utf-8")
    )
    metric_exact = {field: reference.get(field) == candidate.get(field) for field in METRIC_FIELDS}
    target_exact = {
        field: candidate.get(field) == expected
        for field, expected in EXPECTED_METRICS.items()
    }
    confusion_exact = _csv_rows(
        reference_dir / "confusion_matrix.csv"
    ) == _csv_rows(candidate_dir / "confusion_matrix.csv")
    per_class_exact = _csv_rows(
        reference_dir / "per_class_iou.csv"
    ) == _csv_rows(candidate_dir / "per_class_iou.csv")
    reference_manifest = _csv_rows(reference_dir / "evaluation_manifest.csv")
    candidate_manifest = _csv_rows(candidate_dir / "evaluation_manifest.csv")
    reference_ids = [row["sample_id"] for row in reference_manifest]
    candidate_ids = [row["sample_id"] for row in candidate_manifest]
    manifest_gate = {
        "reference_count": len(reference_ids),
        "candidate_count": len(candidate_ids),
        "reference_unique_count": len(set(reference_ids)),
        "candidate_unique_count": len(set(candidate_ids)),
        "ordered_sample_ids_exact": reference_ids == candidate_ids,
        "sample_id_sets_exact": set(reference_ids) == set(candidate_ids),
    }
    source_rows = _source_rows(
        args.old_code_root.resolve(), args.new_code_root.resolve()
    )
    protocol_fields = (
        "integration_protocol_id",
        "representation_protocol_id",
        "checkpoint_epoch",
        "evaluation_sample_count",
        "class_count",
        "ignore_index",
        "eval_scale_array",
        "eval_flip",
        "eval_crop_size",
        "eval_align_corners",
    )
    protocol_exact = {
        field: reference.get(field) == candidate.get(field)
        for field in protocol_fields
    }
    passed = (
        reference.get("status") == "PASS"
        and candidate.get("status") == "PASS"
        and reference.get("world_size") == args.expected_reference_world_size
        and candidate.get("world_size") == args.expected_candidate_world_size
        and all(metric_exact.values())
        and all(target_exact.values())
        and confusion_exact
        and per_class_exact
        and manifest_gate["reference_count"] == 17593
        and manifest_gate["candidate_count"] == 17593
        and manifest_gate["reference_unique_count"] == 17593
        and manifest_gate["candidate_unique_count"] == 17593
        and manifest_gate["sample_id_sets_exact"]
        and all(protocol_exact.values())
        and all(row["exact"] for row in source_rows)
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol": "RELPLUS_EPOCH200_EVALUATOR_EQUIVALENCE_V1",
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "reference_world_size": reference.get("world_size"),
        "candidate_world_size": candidate.get("world_size"),
        "metric_exact": metric_exact,
        "expected_metric_exact": target_exact,
        "confusion_matrix_exact": confusion_exact,
        "per_class_iou_exact": per_class_exact,
        "manifest": manifest_gate,
        "protocol_exact": protocol_exact,
        "evaluator_source_files": source_rows,
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
