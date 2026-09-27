#!/usr/bin/env python3
"""Validate 1/2/8-rank no-pad ownership and exact confusion aggregation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloader.samplers import DistributedEvalSamplerNoPad
from engine.relplus_evaluator import aggregate_rank_evaluations
from utils.metric import hist_info


def _reports(world_size, labels, predictions, class_count):
    dataset = list(range(len(labels)))
    reports = []
    for rank in range(world_size):
        sampler = DistributedEvalSamplerNoPad(
            dataset, num_replicas=world_size, rank=rank
        )
        confusion = np.zeros((class_count, class_count), dtype=np.int64)
        owned = []
        for index in sampler:
            hist, _, _ = hist_info(
                class_count, predictions[index], labels[index]
            )
            confusion += hist.astype(np.int64)
            owned.append("synthetic_{:03d}".format(index))
        reports.append(
            {
                "rank": rank,
                "sample_count": len(owned),
                "owned_sample_ids": owned,
                "confusion_matrix": confusion.tolist(),
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(230819)
    labels = [rng.integers(0, 13, size=(19, 23), dtype=np.uint8) for _ in range(37)]
    predictions = [rng.integers(0, 13, size=value.shape, dtype=np.uint8) for value in labels]
    outcomes = {}
    for world_size in (1, 2, 8):
        reports = _reports(world_size, labels, predictions, 13)
        confusion, count, owned = aggregate_rank_evaluations(reports)
        outcomes[str(world_size)] = {
            "sample_count": count,
            "unique_owned_sample_count": len(set(owned)),
            "confusion_matrix": confusion.tolist(),
            "rank_sample_counts": [report["sample_count"] for report in reports],
        }
    reference = np.asarray(outcomes["1"]["confusion_matrix"], dtype=np.int64)
    exact = all(
        np.array_equal(
            reference,
            np.asarray(outcomes[str(world_size)]["confusion_matrix"], dtype=np.int64),
        )
        for world_size in (2, 8)
    )
    report = {
        "status": "PASS" if exact else "FAIL",
        "claim": "synthetic evaluator distribution plumbing only",
        "scientific_metric_reported": False,
        "sample_count": len(labels),
        "class_count": 13,
        "single_equals_2_rank": exact,
        "single_equals_8_rank": exact,
        "outcomes": outcomes,
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "outcomes"},
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
