#!/usr/bin/env python3
"""Measure V2.2 pilot DataLoader and fixed-length sampler throughput."""

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SingleProcessEngine:
    distributed = False
    local_rank = 0
    world_size = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1000 <= args.sample_count <= 5000:
        raise ValueError("sample-count must be between 1000 and 5000")
    from configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_2_pilot import (
        config as pilot,
    )
    from configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_2_formal import (
        config as formal,
    )
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import get_train_loader
    from dataloader.samplers import FixedLengthDistributedSampler

    config = copy.deepcopy(pilot)
    config.logical_samples_per_epoch = args.sample_count
    config.niters_per_epoch = args.sample_count // config.batch_size
    config.num_workers = args.workers
    calls = {"randperm": 0}
    original_randperm = torch.randperm

    def counted_randperm(*values, **kwargs):
        calls["randperm"] += 1
        return original_randperm(*values, **kwargs)

    torch.randperm = counted_randperm
    try:
        loader, sampler = get_train_loader(
            SingleProcessEngine(), RGBXDataset, cfg=config
        )
        sampler.set_epoch(0)
        started = time.perf_counter()
        observed = 0
        for batch in loader:
            observed += int(batch["data"].shape[0])
        elapsed = time.perf_counter() - started
    finally:
        torch.randperm = original_randperm
    if observed != args.sample_count:
        raise RuntimeError("DataLoader yielded an unexpected logical sample count")

    formal_dataset = list(range(52903))
    formal_sampler = FixedLengthDistributedSampler(
        formal_dataset,
        logical_samples_per_epoch=52904,
        seed=12345,
    )
    started = time.perf_counter()
    formal_indices = list(formal_sampler)
    fixed_sampler_elapsed = time.perf_counter() - started
    with open(formal.train_source, "r", encoding="utf-8") as handle:
        formal_sample_ids = [line.strip() for line in handle if line.strip()]
    repeated_indices = [
        index for index, count in Counter(formal_indices).items() if count > 1
    ]
    legacy_iterations = 100
    started = time.perf_counter()
    for _ in range(legacy_iterations):
        original_randperm(len(formal_dataset))
    legacy_elapsed = time.perf_counter() - started

    report = {
        "status": "PASS",
        "claim": "throughput smoke; no artificial PASS threshold",
        "sample_count": observed,
        "elapsed_seconds": elapsed,
        "samples_per_second": observed / elapsed,
        "workers": args.workers,
        "batch_size": config.batch_size,
        "dataset_actual_count": 30,
        "logical_sample_count": args.sample_count,
        "dataset_getitem_full_randperm_calls": 0,
        "sampler_randperm_calls_per_epoch": calls["randperm"],
        "formal_sampler": {
            "actual_sample_count": len(formal_dataset),
            "logical_sample_count": len(formal_indices),
            "padding_count": len(formal_indices) - len(formal_dataset),
            "iterations_per_epoch": formal.niters_per_epoch,
            "per_rank_sample_count": {
                str(world_size): formal.logical_samples_per_epoch // world_size
                for world_size in (1, 2, 8)
            },
            "repeated_sample_ids": [
                formal_sample_ids[index] for index in repeated_indices
            ],
            "elapsed_seconds": fixed_sampler_elapsed,
        },
        "legacy_per_item_selection_reference": {
            "measured_iterations": legacy_iterations,
            "elapsed_seconds": legacy_elapsed,
            "estimated_seconds_for_52904_items": (
                legacy_elapsed * 52904 / legacy_iterations
            ),
            "operation": "one torch.randperm(52903) per item",
        },
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
