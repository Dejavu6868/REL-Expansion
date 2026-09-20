#!/usr/bin/env python3
"""Trace three independently executed real RGBD/HHA/REL+ DataLoaders."""

import argparse
import copy
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pad_fields(transform):
    cropped_height = min(
        transform.output_height, transform.scaled_height - transform.crop_top
    )
    cropped_width = min(
        transform.output_width, transform.scaled_width - transform.crop_left
    )
    pad_height = max(0, transform.output_height - cropped_height)
    pad_width = max(0, transform.output_width - cropped_width)
    return {
        "pad_top": pad_height // 2,
        "pad_bottom": pad_height - pad_height // 2,
        "pad_left": pad_width // 2,
        "pad_right": pad_width - pad_width // 2,
    }


def _trace_arm(arm, base, source, pilot_cache, limit, epoch, rank, world_size):
    from configs.stanford2d3d_s2d.common import DATASET_ROOT
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.data_setting import build_data_setting
    from dataloader.dataloader import TrainPre
    from dataloader.profiles import author_epoch_seed, transform_trace_row
    from dataloader.samplers import FixedLengthDistributedSampler

    config = copy.deepcopy(base)
    config.train_source = str(source)
    config.eval_source = str(source)
    config.num_workers = 0
    config.batch_size = 1
    if arm == "rgbd":
        config.x_mode = "standard"
        config.x_root_folder = str(Path(DATASET_ROOT) / "RawDepth")
        config.x_is_single_channel = True
        config.representation_protocol_id = "CMX_RGBD_RAWDEPTH_480"
    elif arm == "hha":
        config.x_mode = "standard"
        config.x_root_folder = str(Path(DATASET_ROOT) / "HHA")
        config.x_is_single_channel = False
        config.representation_protocol_id = "CMX_HHA_480"
    else:
        config.x_mode = "rel_plus_v2_1"
        config.x_root_folder = str(pilot_cache / "RELPlus")
        config.x_valid_root_folder = str(pilot_cache / "ValidMask")
        config.x_is_single_channel = False
        config.representation_protocol_id = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
    setting = build_data_setting(config, split="train")
    rng = random.Random(author_epoch_seed(config.seed, epoch, rank))
    preprocess = TrainPre(config.norm_mean, config.norm_std, cfg=config, rng=rng)
    dataset = RGBXDataset(setting, "train", preprocess)
    logical = max(len(dataset), limit * world_size)
    if logical % world_size:
        logical += world_size - logical % world_size
    sampler = FixedLengthDistributedSampler(
        dataset,
        logical_samples_per_epoch=logical,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    sampler.set_epoch(epoch)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        sampler=sampler,
    )
    rows = []
    occurrences = defaultdict(int)
    for batch in loader:
        sample_id = batch["fn"][0]
        occurrence = occurrences[sample_id]
        occurrences[sample_id] += 1
        row = transform_trace_row(
            sample_id,
            epoch,
            rank,
            int(batch["worker_id"][0]),
            preprocess.last_transform,
        )
        row.update(_pad_fields(preprocess.last_transform))
        row["arm"] = arm
        row["occurrence"] = occurrence
        rows.append(row)
        if len(rows) == limit:
            break
    if len(rows) != limit:
        raise RuntimeError("trace arm did not yield requested rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--pilot-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    if not 50 <= args.limit <= 200:
        raise ValueError("trace limit must be between 50 and 200")
    manifest = _read_rows(args.pilot_manifest)
    sample_ids = [row["sample_id"] for row in manifest]
    if len(sample_ids) < 1 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("pilot manifest IDs must be nonempty and unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.output.parent / "three_arm_trace_samples.txt"
    source.write_text("".join(value + "\n" for value in sample_ids), encoding="utf-8")
    from configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_2_formal import (
        config as formal,
    )

    arms = ("rgbd", "hha", "rel_plus_v2_1")
    traces = {
        arm: _trace_arm(
            arm,
            formal,
            source,
            args.pilot_cache,
            args.limit,
            args.epoch,
            args.rank,
            args.world_size,
        )
        for arm in arms
    }
    comparison_fields = (
        "sample_id",
        "occurrence",
        "rank",
        "worker_id",
        "scale",
        "crop_top",
        "crop_left",
        "scaled_height",
        "scaled_width",
        "output_height",
        "output_width",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
    )
    mismatches = []
    reference = traces[arms[0]]
    for arm in arms[1:]:
        for index, (expected, actual) in enumerate(zip(reference, traces[arm])):
            for field in comparison_fields:
                if expected[field] != actual[field]:
                    mismatches.append(
                        {
                            "index": index,
                            "arm": arm,
                            "field": field,
                            "expected": expected[field],
                            "actual": actual[field],
                        }
                    )
    report = {
        "status": "PASS" if not mismatches else "FAIL",
        "claim": "three independently executed real DataLoader traces",
        "arms": list(arms),
        "trace_count_per_arm": args.limit,
        "epoch": args.epoch,
        "rank": args.rank,
        "world_size": args.world_size,
        "base_seed": formal.seed,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "traces": traces,
        "constructed_trace_copy_used": False,
        "file_hash_written": False,
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "traces"}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
