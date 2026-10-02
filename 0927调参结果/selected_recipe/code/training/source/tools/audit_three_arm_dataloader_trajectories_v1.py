#!/usr/bin/env python3
"""Compare real 16-worker three-arm transform traces across epoch boundaries."""

import argparse
import copy
import importlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EPOCHS = (1, 2, 3, 10, 11)
ARMS = {
    "rel_plus": (
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_3_formal"
    ),
    "rgbd": "configs.stanford2d3d_s2d.cmx_mit_b2_rgbd_three_arm_v1",
    "hha": "configs.stanford2d3d_s2d.cmx_mit_b2_hha_three_arm_v1",
}
TRACE_FIELDS = (
    "sample_id",
    "occurrence",
    "epoch",
    "rank",
    "worker_id",
    "worker_seed",
    "scale",
    "scaled_height",
    "scaled_width",
    "crop_top",
    "crop_left",
    "output_height",
    "output_width",
    "pad_top",
    "pad_bottom",
    "pad_left",
    "pad_right",
    "horizontal_flip",
)


def _scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _trace_row(batch, *, epoch, occurrence):
    transform = batch["transform_trace"]
    row = {
        "sample_id": str(batch["fn"][0]),
        "occurrence": int(occurrence),
        "epoch": int(epoch),
        "rank": 0,
        "worker_id": int(_scalar(batch["worker_id"][0])),
        "worker_seed": str(batch["worker_seed"][0]),
    }
    for field in TRACE_FIELDS[6:]:
        value = _scalar(transform[field][0])
        if field == "scale":
            value = float(value)
        elif field == "horizontal_flip":
            value = bool(value)
        else:
            value = int(value)
        row[field] = value
    return row


def _write_progress(output, traces, completed_arms):
    payload = {
        "status": "RUNNING",
        "protocol": "THREE_ARM_REAL_DATALOADER_CROSS_EPOCH_TRACE_V1",
        "completed_arms": list(completed_arms),
        "epochs": list(EPOCHS),
        "rank": 0,
        "world_size": 8,
        "num_workers": 16,
        "head_count_per_epoch": 50,
        "tail_count_per_epoch": 50,
        "traces": traces,
        "file_hash_written": False,
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _trace_arm(module_name):
    import torch

    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import get_train_loader
    from utils.training_protocol import set_author_seed

    config = copy.deepcopy(importlib.import_module(module_name).config)
    config.num_workers = 16
    config.batch_size = 8
    config.emit_transform_trace = True
    engine = SimpleNamespace(world_size=8, local_rank=0, distributed=True)
    loader, sampler = get_train_loader(engine, RGBXDataset, cfg=config)
    rows = []
    for epoch in EPOCHS:
        set_author_seed(
            config.seed,
            epoch=epoch,
            local_rank=0,
            distributed=True,
        )
        sampler.set_epoch(epoch)
        first = []
        last = deque(maxlen=50)
        occurrences = defaultdict(int)
        count = 0
        for batch in loader:
            sample_id = str(batch["fn"][0])
            occurrence = occurrences[sample_id]
            occurrences[sample_id] += 1
            row = _trace_row(batch, epoch=epoch, occurrence=occurrence)
            if count < 50:
                row["epoch_position"] = "head"
                row["position"] = count
                first.append(row)
            tail_row = dict(row)
            tail_row["epoch_position"] = "tail"
            tail_row["position"] = count
            last.append(tail_row)
            count += 1
        if count != config.niters_per_epoch:
            raise RuntimeError(
                "DataLoader epoch length mismatch: expected {}, found {}".format(
                    config.niters_per_epoch, count
                )
            )
        rows.extend(first)
        rows.extend(last)
        del first, last
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    traces = {}
    for arm, module_name in ARMS.items():
        traces[arm] = _trace_arm(module_name)
        _write_progress(output, traces, traces.keys())
        print(
            json.dumps(
                {"arm": arm, "trace_count": len(traces[arm]), "status": "DONE"},
                ensure_ascii=False,
            ),
            flush=True,
        )

    reference = traces["rel_plus"]
    mismatches = []
    comparison_count = 0
    for arm in ("rgbd", "hha"):
        if len(traces[arm]) != len(reference):
            mismatches.append({"arm": arm, "reason": "trace_length"})
            continue
        for index, (expected, actual) in enumerate(zip(reference, traces[arm])):
            for field in TRACE_FIELDS + ("epoch_position", "position"):
                comparison_count += 1
                if expected[field] != actual[field] and len(mismatches) < 50:
                    mismatches.append(
                        {
                            "arm": arm,
                            "index": index,
                            "field": field,
                            "expected": expected[field],
                            "actual": actual[field],
                        }
                    )
    horizontal_flip_all_false = all(
        row["horizontal_flip"] is False
        for arm_rows in traces.values()
        for row in arm_rows
    )
    expected_rows = len(EPOCHS) * 100
    passed = (
        not mismatches
        and horizontal_flip_all_false
        and all(len(rows) == expected_rows for rows in traces.values())
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol": "THREE_ARM_REAL_DATALOADER_CROSS_EPOCH_TRACE_V1",
        "claim": "independently executed real Dataset/Sampler/DataLoader/TrainPre workers",
        "arms": list(ARMS),
        "epochs": list(EPOCHS),
        "rank": 0,
        "world_size": 8,
        "num_workers": 16,
        "worker_seed_policy": "author_default_no_explicit_worker_init",
        "head_count_per_epoch": 50,
        "tail_count_per_epoch": 50,
        "trace_count_per_arm": expected_rows,
        "comparison_field_count": len(TRACE_FIELDS) + 2,
        "element_comparison_count": comparison_count,
        "all_trace_fields_exact": not mismatches,
        "horizontal_flip_all_false": horizontal_flip_all_false,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches,
        "traces": traces,
        "constructed_trace_copy_used": False,
        "gpu_used": False,
        "backpropagation_executed": False,
        "optimizer_step_executed": False,
        "file_hash_written": False,
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "traces"},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
