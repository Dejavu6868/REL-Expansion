#!/usr/bin/env python3
"""Export full ordered sampler sample IDs for required epochs and ranks."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


EPOCHS = (1, 2, 3, 10, 11)
WORLD_SIZE = 8


class SampleIds:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    os.chdir(str(code_root))
    sys.path.insert(0, str(code_root))
    config = importlib.import_module(
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_3_formal"
    ).config
    from dataloader.samplers import FixedLengthDistributedSampler

    with Path(config.train_source).open("r", encoding="utf-8") as handle:
        sample_ids = [line.strip() for line in handle if line.strip()]
    dataset = SampleIds(sample_ids)
    sequences = {}
    for epoch in EPOCHS:
        epoch_rows = {}
        for rank in range(WORLD_SIZE):
            sampler = FixedLengthDistributedSampler(
                dataset,
                logical_samples_per_epoch=config.logical_samples_per_epoch,
                num_replicas=WORLD_SIZE,
                rank=rank,
                shuffle=True,
                seed=config.seed,
            )
            sampler.set_epoch(epoch)
            ordered = [sample_ids[index] for index in sampler]
            if len(ordered) != config.niters_per_epoch:
                raise RuntimeError("sampler rank length mismatch")
            epoch_rows[str(rank)] = ordered
        sequences[str(epoch)] = epoch_rows
    report = {
        "status": "PASS",
        "protocol": "PURE_SAMPLER_EPOCH_SEQUENCE_V1",
        "code_root": str(code_root),
        "seed": int(config.seed),
        "epochs": list(EPOCHS),
        "world_size": WORLD_SIZE,
        "dataset_size": len(sample_ids),
        "logical_samples_per_epoch": int(config.logical_samples_per_epoch),
        "sample_ids_per_rank_epoch": int(config.niters_per_epoch),
        "sequences": sequences,
        "gpu_used": False,
        "backpropagation_executed": False,
        "optimizer_step_executed": False,
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "sequences"},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
