#!/usr/bin/env python3
"""Validate the eight-physical/four-logical DDP batch adapter."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DistributedSampler

from dataloader.distributed_batch import SplitLogicalDistributedSampler


PHYSICAL_WORLD_SIZE = 8
REFERENCE_WORLD_SIZE = 4
GLOBAL_BATCH_SIZE = 12
RANK_BATCH_SIZES = [2, 2, 2, 2, 1, 1, 1, 1]
RANK_SEEDS = list(range(8))
RANK_PAIRS = [[0, 4], [1, 5], [2, 6], [3, 7]]
VERIFIED_EPOCHS = [1, 2, 17, 32]
LOSS_WEIGHTING = "2 * local_cross_entropy_sum / paired_valid_pixels"
SCHEMA = "cmx.training_topology/v1"
TOPOLOGY_ID = "8-physical-4-logical-global-batch-12"
STOCHASTIC_DEVIATION = (
    "eight independent physical-rank RNG streams; not bitwise equivalent to the "
    "four-rank stochastic trajectory"
)


class _Dataset:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gpu_inventory(path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 4:
                raise ValueError(
                    "GPU inventory line {} must contain four columns".format(line_number)
                )
            record = {
                "physical_gpu_id": int(row[0].strip()),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "memory_total_mib": int(row[3].strip()),
            }
            if not record["uuid"] or not record["name"]:
                raise ValueError("GPU inventory contains an empty UUID or name")
            if record["memory_total_mib"] <= 0:
                raise ValueError("GPU inventory contains invalid total memory")
            records.append(record)
    if [record["physical_gpu_id"] for record in records] != list(
        range(PHYSICAL_WORLD_SIZE)
    ):
        raise ValueError("GPU inventory must list physical GPU IDs 0 through 7 in order")
    uuids = [record["uuid"] for record in records]
    if len(set(uuids)) != PHYSICAL_WORLD_SIZE:
        raise ValueError("GPU inventory must contain eight distinct GPU UUIDs")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": records,
    }


def validate_smoke_topology(path, gpu_inventory):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    topology = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": SCHEMA,
        "topology_id": TOPOLOGY_ID,
        "status": "passed",
        "physical_world_size": PHYSICAL_WORLD_SIZE,
        "reference_world_size": REFERENCE_WORLD_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "niters_per_epoch": 1,
        "runtime_required": True,
        "runtime_verified": True,
        "physical_gpu_ids": list(range(PHYSICAL_WORLD_SIZE)),
        "physical_gpu_uuids": [
            record["uuid"] for record in gpu_inventory["records"]
        ],
    }
    differences = {
        key: {"expected": value, "actual": topology.get(key)}
        for key, value in expected.items()
        if topology.get(key) != value
    }
    if differences:
        raise ValueError("smoke topology mismatch: {}".format(differences))
    if topology.get("gpu_inventory", {}).get("sha256") != gpu_inventory["sha256"]:
        raise ValueError("smoke topology GPU-inventory hash mismatch")
    runtime_records = topology.get("runtime_records", [])
    if len(runtime_records) != PHYSICAL_WORLD_SIZE:
        raise ValueError("smoke topology must contain eight runtime records")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": topology["status"],
        "runtime_verified": topology["runtime_verified"],
    }


def build_samplers(dataset, steps):
    return [
        SplitLogicalDistributedSampler(
            dataset,
            global_batch_size=GLOBAL_BATCH_SIZE,
            num_steps=steps,
            num_replicas=PHYSICAL_WORLD_SIZE,
            rank=rank,
            reference_replicas=REFERENCE_WORLD_SIZE,
        )
        for rank in range(PHYSICAL_WORLD_SIZE)
    ]


def validate_sampling(steps):
    sample_count = GLOBAL_BATCH_SIZE * steps
    dataset = _Dataset(sample_count)
    samplers = build_samplers(dataset, steps)
    if [sampler.local_batch_size for sampler in samplers] != RANK_BATCH_SIZES:
        raise ValueError("physical rank batch sizes do not match the contract")

    epoch_reports = []
    for epoch in VERIFIED_EPOCHS:
        for sampler in samplers:
            sampler.set_epoch(epoch)
        physical_indices = [list(sampler) for sampler in samplers]
        flat = [index for rank_indices in physical_indices for index in rank_indices]
        counts = Counter(flat)
        duplicates = sum(count - 1 for count in counts.values() if count > 1)
        missing = sample_count - len(counts)
        if len(flat) != sample_count or duplicates or missing:
            raise ValueError("physical samplers do not cover one epoch exactly")

        generator = torch.Generator()
        generator.manual_seed(epoch)
        permutation = torch.randperm(sample_count, generator=generator).tolist()
        for reference_rank in range(REFERENCE_WORLD_SIZE):
            reference = DistributedSampler(
                dataset,
                num_replicas=REFERENCE_WORLD_SIZE,
                rank=reference_rank,
                shuffle=True,
                seed=0,
                drop_last=False,
            )
            reference.set_epoch(epoch)
            expected = list(reference)
            two = physical_indices[reference_rank]
            one = physical_indices[reference_rank + REFERENCE_WORLD_SIZE]
            reconstructed = []
            for step in range(steps):
                reconstructed.extend(two[2 * step : 2 * step + 2])
                reconstructed.append(one[step])
            if reconstructed != expected:
                raise ValueError("physical split does not reconstruct reference-rank batches")

        for step in range(steps):
            physical_step = []
            for reference_rank in range(REFERENCE_WORLD_SIZE):
                physical_step.extend(
                    physical_indices[reference_rank][2 * step : 2 * step + 2]
                )
                physical_step.append(
                    physical_indices[reference_rank + REFERENCE_WORLD_SIZE][step]
                )
            expected_step = permutation[
                GLOBAL_BATCH_SIZE * step : GLOBAL_BATCH_SIZE * (step + 1)
            ]
            if sorted(physical_step) != sorted(expected_step):
                raise ValueError("physical step does not reconstruct the global batch")

        epoch_reports.append(
            {
                "epoch": epoch,
                "sample_count": len(flat),
                "unique_indices": len(counts),
                "duplicate_indices": duplicates,
                "missing_indices": missing,
                "reference_batches_reconstructed": True,
                "global_batches_reconstructed": True,
            }
        )
    return samplers, epoch_reports


def validate_runtime(run, steps, gpu_inventory):
    records = []
    for rank in range(PHYSICAL_WORLD_SIZE):
        path = run / "status" / "topology_rank_{}.json".format(rank)
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "physical_rank": rank,
            "physical_world_size": PHYSICAL_WORLD_SIZE,
            "physical_gpu_id": rank,
            "physical_gpu_uuid": gpu_inventory["records"][rank]["uuid"],
            "reference_rank": rank % REFERENCE_WORLD_SIZE,
            "reference_world_size": REFERENCE_WORLD_SIZE,
            "reference_group_ranks": RANK_PAIRS[rank % REFERENCE_WORLD_SIZE],
            "local_batch_size": RANK_BATCH_SIZES[rank],
            "sampler_samples": RANK_BATCH_SIZES[rank] * steps,
            "loader_steps": steps,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "rank_seed": RANK_SEEDS[rank],
            "sampler_class": "SplitLogicalDistributedSampler",
            "loss_weighting": LOSS_WEIGHTING,
            "ignore_index": 255,
            "first_optimizer_step_completed": True,
            "first_optimizer_step_iteration": 0,
        }
        differences = {
            key: {"expected": value, "actual": record.get(key)}
            for key, value in expected.items()
            if record.get(key) != value
        }
        if differences:
            raise ValueError("rank {} runtime topology mismatch: {}".format(rank, differences))
        records.append({"path": str(path.resolve()), **record})
    if len({record["physical_gpu_uuid"] for record in records}) != PHYSICAL_WORLD_SIZE:
        raise ValueError("runtime topology must contain eight distinct GPU UUIDs")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--gpu-inventory", required=True)
    parser.add_argument("--smoke-topology")
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()

    run = Path(args.run_dir).resolve()
    output = run / "configs" / "training_topology.json"
    report = {
        "schema": SCHEMA,
        "topology_id": TOPOLOGY_ID,
        "status": "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_world_size": PHYSICAL_WORLD_SIZE,
        "physical_gpu_ids": list(range(PHYSICAL_WORLD_SIZE)),
        "reference_world_size": REFERENCE_WORLD_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "niters_per_epoch": args.steps,
        "epoch_sample_count": GLOBAL_BATCH_SIZE * args.steps,
        "physical_rank_batch_sizes": RANK_BATCH_SIZES,
        "physical_rank_samples_per_epoch": [
            batch * args.steps for batch in RANK_BATCH_SIZES
        ],
        "physical_rank_seeds": RANK_SEEDS,
        "effective_distributed_seeds": [0, 1, 2, 3],
        "reference_rank_pairs": RANK_PAIRS,
        "sampler": "SplitLogicalDistributedSampler",
        "sampler_reference": "torch DistributedSampler(seed=0, drop_last=False)",
        "loss_weighting": LOSS_WEIGHTING,
        "ignore_index": 255,
        "verified_epochs": VERIFIED_EPOCHS,
        "runtime_required": args.require_runtime,
        "reference_stochastic_trajectory_equivalent": False,
        "stochastic_deviation": STOCHASTIC_DEVIATION,
    }
    try:
        gpu_inventory = validate_gpu_inventory(Path(args.gpu_inventory))
        report["gpu_inventory"] = gpu_inventory
        report["physical_gpu_uuids"] = [
            record["uuid"] for record in gpu_inventory["records"]
        ]
        report["smoke_evidence"] = (
            validate_smoke_topology(Path(args.smoke_topology), gpu_inventory)
            if args.smoke_topology
            else None
        )
        samplers, epoch_reports = validate_sampling(args.steps)
        report["physical_rank_sampler_lengths"] = [len(sampler) for sampler in samplers]
        report["epoch_sampling_checks"] = epoch_reports
        report["runtime_records"] = (
            validate_runtime(run, args.steps, gpu_inventory)
            if args.require_runtime
            else []
        )
        report["runtime_verified"] = args.require_runtime
        report["status"] = "passed"
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
        atomic_json(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise

    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
