#!/usr/bin/env python3
"""Enforce protocol parity before atomically publishing baseline comparisons."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex

from relplus.spec import RELPLUS_SPEC, RELPLUS_SPEC_SHA256


COMPARABLE_KEYS = (
    "dataset_name",
    "dataset_path",
    "rgb_root_folder",
    "rgb_format",
    "gt_root_folder",
    "gt_format",
    "gt_transform",
    "train_source",
    "eval_source",
    "is_test",
    "num_train_imgs",
    "num_eval_imgs",
    "num_classes",
    "class_names",
    "background",
    "image_height",
    "image_width",
    "norm_mean",
    "norm_std",
    "backbone",
    "pretrained_model",
    "pretrained_sha256",
    "decoder",
    "decoder_embed_dim",
    "optimizer",
    "lr",
    "lr_power",
    "momentum",
    "weight_decay",
    "batch_size",
    "nepochs",
    "niters_per_epoch",
    "num_workers",
    "train_scale_array",
    "warm_up_epoch",
    "fix_bias",
    "bn_eps",
    "bn_momentum",
    "eval_iter",
    "eval_stride_rate",
    "eval_scale_array",
    "eval_flip",
    "eval_crop_size",
    "checkpoint_start_epoch",
    "checkpoint_step",
    "seed",
    "effective_distributed_seeds",
)

BASELINES = {
    "HHA": {
        "mode": "hha",
        "config_module": "configs.stanford2d3d_b2_hha",
        "run": Path(
            "/data/zhuzhaoziao/cmx/outputs/"
            "stanford2d3d_b2_hha_formal_seed12345_20260711_004244"
        ),
        "expected_sha256": "37df767cb312981e86e8266ff6e552263ebf7b5efc276a1d121d526c3bea0e3e",
    },
    "RawDepth": {
        "mode": "rawdepth",
        "config_module": "configs.stanford2d3d_b2_rawdepth",
        "run": Path(
            "/data/zhuzhaoziao/cmx/outputs/"
            "stanford2d3d_b2_rawdepth_formal_seed12345_20260711_163209"
        ),
        "expected_sha256": "1f535608ec16b2d585cdd64f432d3dcd2a34cc20ddca4504019fc6acdb2b295b",
    },
}

METRIC_KEYS = ("miou", "pixel_accuracy", "mean_pixel_accuracy")
TOPOLOGY_EXPECTED = {
    "schema": "cmx.training_topology/v1",
    "topology_id": "8-physical-4-logical-global-batch-12",
    "status": "passed",
    "physical_world_size": 8,
    "physical_gpu_ids": list(range(8)),
    "reference_world_size": 4,
    "global_batch_size": 12,
    "niters_per_epoch": 4409,
    "epoch_sample_count": 52908,
    "physical_rank_batch_sizes": [2, 2, 2, 2, 1, 1, 1, 1],
    "physical_rank_sampler_lengths": [8818, 8818, 8818, 8818, 4409, 4409, 4409, 4409],
    "physical_rank_seeds": list(range(8)),
    "effective_distributed_seeds": [0, 1, 2, 3],
    "reference_rank_pairs": [[0, 4], [1, 5], [2, 6], [3, 7]],
    "sampler": "SplitLogicalDistributedSampler",
    "loss_weighting": "2 * local_cross_entropy_sum / paired_valid_pixels",
    "ignore_index": 255,
    "verified_epochs": [1, 2, 17, 32],
    "runtime_required": True,
    "runtime_verified": True,
    "reference_stochastic_trajectory_equivalent": False,
    "stochastic_deviation": (
        "eight independent physical-rank RNG streams; not bitwise equivalent to the "
        "four-rank stochastic trajectory"
    ),
}


def load(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_json(path, payload):
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def differences(left, right, left_name, right_name):
    result = {}
    for key in COMPARABLE_KEYS:
        if key not in left or key not in right:
            result[key] = {
                left_name: left.get(key, "<missing>"),
                right_name: right.get(key, "<missing>"),
            }
        elif left[key] != right[key]:
            result[key] = {left_name: left[key], right_name: right[key]}
    return result


def checked_metrics(path, label):
    metrics = load(path)
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("{} {} is not numeric: {!r}".format(label, key, value))
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("{} {} is outside [0, 1]: {!r}".format(label, key, value))
    return metrics


def require_command(path, required_tokens):
    path = Path(path)
    command = path.read_text(encoding="utf-8").strip()
    if not command:
        raise ValueError("empty command evidence: {}".format(path))
    missing = [token for token in required_tokens if token not in command]
    if missing:
        raise ValueError("command {} lacks tokens {}".format(path, missing))
    return {"path": str(path.resolve()), "sha256": sha256(path), "command": command}


def require_formal_command(path):
    path = Path(path)
    content = path.read_text(encoding="utf-8").strip()
    command_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(command_lines) != 1:
        raise ValueError("formal command evidence must contain exactly one command")
    argv = shlex.split(command_lines[0])
    if len(argv) not in (11, 13):
        raise ValueError("formal command has an unexpected argument count")
    expected_prefix = [
        "/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=8",
    ]
    if argv[:4] != expected_prefix:
        raise ValueError("formal command launcher prefix mismatch")
    if not argv[4].startswith("--master_port=") or not argv[4].split("=", 1)[1].isdigit():
        raise ValueError("formal command master port is invalid")
    expected_body = [
        "tools/run_with_config.py",
        "--config",
        "configs.cmx_relplus_2d",
        "train.py",
        "-p",
        argv[4].split("=", 1)[1],
    ]
    if argv[5:11] != expected_body:
        raise ValueError("formal command training arguments mismatch")
    if len(argv) == 13 and (argv[11] != "-c" or not Path(argv[12]).is_absolute()):
        raise ValueError("formal resume arguments are invalid")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "command": content,
        "argv": argv,
    }


def validate_topology_gpu_evidence(run, report):
    inventory_path = run / "environment" / "gpu_inventory.csv"
    if not inventory_path.is_file() or inventory_path.stat().st_size == 0:
        raise FileNotFoundError(inventory_path)
    inventory = report.get("gpu_inventory", {})
    if Path(inventory.get("path", "")).resolve() != inventory_path.resolve():
        raise ValueError("training topology GPU-inventory path mismatch")
    if inventory.get("sha256") != sha256(inventory_path):
        raise ValueError("training topology GPU-inventory hash mismatch")
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        csv_records = []
        for row in csv.reader(handle):
            if len(row) != 4:
                raise ValueError("audited GPU inventory has an invalid row")
            csv_records.append(
                {
                    "physical_gpu_id": int(row[0].strip()),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "memory_total_mib": int(row[3].strip()),
                }
            )
    records = inventory.get("records", [])
    if records != csv_records:
        raise ValueError("training topology GPU records differ from audited CSV")
    ids = [record.get("physical_gpu_id") for record in records]
    uuids = [record.get("uuid") for record in records]
    if ids != list(range(8)) or len(set(uuids)) != 8:
        raise ValueError("training topology GPU inventory is incomplete")
    if report.get("physical_gpu_uuids") != uuids:
        raise ValueError("training topology physical GPU UUIDs mismatch")

    smoke_path = run / "smoke" / "ddp" / "configs" / "training_topology.json"
    smoke = report.get("smoke_evidence", {})
    if Path(smoke.get("path", "")).resolve() != smoke_path.resolve():
        raise ValueError("training topology smoke-evidence path mismatch")
    if smoke.get("sha256") != sha256(smoke_path):
        raise ValueError("training topology smoke-evidence hash mismatch")
    if smoke.get("status") != "passed" or smoke.get("runtime_verified") is not True:
        raise ValueError("training topology smoke evidence did not pass")
    smoke_report = load(smoke_path)
    smoke_expected = {
        "schema": "cmx.training_topology/v1",
        "topology_id": "8-physical-4-logical-global-batch-12",
        "status": "passed",
        "physical_world_size": 8,
        "reference_world_size": 4,
        "global_batch_size": 12,
        "niters_per_epoch": 1,
        "runtime_required": True,
        "runtime_verified": True,
        "physical_gpu_ids": list(range(8)),
        "physical_gpu_uuids": uuids,
        "epoch_sample_count": 12,
        "physical_rank_batch_sizes": [2, 2, 2, 2, 1, 1, 1, 1],
        "physical_rank_sampler_lengths": [2, 2, 2, 2, 1, 1, 1, 1],
        "physical_rank_seeds": list(range(8)),
        "reference_rank_pairs": [[0, 4], [1, 5], [2, 6], [3, 7]],
        "sampler": "SplitLogicalDistributedSampler",
        "loss_weighting": "2 * local_cross_entropy_sum / paired_valid_pixels",
        "ignore_index": 255,
        "verified_epochs": [1, 2, 17, 32],
        "reference_stochastic_trajectory_equivalent": False,
    }
    if any(smoke_report.get(key) != value for key, value in smoke_expected.items()):
        raise ValueError("smoke topology contract mismatch")
    if smoke_report.get("gpu_inventory", {}).get("sha256") != inventory["sha256"]:
        raise ValueError("smoke topology GPU-inventory hash mismatch")
    smoke_checks = smoke_report.get("epoch_sampling_checks", [])
    if [check.get("epoch") for check in smoke_checks] != [1, 2, 17, 32]:
        raise ValueError("smoke topology sampled epochs mismatch")
    if any(
        check.get("sample_count") != 12
        or check.get("unique_indices") != 12
        or check.get("duplicate_indices") != 0
        or check.get("missing_indices") != 0
        or check.get("reference_batches_reconstructed") is not True
        or check.get("global_batches_reconstructed") is not True
        for check in smoke_checks
    ):
        raise ValueError("smoke topology sampling checks failed")
    smoke_runtime = smoke_report.get("runtime_records", [])
    if len(smoke_runtime) != 8:
        raise ValueError("smoke topology runtime records are incomplete")
    for rank, record in enumerate(smoke_runtime):
        expected_path = run / "smoke" / "ddp" / "status" / "topology_rank_{}.json".format(rank)
        if Path(record.get("path", "")).resolve() != expected_path.resolve():
            raise ValueError("smoke runtime rank {} path mismatch".format(rank))
        source = load(expected_path)
        if source != {key: value for key, value in record.items() if key != "path"}:
            raise ValueError("smoke runtime rank {} content mismatch".format(rank))
        expected_runtime = {
            "physical_rank": rank,
            "physical_world_size": 8,
            "physical_gpu_id": rank,
            "physical_gpu_uuid": uuids[rank],
            "reference_rank": rank % 4,
            "reference_world_size": 4,
            "reference_group_ranks": [[0, 4], [1, 5], [2, 6], [3, 7]][rank % 4],
            "local_batch_size": [2, 2, 2, 2, 1, 1, 1, 1][rank],
            "sampler_samples": [2, 2, 2, 2, 1, 1, 1, 1][rank],
            "loader_steps": 1,
            "global_batch_size": 12,
            "rank_seed": rank,
            "sampler_class": "SplitLogicalDistributedSampler",
            "loss_weighting": "2 * local_cross_entropy_sum / paired_valid_pixels",
            "ignore_index": 255,
            "first_optimizer_step_completed": True,
            "first_optimizer_step_iteration": 0,
        }
        if any(record.get(key) != value for key, value in expected_runtime.items()):
            raise ValueError("smoke runtime rank {} contract mismatch".format(rank))

    runtime_records = report.get("runtime_records", [])
    for rank, record in enumerate(runtime_records):
        expected_path = run / "status" / "topology_rank_{}.json".format(rank)
        if Path(record.get("path", "")).resolve() != expected_path.resolve():
            raise ValueError("runtime topology rank {} path mismatch".format(rank))
        source = load(expected_path)
        reported = {key: value for key, value in record.items() if key != "path"}
        if source != reported:
            raise ValueError("runtime topology rank {} content mismatch".format(rank))
        expected_record = {
            "physical_rank": rank,
            "physical_world_size": 8,
            "physical_gpu_id": rank,
            "physical_gpu_uuid": uuids[rank],
            "reference_rank": rank % 4,
            "reference_world_size": 4,
            "reference_group_ranks": [[0, 4], [1, 5], [2, 6], [3, 7]][rank % 4],
            "local_batch_size": [2, 2, 2, 2, 1, 1, 1, 1][rank],
            "sampler_samples": [8818, 8818, 8818, 8818, 4409, 4409, 4409, 4409][rank],
            "loader_steps": 4409,
            "global_batch_size": 12,
            "rank_seed": rank,
            "sampler_class": "SplitLogicalDistributedSampler",
            "loss_weighting": "2 * local_cross_entropy_sum / paired_valid_pixels",
            "ignore_index": 255,
            "first_optimizer_step_completed": True,
            "first_optimizer_step_iteration": 0,
        }
        if any(record.get(key) != value for key, value in expected_record.items()):
            raise ValueError("runtime topology rank {} contract mismatch".format(rank))


def validate_training_topology(run):
    path = run / "configs" / "training_topology.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    report = load(path)
    differences = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in TOPOLOGY_EXPECTED.items()
        if report.get(key) != expected
    }
    if differences:
        raise ValueError("training topology mismatch: {}".format(differences))
    checks = report.get("epoch_sampling_checks")
    if not isinstance(checks, list) or len(checks) != 4:
        raise ValueError("training topology must contain four epoch sampling checks")
    if [check.get("epoch") for check in checks] != [1, 2, 17, 32]:
        raise ValueError("training topology sampled epochs mismatch")
    for check in checks:
        if not (
            check.get("sample_count") == 52908
            and check.get("unique_indices") == 52908
            and check.get("duplicate_indices") == 0
            and check.get("missing_indices") == 0
            and check.get("reference_batches_reconstructed") is True
            and check.get("global_batches_reconstructed") is True
        ):
            raise ValueError("training topology epoch coverage check failed")
    records = report.get("runtime_records")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("training topology must contain eight runtime rank records")
    if sorted(record.get("physical_rank") for record in records) != list(range(8)):
        raise ValueError("training topology runtime ranks are incomplete")
    validate_topology_gpu_evidence(run, report)
    return {"path": str(path.resolve()), "sha256": sha256(path), "report": report}


def build_comparison(run):
    formal_metadata_path = run / "configs" / "resolved_config.json"
    formal_metadata = load(formal_metadata_path)
    if formal_metadata.get("config_module") != "configs.cmx_relplus_2d":
        raise ValueError("unexpected formal config module")
    formal_config = formal_metadata["resolved_config"]
    expected_identity = {
        "model_name": RELPLUS_SPEC["model_name"],
        "config_name": RELPLUS_SPEC["config_name"],
        "representation_semantics": RELPLUS_SPEC["representation_semantics"],
        "relplus_representation_version": RELPLUS_SPEC["representation_version"],
        "relplus_point_frame": RELPLUS_SPEC["point_frame"],
        "relplus_translation_in_red_loa": RELPLUS_SPEC["translation_in_red_loa"],
        "relplus_representation_spec_sha256": RELPLUS_SPEC_SHA256,
    }
    for key, expected in expected_identity.items():
        if formal_config.get(key) != expected:
            raise ValueError("formal REL+ identity mismatch for {}".format(key))
    formal_command = require_formal_command(run / "configs" / "command.txt")
    formal_topology = validate_training_topology(run)
    rel_metric_path = run / "metrics" / "metrics.json"
    rel = checked_metrics(rel_metric_path, "REL+")
    formal_checkpoint = run / "checkpoints" / "epoch-32.pth"
    if not formal_checkpoint.is_file() or formal_checkpoint.stat().st_size == 0:
        raise FileNotFoundError(formal_checkpoint)
    formal_checkpoint_sha = sha256(formal_checkpoint)

    fairness = {
        "passed": False,
        "comparable_keys": list(COMPARABLE_KEYS),
        "formal": {
            "resolved_config": str(formal_metadata_path.resolve()),
            "resolved_config_sha256": sha256(formal_metadata_path),
            "command": formal_command,
            "training_topology": formal_topology,
            "checkpoint": str(formal_checkpoint.resolve()),
            "checkpoint_sha256": formal_checkpoint_sha,
        },
        "baselines": {},
    }
    rows = []
    failures = {}
    for name, specification in BASELINES.items():
        mode = specification["mode"]
        baseline_run = specification["run"]
        reeval_dir = run / "baseline_reeval" / mode
        source_metadata_path = baseline_run / "metadata.json"
        reeval_metadata_path = reeval_dir / "configs" / "resolved_config.json"
        source_metadata = load(source_metadata_path)
        reeval_metadata = load(reeval_metadata_path)
        if source_metadata.get("config_module") != specification["config_module"]:
            raise ValueError("{} source config module mismatch".format(name))
        if reeval_metadata.get("config_module") != specification["config_module"]:
            raise ValueError("{} re-evaluation config module mismatch".format(name))
        source_command = str(source_metadata.get("command", "")).strip()
        if not source_command:
            raise ValueError("{} source training command evidence is absent".format(name))
        source_config = source_metadata["resolved_config"]
        reeval_config = reeval_metadata["resolved_config"]
        command = require_command(
            reeval_dir / "configs" / "command.txt",
            (specification["config_module"], str(baseline_run / "checkpoints" / "epoch-32.pth")),
        )
        validation_path = reeval_dir / "checkpoint_validation.json"
        validation = load(validation_path)
        expected_checkpoint = (baseline_run / "checkpoints" / "epoch-32.pth").resolve()
        expected_sha = specification["expected_sha256"]
        validation_ok = (
            validation.get("verified") is True
            and validation.get("sha256_match") is True
            and validation.get("actual_sha256") == expected_sha
            and validation.get("expected_sha256") == expected_sha
            and type(validation.get("checkpoint_epoch")) is int
            and validation.get("checkpoint_epoch") == 32
            and type(validation.get("checkpoint_iteration")) is int
            and validation.get("checkpoint_iteration") == 4408
            and type(validation.get("model_key_count")) is int
            and validation.get("model_key_count") > 0
            and Path(validation.get("checkpoint", "")).resolve() == expected_checkpoint
            and validation.get("source_metadata_sha256") == sha256(source_metadata_path)
        )
        parity = {
            "formal_vs_source": differences(
                formal_config, source_config, "relplus", "baseline_source"
            ),
            "formal_vs_reevaluation": differences(
                formal_config, reeval_config, "relplus", "baseline_reevaluation"
            ),
            "source_vs_reevaluation": differences(
                source_config, reeval_config, "baseline_source", "baseline_reevaluation"
            ),
        }
        comparable = validation_ok and not any(parity.values())
        fairness["baselines"][name] = {
            "comparable": comparable,
            "differences": parity,
            "source_resolved_config": str(source_metadata_path.resolve()),
            "source_resolved_config_sha256": sha256(source_metadata_path),
            "source_training_command": source_command,
            "reevaluation_resolved_config": str(reeval_metadata_path.resolve()),
            "reevaluation_resolved_config_sha256": sha256(reeval_metadata_path),
            "reevaluation_command": command,
            "checkpoint": str(expected_checkpoint),
            "checkpoint_sha256": validation.get("actual_sha256"),
            "checkpoint_expected_sha256": expected_sha,
            "checkpoint_validation": str(validation_path.resolve()),
            "checkpoint_validation_ok": validation_ok,
        }
        if not comparable:
            failures[name] = fairness["baselines"][name]

        metric_path = reeval_dir / "metrics.json"
        metrics = checked_metrics(metric_path, name)
        rows.append(
            {
                "method": name,
                "miou_percent": metrics["miou"] * 100.0,
                "pixel_accuracy_percent": metrics["pixel_accuracy"] * 100.0,
                "mean_accuracy_percent": metrics["mean_pixel_accuracy"] * 100.0,
                "relplus_delta_miou_points": (rel["miou"] - metrics["miou"]) * 100.0,
                "relplus_delta_pixel_accuracy_points": (
                    rel["pixel_accuracy"] - metrics["pixel_accuracy"]
                ) * 100.0,
                "comparable": comparable,
                "checkpoint": str(expected_checkpoint),
                "checkpoint_sha256": validation.get("actual_sha256"),
                "metrics_source": str(metric_path.resolve()),
            }
        )

    if failures:
        raise RuntimeError(
            "fairness gate failed before publication: {}".format(
                json.dumps(failures, sort_keys=True)
            )
        )
    fairness["passed"] = True
    rows.insert(
        0,
        {
            "method": "REL+",
            "miou_percent": rel["miou"] * 100.0,
            "pixel_accuracy_percent": rel["pixel_accuracy"] * 100.0,
            "mean_accuracy_percent": rel["mean_pixel_accuracy"] * 100.0,
            "relplus_delta_miou_points": 0.0,
            "relplus_delta_pixel_accuracy_points": 0.0,
            "comparable": True,
            "checkpoint": str(formal_checkpoint.resolve()),
            "checkpoint_sha256": formal_checkpoint_sha,
            "metrics_source": str(rel_metric_path.resolve()),
        },
    )
    return fairness, rows


def publish(run, fairness, rows):
    metrics_dir = run / "metrics"
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    lines = [
        "# Fair 2D CMX comparison",
        "",
        "Primary checkpoint policy: epoch 32 is the only eligible checkpoint; best equals last.",
        "All deltas below were released only after source-training and re-evaluation config parity passed.",
        "",
        "| Method | mIoU (%) | Pixel Acc. (%) | Mean Acc. (%) | REL+ delta mIoU (pt) | Checkpoint SHA-256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {miou_percent:.3f} | {pixel_accuracy_percent:.3f} | "
            "{mean_accuracy_percent:.3f} | {relplus_delta_miou_points:.3f} | "
            "`{checkpoint_sha256}` |".format(**row)
        )
    # Every published artifact is written to a same-directory temporary file
    # and renamed only after the complete fairness payload has passed.
    atomic_json(metrics_dir / "fairness_gate.json", fairness)
    atomic_write(metrics_dir / "comparison.csv", csv_buffer.getvalue())
    atomic_write(metrics_dir / "comparison.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    status = run / "status" / "compare.exitcode"
    atomic_write(status, "1\n")
    try:
        fairness, rows = build_comparison(run)
        publish(run, fairness, rows)
    except BaseException:
        atomic_write(status, "1\n")
        raise
    atomic_write(status, "0\n")


if __name__ == "__main__":
    main()
