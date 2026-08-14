#!/usr/bin/env python3
"""Publish COMPLETE only after independently rechecking every formal gate."""

import argparse
from collections.abc import Mapping
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex

from relplus.spec import RELPLUS_SPEC, RELPLUS_SPEC_SHA256


EXPECTED_SAMPLE_COUNT = 70496
EXPECTED_EVAL_SAMPLE_COUNT = 17593
EXPECTED_EPOCHS = tuple(range(4, 33, 4))
TRAIN_LOSS_COLUMN = "train_logged_epoch_mean_ddp_batch_ce"
VALIDATION_LOSS_COLUMN = "validation_pixel_weighted_mean_ce"
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
REQUIRED_EXITS = (
    "audit",
    "tests",
    "prepare",
    "semantics_validation",
    "semantics_validation_initial",
    "cache_validation",
    "cache_validation_initial",
    "cache_revalidation",
    "semantics_revalidation",
    "smoke",
    "topology_preflight",
    "topology_runtime",
    "train",
    "eval",
    "baseline_preflight",
    "baseline_hha_eval",
    "baseline_rawdepth_eval",
    "diagnostics",
    "compare",
)


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
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_json(path, payload):
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def require_regular_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("required non-empty file is absent: {}".format(path))
    return path


def _code_files(repo_root):
    repo_root = Path(repo_root).resolve()
    files = set()
    for path in repo_root.rglob("*"):
        relative = path.relative_to(repo_root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("repository tree contains a symlink: {}".format(path))
        if path.is_file() and path.suffix != ".pyc":
            files.add(relative.as_posix())
    return files


def validate_code_manifest(run, repo_root=None):
    repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    manifest_path = require_regular_file(run / "code_manifest.sha256")
    recorded = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(
                "invalid code-manifest entry at line {}".format(line_number)
            )
        digest = fields[0].lower()
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                "invalid code-manifest digest at line {}".format(line_number)
            )
        raw_path = fields[1].lstrip("*").strip()
        pure = Path(raw_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(
                "unsafe code-manifest path at line {}: {}".format(
                    line_number, raw_path
                )
            )
        normalized = pure.as_posix()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized or normalized in recorded:
            raise ValueError(
                "empty or duplicate code-manifest path at line {}".format(line_number)
            )
        recorded[normalized] = digest

    current_files = _code_files(repo_root)
    recorded_files = set(recorded)
    if current_files != recorded_files:
        raise ValueError(
            "repository file set changed during run: added={} removed={}".format(
                sorted(current_files - recorded_files),
                sorted(recorded_files - current_files),
            )
        )
    mismatches = []
    for relative in sorted(recorded):
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            mismatches.append(relative)
        elif sha256(path) != recorded[relative]:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(
            "repository content changed during run: {}".format(mismatches)
        )
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
        "file_count": len(recorded),
        "repository_root": str(repo_root),
        "status": "passed",
    }


def validate_initialization_report(run, config):
    path = require_regular_file(run / "initialization_report.json")
    report = load(path)
    expected_checkpoint = require_regular_file(config["pretrained_model"]).resolve()
    if sha256(expected_checkpoint) != config.get("pretrained_sha256"):
        raise ValueError("pretrained checkpoint file SHA-256 mismatch")
    if Path(report.get("checkpoint_path", "")).resolve() != expected_checkpoint:
        raise ValueError("initialization checkpoint path mismatch")
    if report.get("checkpoint_sha256") != config.get("pretrained_sha256"):
        raise ValueError("initialization checkpoint SHA-256 mismatch")
    if report.get("loading_module") != (
        "dual MiT backbone: identical weights copied to RGB and REL+ encoders"
    ):
        raise ValueError("unexpected initialization loading module")
    expected_counts = {
        "loaded_tensor_count": 664,
        "loaded_parameter_count": 48392576,
        "model_state_parameter_count": 64988560,
    }
    for key, expected in expected_counts.items():
        if report.get(key) != expected:
            raise ValueError(
                "initialization {} is {!r}, expected {}".format(
                    key, report.get(key), expected
                )
            )
    expected_ratio = expected_counts["loaded_parameter_count"] / float(
        expected_counts["model_state_parameter_count"]
    )
    ratio = finite_number(
        report.get("loaded_parameter_ratio"),
        "initialization loaded_parameter_ratio",
        lower=0.0,
        upper=1.0,
    )
    if not math.isclose(ratio, expected_ratio, rel_tol=1e-15, abs_tol=0.0):
        raise ValueError("initialization loaded_parameter_ratio is inconsistent")
    loaded_keys = report.get("loaded_keys")
    if not isinstance(loaded_keys, list) or len(loaded_keys) != 664:
        raise ValueError("initialization loaded_keys must contain exactly 664 keys")
    if len(set(loaded_keys)) != 664 or not all(
        isinstance(key, str) and key for key in loaded_keys
    ):
        raise ValueError("initialization loaded_keys are invalid or duplicated")
    base_keys = [key for key in loaded_keys if not key.startswith("extra_")]
    extra_keys = [key for key in loaded_keys if key.startswith("extra_")]
    if len(base_keys) != 332 or len(extra_keys) != 332:
        raise ValueError("initialization did not load two complete MiT encoder copies")

    def paired_key(key):
        if "patch_embed" in key:
            return key.replace("patch_embed", "extra_patch_embed")
        if "block" in key:
            return key.replace("block", "extra_block")
        if "norm" in key:
            return key.replace("norm", "extra_norm")
        raise ValueError("unexpected MiT backbone key: {}".format(key))

    if {paired_key(key) for key in base_keys} != set(extra_keys):
        raise ValueError("RGB and REL+ MiT initialization keys are not exact pairs")
    missing_keys = report.get("missing_keys")
    if not isinstance(missing_keys, list) or len(missing_keys) != 148:
        raise ValueError("initialization missing_keys must contain exactly 148 fusion keys")
    if len(set(missing_keys)) != 148 or not all(
        isinstance(key, str) and key.startswith(("FRMs.", "FFMs."))
        for key in missing_keys
    ):
        raise ValueError("initialization missing_keys contain non-fusion parameters")
    if sum(key.startswith("FRMs.") for key in missing_keys) != 32 or sum(
        key.startswith("FFMs.") for key in missing_keys
    ) != 116:
        raise ValueError("initialization fusion missing-key counts are unexpected")
    for key in ("unexpected_keys", "shape_mismatch"):
        if report.get(key) != []:
            raise ValueError("initialization {} is not empty".format(key))
    if report.get("checkpoint_unmapped_keys") != ["head.bias", "head.weight"]:
        raise ValueError("initialization checkpoint unmapped keys are unexpected")
    if report.get("strict") is not False:
        raise ValueError("initialization strict flag must document REL-default loading")
    if report.get("decoder_and_fusion_initialization") != (
        "unchanged CMX defaults; decoder Kaiming, fusion module constructors"
    ):
        raise ValueError("decoder/fusion initialization is not REL-default")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "report": report,
    }


def finite_number(value, label, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} is not numeric: {!r}".format(label, value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} is not finite: {!r}".format(label, value))
    if lower is not None and value < lower:
        raise ValueError("{} is below {}: {}".format(label, lower, value))
    if upper is not None and value > upper:
        raise ValueError("{} is above {}: {}".format(label, upper, value))
    return value


def validate_exit_codes(run):
    exit_codes = {}
    for name in REQUIRED_EXITS:
        path = require_regular_file(run / "status" / (name + ".exitcode"))
        text = path.read_text(encoding="utf-8").strip()
        if not text or any(character not in "-0123456789" for character in text):
            raise ValueError("invalid exit-code evidence {}: {!r}".format(path, text))
        exit_codes[name] = int(text)
    if any(exit_codes.values()):
        raise RuntimeError("nonzero stage exit: {}".format(exit_codes))
    return exit_codes


def validate_cache_report(run, report_name):
    report_path = run / "data_reports" / report_name
    report = load(require_regular_file(report_path))
    exact_counts = ("sample_count", "manifest_count", "png_count", "validated_files")
    for key in exact_counts:
        if report.get(key) != EXPECTED_SAMPLE_COUNT:
            raise ValueError("cache validation {} is {!r}".format(key, report.get(key)))
    for key in (
        "all_sha256_match",
        "all_decodable",
        "representation_semantics_valid",
        "representation_generator_valid",
        "cache_tree_symlink_free",
        "all_cache_entries_generated",
    ):
        if report.get(key) is not True:
            raise ValueError("cache validation {} is not true".format(key))
    if report.get("cache_is_symlink") is not False:
        raise ValueError("REL-default cache must be a run-local directory")
    if report.get("representation_spec_sha256") != RELPLUS_SPEC_SHA256:
        raise ValueError("cache validation representation spec SHA-256 mismatch")
    return report


def validate_cache(run):
    initial = validate_cache_report(run, "cache_validation_initial.json")
    revalidation = validate_cache_report(run, "cache_revalidation.json")
    initial_time = datetime.fromisoformat(initial["completed_at_utc"])
    revalidation_time = datetime.fromisoformat(revalidation["completed_at_utc"])
    if revalidation_time <= initial_time:
        raise ValueError("cache revalidation is not newer than initial validation")
    current_path = require_regular_file(run / "data_reports" / "cache_validation.json")
    revalidation_path = run / "data_reports" / "cache_revalidation.json"
    if sha256(current_path) != sha256(revalidation_path):
        raise ValueError("current cache validation report is not the final revalidation")
    diagnostics_exit = require_regular_file(run / "status" / "diagnostics.exitcode")
    if revalidation_path.stat().st_mtime_ns < diagnostics_exit.stat().st_mtime_ns:
        raise ValueError("cache revalidation was not recorded after diagnostics")

    stats_path = run / "data_reports" / "relplus_statistics.json"
    stats = load(require_regular_file(stats_path))
    if stats.get("sample_count") != EXPECTED_SAMPLE_COUNT:
        raise ValueError("REL+ statistics sample_count mismatch")
    generated = stats.get("generated_count")
    skipped = stats.get("skipped_count")
    if not isinstance(generated, int) or not isinstance(skipped, int):
        raise TypeError("generated_count and skipped_count must be integers")
    if generated != EXPECTED_SAMPLE_COUNT or skipped != 0:
        raise ValueError("REL-default cache must be generated fresh with no skipped entries")
    for key in (
        "representation_semantics",
        "representation_version",
        "point_frame",
        "translation_in_red_loa",
    ):
        if stats.get(key) != RELPLUS_SPEC[key]:
            raise ValueError("REL+ statistics {} mismatch".format(key))
    if stats.get("representation_spec_sha256") != RELPLUS_SPEC_SHA256:
        raise ValueError("REL+ statistics representation spec SHA-256 mismatch")
    if (
        stats.get("representation_generator_bundle_sha256")
        != initial.get("representation_generator_bundle_sha256")
    ):
        raise ValueError("initial REL+ representation generator bundle SHA-256 mismatch")
    if (
        stats.get("representation_generator_bundle_sha256")
        != revalidation.get("representation_generator_bundle_sha256")
    ):
        raise ValueError("final REL+ representation generator bundle SHA-256 mismatch")
    return {"initial": initial, "revalidation": revalidation}, stats


def validate_semantics_report(run, report_name):
    report = load(require_regular_file(run / "data_reports" / report_name))
    if report.get("status") != "passed" or report.get("exit_code") != 0:
        raise ValueError("REL-default semantic recomputation did not pass")
    if report.get("sample_count") != 16 or report.get("matched_samples") != 16:
        raise ValueError("REL-default semantic recomputation must match all 16 samples")
    if report.get("all_exact") is not True:
        raise ValueError("REL-default semantic recomputation is not byte-exact")
    expected = {
        "representation_semantics": RELPLUS_SPEC["representation_semantics"],
        "representation_version": RELPLUS_SPEC["representation_version"],
        "representation_spec_sha256": RELPLUS_SPEC_SHA256,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError("semantic recomputation {} mismatch".format(key))
    return report


def validate_semantics(run):
    initial = validate_semantics_report(run, "semantics_validation_initial.json")
    revalidation = validate_semantics_report(run, "semantics_revalidation.json")
    initial_time = datetime.fromisoformat(initial["validated_at_utc"])
    revalidation_time = datetime.fromisoformat(revalidation["validated_at_utc"])
    if revalidation_time <= initial_time:
        raise ValueError("semantic revalidation is not newer than initial validation")
    current_path = require_regular_file(
        run / "data_reports" / "semantics_validation.json"
    )
    revalidation_path = run / "data_reports" / "semantics_revalidation.json"
    if sha256(current_path) != sha256(revalidation_path):
        raise ValueError("current semantic report is not the final revalidation")
    diagnostics_exit = require_regular_file(run / "status" / "diagnostics.exitcode")
    if revalidation_path.stat().st_mtime_ns < diagnostics_exit.stat().st_mtime_ns:
        raise ValueError("semantic revalidation was not recorded after diagnostics")
    return {"initial": initial, "revalidation": revalidation}


def parse_sha256_file(path, expected_path):
    text = require_regular_file(path).read_text(encoding="utf-8").strip()
    fields = text.split(maxsplit=1)
    if len(fields) != 2 or len(fields[0]) != 64:
        raise ValueError("invalid sha256sum evidence: {}".format(path))
    recorded_path = fields[1].lstrip("*").strip()
    if Path(recorded_path).resolve() != Path(expected_path).resolve():
        raise ValueError("sha256sum path does not identify epoch-32.pth")
    return fields[0].lower()


def inspect_checkpoint(path):
    import torch

    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint root is not a mapping")
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch != 32:
        raise ValueError("checkpoint internal epoch is {!r}, expected 32".format(epoch))
    iteration = payload.get("iteration")
    if type(iteration) is not int or iteration != 4408:
        raise ValueError(
            "checkpoint internal iteration is {!r}, expected 4408".format(iteration)
        )
    model = payload.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("checkpoint model mapping is absent or empty")
    return {
        "epoch": int(epoch),
        "iteration": iteration,
        "model_key_count": len(model),
    }


def validate_primary_checkpoint(run):
    checkpoint = require_regular_file(run / "checkpoints" / "epoch-32.pth")
    actual = sha256(checkpoint)
    recorded = parse_sha256_file(run / "checkpoints" / "epoch-32.sha256", checkpoint)
    if actual != recorded:
        raise ValueError("epoch-32 checkpoint SHA-256 differs from recorded sha256sum")
    aliases = {}
    for name in ("best.pth", "last.pth"):
        alias = require_regular_file(run / "checkpoints" / name)
        if alias.resolve() != checkpoint.resolve():
            raise ValueError("{} does not resolve to epoch-32.pth".format(name))
        alias_sha = sha256(alias)
        if alias_sha != actual:
            raise ValueError("{} content differs from epoch-32.pth".format(name))
        aliases[name] = {"path": str(alias), "sha256": alias_sha}
    internal = inspect_checkpoint(checkpoint)
    return checkpoint, actual, recorded, aliases, internal


def validate_metric_bundle(metrics_path, csv_path, class_names, label):
    metrics = load(require_regular_file(metrics_path))
    totals = {}
    for key in (
        "miou",
        "pixel_accuracy",
        "mean_pixel_accuracy",
        "frequency_weighted_iou",
    ):
        totals[key] = finite_number(metrics.get(key), "{} {}".format(label, key), 0.0, 1.0)
    per_class = metrics.get("per_class_iou")
    if not isinstance(per_class, Mapping) or len(per_class) != 13:
        raise ValueError("{} must contain exactly 13 per-class IoUs".format(label))
    if set(per_class) != set(class_names):
        raise ValueError("{} per-class names do not match resolved config".format(label))
    checked_per_class = {
        name: finite_number(per_class[name], "{} IoU {}".format(label, name), 0.0, 1.0)
        for name in class_names
    }
    confusion = metrics.get("confusion_matrix")
    if not isinstance(confusion, list) or len(confusion) != 13:
        raise ValueError("{} confusion matrix must have 13 rows".format(label))
    matrix_sum = 0
    diagonal = 0
    for row_index, row in enumerate(confusion):
        if not isinstance(row, list) or len(row) != 13:
            raise ValueError("{} confusion matrix row {} is not length 13".format(label, row_index))
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("{} confusion entry [{},{}] is invalid".format(label, row_index, column_index))
            matrix_sum += value
            if row_index == column_index:
                diagonal += value
    labeled = metrics.get("labeled_pixels")
    correct = metrics.get("correct_pixels")
    if isinstance(labeled, bool) or not isinstance(labeled, int) or labeled <= 0:
        raise ValueError("{} labeled_pixels is invalid".format(label))
    if isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct <= labeled:
        raise ValueError("{} correct_pixels is invalid".format(label))
    if matrix_sum != labeled or diagonal != correct:
        raise ValueError("{} confusion matrix totals disagree with pixel counts".format(label))

    with require_regular_file(csv_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 13 or set(rows[0]) != {"class", "iou"}:
        raise ValueError("{} per-class CSV must contain exactly 13 class/iou rows".format(label))
    if [row["class"] for row in rows] != list(class_names):
        raise ValueError("{} per-class CSV order/names mismatch".format(label))
    for row in rows:
        value = finite_number(
            float(row["iou"]), "{} CSV IoU".format(label), 0.0, 1.0
        )
        if not math.isclose(value, checked_per_class[row["class"]], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("{} per-class CSV disagrees with metrics JSON".format(label))
    return metrics, totals


def config_differences(left, right):
    result = {}
    for key in COMPARABLE_KEYS:
        if key not in left or key not in right:
            result[key] = {
                "left": left.get(key, "<missing>"),
                "right": right.get(key, "<missing>"),
            }
        elif left[key] != right[key]:
            result[key] = {"left": left[key], "right": right[key]}
    return result


def validate_command(path, required_tokens):
    path = require_regular_file(path)
    content = path.read_text(encoding="utf-8").strip()
    missing = [token for token in required_tokens if token not in content]
    if missing:
        raise ValueError("command evidence {} lacks {}".format(path, missing))
    return {"path": str(path.resolve()), "sha256": sha256(path), "command": content}


def validate_formal_command(path):
    path = require_regular_file(path)
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


def validate_resume_evidence(run, command):
    report_path = run / "status" / "resume_checkpoint_validation.json"
    sha_path = run / "status" / "resume_checkpoint.sha256"
    argv = command["argv"]
    if len(argv) == 11:
        if report_path.exists() or sha_path.exists():
            raise ValueError("non-resumed formal command has stale resume evidence")
        return {"resumed": False}

    checkpoint = require_regular_file(Path(argv[12]))
    expected_parent = (run / "checkpoints").resolve()
    if checkpoint.resolve().parent != expected_parent:
        raise ValueError("resume checkpoint is not owned by this run")
    report = load(require_regular_file(report_path))
    if report.get("status") != "passed" or report.get("exit_code") != 0:
        raise ValueError("resume checkpoint validation did not pass")
    if Path(report.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError("resume report checkpoint path mismatch")
    actual_sha = sha256(checkpoint)
    if report.get("sha256") != actual_sha:
        raise ValueError("resume report checkpoint SHA-256 mismatch")
    if parse_sha256_file(sha_path, checkpoint) != actual_sha:
        raise ValueError("resume sha256sum evidence mismatch")
    epoch = report.get("epoch")
    expected_epochs = set(range(4, 32, 4))
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch not in expected_epochs:
        raise ValueError("resume report epoch is not an allowed boundary")
    if checkpoint.name != "epoch-{}.pth".format(epoch):
        raise ValueError("resume report epoch and checkpoint filename mismatch")
    if report.get("iteration") != 4408:
        raise ValueError("resume report iteration must equal 4408")
    if report.get("model_entry_count", 0) <= 0:
        raise ValueError("resume report model is empty")
    optimizer_keys = report.get("optimizer_keys")
    if not isinstance(optimizer_keys, list) or not {"state", "param_groups"}.issubset(
        optimizer_keys
    ):
        raise ValueError("resume report optimizer evidence is incomplete")
    return {
        "resumed": True,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha,
        "epoch": epoch,
        "iteration": 4408,
        "report": report,
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256(report_path),
    }


def validate_topology_gpu_evidence(run, report):
    inventory_path = require_regular_file(run / "environment" / "gpu_inventory.csv")
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

    smoke_path = require_regular_file(
        run / "smoke" / "ddp" / "configs" / "training_topology.json"
    )
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
        expected_path = require_regular_file(
            run / "status" / "topology_rank_{}.json".format(rank)
        )
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
    path = require_regular_file(run / "configs" / "training_topology.json")
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


def validate_fairness_and_baselines(run, formal_config, formal_checkpoint_sha, class_names):
    gate_path = run / "metrics" / "fairness_gate.json"
    gate = load(require_regular_file(gate_path))
    if gate.get("passed") is not True:
        raise ValueError("published fairness gate did not pass")
    if gate.get("comparable_keys") != list(COMPARABLE_KEYS):
        raise ValueError("published fairness key set differs from finalizer key set")
    formal_metadata_path = run / "configs" / "resolved_config.json"
    if gate.get("formal", {}).get("resolved_config_sha256") != sha256(formal_metadata_path):
        raise ValueError("fairness gate formal resolved-config hash mismatch")
    if gate.get("formal", {}).get("checkpoint_sha256") != formal_checkpoint_sha:
        raise ValueError("fairness gate formal checkpoint hash mismatch")
    formal_command = validate_formal_command(run / "configs" / "command.txt")
    formal_topology = validate_training_topology(run)
    if gate.get("formal", {}).get("command", {}).get("sha256") != formal_command["sha256"]:
        raise ValueError("fairness gate formal command hash mismatch")
    if (
        gate.get("formal", {}).get("training_topology", {}).get("sha256")
        != formal_topology["sha256"]
    ):
        raise ValueError("fairness gate training-topology hash mismatch")

    baseline_evidence = {}
    for name, specification in BASELINES.items():
        mode = specification["mode"]
        baseline_run = specification["run"]
        source_metadata_path = require_regular_file(baseline_run / "metadata.json")
        source_metadata = load(source_metadata_path)
        if source_metadata.get("config_module") != specification["config_module"]:
            raise ValueError("{} source config module mismatch".format(name))
        source_command = str(source_metadata.get("command", "")).strip()
        if not source_command:
            raise ValueError("{} source training command is absent".format(name))
        reeval_dir = run / "baseline_reeval" / mode
        reeval_metadata_path = require_regular_file(
            reeval_dir / "configs" / "resolved_config.json"
        )
        reeval_metadata = load(reeval_metadata_path)
        if reeval_metadata.get("config_module") != specification["config_module"]:
            raise ValueError("{} re-evaluation config module mismatch".format(name))
        source_config = source_metadata["resolved_config"]
        reeval_config = reeval_metadata["resolved_config"]
        parity = {
            "formal_vs_source": config_differences(formal_config, source_config),
            "formal_vs_reevaluation": config_differences(formal_config, reeval_config),
            "source_vs_reevaluation": config_differences(source_config, reeval_config),
        }
        if any(parity.values()):
            raise ValueError("{} independent config parity failed: {}".format(name, parity))
        command = validate_command(
            reeval_dir / "configs" / "command.txt",
            (specification["config_module"], str(baseline_run / "checkpoints" / "epoch-32.pth")),
        )
        validation_path = require_regular_file(reeval_dir / "checkpoint_validation.json")
        validation = load(validation_path)
        checkpoint = require_regular_file(baseline_run / "checkpoints" / "epoch-32.pth")
        actual_sha = sha256(checkpoint)
        expected_sha = specification["expected_sha256"]
        if actual_sha != expected_sha:
            raise ValueError("{} current checkpoint hash differs from preregistered hash".format(name))
        if not (
            validation.get("verified") is True
            and validation.get("sha256_match") is True
            and validation.get("actual_sha256") == actual_sha
            and validation.get("expected_sha256") == expected_sha
            and type(validation.get("checkpoint_epoch")) is int
            and validation.get("checkpoint_epoch") == 32
            and type(validation.get("checkpoint_iteration")) is int
            and validation.get("checkpoint_iteration") == 4408
            and type(validation.get("model_key_count")) is int
            and validation.get("model_key_count") > 0
            and Path(validation.get("checkpoint", "")).resolve() == checkpoint.resolve()
            and validation.get("source_metadata_sha256") == sha256(source_metadata_path)
        ):
            raise ValueError("{} checkpoint preflight evidence is inconsistent".format(name))
        published = gate.get("baselines", {}).get(name, {})
        if not (
            published.get("comparable") is True
            and published.get("checkpoint_validation_ok") is True
            and published.get("checkpoint_sha256") == actual_sha
            and published.get("checkpoint_expected_sha256") == expected_sha
            and not any(published.get("differences", {}).values())
            and published.get("source_resolved_config_sha256") == sha256(source_metadata_path)
            and published.get("source_training_command") == source_command
            and published.get("reevaluation_resolved_config_sha256") == sha256(reeval_metadata_path)
            and published.get("reevaluation_command", {}).get("sha256") == command["sha256"]
        ):
            raise ValueError("{} published fairness evidence is inconsistent".format(name))
        metrics, totals = validate_metric_bundle(
            reeval_dir / "metrics.json",
            reeval_dir / "per_class_iou.csv",
            class_names,
            name,
        )
        baseline_evidence[name] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": actual_sha,
            "expected_checkpoint_sha256": expected_sha,
            "checkpoint_validation": str(validation_path.resolve()),
            "source_metadata": str(source_metadata_path.resolve()),
            "source_metadata_sha256": sha256(source_metadata_path),
            "reevaluation_metadata": str(reeval_metadata_path.resolve()),
            "reevaluation_metadata_sha256": sha256(reeval_metadata_path),
            "reevaluation_command": command,
            "metrics": metrics,
            "total_metrics": totals,
        }
    return gate, baseline_evidence


def validate_diagnostics(run, epoch32_sha):
    expected_names = {
        "validation_loss_epoch_{}.json".format(epoch) for epoch in EXPECTED_EPOCHS
    }
    observed_names = {path.name for path in (run / "metrics").glob("validation_loss_epoch_*.json")}
    if observed_names != expected_names:
        raise ValueError("expected exactly eight fixed validation-loss JSON files")
    expected_preflight_names = {
        "validation_loss_preflight_epoch_{}.json".format(epoch)
        for epoch in EXPECTED_EPOCHS
    }
    observed_preflight_names = {
        path.name
        for path in (run / "metrics").glob("validation_loss_preflight_epoch_*.json")
    }
    if observed_preflight_names != expected_preflight_names:
        raise ValueError("expected exactly eight fixed validation-loss preflight JSON files")
    reports = {}
    preflights = {}
    checkpoint_hashes = {}
    for epoch in EXPECTED_EPOCHS:
        path = require_regular_file(
            run / "metrics" / "validation_loss_epoch_{}.json".format(epoch)
        )
        report = load(path)
        preflight_path = require_regular_file(
            run / "metrics" / "validation_loss_preflight_epoch_{}.json".format(epoch)
        )
        preflight = load(preflight_path)
        if type(report.get("epoch")) is not int or report.get("epoch") != epoch:
            raise ValueError("validation-loss epoch field mismatch for {}".format(epoch))
        checkpoint = require_regular_file(run / "checkpoints" / "epoch-{}.pth".format(epoch))
        current_sha = epoch32_sha if epoch == 32 else sha256(checkpoint)
        for payload, label in ((preflight, "preflight"), (report, "validation")):
            if type(payload.get("checkpoint_epoch")) is not int or payload.get("checkpoint_epoch") != epoch:
                raise ValueError("{} checkpoint epoch mismatch for {}".format(label, epoch))
            if type(payload.get("expected_epoch")) is not int or payload.get("expected_epoch") != epoch:
                raise ValueError("{} expected epoch mismatch for {}".format(label, epoch))
            if Path(payload.get("checkpoint", "")).resolve() != checkpoint.resolve():
                raise ValueError("{} checkpoint path mismatch for {}".format(label, epoch))
            if payload.get("checkpoint_sha256") != current_sha:
                raise ValueError("{} current checkpoint SHA-256 mismatch for {}".format(label, epoch))
            if payload.get("checkpoint_size_bytes") != checkpoint.stat().st_size:
                raise ValueError("{} checkpoint size mismatch for {}".format(label, epoch))
        mean_ce = finite_number(report.get("mean_cross_entropy"), "epoch {} validation CE".format(epoch), 0.0)
        ce_sum = finite_number(report.get("cross_entropy_sum"), "epoch {} CE sum".format(epoch), 0.0)
        valid_pixels = report.get("valid_pixels")
        if isinstance(valid_pixels, bool) or not isinstance(valid_pixels, int) or valid_pixels <= 0:
            raise ValueError("epoch {} valid pixel count is invalid".format(epoch))
        if report.get("sample_count") != EXPECTED_EVAL_SAMPLE_COUNT:
            raise ValueError("epoch {} validation sample count mismatch".format(epoch))
        if not math.isclose(mean_ce, ce_sum / valid_pixels, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("epoch {} validation loss arithmetic mismatch".format(epoch))
        if not str(report.get("protocol", "")).strip():
            raise ValueError("epoch {} validation protocol is absent".format(epoch))
        reports[epoch] = report
        preflights[epoch] = preflight
        checkpoint_hashes[str(epoch)] = current_sha

    protocol_path = require_regular_file(run / "metrics" / "loss_curves_protocol.json")
    protocol = load(protocol_path)
    expected_validation_protocol = reports[EXPECTED_EPOCHS[0]]["protocol"]
    if not (
        protocol.get("checkpoint_epochs") == list(EXPECTED_EPOCHS)
        and protocol.get("checkpoint_sha256") == checkpoint_hashes
        and protocol.get("train_column") == TRAIN_LOSS_COLUMN
        and protocol.get("validation_column") == VALIDATION_LOSS_COLUMN
        and isinstance(protocol.get("train_protocol"), str)
        and bool(protocol.get("train_protocol", "").strip())
        and protocol.get("validation_protocol") == expected_validation_protocol
        and "not used for checkpoint selection" in str(protocol.get("purpose", ""))
        and all(report["protocol"] == expected_validation_protocol for report in reports.values())
    ):
        raise ValueError("loss_curves_protocol.json is inconsistent with fixed diagnostics")

    csv_path = require_regular_file(run / "metrics" / "loss_curves.csv")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        "epoch",
        TRAIN_LOSS_COLUMN,
        VALIDATION_LOSS_COLUMN,
        "train_protocol",
        "validation_protocol",
    }
    if len(rows) != 8 or not rows or set(rows[0]) != required_columns:
        raise ValueError("loss_curves.csv must contain eight rows and the frozen columns")
    if [int(row["epoch"]) for row in rows] != list(EXPECTED_EPOCHS):
        raise ValueError("loss_curves.csv epoch sequence mismatch")
    for row in rows:
        epoch = int(row["epoch"])
        finite_number(
            float(row[TRAIN_LOSS_COLUMN]), "epoch {} train CE".format(epoch), 0.0
        )
        csv_validation = finite_number(
            float(row[VALIDATION_LOSS_COLUMN]),
            "epoch {} CSV validation CE".format(epoch),
            0.0,
        )
        if not math.isclose(
            csv_validation,
            float(reports[epoch]["mean_cross_entropy"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("loss_curves.csv validation CE mismatch for epoch {}".format(epoch))
        if row["validation_protocol"] != reports[epoch]["protocol"]:
            raise ValueError("loss_curves.csv protocol mismatch for epoch {}".format(epoch))
        if row["train_protocol"] != protocol["train_protocol"]:
            raise ValueError("loss_curves.csv train protocol mismatch for epoch {}".format(epoch))
    png_path = require_regular_file(run / "metrics" / "loss_curves.png")
    with png_path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("loss_curves.png has an invalid PNG signature")
    return {
        "epochs": list(EXPECTED_EPOCHS),
        "reports": {str(epoch): reports[epoch] for epoch in EXPECTED_EPOCHS},
        "preflight_reports": {
            str(epoch): preflights[epoch] for epoch in EXPECTED_EPOCHS
        },
        "checkpoint_sha256": checkpoint_hashes,
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256(csv_path),
        "plot": str(png_path.resolve()),
        "plot_sha256": sha256(png_path),
        "protocol": protocol,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
    }


def validate_comparison(run, formal_sha, formal_metrics, baseline_evidence):
    path = require_regular_file(run / "metrics" / "comparison.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3 or {row.get("method") for row in rows} != {"REL+", "HHA", "RawDepth"}:
        raise ValueError("comparison.csv must contain exactly REL+, HHA, and RawDepth")
    by_method = {row["method"]: row for row in rows}
    expected_hashes = {"REL+": formal_sha}
    expected_hashes.update(
        {name: evidence["checkpoint_sha256"] for name, evidence in baseline_evidence.items()}
    )
    expected_metrics = {"REL+": formal_metrics}
    expected_metrics.update(
        {name: evidence["metrics"] for name, evidence in baseline_evidence.items()}
    )
    for name, row in by_method.items():
        if row.get("comparable") != "True":
            raise ValueError("{} comparison row is not marked comparable".format(name))
        if row.get("checkpoint_sha256") != expected_hashes[name]:
            raise ValueError("{} comparison checkpoint hash mismatch".format(name))
        for column, key in (
            ("miou_percent", "miou"),
            ("pixel_accuracy_percent", "pixel_accuracy"),
            ("mean_accuracy_percent", "mean_pixel_accuracy"),
        ):
            observed = finite_number(float(row[column]), "{} {}".format(name, column))
            expected = float(expected_metrics[name][key]) * 100.0
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("{} comparison metric mismatch".format(name))
    return rows


def finalize(run):
    exit_codes = validate_exit_codes(run)
    code_manifest = validate_code_manifest(run)
    cache_validation, stats = validate_cache(run)
    semantics_validation = validate_semantics(run)
    checkpoint, checkpoint_sha, recorded_sha, aliases, checkpoint_internal = (
        validate_primary_checkpoint(run)
    )
    formal_metadata_path = require_regular_file(run / "configs" / "resolved_config.json")
    formal_metadata = load(formal_metadata_path)
    if formal_metadata.get("config_module") != "configs.cmx_relplus_2d":
        raise ValueError("unexpected formal config module")
    config = formal_metadata["resolved_config"]
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
        if config.get(key) != expected:
            raise ValueError("formal REL+ identity mismatch for {}".format(key))
    initialization = validate_initialization_report(run, config)
    class_names = config.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != 13 or len(set(class_names)) != 13:
        raise ValueError("resolved config must define 13 unique class names")
    formal_metrics, formal_totals = validate_metric_bundle(
        run / "metrics" / "metrics.json",
        run / "metrics" / "per_class_iou.csv",
        class_names,
        "REL+",
    )
    fairness, baseline_evidence = validate_fairness_and_baselines(
        run, config, checkpoint_sha, class_names
    )
    diagnostics = validate_diagnostics(run, checkpoint_sha)
    comparison = validate_comparison(
        run, checkpoint_sha, formal_metrics, baseline_evidence
    )
    topology = validate_training_topology(run)
    command = validate_formal_command(run / "configs" / "command.txt")
    resume_evidence = validate_resume_evidence(run, command)
    manifest = {
        "experiment_id": run.name,
        "model_name": RELPLUS_SPEC["model_name"],
        "config_name": RELPLUS_SPEC["config_name"],
        "config_module": "configs.cmx_relplus_2d",
        "modality": "relplus",
        "representation_semantics": RELPLUS_SPEC["representation_semantics"],
        "representation_version": RELPLUS_SPEC["representation_version"],
        "representation_spec_sha256": RELPLUS_SPEC_SHA256,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_path": "/home/zhuzhaoziao/rel_exp/cmx_rel+",
        "source_cmx_path": "/home/zhuzhaoziao/rel_exp/cmx",
        "source_cmx_commit": "e251d860aebc2f583a6c4919877e6bebe7f1aff3",
        "source_rel_path": "/data/bxh_copy/Pano_MA_Seg",
        "source_rel_commit": "unknown-no-git-ref",
        "dataset": config["dataset_path"],
        "train_split": config["train_source"],
        "test_split": config["eval_source"],
        "backbone": config["backbone"],
        "seed_configured": config["seed"],
        "effective_distributed_seeds": config["effective_distributed_seeds"],
        "camera_metadata": "Pose/<area>/<sample>.json: K and W2C [R|t]",
        "rel_parameters": {
            "channel_order": ["ReD", "EGVIA", "LOA"],
            "alpha_degrees": 45.0,
            "lambda": 0.5,
            "normal_radius": 3,
            "point_frame": RELPLUS_SPEC["point_frame"],
            "translation_in_red_loa": RELPLUS_SPEC["translation_in_red_loa"],
        },
        "initialization_checkpoint": config["pretrained_model"],
        "resolved_config": str(formal_metadata_path.resolve()),
        "resolved_config_sha256": sha256(formal_metadata_path),
        "code_manifest": code_manifest,
        "initialization_evidence": initialization,
        "training_command": command,
        "resume_evidence": resume_evidence,
        "training_topology": topology,
        "output_checkpoint": str(checkpoint.resolve()),
        "output_checkpoint_sha256": checkpoint_sha,
        "recorded_checkpoint_sha256": recorded_sha,
        "checkpoint_internal": checkpoint_internal,
        "checkpoint_aliases": aliases,
        "best_epoch": 32,
        "last_epoch": 32,
        "checkpoint_selection": (
            "epoch 32 was the only preregistered eligible checkpoint; best == last"
        ),
        "best_miou": formal_metrics["miou"],
        "pixel_accuracy": formal_metrics["pixel_accuracy"],
        "mean_pixel_accuracy": formal_metrics["mean_pixel_accuracy"],
        "formal_metrics": formal_metrics,
        "formal_total_metrics": formal_totals,
        "baseline_evidence": baseline_evidence,
        "comparison": comparison,
        "fairness_gate": fairness,
        "diagnostic_loss_curves": diagnostics,
        "cache_validation": cache_validation,
        "cache_statistics": stats,
        "semantics_validation": semantics_validation,
        "stage_exit_codes": exit_codes,
        "configuration_deviations": True,
    }
    status = """# COMPLETE

All preregistered gates passed, including cache integrity, epoch-32 checkpoint integrity,
exact metrics, eight fixed loss diagnostics, baseline hashes, and independent fairness checks.

- Primary checkpoint: `{checkpoint}`
- Checkpoint SHA-256: `{digest}`
- mIoU: `{miou:.6f}%`
- Pixel accuracy: `{pixel:.6f}%`
- Mean accuracy: `{mean:.6f}%`
- Selection rule: epoch 32 only; best equals last.
""".format(
        checkpoint=checkpoint.resolve(),
        digest=checkpoint_sha,
        miou=formal_metrics["miou"] * 100.0,
        pixel=formal_metrics["pixel_accuracy"] * 100.0,
        mean=formal_metrics["mean_pixel_accuracy"] * 100.0,
    )
    atomic_json(run / "RUN_MANIFEST.yaml", manifest)
    atomic_write(run / "STATUS.md", status)
    atomic_write(run / "status" / "current_stage", "complete\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    atomic_write(run / "status" / "finalize.exitcode", "1\n")
    atomic_write(run / "status" / "current_stage", "finalize\n")
    atomic_write(
        run / "STATUS.md",
        "# FINALIZATION IN PROGRESS\n\nNo COMPLETE marker is valid until every final gate passes.\n",
    )
    try:
        finalize(run)
    except BaseException:
        atomic_write(run / "status" / "finalize.exitcode", "1\n")
        raise
    atomic_write(run / "status" / "finalize.exitcode", "0\n")


if __name__ == "__main__":
    main()
