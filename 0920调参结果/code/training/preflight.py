#!/usr/bin/env python3
"""CPU-only evidence and resource checks for the separately authorized suite."""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path


sys.dont_write_bytecode = True

from suite_common import (
    build_config,
    configure_imports,
    digest,
    dump_json,
    load_suite,
    validate_config,
    baseline_config_comparison,
)
import resource_guard


ARMS = ("hha", "relplus")
EXPECTED_SOURCE_COUNT = 174
MINIMUM_OUTPUT_FREE_BYTES = 250 * 1024 ** 3
REQUIRED_SHARED_FILES = (
    "full_manifest",
    "resolved_manifest_path",
    "train_source",
    "eval_source",
    "class_mapping",
    "training_data_preflight_report",
)


def json_value(value):
    """Preserve complete configurations, including NumPy values and tuples."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "item"):
        return json_value(value.item())
    raise TypeError("Unsupported resolved configuration value: {}".format(type(value)))


def hash_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_integrity(root, expected):
    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError("Source root is missing: {}".format(root))
    actual = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if len(expected) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError("Frozen source manifest must contain exactly 174 Python files")
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
    if missing or unexpected or changed:
        raise RuntimeError("Source integrity mismatch at {}: {}".format(root, {
            "missing": missing, "unexpected": unexpected, "changed": changed,
        }))
    return {"status": "PASS", "root": str(root), "file_count": len(actual),
            "source_hashes": actual, "ordered_manifest_sha256": hash_json(actual)}


def readonly_inputs(expected):
    rows = expected.get("additional_readonly_inputs")
    if not isinstance(rows, list) or len(rows) < 3:
        raise RuntimeError("Frozen pretrained and split fingerprints are incomplete")
    result = []
    for row in rows:
        path = Path(row["path"])
        actual = digest(path)
        if actual != row["sha256"]:
            raise RuntimeError("Read-only input SHA-256 mismatch: {}".format(path))
        result.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                       "sha256": actual, "status": "PASS"})
    return result


def ordered_ids(path, expected_count):
    with Path(path).open("r", encoding="utf-8") as handle:
        values = [line.strip() for line in handle if line.strip()]
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise RuntimeError("Split count or uniqueness mismatch: {}".format(path))
    return values


def file_evidence(config, arm):
    required = set(REQUIRED_SHARED_FILES)
    if arm == "relplus":
        required.update(("cache_generation_report", "generation_resolved_manifest_path",
                         "cache_audit_report"))
    else:
        required.add("input_audit_report")
    fields = required | {
        key for key in config
        if key.endswith("_report") or "manifest" in key
    }
    result = {}
    for field in sorted(fields):
        value = getattr(config, field, None)
        if not isinstance(value, (str, Path)) or not str(value):
            if field in required:
                raise RuntimeError("Missing required data evidence field: {}".format(field))
            continue
        path = Path(value).resolve()
        if not path.is_file():
            if field in required:
                raise RuntimeError("Required data evidence is missing: {}".format(path))
            result[field] = {"path": str(path), "exists": False,
                             "required_for_data_preflight": False}
            continue
        result[field] = {"path": str(path), "exists": True,
                         "required_for_data_preflight": field in required,
                         "size_bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def sampler_evidence(config, train_ids):
    from dataloader.samplers import FixedLengthDistributedSampler

    if (config.batch_size, config.niters_per_epoch,
            config.logical_samples_per_epoch, config.sampler_padding_count) != (56, 945, 52920, 17):
        raise RuntimeError("Batch/sampler budget is inconsistent")
    epochs = {}
    for epoch in (1, 2):
        rank_rows = []
        merged = []
        for rank in range(8):
            sampler = FixedLengthDistributedSampler(
                train_ids, logical_samples_per_epoch=config.logical_samples_per_epoch,
                num_replicas=8, rank=rank, shuffle=True, seed=config.seed,
            )
            sampler.set_epoch(epoch)
            indices = list(sampler)
            if len(indices) != 6615 or len(sampler) != 6615 or sampler.padding_count != 17:
                raise RuntimeError("Unexpected sampler rank length or padding")
            if len(indices) // 7 != config.niters_per_epoch or len(indices) % 7:
                raise RuntimeError("Sampler would yield a partial or missing training batch")
            if any(index < 0 or index >= len(train_ids) for index in indices):
                raise RuntimeError("Sampler index is outside the real training set")
            sample_ids = [train_ids[index] for index in indices]
            merged.extend(indices)
            rank_rows.append({"rank": rank, "sample_count": len(indices),
                              "complete_batches": len(indices) // 7,
                              "ordered_sample_ids_sha256": hash_json(sample_ids),
                              "first_sample_id": sample_ids[0], "last_sample_id": sample_ids[-1]})
        unique = set(merged)
        if len(merged) != 52920 or unique != set(range(52903)):
            raise RuntimeError("Sampler fails complete real-sample coverage")
        epochs[str(epoch)] = {"rank_sequences": rank_rows, "logical_count": len(merged),
                              "real_unique_count": len(unique),
                              "padding_occurrences": len(merged) - len(unique),
                              "complete_real_sample_coverage": True}
    return {"status": "PASS", "world_size": 8, "epochs": epochs,
            "sequence_digest": hash_json(epochs), "image_decoding_performed": False}


def eval_bn_evidence(config):
    import torch
    import torch.nn as nn
    from models.builder import EncoderDecoder

    torch.set_num_threads(1)
    model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    model.eval()
    rows = []
    differences = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            row = {"name": name, "type": type(layer).__name__, "eps": float(layer.eps),
                   "momentum": layer.momentum, "track_running_stats": layer.track_running_stats,
                   "affine": layer.affine, "num_features": layer.num_features}
            rows.append(row)
            if name.startswith("decode_head.") and (
                    layer.eps != config.bn_eps or layer.momentum != config.bn_momentum):
                differences.append(dict(row, configured_training_eps=config.bn_eps,
                                        configured_training_momentum=config.bn_momentum))
    if not rows or not any(row["name"].startswith("decode_head.") for row in rows):
        raise RuntimeError("Expected BN layers were not found in the CPU evaluation model")
    result = {
        "status": "KNOWN_FROZEN_EVALUATOR_DIFFERENCE" if differences else "NO_CONSTRUCTOR_DIFFERENCE_OBSERVED",
        "constructor": "EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)",
        "device": "cpu", "checkpoint_loaded": False, "forward_executed": False,
        "pretrained_loaded": False, "layers": rows, "decoder_configuration_differences": differences,
        "scientific_effect": "UNKNOWN; no checkpoint forward or score comparison was performed",
        "frozen_evaluator_modified": False,
        "training_runtime_bn_confirmation": "Recorded separately by the new-suite DDP smoke",
    }
    del model
    return result


def available_output_space(output_root):
    existing = Path(output_root)
    while not existing.exists():
        if existing.parent == existing:
            raise RuntimeError("Cannot find an existing output filesystem")
        existing = existing.parent
    disk = shutil.disk_usage(str(existing))
    result = {"queried_path": str(existing), "free_bytes": disk.free,
              "total_bytes": disk.total, "required_free_bytes": MINIMUM_OUTPUT_FREE_BYTES}
    if disk.free < MINIMUM_OUTPUT_FREE_BYTES:
        raise RuntimeError("Output filesystem has less than 250 GiB free: {}".format(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    args = parser.parse_args(argv)
    report = {"status": "RUNNING", "started_at_unix": time.time(),
              "suite_path": str(args.suite.resolve()), "optimizer_steps_executed": 0,
              "old_scientific_checkpoint_rehashed": False}
    report_path = None
    try:
        suite = load_suite(args.suite)
        if suite.get("authorization", {}).get("approved") is not True:
            raise RuntimeError("Suite authorization.approved must be explicitly true")
        if tuple(suite.get("arms", [])) != ARMS:
            raise RuntimeError("Suite must specify the ordered arms hha, relplus")
        bundle = Path(__file__).resolve().parent
        if bundle != Path(suite["remote_source_root"]).resolve():
            raise RuntimeError("Preflight is not running from the authorized bundle")
        output_root = Path(suite["output_root"]).resolve()
        report_path = output_root / "preflight.json"
        report["suite_sha256"] = digest(args.suite)
        report["authorization"] = suite["authorization"]
        report["output_disk_before"] = available_output_space(output_root)
        snapshot = resource_guard.resources()
        report["resources_before"] = snapshot
        resource_guard.require_idle(snapshot)
        expected_path = bundle / "frozen_integrity.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        report["frozen_integrity_sha256"] = digest(expected_path)
        report["source_integrity"] = {
            "bundle": source_integrity(bundle / "source", expected["source_hashes"]),
            "frozen_original": source_integrity(suite["frozen_source_root"], expected["source_hashes"]),
        }
        report["readonly_inputs"] = readonly_inputs(expected)
        protected_input_paths = {row["path"] for row in report["readonly_inputs"]}
        configure_imports(suite)
        import cv2
        import numpy as np
        import torch
        import utils.training_protocol as training_protocol

        if Path(training_protocol.__file__).resolve().parent.parent != bundle / "source":
            raise RuntimeError("Training contract was imported outside the verified source snapshot")
        report["environment"] = {
            "python": platform.python_version(), "python_executable": sys.executable,
            "pytorch": torch.__version__, "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "opencv": cv2.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "training_protocol_source": str(Path(training_protocol.__file__).resolve()),
        }
        report["arms"] = {}
        baseline_ids = None
        baseline_sampler = None
        bn_config = None
        for arm in ARMS:
            config = build_config(suite, arm, mode="formal")
            validate_config(config, suite, arm)
            for field in ("pretrained_model", "train_source", "eval_source"):
                if str(Path(getattr(config, field)).resolve()) not in protected_input_paths:
                    raise RuntimeError("{} does not refer to a fingerprinted frozen input".format(field))
            training_protocol.assert_training_ready(config)
            training_protocol.assert_runtime_dataset_contract(config, require_cache_audit=True)
            train_ids = ordered_ids(config.train_source, 52903)
            test_ids = ordered_ids(config.eval_source, 17593)
            if set(train_ids) & set(test_ids):
                raise RuntimeError("Train/test sample IDs overlap")
            pair = (train_ids, test_ids)
            if baseline_ids is None:
                baseline_ids = pair
            elif pair != baseline_ids:
                raise RuntimeError("The two arms do not share identical ordered train/test IDs")
            sample_report = sampler_evidence(config, train_ids)
            if baseline_sampler is None:
                baseline_sampler = sample_report["sequence_digest"]
            elif baseline_sampler != sample_report["sequence_digest"]:
                raise RuntimeError("Two-arm epoch 1/2 sampler sequences differ")
            resolved_path = output_root / "configs" / "{}.json".format(arm)
            dump_json(resolved_path, json_value(dict(config)))
            report["arms"][arm] = {
                "status": "PASS", "resolved_config": str(resolved_path),
                "baseline_comparison": baseline_config_comparison(config, arm),
                "resolved_config_sha256": digest(resolved_path),
                "train_count": len(train_ids), "test_count": len(test_ids),
                "train_test_disjoint": True, "train_ordered_ids_sha256": hash_json(train_ids),
                "test_ordered_ids_sha256": hash_json(test_ids),
                "data_contract_checks": ["assert_training_ready", "assert_runtime_dataset_contract"],
                "data_evidence_files": file_evidence(config, arm), "sampler": sample_report,
                "representation_protocol_id": config.representation_protocol_id,
                "integration_protocol_id": config.integration_protocol_id,
            }
            if bn_config is None:
                bn_config = config
        report["cross_arm_ordered_splits_exact"] = True
        report["cross_arm_epoch_1_2_sampler_sequences_exact"] = True
        report["frozen_evaluator_bn"] = eval_bn_evidence(bn_config)
        snapshot = resource_guard.resources()
        report["resources_after"] = snapshot
        resource_guard.require_idle(snapshot)
        report["output_disk_after"] = available_output_space(output_root)
        report.update(status="PASS", completed_at_unix=time.time())
        dump_json(report_path, json_value(report))
        print(json.dumps({"status": "PASS", "report": str(report_path)}, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        report.update(status="FAIL", completed_at_unix=time.time(),
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if report_path is not None:
            dump_json(report_path, json_value(report))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
