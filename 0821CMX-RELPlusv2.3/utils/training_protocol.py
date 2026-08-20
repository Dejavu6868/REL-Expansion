"""Executable training defaults aligned to the frozen REL author source."""

import os
import random
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from utils.init_func import group_weight
from utils.loss_opr import FocalLoss2d


V2_3_INTEGRATION_PROTOCOL_ID = "CMX_RELPLUS_V2_3"


def _load_json_report(path, label):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("{} report does not exist: {}".format(label, path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("{} report is unreadable: {}".format(label, error))


def _absolute(value):
    return str(Path(value).expanduser().resolve())


def _read_nonempty_lines(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _manifest_split_ids(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [row.get("sample_id", "") for row in rows]
    if not sample_ids or any(not sample_id for sample_id in sample_ids):
        raise RuntimeError("manifest contains an empty sample ID")
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("manifest sample IDs are not unique")
    unexpected = sorted({row.get("split") for row in rows} - {"train", "test"})
    if unexpected:
        raise RuntimeError("manifest contains unexpected splits: {}".format(unexpected))
    return {
        "all": sample_ids,
        "train": [row["sample_id"] for row in rows if row["split"] == "train"],
        "test": [row["sample_id"] for row in rows if row["split"] == "test"],
    }


def validate_ordered_runtime_splits(config):
    manifest = _manifest_split_ids(config.full_manifest)
    resolved = _manifest_split_ids(config.resolved_manifest_path)
    if resolved != manifest:
        raise RuntimeError("resolved manifest ordered sample IDs differ from the manifest")
    train_ids = _read_nonempty_lines(config.train_source)
    test_ids = _read_nonempty_lines(config.eval_source)
    if train_ids != manifest["train"]:
        raise RuntimeError("train ordered sample IDs differ from the manifest")
    if test_ids != manifest["test"]:
        raise RuntimeError("test ordered sample IDs differ from the manifest")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise RuntimeError("runtime split sample IDs are not unique")
    if set(train_ids) & set(test_ids):
        raise RuntimeError("runtime train/test sample IDs overlap")
    if set(train_ids) | set(test_ids) != set(manifest["all"]):
        raise RuntimeError("runtime split union differs from the manifest")
    if len(train_ids) != int(config.num_train_imgs):
        raise RuntimeError("runtime train count differs from config")
    if len(test_ids) != int(config.num_eval_imgs):
        raise RuntimeError("runtime test count differs from config")
    return {
        "manifest_count": len(manifest["all"]),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
    }


def assert_v2_3_data_ready(config):
    generation = _load_json_report(
        config.cache_generation_report, "cache generation"
    )
    audit = _load_json_report(config.cache_audit_report, "cache audit")
    preflight = _load_json_report(
        config.training_data_preflight_report, "training-data preflight"
    )
    expected_identity = {
        "integration_protocol_id": config.integration_protocol_id,
        "representation_protocol_id": config.representation_protocol_id,
        "cache_root": _absolute(config.formal_cache_root),
        "rel_plus_root": _absolute(config.x_root_folder),
        "valid_mask_root": _absolute(config.x_valid_root_folder),
        "manifest_path": _absolute(config.full_manifest),
        "resolved_manifest_path": _absolute(config.resolved_manifest_path),
        "train_source": _absolute(config.train_source),
        "eval_source": _absolute(config.eval_source),
        "manifest_count": int(config.num_train_imgs + config.num_eval_imgs),
        "train_count": int(config.num_train_imgs),
        "test_count": int(config.num_eval_imgs),
        "failure_count": 0,
    }
    expected_generation_identity = dict(
        expected_identity,
        resolved_manifest_path=_absolute(
            config.generation_resolved_manifest_path
        ),
    )
    for field, expected in expected_generation_identity.items():
        found = generation.get(field)
        if found != expected:
            raise RuntimeError(
                "cache generation {} mismatch: expected {!r}, found {!r}".format(
                    field, expected, found
                )
            )
    expected_count = expected_identity["manifest_count"]
    expected_generation_state = {
        "status": "PASS",
        "selected_count": expected_count,
        "generated_or_resumed_count": expected_count,
        "full_cache_generated": True,
        "dry_run": False,
    }
    for field, expected in expected_generation_state.items():
        found = generation.get(field)
        if found != expected:
            raise RuntimeError(
                "cache generation {} mismatch: expected {!r}, found {!r}".format(
                    field, expected, found
                )
            )
    for label, report in (("cache audit", audit), ("training-data preflight", preflight)):
        if report.get("status") != "PASS":
            raise RuntimeError("{} status must be PASS".format(label))
        for field, expected in expected_identity.items():
            found = report.get(field)
            if found != expected:
                raise RuntimeError(
                    "{} {} mismatch: expected {!r}, found {!r}".format(
                        label, field, expected, found
                    )
                )
    if audit.get("regeneration_count") != 70:
        raise RuntimeError("cache audit regeneration_count must be 70")
    if audit.get("regeneration_failure_count") != 0:
        raise RuntimeError("cache audit regeneration failures must be zero")
    if preflight.get("sample_count") != expected_identity["manifest_count"]:
        raise RuntimeError("training-data preflight sample_count mismatch")
    if preflight.get("all_samples_decoded_this_run") is not True:
        raise RuntimeError("training-data preflight must decode every sample this run")
    if _absolute(preflight.get("class_mapping", "")) != _absolute(config.class_mapping):
        raise RuntimeError("training-data preflight class_mapping mismatch")
    split_counts = validate_ordered_runtime_splits(config)
    return {
        "cache_generation": generation,
        "cache_audit": audit,
        "training_data_preflight": preflight,
        "split_counts": split_counts,
    }


def _assert_legacy_data_ready(config):
    audit_path = getattr(config, "cache_audit_report", None)
    if audit_path:
        report = _load_json_report(audit_path, "cache audit")
        expected = {
            "status": "PASS",
            "integration_protocol_id": getattr(
                config, "integration_protocol_id", getattr(config, "protocol_id", None)
            ),
            "representation_protocol_id": getattr(
                config, "representation_protocol_id", None
            ),
            "manifest_count": int(config.num_train_imgs + config.num_eval_imgs),
            "train_count": int(config.num_train_imgs),
            "test_count": int(config.num_eval_imgs),
            "failure_count": 0,
        }
        for field, value in expected.items():
            if report.get(field) != value:
                raise RuntimeError(
                    "cache audit report {} mismatch: expected {!r}, found {!r}".format(
                        field, value, report.get(field)
                    )
                )
        return report
    if not getattr(config, "data_ready", False):
        raise RuntimeError(
            "Formal data is not ready. Keep data_ready=False until the full "
            "REL+ cache has passed its separate audit."
        )
    return {"status": "PASS", "source": "legacy_data_ready_flag"}


def assert_training_ready(config):
    if not getattr(config, "training_authorized", False):
        raise RuntimeError(
            "Formal training is not authorized. Set training_authorized=True "
            "only after explicit user approval."
        )
    if getattr(config, "integration_protocol_id", None) == V2_3_INTEGRATION_PROTOCOL_ID:
        if not getattr(config, "source_compatible_invalid_accepted", False):
            raise RuntimeError(
                "SOURCE_COMPAT_STORAGE_255 must be explicitly accepted for V2.3"
            )
        return assert_v2_3_data_ready(config)
    return _assert_legacy_data_ready(config)


def assert_runtime_dataset_contract(config, *, require_cache_audit=False):
    counts = {}
    for field, expected in (
        ("train_source", int(config.num_train_imgs)),
        ("eval_source", int(config.num_eval_imgs)),
    ):
        path = Path(getattr(config, field))
        if not path.is_file():
            raise RuntimeError("runtime dataset list is missing: {}".format(path))
        with path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != expected:
            raise RuntimeError(
                "{} count mismatch: expected {}, found {}".format(
                    field, expected, count
                )
            )
        counts[field] = count
    if require_cache_audit:
        if getattr(config, "integration_protocol_id", None) == V2_3_INTEGRATION_PROTOCOL_ID:
            reports = assert_v2_3_data_ready(config)
            counts.update(reports)
        else:
            authorized = getattr(config, "training_authorized", False)
            try:
                config.training_authorized = True
                audit = assert_training_ready(config)
            finally:
                config.training_authorized = authorized
            counts["cache_audit"] = audit
    return counts


def set_author_seed(
    base_seed, *, epoch=None, local_rank=0, distributed=False
):
    seed = int(base_seed)
    if epoch is not None:
        seed += int(epoch)
    if distributed:
        seed += int(local_rank) * 1000
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def configure_author_cudnn():
    cudnn.benchmark = False
    cudnn.deterministic = False
    return {
        "benchmark": bool(cudnn.benchmark),
        "deterministic": bool(cudnn.deterministic),
    }


def build_author_criterion(config):
    if config.criterion == "Focal":
        return FocalLoss2d(
            gamma=getattr(config, "focal_gamma", 2),
            reduction="none",
            ignore_index=config.background,
        )
    if config.criterion in ("CE", "CrossEntropy"):
        return nn.CrossEntropyLoss(
            reduction="none", ignore_index=config.background
        )
    raise NotImplementedError("unsupported criterion: {}".format(config.criterion))


def build_author_optimizer(model, norm_layer, config):
    params = group_weight([], model, norm_layer, config.lr)
    if config.optimizer == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    if config.optimizer in ("SGD", "SGDM"):
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise NotImplementedError("unsupported optimizer: {}".format(config.optimizer))
