"""Executable training defaults aligned to the frozen REL author source."""

import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from utils.init_func import group_weight
from utils.loss_opr import FocalLoss2d


def assert_training_ready(config):
    if not getattr(config, "training_authorized", False):
        raise RuntimeError(
            "Formal training is not authorized. Set training_authorized=True "
            "only after explicit user approval."
        )
    audit_path = getattr(config, "cache_audit_report", None)
    if audit_path:
        path = Path(audit_path)
        if not path.is_file():
            raise RuntimeError("cache audit report does not exist: {}".format(path))
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError("cache audit report is unreadable: {}".format(error))
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
