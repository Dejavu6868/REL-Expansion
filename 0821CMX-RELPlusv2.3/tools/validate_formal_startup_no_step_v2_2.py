#!/usr/bin/env python3
"""Build the shared formal runtime and run one backward without any step."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _gradient_contract(model):
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return {
        "tensor_count": len(gradients),
        "all_finite": bool(gradients)
        and all(bool(torch.isfinite(value).all()) for value in gradients),
        "any_nonzero": bool(gradients)
        and any(bool(torch.count_nonzero(value)) for value in gradients),
    }


def _pilot_config(base, pilot_root, audit_report, output_root):
    config = copy.deepcopy(base)
    config.experiment_name = "CMX_RELPlus_v2_2_formal_startup_no_step"
    config.x_root_folder = str(pilot_root / "RELPlus")
    config.x_valid_root_folder = str(pilot_root / "ValidMask")
    config.train_source = str(pilot_root / "train.txt")
    config.eval_source = str(pilot_root / "test.txt")
    config.num_train_imgs = 30
    config.num_eval_imgs = 6
    config.logical_samples_per_epoch = 32
    config.niters_per_epoch = 4
    config.num_workers = 0
    config.cache_audit_report = str(audit_report)
    config.training_authorized = True
    config.full_cache_authorized = False
    config.data_ready = False
    config.output_dir = str(output_root)
    config.log_dir = str(output_root / "logs")
    config.log_file = str(output_root / "logs" / "train.log")
    config.tb_dir = str(output_root / "tensorboard")
    config.checkpoint_dir = str(output_root / "checkpoints_not_written")
    config.log_dir_link = str(output_root / "latest_logs")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--cache-audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("single", "ddp_mock"), default="single")
    from engine.engine import Engine

    with Engine(custom_parser=parser) as engine:
        args = engine.args
        if args.mode == "ddp_mock":
            engine.distributed = True
            engine.local_rank = 0
            engine.world_size = 2

        from configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_2_formal import (
            config as formal,
        )
        from utils.training_protocol import (
            assert_runtime_dataset_contract,
            assert_training_ready,
        )
        from utils.training_runtime import (
            build_training_runtime,
            sanitize_author_loss_map,
        )

        config = _pilot_config(
            formal,
            args.pilot_root.resolve(),
            args.cache_audit_report.resolve(),
            args.output.resolve().parent / (args.mode + "_runtime"),
        )
        audit = assert_training_ready(config)
        lists = assert_runtime_dataset_contract(
            config, require_cache_audit=True
        )
        runtime = build_training_runtime(
            config,
            engine,
            device=args.device,
            wrap_distributed=False,
            norm_layer_override=(
                nn.BatchNorm2d if args.mode == "ddp_mock" else None
            ),
        )
        runtime.train_sampler.set_epoch(0)
        batch = next(iter(runtime.train_loader))
        rgb = batch["data"].to(runtime.device, non_blocking=True)
        modal_x = batch["modal_x"].to(runtime.device, non_blocking=True)
        label = batch["label"].to(runtime.device, non_blocking=True)
        runtime.optimizer.zero_grad()
        loss_map = runtime.model(rgb, modal_x, label)
        loss_map, nan_count = sanitize_author_loss_map(loss_map)
        loss = loss_map.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("formal startup smoke loss is not finite")
        loss.backward()
        gradients = _gradient_contract(runtime.model)
        if not gradients["all_finite"] or not gradients["any_nonzero"]:
            raise RuntimeError("formal startup smoke gradients failed")

        report = {
            "status": "PASS",
            "claim": "formal startup no-step plumbing PASS",
            "mode": args.mode,
            "integration_protocol_id": config.integration_protocol_id,
            "representation_protocol_id": config.representation_protocol_id,
            "authorization_scope": "ephemeral pilot no-step smoke only",
            "engine": {
                "distributed": engine.distributed,
                "local_rank": engine.local_rank,
                "world_size": engine.world_size,
            },
            "ddp_mock_scope": (
                "rank/world-size seed, sampler partition, per-rank batch and "
                "shared runtime; process-group SyncBatchNorm/DDP communication "
                "is intentionally not executed"
                if args.mode == "ddp_mock"
                else None
            ),
            "runtime_components": [
                "Engine",
                "file logger",
                "author seed",
                "fixed-length DataLoader",
                "Original CMX MiT-B2",
                "pretrained initialization",
                "AdamW optimizer construction",
                "WarmUpPolyLR construction",
                "FocalLoss2d none_then_mean",
            ],
            "audit_status": audit["status"],
            "runtime_lists": lists,
            "batch_sample_ids": list(batch["fn"]),
            "rgb_shape": list(rgb.shape),
            "modal_x_shape": list(modal_x.shape),
            "label_shape": list(label.shape),
            "loss_map_shape": list(loss_map.shape),
            "loss": float(loss.detach().cpu()),
            "author_nan_replacement_count": nan_count,
            "gradients": gradients,
            "backward_executed": True,
            "optimizer_constructed": True,
            "scheduler_constructed": True,
            "optimizer_step_executed": False,
            "scheduler_step_executed": False,
            "checkpoint_written": False,
            "epoch_loop_started": False,
            "formal_training_started": False,
            "file_hash_written": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False))
        del loss, loss_map, runtime, rgb, modal_x, label
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
