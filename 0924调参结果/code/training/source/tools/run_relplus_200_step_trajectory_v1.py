#!/usr/bin/env python3
"""Run a deterministic single-GPU REL+ optimizer trajectory for code comparison."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


def _clone_tensors(mapping):
    return {
        name: value.detach().cpu().clone()
        for name, value in mapping.items()
    }


class AuditEngine:
    def __init__(self):
        self.world_size = 1
        self.local_rank = 0
        self.distributed = False
        self.state = SimpleNamespace()

    def register_state(self, **values):
        for name, value in values.items():
            setattr(self.state, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic-audit", action="store_true")
    args = parser.parse_args()
    if args.steps != 200:
        raise ValueError("the equivalence gate requires exactly 200 steps")

    code_root = args.code_root.resolve()
    if not (code_root / "models" / "builder.py").is_file():
        raise RuntimeError("CMX code root is incomplete: {}".format(code_root))
    os.chdir(str(code_root))
    sys.path.insert(0, str(code_root))

    import importlib

    selected = importlib.import_module(
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_3_formal"
    )
    sys.modules["config"] = selected
    from utils.training_protocol import set_author_seed
    from utils.training_runtime import (
        build_training_runtime,
        sanitize_author_loss_map,
    )

    config = selected.config
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output.parent / (output.stem + "_runtime")
    config.batch_size = 1
    config.num_workers = 0
    config.output_dir = str(run_dir)
    config.log_dir = str(run_dir / "logs")
    config.log_file = str(run_dir / "logs" / "trajectory.log")
    config.log_dir_link = ""
    config.tb_dir = str(run_dir / "tensorboard")
    config.checkpoint_dir = str(run_dir / "checkpoints")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 200-step equivalence gate")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    engine = AuditEngine()
    runtime = build_training_runtime(
        config,
        engine,
        device=device,
        norm_layer_override=nn.BatchNorm2d,
    )
    if args.deterministic_audit:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    set_author_seed(config.seed, epoch=1, local_rank=0, distributed=False)
    runtime.train_sampler.set_epoch(1)
    runtime.model.train()
    iterator = iter(runtime.train_loader)

    losses = []
    learning_rates = []
    sample_ids = []
    first = None
    nan_replacement_count = 0
    for step in range(args.steps):
        batch = next(iterator)
        rgb = batch["data"].to(device, non_blocking=True)
        modal_x = batch["modal_x"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = runtime.model(rgb, modal_x)
        raw_loss_map = runtime.criterion(logits, label.long())
        loss_map, nan_count = sanitize_author_loss_map(raw_loss_map)
        nan_replacement_count += nan_count
        loss = loss_map.mean()
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("trajectory logits are not finite")
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("trajectory loss is not finite")
        runtime.optimizer.zero_grad()
        loss.backward()
        gradients = {
            name: parameter.grad
            for name, parameter in runtime.model.named_parameters()
            if parameter.grad is not None
        }
        if not gradients or not all(
            bool(torch.isfinite(value).all().item()) for value in gradients.values()
        ):
            raise FloatingPointError("trajectory gradients are incomplete or non-finite")
        if step == 0:
            first = {
                "sample_id": list(batch["fn"]),
                "worker_id": batch["worker_id"].detach().cpu().clone(),
                "rgb": rgb.detach().cpu().clone(),
                "modal_x": modal_x.detach().cpu().clone(),
                "label": label.detach().cpu().clone(),
                "logits": logits.detach().cpu().clone(),
                "focal_loss_map": raw_loss_map.detach().cpu().clone(),
                "gradients": _clone_tensors(gradients),
            }
        runtime.optimizer.step()
        iteration = step
        learning_rate = runtime.scheduler.get_lr(iteration)
        for group in runtime.optimizer.param_groups:
            group["lr"] = learning_rate
        if step == 0:
            first["model_after_adamw_step"] = _clone_tensors(
                runtime.model.state_dict()
            )
        losses.append(float(loss.detach().cpu().item()))
        learning_rates.append(float(learning_rate))
        sample_ids.extend(str(value) for value in batch["fn"])

    torch.cuda.synchronize(device)
    packet = {
        "protocol": "RELPLUS_OLD_VS_THREE_ARM_200_STEP_ATOL0_V1",
        "code_root": str(code_root),
        "steps": args.steps,
        "seed": int(config.seed),
        "epoch": 1,
        "batch_size": 1,
        "num_workers": 0,
        "device": str(device),
        "sample_ids": sample_ids,
        "losses": losses,
        "learning_rates": learning_rates,
        "first": first,
        "final_model": _clone_tensors(runtime.model.state_dict()),
        "nan_replacement_count": nan_replacement_count,
        "torch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_audit": bool(args.deterministic_audit),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "cudnn_allow_tf32": bool(
            getattr(torch.backends.cudnn, "allow_tf32", False)
        ),
        "file_hash_written": False,
    }
    torch.save(packet, str(output))
    report = {
        key: value
        for key, value in packet.items()
        if key
        not in (
            "first",
            "final_model",
            "losses",
            "learning_rates",
            "sample_ids",
        )
    }
    report.update(
        {
            "status": "PASS",
            "output": str(output),
            "loss_count": len(losses),
            "learning_rate_count": len(learning_rates),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "first_learning_rate": learning_rates[0],
            "last_learning_rate": learning_rates[-1],
            "first_gradient_tensor_count": len(first["gradients"]),
            "final_model_tensor_count": len(packet["final_model"]),
        }
    )
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
