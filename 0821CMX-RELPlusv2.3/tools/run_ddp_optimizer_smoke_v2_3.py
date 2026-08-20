#!/usr/bin/env python3
"""Real 8-rank CMX optimizer/save/restore smoke on the formal cache."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.engine import Engine
from utils.training_protocol import assert_training_ready
from utils.training_runtime import build_training_runtime, sanitize_author_loss_map


def _base_model(model):
    return model.module if hasattr(model, "module") else model


def _parameter_group(name):
    if name.startswith("decode_head."):
        return "decoder"
    if ".FRMs." in name or ".FFMs." in name:
        return "fusion"
    if "backbone.extra_" in name:
        return "x_encoder"
    if name.startswith("backbone."):
        return "rgb_encoder"
    return None


def _snapshots(model, per_group=4):
    snapshots = {name: [] for name in ("rgb_encoder", "x_encoder", "fusion", "decoder")}
    for name, parameter in _base_model(model).named_parameters():
        group = _parameter_group(name)
        if group and len(snapshots[group]) < per_group:
            snapshots[group].append((name, parameter.detach().cpu().clone()))
    missing = [name for name, values in snapshots.items() if not values]
    if missing:
        raise RuntimeError("missing smoke parameter groups: {}".format(missing))
    return snapshots


def _changed_groups(model, snapshots):
    current = dict(_base_model(model).named_parameters())
    return {
        group: any(
            not torch.equal(before, current[name].detach().cpu())
            for name, before in values
        )
        for group, values in snapshots.items()
    }


def _snapshots_match(model, snapshots):
    current = dict(_base_model(model).named_parameters())
    return all(
        torch.equal(before, current[name].detach().cpu())
        for values in snapshots.values()
        for name, before in values
    )


def _all_gradients_finite(model):
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _run_steps(runtime, start_iteration, count):
    iterator = iter(runtime.train_loader)
    losses = []
    nan_replacement_count = 0
    gradients_finite = True
    logits_finite = True
    for offset in range(count):
        minibatch, iterator = _next_batch(iterator, runtime.train_loader)
        rgb = minibatch["data"].to(runtime.device, non_blocking=True)
        label = minibatch["label"].to(runtime.device, non_blocking=True)
        modal_x = minibatch["modal_x"].to(runtime.device, non_blocking=True)
        if offset == 0:
            with torch.no_grad():
                logits = _base_model(runtime.model).encode_decode(rgb, modal_x)
            logits_finite = logits_finite and bool(torch.isfinite(logits).all().item())
        loss_map = runtime.model(rgb, modal_x, label)
        loss_map, nan_count = sanitize_author_loss_map(loss_map)
        nan_replacement_count += nan_count
        loss = loss_map.mean()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("DDP smoke loss is NaN or Inf")
        runtime.optimizer.zero_grad()
        loss.backward()
        gradients_finite = gradients_finite and _all_gradients_finite(runtime.model)
        if not gradients_finite:
            raise FloatingPointError("DDP smoke gradient is NaN or Inf")
        runtime.optimizer.step()
        iteration = int(start_iteration) + offset
        learning_rate = runtime.scheduler.get_lr(iteration)
        for group in runtime.optimizer.param_groups:
            group["lr"] = learning_rate
        losses.append(float(loss.detach().item()))
    return {
        "losses": losses,
        "loss_finite": all(np_value == np_value and abs(np_value) != float("inf") for np_value in losses),
        "logits_finite": logits_finite,
        "gradients_finite": gradients_finite,
        "nan_replacement_count": nan_replacement_count,
        "last_lr": float(runtime.optimizer.param_groups[0]["lr"]),
    }


def _save_checkpoint(runtime, path, *, epoch, iteration):
    model = _base_model(runtime.model)
    payload = {
        "model": model.state_dict(),
        "optimizer": runtime.optimizer.state_dict(),
        "epoch": int(epoch),
        "iteration": int(iteration),
        "disposable": True,
        "integration_protocol_id": "CMX_RELPLUS_V2_3",
    }
    torch.save(payload, str(path))


def _restore_checkpoint(runtime, path, *, expected_epoch, expected_iteration):
    payload = torch.load(str(path), map_location="cpu")
    if int(payload.get("epoch", -1)) != int(expected_epoch):
        raise RuntimeError("disposable checkpoint epoch mismatch")
    if int(payload.get("iteration", -1)) != int(expected_iteration):
        raise RuntimeError("disposable checkpoint iteration mismatch")
    _base_model(runtime.model).load_state_dict(payload["model"], strict=True)
    runtime.optimizer.load_state_dict(payload["optimizer"])
    return payload


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config-module",
        default=(
            "configs.stanford2d3d_s2d."
            "cmx_mit_b2_rel_plus_v2_3_formal"
        ),
    )
    known, _ = bootstrap.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    parser.add_argument("--cache-audit", required=True, type=Path)
    parser.add_argument("--training-data-preflight", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--resume-iterations", type=int, default=3)
    parser.add_argument("--authorize-ddp-smoke", action="store_true")
    parser.add_argument("--accept-source-compatible-invalid", action="store_true")

    module = importlib.import_module(known.config_module)
    config = module.config
    with Engine(custom_parser=parser) as engine:
        args = engine.args
        if engine.world_size != 8:
            raise RuntimeError("V2.3 DDP smoke requires exactly 8 ranks")
        if not args.authorize_ddp_smoke:
            raise RuntimeError("--authorize-ddp-smoke is required")
        if not args.accept_source_compatible_invalid:
            raise RuntimeError("--accept-source-compatible-invalid is required")
        if not 50 <= args.iterations <= 100:
            raise ValueError("DDP smoke iterations must be between 50 and 100")
        if not 2 <= args.resume_iterations <= 5:
            raise ValueError("resume iterations must be between 2 and 5")

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        config.cache_audit_report = str(args.cache_audit.resolve())
        config.training_data_preflight_report = str(
            args.training_data_preflight.resolve()
        )
        config.training_authorized = True
        config.source_compatible_invalid_accepted = True
        config.output_dir = str(output_dir / "runtime")
        config.log_dir = str(output_dir / "runtime" / "logs")
        config.log_file = str(output_dir / "runtime" / "logs" / "train.log")
        config.log_dir_link = str(output_dir / "latest_runtime_logs")
        config.tb_dir = str(output_dir / "runtime" / "tensorboard")
        config.checkpoint_dir = str(output_dir / "runtime" / "checkpoints")
        assert_training_ready(config)
        if not Path(config.pretrained_model).is_file():
            raise FileNotFoundError(config.pretrained_model)

        runtime = build_training_runtime(config, engine)
        runtime.train_sampler.set_epoch(0)
        runtime.model.train()
        initial = _snapshots(runtime.model)
        first = _run_steps(runtime, 0, args.iterations)
        changed = _changed_groups(runtime.model, initial)
        if not all(changed.values()):
            raise RuntimeError("not all model parameter groups changed: {}".format(changed))
        checkpoint_parameters = _snapshots(runtime.model)

        checkpoint = output_dir / (
            "DISPOSABLE_DDP_SMOKE_epoch-0_iter-{}.pth".format(args.iterations)
        )
        if engine.local_rank == 0:
            _save_checkpoint(
                runtime,
                checkpoint,
                epoch=0,
                iteration=args.iterations,
            )
        dist.barrier()
        lr_before_restore = float(runtime.optimizer.param_groups[0]["lr"])
        _restore_checkpoint(
            runtime,
            checkpoint,
            expected_epoch=0,
            expected_iteration=args.iterations,
        )
        parameters_match_after_restore = _snapshots_match(
            runtime.model, checkpoint_parameters
        )
        if not parameters_match_after_restore:
            raise RuntimeError("model parameters changed across checkpoint restore")
        lr_after_restore = float(runtime.optimizer.param_groups[0]["lr"])
        if lr_after_restore != lr_before_restore:
            raise RuntimeError("optimizer LR changed across checkpoint restore")
        resume_initial = _snapshots(runtime.model)
        resume_start = args.iterations
        resumed = _run_steps(runtime, resume_start, args.resume_iterations)
        resume_changed = _changed_groups(runtime.model, resume_initial)
        if not all(resume_changed.values()):
            raise RuntimeError(
                "not all parameter groups changed after resume: {}".format(
                    resume_changed
                )
            )
        lr_updated_after_resume = resumed["last_lr"] != lr_after_restore
        if not lr_updated_after_resume:
            raise RuntimeError("learning rate did not update after resume")

        rank_report = {
            "status": "PASS",
            "rank": engine.local_rank,
            "world_size": engine.world_size,
            "iterations_before_checkpoint": args.iterations,
            "iterations_after_resume": args.resume_iterations,
            "optimizer_step_executed": True,
            "checkpoint_saved": checkpoint.is_file(),
            "checkpoint_resumed": True,
            "parameters_match_after_restore": parameters_match_after_restore,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": 0,
            "checkpoint_iteration": args.iterations,
            "parameter_groups_changed": changed,
            "parameter_groups_changed_after_resume": resume_changed,
            "loss_finite": first["loss_finite"] and resumed["loss_finite"],
            "logits_finite": first["logits_finite"] and resumed["logits_finite"],
            "gradients_finite": first["gradients_finite"] and resumed["gradients_finite"],
            "lr_before_restore": lr_before_restore,
            "lr_after_restore": lr_after_restore,
            "lr_after_resume": resumed["last_lr"],
            "lr_continuous_across_restore": lr_after_restore == lr_before_restore,
            "lr_updated_after_resume": lr_updated_after_resume,
            "pretrained_model_loaded": True,
            "author_nan_replacement_count": (
                first["nan_replacement_count"] + resumed["nan_replacement_count"]
            ),
            "pid": os.getpid(),
        }
        (output_dir / "rank_{:02d}.json".format(engine.local_rank)).write_text(
            json.dumps(rank_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        dist.barrier()
        if engine.local_rank == 0:
            ranks = [
                json.loads(
                    (output_dir / "rank_{:02d}.json".format(rank)).read_text(
                        encoding="utf-8"
                    )
                )
                for rank in range(engine.world_size)
            ]
            summary = {
                "status": "PASS" if all(row["status"] == "PASS" for row in ranks) else "FAIL",
                "integration_protocol_id": config.integration_protocol_id,
                "representation_protocol_id": config.representation_protocol_id,
                "gpu_count": engine.world_size,
                "rank_count": len(ranks),
                "iterations": args.iterations,
                "resume_iterations": args.resume_iterations,
                "optimizer_step_executed": all(row["optimizer_step_executed"] for row in ranks),
                "checkpoint_saved": checkpoint.is_file(),
                "checkpoint_resumed": all(row["checkpoint_resumed"] for row in ranks),
                "parameters_match_after_restore": all(
                    row["parameters_match_after_restore"] for row in ranks
                ),
                "parameter_groups_changed": {
                    name: all(row["parameter_groups_changed"][name] for row in ranks)
                    for name in ("rgb_encoder", "x_encoder", "fusion", "decoder")
                },
                "parameter_groups_changed_after_resume": {
                    name: all(
                        row["parameter_groups_changed_after_resume"][name]
                        for row in ranks
                    )
                    for name in ("rgb_encoder", "x_encoder", "fusion", "decoder")
                },
                "loss_finite": all(row["loss_finite"] for row in ranks),
                "logits_finite": all(row["logits_finite"] for row in ranks),
                "gradients_finite": all(row["gradients_finite"] for row in ranks),
                "lr_continuous_across_restore": all(
                    row["lr_continuous_across_restore"] for row in ranks
                ),
                "lr_updated_after_resume": all(
                    row["lr_updated_after_resume"] for row in ranks
                ),
                "pretrained_model_loaded": all(
                    row["pretrained_model_loaded"] for row in ranks
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": 0,
                "checkpoint_iteration": args.iterations,
                "pretrained_model": config.pretrained_model,
                "cache_generation_report": config.cache_generation_report,
                "cache_audit_report": config.cache_audit_report,
                "training_data_preflight_report": config.training_data_preflight_report,
                "ranks": ranks,
                "file_hash_written": False,
            }
            (output_dir / "ddp_optimizer_smoke_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False))
        dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
