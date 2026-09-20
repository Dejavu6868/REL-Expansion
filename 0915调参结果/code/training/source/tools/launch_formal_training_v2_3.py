#!/usr/bin/env python3
"""Fail-closed launcher for one authorized CMX-REL+ V2.3 formal run."""

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_checkpoint_sweep_v2_3 import resolve_distributed_launcher
from utils.training_protocol import assert_training_ready


EXPECTED_CONTROLS = {
    "architecture": "Original CMX",
    "x_mode": "rel_plus_v2_1",
    "channel_order": ("EGVIA", "LOA", "ReD"),
    "num_classes": 13,
    "backbone": "mit_b2",
    "decoder": "MLPDecoder",
    "using_gate": False,
    "using_smmf": False,
    "using_dymm": False,
    "using_sga": False,
    "seed": 12345,
    "nepochs": 200,
    "batch_size": 8,
    "amp": False,
    "sync_bn": True,
    "criterion": "Focal",
    "focal_gamma": 2,
    "loss_reduction": "none_then_mean",
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "warm_up_epoch": 10,
    "scheduler": "WarmUpPolyLR",
    "invalid_policy": "SOURCE_COMPAT_STORAGE_255",
    "augmentation_profile": "S2D_RELPLUS_COMPARISON_NO_FLIP",
    "primary_endpoint": "epoch_200",
    "secondary_endpoint": "test_selected_best",
}


def _load_report(path, label):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("{} is missing: {}".format(label, path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("{} is unreadable: {}".format(label, error))


def validate_ddp_smoke(report, config):
    expected = {
        "status": "PASS",
        "integration_protocol_id": config.integration_protocol_id,
        "representation_protocol_id": config.representation_protocol_id,
        "gpu_count": 8,
        "rank_count": 8,
        "optimizer_step_executed": True,
        "checkpoint_saved": True,
        "checkpoint_resumed": True,
        "parameters_match_after_restore": True,
        "loss_finite": True,
        "logits_finite": True,
        "gradients_finite": True,
        "lr_continuous_across_restore": True,
        "lr_updated_after_resume": True,
        "pretrained_model_loaded": True,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise RuntimeError(
                "DDP smoke {} mismatch: expected {!r}, found {!r}".format(
                    field, value, report.get(field)
                )
            )
    if not 50 <= int(report.get("iterations", 0)) <= 100:
        raise RuntimeError("DDP smoke iteration count is outside 50..100")
    if not 2 <= int(report.get("resume_iterations", 0)) <= 5:
        raise RuntimeError("DDP smoke resume iteration count is outside 2..5")
    groups = report.get("parameter_groups_changed", {})
    resume_groups = report.get("parameter_groups_changed_after_resume", {})
    for name in ("rgb_encoder", "x_encoder", "fusion", "decoder"):
        if groups.get(name) is not True:
            raise RuntimeError("DDP smoke parameter group did not change: {}".format(name))
        if resume_groups.get(name) is not True:
            raise RuntimeError(
                "DDP smoke parameter group did not change after resume: {}".format(
                    name
                )
            )
    if str(Path(report.get("cache_audit_report", "")).resolve()) != str(
        Path(config.cache_audit_report).resolve()
    ):
        raise RuntimeError("DDP smoke cache audit identity mismatch")
    if str(Path(report.get("cache_generation_report", "")).resolve()) != str(
        Path(config.cache_generation_report).resolve()
    ):
        raise RuntimeError("DDP smoke cache generation identity mismatch")
    if str(Path(report.get("training_data_preflight_report", "")).resolve()) != str(
        Path(config.training_data_preflight_report).resolve()
    ):
        raise RuntimeError("DDP smoke preflight identity mismatch")
    return report


def validate_frozen_controls(config):
    for field, value in EXPECTED_CONTROLS.items():
        if getattr(config, field, None) != value:
            raise RuntimeError(
                "frozen control {} mismatch: expected {!r}, found {!r}".format(
                    field, value, getattr(config, field, None)
                )
            )
    if float(config.lr) != 6e-5:
        raise RuntimeError("frozen learning rate must remain 6e-5")
    if list(config.checkpoint_epochs) != list(range(100, 201, 5)):
        raise RuntimeError("formal checkpoint schedule must be 100..200 step 5")


def _process_is_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def validate_resume_checkpoint(path, config):
    path = Path(path).resolve()
    checkpoint_dir = Path(config.checkpoint_dir).resolve()
    if not path.is_file():
        raise RuntimeError("resume checkpoint is missing: {}".format(path))
    if path.parent != checkpoint_dir:
        raise RuntimeError("resume checkpoint is outside the formal checkpoint directory")
    name = path.name
    if not name.startswith("epoch-") or not name.endswith(".pth"):
        raise RuntimeError("resume checkpoint filename must be epoch-N.pth")
    try:
        filename_epoch = int(name[len("epoch-") : -len(".pth")])
    except ValueError:
        raise RuntimeError("resume checkpoint filename epoch is invalid")
    if filename_epoch not in set(config.checkpoint_epochs):
        raise RuntimeError("resume checkpoint epoch is outside the formal schedule")
    if filename_epoch >= int(config.nepochs):
        raise RuntimeError("resume checkpoint must precede the final epoch")
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError("resume checkpoint payload must be a mapping")
    if payload.get("epoch") != filename_epoch:
        raise RuntimeError("resume checkpoint filename/payload epoch mismatch")
    expected_iteration = int(config.niters_per_epoch) - 1
    if payload.get("iteration") != expected_iteration:
        raise RuntimeError("resume checkpoint iteration mismatch")
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    if not isinstance(model, dict) or not model:
        raise RuntimeError("resume checkpoint model state is missing")
    if not isinstance(optimizer, dict):
        raise RuntimeError("resume checkpoint optimizer state is missing")
    param_groups = optimizer.get("param_groups")
    optimizer_state = optimizer.get("state")
    if not isinstance(param_groups, list) or not param_groups:
        raise RuntimeError("resume checkpoint optimizer parameter groups are missing")
    if not isinstance(optimizer_state, dict) or not optimizer_state:
        raise RuntimeError("resume checkpoint optimizer entries are missing")
    identity = {
        "path": str(path),
        "epoch": filename_epoch,
        "iteration": expected_iteration,
        "model_state_key_count": len(model),
        "optimizer_param_group_count": len(param_groups),
        "optimizer_state_entry_count": len(optimizer_state),
    }
    del payload
    return identity


def build_training_command(
    *,
    python,
    launcher,
    nproc_per_node,
    config_module,
    resolved_path,
    resume_checkpoint=None
):
    command = [
        str(python),
        "-m",
        str(launcher),
        "--nproc_per_node",
        str(int(nproc_per_node)),
        str(ROOT / "tools" / "run_with_config.py"),
        "--config-module",
        str(config_module),
        "--resolved-config",
        str(resolved_path),
        "train.py",
    ]
    if resume_checkpoint is not None:
        command.extend(["--continue", str(Path(resume_checkpoint).resolve())])
    return command


def build_resolved_payload(config, *, launch_id, launcher, nproc_per_node):
    run_dir = Path(config.output_dir).resolve()
    runtime_overrides = {
        "training_authorized": True,
        "source_compatible_invalid_accepted": True,
        "formal_cache_root": str(Path(config.formal_cache_root).resolve()),
        "x_root_folder": str(Path(config.x_root_folder).resolve()),
        "x_valid_root_folder": str(Path(config.x_valid_root_folder).resolve()),
        "full_manifest": str(Path(config.full_manifest).resolve()),
        "cache_generation_report": str(
            Path(config.cache_generation_report).resolve()
        ),
        "generation_resolved_manifest_path": str(
            Path(config.generation_resolved_manifest_path).resolve()
        ),
        "resolved_manifest_path": str(Path(config.resolved_manifest_path).resolve()),
        "train_source": str(Path(config.train_source).resolve()),
        "eval_source": str(Path(config.eval_source).resolve()),
        "class_mapping": str(Path(config.class_mapping).resolve()),
        "cache_audit_report": str(Path(config.cache_audit_report).resolve()),
        "training_data_preflight_report": str(
            Path(config.training_data_preflight_report).resolve()
        ),
        "ddp_smoke_report": str(Path(config.ddp_smoke_report).resolve()),
        "output_dir": str(run_dir),
        "log_dir": str(run_dir / "logs"),
        "tb_dir": str(run_dir / "tensorboard"),
        "log_dir_link": str(run_dir.parent / "latest_CMX_RELPlus_v2_3_logs"),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "log_file": str(run_dir / "logs" / "train.log"),
        "link_log_file": str(run_dir / "logs" / "train_last.log"),
        "val_log_file": str(run_dir / "logs" / "val.log"),
        "link_val_log_file": str(run_dir / "logs" / "val_last.log"),
        "launch_id": launch_id,
    }
    return {
        "status": "PASS",
        "resolved_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "launch_id": launch_id,
        "integration_protocol_id": config.integration_protocol_id,
        "representation_protocol_id": config.representation_protocol_id,
        "runtime_overrides": runtime_overrides,
        "seed": config.seed,
        "global_batch": config.batch_size,
        "epochs": config.nepochs,
        "architecture": config.architecture,
        "backbone": config.backbone,
        "decoder": config.decoder,
        "loss": {
            "name": config.criterion,
            "gamma": config.focal_gamma,
            "reduction": config.loss_reduction,
        },
        "optimizer": {
            "name": config.optimizer,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
        },
        "scheduler": {
            "name": config.scheduler,
            "warm_up_epoch": config.warm_up_epoch,
            "iteration_wise": True,
        },
        "invalid_policy": config.invalid_policy,
        "augmentation_profile": config.augmentation_profile,
        "checkpoint_endpoint": {
            "primary": config.primary_endpoint,
            "secondary": config.secondary_endpoint,
            "epochs": list(config.checkpoint_epochs),
        },
        "nproc_per_node": int(nproc_per_node),
        "distributed_launcher": launcher,
        "amp": config.amp,
        "sync_bn": config.sync_bn,
        "file_hash_written": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-module",
        default=(
            "configs.stanford2d3d_s2d."
            "cmx_mit_b2_rel_plus_v2_3_formal"
        ),
    )
    parser.add_argument("--cache-audit", required=True, type=Path)
    parser.add_argument("--training-data-preflight", required=True, type=Path)
    parser.add_argument("--ddp-smoke", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--authorize-formal-training", action="store_true")
    parser.add_argument("--accept-source-compatible-invalid", action="store_true")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument(
        "--launcher",
        choices=("auto", "torch.distributed.run", "torch.distributed.launch"),
        default="auto",
    )
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.authorize_formal_training:
        raise RuntimeError("--authorize-formal-training is required")
    if not args.accept_source_compatible_invalid:
        raise RuntimeError("--accept-source-compatible-invalid is required")
    if args.nproc_per_node != 8:
        raise RuntimeError("formal V2.3 training requires exactly 8 processes")

    module = importlib.import_module(args.config_module)
    config = module.config
    if config.training_authorized or config.full_cache_authorized:
        raise RuntimeError("repository formal config must remain fail-closed")
    validate_frozen_controls(config)
    config.cache_audit_report = str(args.cache_audit.resolve())
    config.training_data_preflight_report = str(
        args.training_data_preflight.resolve()
    )
    config.ddp_smoke_report = str(args.ddp_smoke.resolve())
    config.training_authorized = True
    config.source_compatible_invalid_accepted = True
    assert_training_ready(config)
    smoke = validate_ddp_smoke(
        _load_report(args.ddp_smoke, "DDP smoke report"), config
    )
    if torch.cuda.device_count() != 8:
        raise RuntimeError("formal training requires exactly 8 visible GPUs")
    if not Path(config.pretrained_model).is_file():
        raise RuntimeError("MiT-B2 pretrained model is missing")
    resume_identity = None
    if args.resume_checkpoint is not None:
        resume_identity = validate_resume_checkpoint(args.resume_checkpoint, config)

    launcher = resolve_distributed_launcher(args.launcher)
    launch_id = "CMX_RELPlus_v2_3_seed12345_{}".format(
        datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    resolved_path = args.resolved_config or (
        Path(config.output_dir).parents[1]
        / "resolved_configs"
        / "resolved_formal_config.json"
    )
    resolved_path = resolved_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_resolved_payload(
        config,
        launch_id=launch_id,
        launcher=launcher,
        nproc_per_node=args.nproc_per_node,
    )
    payload["resume_checkpoint"] = resume_identity
    resolved_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    command = build_training_command(
        python=sys.executable,
        launcher=launcher,
        nproc_per_node=args.nproc_per_node,
        config_module=args.config_module,
        resolved_path=resolved_path,
        resume_checkpoint=(
            resume_identity["path"] if resume_identity is not None else None
        ),
    )
    run_dir = Path(payload["runtime_overrides"]["output_dir"])
    report_path = run_dir.parent / "formal_training_launch_report.json"
    pid_path = run_dir.parent / "formal_training.pid"
    previous_stale_pid = None
    if pid_path.is_file():
        text = pid_path.read_text(encoding="utf-8").strip()
        if text and _process_is_alive(text):
            raise RuntimeError("a formal training process is already alive: {}".format(text))
        if resume_identity is None:
            raise RuntimeError("stale formal training PID file requires explicit review")
        previous_stale_pid = text or None
    launch_report = {
        "status": "VALIDATED_ONLY" if args.validate_only else "FORMAL_TRAINING_LAUNCHING",
        "launch_id": launch_id,
        "command": command,
        "pid": os.getpid(),
        "resolved_config": str(resolved_path),
        "output_dir": str(run_dir),
        "log_path": str(run_dir / "logs" / "train.log"),
        "gpu_count": 8,
        "cache_generation": str(Path(config.cache_generation_report).resolve()),
        "cache_audit": str(args.cache_audit.resolve()),
        "training_data_preflight": str(args.training_data_preflight.resolve()),
        "ddp_smoke": str(args.ddp_smoke.resolve()),
        "ddp_smoke_checkpoint": smoke.get("checkpoint"),
        "resume_checkpoint": resume_identity,
        "previous_stale_pid": previous_stale_pid,
        "file_hash_written": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(launch_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(launch_report, ensure_ascii=False), flush=True)
    if args.validate_only:
        return 0
    pid_path.write_text("{}\n".format(os.getpid()), encoding="utf-8")
    os.execvpe(command[0], command, os.environ.copy())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
