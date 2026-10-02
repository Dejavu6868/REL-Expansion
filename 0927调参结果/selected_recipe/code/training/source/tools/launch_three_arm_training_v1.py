#!/usr/bin/env python3
"""Fail-closed launcher for one authorized S2D RGBD or HHA formal run."""

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
from tools.launch_formal_training_v2_3 import build_training_command
from utils.training_protocol import assert_training_ready


EXPECTED_SHARED_CONTROLS = {
    "architecture": "Original CMX",
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
    "num_workers": 16,
    "logical_samples_per_epoch": 52904,
    "niters_per_epoch": 6613,
    "amp": False,
    "sync_bn": True,
    "criterion": "Focal",
    "focal_gamma": 2,
    "loss_reduction": "none_then_mean",
    "optimizer": "AdamW",
    "lr": 6e-5,
    "weight_decay": 0.01,
    "warm_up_epoch": 10,
    "scheduler": "WarmUpPolyLR",
    "augmentation_profile": "S2D_RELPLUS_COMPARISON_NO_FLIP",
    "train_horizontal_flip": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": False,
    "primary_endpoint": "epoch_200",
    "secondary_endpoint": "test_selected_best",
    "recovery_checkpoint_step": 5,
    "recovery_checkpoint_before_epoch": 100,
}

EXPECTED_ARM_CONTROLS = {
    "rgbd": {
        "x_mode": "rawdepth_uint8_repeat3",
        "representation_protocol_id": "RGBD_UINT8_REPEAT3_SOURCECOMPAT",
        "x_is_single_channel": True,
        "x_loader_adapter": "IMREAD_UNCHANGED_UINT8_REPEAT3",
    },
    "hha": {
        "x_mode": "hha_frozen_cache",
        "representation_protocol_id": "HHA_FROZEN_CACHE_EXECUTABLE_ORDER",
        "x_is_single_channel": False,
        "x_loader_adapter": "IMREAD_UNCHANGED_PRESERVE_ARRAY_ORDER",
    },
}


def _load_report(path, label):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("{} is missing: {}".format(label, path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("{} is unreadable: {}".format(label, error))


def _absolute(value):
    return str(Path(value).expanduser().resolve())


def _process_is_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def validate_frozen_controls(config):
    if config.integration_protocol_id != "CMX_S2D_THREE_ARM_V1":
        raise RuntimeError("three-arm integration protocol mismatch")
    arm = getattr(config, "comparison_arm", None)
    if arm not in EXPECTED_ARM_CONTROLS:
        raise RuntimeError("unknown comparison_arm")
    for field, expected in EXPECTED_SHARED_CONTROLS.items():
        found = getattr(config, field, None)
        if found != expected:
            raise RuntimeError(
                "frozen control {} mismatch: expected {!r}, found {!r}".format(
                    field, expected, found
                )
            )
    for field, expected in EXPECTED_ARM_CONTROLS[arm].items():
        found = getattr(config, field, None)
        if found != expected:
            raise RuntimeError(
                "{} {} mismatch: expected {!r}, found {!r}".format(
                    arm, field, expected, found
                )
            )
    if list(config.train_scale_array) != [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]:
        raise RuntimeError("frozen train_scale_array mismatch")
    if list(config.checkpoint_epochs) != list(range(100, 201, 5)):
        raise RuntimeError("formal checkpoint schedule must be 100..200 step 5")


def validate_ddp_smoke(report, config):
    expected = {
        "status": "PASS",
        "integration_protocol_id": config.integration_protocol_id,
        "representation_protocol_id": config.representation_protocol_id,
        "comparison_arm": config.comparison_arm,
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
    for field, expected_value in expected.items():
        if report.get(field) != expected_value:
            raise RuntimeError(
                "DDP smoke {} mismatch: expected {!r}, found {!r}".format(
                    field, expected_value, report.get(field)
                )
            )
    for name in ("rgb_encoder", "x_encoder", "fusion", "decoder"):
        if report.get("parameter_groups_changed", {}).get(name) is not True:
            raise RuntimeError("DDP smoke parameter group did not change: {}".format(name))
        if report.get("parameter_groups_changed_after_resume", {}).get(name) is not True:
            raise RuntimeError(
                "DDP smoke parameter group did not change after resume: {}".format(
                    name
                )
            )
    if _absolute(report.get("input_audit_report", "")) != _absolute(
        config.input_audit_report
    ):
        raise RuntimeError("DDP smoke input audit identity mismatch")
    return report


def validate_shared_prerequisites(config):
    equivalence = _load_report(
        config.training_code_equivalence_report,
        "training-code equivalence report",
    )
    expected_equivalence = {
        "status": "PASS",
        "decision": "PASS_WITH_EXPLAINED_CUDA_BACKWARD_NONDETERMINISM",
        "steps": 200,
        "single_gpu": True,
        "forward_and_loss_map_bitwise_exact": True,
        "formal_training_flags_unchanged": True,
    }
    for field, expected in expected_equivalence.items():
        if equivalence.get(field) != expected:
            raise RuntimeError("training-code equivalence {} mismatch".format(field))
    if not all(equivalence.get("envelope_checks", {}).values()):
        raise RuntimeError("training-code equivalence envelope is incomplete")
    if not all(
        row.get("exact") is True
        for row in equivalence.get("numerical_source_files", [])
    ):
        raise RuntimeError("training numerical source files are not exact")

    initialization = _load_report(
        config.model_initialization_report,
        "model-initialization report",
    )
    expected_initialization = {
        "status": "PASS",
        "seed_reset_before_each_model": True,
        "state_dict_order_exact": True,
        "state_dict_shapes_exact": True,
        "all_state_tensors_bitwise_exact": True,
        "parameter_count_exact": True,
        "missing_keys_identical": True,
        "unexpected_keys_identical": True,
        "cross_arm_mismatch_count": 0,
    }
    for field, expected in expected_initialization.items():
        if initialization.get(field) != expected:
            raise RuntimeError("model initialization {} mismatch".format(field))

    sampler = _load_report(
        config.sampler_sequence_report,
        "sampler-sequence report",
    )
    expected_sampler = {
        "status": "PASS",
        "epochs": [1, 2, 3, 10, 11],
        "world_size": 8,
        "sample_ids_per_rank_epoch": 6613,
        "element_comparison_count": 264520,
        "all_ordered_sample_ids_exact": True,
        "mismatch_count": 0,
    }
    for field, expected in expected_sampler.items():
        if sampler.get(field) != expected:
            raise RuntimeError("sampler sequence {} mismatch".format(field))

    dataloader = _load_report(
        config.dataloader_trace_report,
        "DataLoader trajectory report",
    )
    expected_dataloader = {
        "status": "PASS",
        "epochs": [1, 2, 3, 10, 11],
        "rank": 0,
        "world_size": 8,
        "num_workers": 16,
        "head_count_per_epoch": 50,
        "tail_count_per_epoch": 50,
        "trace_count_per_arm": 500,
        "all_trace_fields_exact": True,
        "horizontal_flip_all_false": True,
        "mismatch_count": 0,
        "constructed_trace_copy_used": False,
    }
    for field, expected in expected_dataloader.items():
        if dataloader.get(field) != expected:
            raise RuntimeError("DataLoader trajectory {} mismatch".format(field))

    evaluator = _load_report(
        config.evaluator_equivalence_report,
        "epoch-200 evaluator-equivalence report",
    )
    expected_evaluator = {
        "status": "PASS",
        "reference_world_size": 8,
        "candidate_world_size": 8,
        "confusion_matrix_exact": True,
        "per_class_iou_exact": True,
    }
    for field, expected in expected_evaluator.items():
        if evaluator.get(field) != expected:
            raise RuntimeError("evaluator equivalence {} mismatch".format(field))
    for field in ("metric_exact", "expected_metric_exact", "protocol_exact"):
        values = evaluator.get(field, {})
        if not values or not all(values.values()):
            raise RuntimeError("evaluator equivalence {} is incomplete".format(field))
    manifest = evaluator.get("manifest", {})
    for field in (
        "reference_count",
        "candidate_count",
        "reference_unique_count",
        "candidate_unique_count",
    ):
        if manifest.get(field) != 17593:
            raise RuntimeError("evaluator manifest {} mismatch".format(field))
    if manifest.get("sample_id_sets_exact") is not True:
        raise RuntimeError("evaluator sample-ID sets differ")
    if not all(
        row.get("exact") is True
        for row in evaluator.get("evaluator_source_files", [])
    ):
        raise RuntimeError("evaluator numerical source files are not exact")

    rank_consistency = _load_report(
        config.evaluator_rank_consistency_report,
        "1-GPU-vs-8-GPU evaluator report",
    )
    expected_rank_consistency = {
        "status": "PASS",
        "reference_world_size": 8,
        "candidate_world_size": 1,
        "confusion_matrix_exact": True,
        "per_class_iou_exact": True,
    }
    for field, expected in expected_rank_consistency.items():
        if rank_consistency.get(field) != expected:
            raise RuntimeError("evaluator rank consistency {} mismatch".format(field))
    for field in ("metric_exact", "expected_metric_exact", "protocol_exact"):
        values = rank_consistency.get(field, {})
        if not values or not all(values.values()):
            raise RuntimeError(
                "evaluator rank consistency {} is incomplete".format(field)
            )
    rank_manifest = rank_consistency.get("manifest", {})
    for field in (
        "reference_count",
        "candidate_count",
        "reference_unique_count",
        "candidate_unique_count",
    ):
        if rank_manifest.get(field) != 17593:
            raise RuntimeError("rank-consistency manifest {} mismatch".format(field))
    if rank_manifest.get("sample_id_sets_exact") is not True:
        raise RuntimeError("rank-consistency sample-ID sets differ")
    if not all(
        row.get("exact") is True
        for row in rank_consistency.get("evaluator_source_files", [])
    ):
        raise RuntimeError("rank-consistency evaluator source files are not exact")
    return {
        "training_code_equivalence": equivalence,
        "model_initialization": initialization,
        "sampler_sequence": sampler,
        "dataloader_trajectory": dataloader,
        "evaluator_equivalence": evaluator,
        "evaluator_rank_consistency": rank_consistency,
    }


def build_resolved_payload(config, *, launch_id, launcher, nproc_per_node):
    run_dir = Path(config.output_dir).resolve()
    runtime_overrides = {
        "training_authorized": True,
        "source_compatible_invalid_accepted": False,
        "input_audit_report": _absolute(config.input_audit_report),
        "x_root_folder": _absolute(config.x_root_folder),
        "full_manifest": _absolute(config.full_manifest),
        "resolved_manifest_path": _absolute(config.resolved_manifest_path),
        "train_source": _absolute(config.train_source),
        "eval_source": _absolute(config.eval_source),
        "class_mapping": _absolute(config.class_mapping),
        "training_data_preflight_report": _absolute(
            config.training_data_preflight_report
        ),
        "ddp_smoke_report": _absolute(config.ddp_smoke_report),
        "output_dir": str(run_dir),
        "log_dir": str(run_dir / "logs"),
        "tb_dir": str(run_dir / "tensorboard"),
        "log_dir_link": _absolute(config.log_dir_link),
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
        "comparison_protocol_id": config.comparison_protocol_id,
        "comparison_arm": config.comparison_arm,
        "representation_protocol_id": config.representation_protocol_id,
        "runtime_overrides": runtime_overrides,
        "seed": config.seed,
        "global_batch": config.batch_size,
        "num_workers": config.num_workers,
        "worker_seed_policy": config.worker_seed_policy,
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
        "augmentation_profile": config.augmentation_profile,
        "checkpoint_endpoint": {
            "primary": config.primary_endpoint,
            "secondary": config.secondary_endpoint,
            "epochs": list(config.checkpoint_epochs),
        },
        "recovery": {
            "step": config.recovery_checkpoint_step,
            "before_epoch": config.recovery_checkpoint_before_epoch,
            "slots": ["recovery_epoch_odd.pth", "recovery_epoch_even.pth"],
            "atomic_replace": True,
            "rank_rng_and_sampler_state": True,
        },
        "prerequisite_reports": {
            "training_code_equivalence": _absolute(
                config.training_code_equivalence_report
            ),
            "model_initialization": _absolute(config.model_initialization_report),
            "sampler_sequence": _absolute(config.sampler_sequence_report),
            "dataloader_trajectory": _absolute(config.dataloader_trace_report),
            "evaluator_equivalence": _absolute(
                config.evaluator_equivalence_report
            ),
            "evaluator_rank_consistency": _absolute(
                config.evaluator_rank_consistency_report
            ),
        },
        "nproc_per_node": int(nproc_per_node),
        "distributed_launcher": launcher,
        "amp": config.amp,
        "sync_bn": config.sync_bn,
        "file_hash_written": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-module", required=True)
    parser.add_argument("--input-audit", required=True, type=Path)
    parser.add_argument("--ddp-smoke", required=True, type=Path)
    parser.add_argument("--authorize-formal-training", action="store_true")
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
    if args.nproc_per_node != 8:
        raise RuntimeError("formal three-arm training requires exactly 8 processes")
    module = importlib.import_module(args.config_module)
    config = module.config
    if config.training_authorized or config.full_cache_authorized:
        raise RuntimeError("repository formal config must remain fail-closed")
    validate_frozen_controls(config)
    validate_shared_prerequisites(config)
    config.input_audit_report = _absolute(args.input_audit)
    config.ddp_smoke_report = _absolute(args.ddp_smoke)
    config.training_authorized = True
    assert_training_ready(config)
    smoke = validate_ddp_smoke(
        _load_report(args.ddp_smoke, "DDP smoke report"), config
    )
    if torch.cuda.device_count() != 8:
        raise RuntimeError("formal training requires exactly 8 visible GPUs")
    if not Path(config.pretrained_model).is_file():
        raise RuntimeError("MiT-B2 pretrained model is missing")

    run_dir = Path(config.output_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("formal output directory is not empty: {}".format(run_dir))
    launcher = resolve_distributed_launcher(args.launcher)
    launch_id = "CMX_{}_S2D_seed12345_{}".format(
        config.comparison_arm.upper(),
        datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"),
    )
    resolved_path = args.resolved_config or (
        run_dir.parent
        / "shared_configs"
        / "{}_resolved_formal_config.json".format(config.comparison_arm)
    )
    resolved_path = resolved_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_resolved_payload(
        config,
        launch_id=launch_id,
        launcher=launcher,
        nproc_per_node=args.nproc_per_node,
    )
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
    )
    launch_reports = run_dir.parent / "launch_reports"
    launch_reports.mkdir(parents=True, exist_ok=True)
    report_path = launch_reports / "{}_formal_training_launch.json".format(
        config.comparison_arm
    )
    pid_dir = run_dir.parent / "pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_path = pid_dir / "{}_formal_training.pid".format(config.comparison_arm)
    if pid_path.is_file():
        text = pid_path.read_text(encoding="utf-8").strip()
        if text and _process_is_alive(text):
            raise RuntimeError("a formal training process is already alive: {}".format(text))
        raise RuntimeError("stale formal training PID file requires explicit review")
    launch_report = {
        "status": "VALIDATED_ONLY" if args.validate_only else "FORMAL_TRAINING_LAUNCHING",
        "launch_id": launch_id,
        "comparison_arm": config.comparison_arm,
        "command": command,
        "pid": os.getpid(),
        "resolved_config": str(resolved_path),
        "output_dir": str(run_dir),
        "log_path": str(run_dir / "logs" / "train.log"),
        "gpu_count": 8,
        "input_audit": _absolute(args.input_audit),
        "ddp_smoke": _absolute(args.ddp_smoke),
        "ddp_smoke_checkpoint": smoke.get("checkpoint"),
        "training_code_equivalence": _absolute(
            config.training_code_equivalence_report
        ),
        "model_initialization": _absolute(config.model_initialization_report),
        "sampler_sequence": _absolute(config.sampler_sequence_report),
        "dataloader_trajectory": _absolute(config.dataloader_trace_report),
        "evaluator_equivalence": _absolute(config.evaluator_equivalence_report),
        "evaluator_rank_consistency": _absolute(
            config.evaluator_rank_consistency_report
        ),
        "file_hash_written": False,
    }
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
