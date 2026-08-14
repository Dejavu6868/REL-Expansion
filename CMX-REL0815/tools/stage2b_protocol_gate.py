#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ARMS = ("rawdepth", "hha", "relplus_local", "relplus_pose")
FROZEN_FIELDS = (
    "train_source", "eval_source", "num_train_imgs", "num_eval_imgs", "num_classes",
    "class_names", "background", "image_height", "image_width", "norm_mean", "norm_std",
    "backbone", "pretrained_model", "decoder", "decoder_embed_dim", "optimizer", "lr",
    "lr_power", "weight_decay", "batch_size", "nepochs", "niters_per_epoch", "num_workers",
    "train_scale_array", "train_horizontal_flip", "warm_up_epoch", "fix_bias", "bn_eps",
    "bn_momentum", "eval_stride_rate", "eval_scale_array", "eval_flip", "eval_crop_size",
    "checkpoint_start_epoch", "checkpoint_step", "deterministic_training", "amp_enabled",
    "gradient_clipping", "checkpoint_selection_rule", "physical_world_size", "physical_gpu_ids",
    "reference_world_size", "physical_rank_batch_sizes", "reference_rank_pairs",
    "distributed_batch_adapter", "distributed_loss_adapter", "reference_stochastic_trajectory_equivalent",
    "second_modality_identity", "gravity_source",
)


def json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return str(value)


def snapshot(module_name, seed=None):
    environment = dict(os.environ)
    environment["CMX_RUN_DIR"] = "/declared/config-snapshot-run"
    if seed is not None:
        environment["STAGE2B_SEED"] = str(seed)
        environment["STAGE2B_COMMON_INITIAL_MODEL"] = "/declared/seed-common-initial.pth"
    code = (
        "import importlib,json; "
        "c=importlib.import_module(%r).config; "
        "fields=%r; "
        "print(json.dumps({k:getattr(c,k,None) for k in fields}, "
        "default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))"
    ) % (module_name, FROZEN_FIELDS)
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=True
    )
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    differences = []
    for seed in (23456, 34567):
        for arm in ARMS:
            baseline = snapshot("configs.stage2a_{}".format(arm))
            candidate = snapshot("configs.stage2b_{}".format(arm), seed=seed)
            for field in FROZEN_FIELDS:
                if json_value(candidate[field]) != json_value(baseline[field]):
                    differences.append({
                        "seed": seed, "arm": arm, "field": field,
                        "stage2a": json_value(baseline[field]), "stage2b": json_value(candidate[field]),
                    })
    report = {
        "status": "PASS_STAGE2B_PROTOCOL_GATE" if not differences else "FAIL_STAGE2B_PROTOCOL_GATE",
        "baseline_seed": 12345,
        "new_paired_seeds": [23456, 34567],
        "frozen_fields_checked": list(FROZEN_FIELDS),
        "allowed_changes": ["seed", "rank-derived seeds", "output paths", "stage label"],
        "differences": differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
