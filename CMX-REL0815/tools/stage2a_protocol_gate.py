#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ARMS = ("rawdepth", "hha", "relplus_local", "relplus_pose")
COMMON_FIELDS = (
    "seed", "physical_world_size", "physical_gpu_ids", "reference_world_size",
    "physical_rank_batch_sizes", "physical_rank_seeds", "reference_rank_pairs",
    "train_source", "eval_source", "num_train_imgs", "num_eval_imgs", "num_classes",
    "class_names", "background", "image_height", "image_width", "norm_mean", "norm_std",
    "backbone", "pretrained_model", "decoder", "decoder_embed_dim", "optimizer", "lr",
    "lr_power", "weight_decay", "batch_size", "nepochs", "niters_per_epoch", "num_workers",
    "train_scale_array", "train_horizontal_flip", "warm_up_epoch", "fix_bias", "bn_eps",
    "bn_momentum", "eval_stride_rate", "eval_scale_array", "eval_flip", "eval_crop_size",
    "checkpoint_start_epoch", "checkpoint_step", "deterministic_training", "amp_enabled",
    "gradient_clipping", "checkpoint_selection_rule", "common_initial_model",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    configs = {}
    for arm in ARMS:
        metadata = json.loads((args.root / "configs" / (arm + "_resolved.json")).read_text())
        configs[arm] = metadata["resolved_config"]
    reference = configs[ARMS[0]]
    differences = []
    for arm in ARMS[1:]:
        for field in COMMON_FIELDS:
            if configs[arm].get(field) != reference.get(field):
                differences.append({"arm": arm, "field": field,
                                    "reference": reference.get(field), "actual": configs[arm].get(field)})
    expected_gravity = {
        "rawdepth": "not_applicable", "hha": "HHA baseline EstGravity",
        "relplus_local": "REL-default EstGravity", "relplus_pose": "R_w2c @ [0,0,-1]",
    }
    for arm in ARMS:
        if configs[arm].get("gravity_source") != expected_gravity[arm]:
            differences.append({"arm": arm, "field": "gravity_source",
                                "expected": expected_gravity[arm], "actual": configs[arm].get("gravity_source")})
    report = {
        "status": "PASS_STAGE2A_PROTOCOL_GATE" if not differences else "FAIL_STAGE2A_PROTOCOL_GATE",
        "arms": list(ARMS), "common_fields_checked": list(COMMON_FIELDS),
        "differences": differences, "unique_changed_factor": "second modality and contracted gravity source",
    }
    output = args.root / "configs/protocol_gate.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
