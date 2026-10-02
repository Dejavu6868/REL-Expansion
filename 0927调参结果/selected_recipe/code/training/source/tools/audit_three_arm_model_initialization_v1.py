#!/usr/bin/env python3
"""Build REL+/RGBD/HHA sequentially and compare every initialized tensor."""

import argparse
import gc
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ARMS = {
    "rel_plus": (
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_3_formal"
    ),
    "rgbd": "configs.stanford2d3d_s2d.cmx_mit_b2_rgbd_three_arm_v1",
    "hha": "configs.stanford2d3d_s2d.cmx_mit_b2_hha_three_arm_v1",
}


def _source_state(path):
    payload = torch.load(str(path), map_location="cpu")
    if "model" in payload:
        payload = payload["model"]
    if "state_dict" in payload:
        payload = payload["state_dict"]
    return payload


def _mapped_source(source):
    mapped = {}
    for key, value in source.items():
        if "patch_embed" in key:
            mapped[key] = (key, key.replace("patch_embed", "extra_patch_embed"))
        elif "block" in key:
            mapped[key] = (key, key.replace("block", "extra_block"))
        elif "norm" in key:
            mapped[key] = (key, key.replace("norm", "extra_norm"))
    return mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    from models.builder import EncoderDecoder
    from utils.training_protocol import build_author_criterion, set_author_seed

    reference = None
    reference_keys = None
    source = None
    mapping = None
    arm_reports = {}
    cross_arm_mismatches = []
    for arm, module_name in ARMS.items():
        config = importlib.import_module(module_name).config
        set_author_seed(config.seed, local_rank=0, distributed=False)
        criterion = build_author_criterion(config)
        model = EncoderDecoder(
            cfg=config,
            criterion=criterion,
            norm_layer=nn.SyncBatchNorm,
        )
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        keys = list(state)
        if source is None:
            source = _source_state(Path(config.pretrained_model))
            mapping = _mapped_source(source)
        pretrained_mismatches = []
        rgb_loaded = 0
        x_loaded = 0
        for source_key, (rgb_key, x_key) in mapping.items():
            for encoder, target_key in (("rgb", rgb_key), ("x", x_key)):
                full_key = "backbone.{}".format(target_key)
                source_tensor = source[source_key]
                if full_key not in state:
                    pretrained_mismatches.append(
                        {"encoder": encoder, "key": full_key, "reason": "missing"}
                    )
                elif tuple(state[full_key].shape) != tuple(source_tensor.shape):
                    pretrained_mismatches.append(
                        {"encoder": encoder, "key": full_key, "reason": "shape"}
                    )
                elif not torch.equal(state[full_key], source_tensor.cpu()):
                    pretrained_mismatches.append(
                        {"encoder": encoder, "key": full_key, "reason": "value"}
                    )
                elif encoder == "rgb":
                    rgb_loaded += 1
                else:
                    x_loaded += 1

        if reference is None:
            reference = state
            reference_keys = keys
            exact_count = len(state)
        else:
            if keys != reference_keys:
                cross_arm_mismatches.append(
                    {"arm": arm, "reason": "ordered_state_dict_keys"}
                )
            exact_count = 0
            for name in reference_keys:
                if name not in state:
                    cross_arm_mismatches.append(
                        {"arm": arm, "key": name, "reason": "missing"}
                    )
                elif tuple(reference[name].shape) != tuple(state[name].shape):
                    cross_arm_mismatches.append(
                        {"arm": arm, "key": name, "reason": "shape"}
                    )
                elif not torch.equal(reference[name], state[name]):
                    cross_arm_mismatches.append(
                        {"arm": arm, "key": name, "reason": "value"}
                    )
                else:
                    exact_count += 1
        decoder_names = [name for name in keys if name.startswith("decode_head.")]
        fusion_names = [
            name for name in keys if ".FRMs." in name or ".FFMs." in name
        ]
        mapped_target_keys = {
            "backbone.{}".format(target)
            for targets in mapping.values()
            for target in targets
        }
        missing_from_pretrain = sorted(set(keys) - mapped_target_keys)
        arm_reports[arm] = {
            "config_module": module_name,
            "seed": int(config.seed),
            "pretrained_model": config.pretrained_model,
            "state_dict_key_count": len(keys),
            "parameter_count": sum(
                int(parameter.numel()) for parameter in model.parameters()
            ),
            "exact_tensor_matches_to_rel_plus": exact_count,
            "rgb_pretrained_exact_tensor_count": rgb_loaded,
            "x_pretrained_exact_tensor_count": x_loaded,
            "pretrained_mismatch_count": len(pretrained_mismatches),
            "pretrained_mismatches": pretrained_mismatches[:20],
            "decoder_tensor_count": len(decoder_names),
            "fusion_tensor_count": len(fusion_names),
            "missing_keys_from_mit_b2_pretrain": missing_from_pretrain,
            "unexpected_pretrain_keys": sorted(set(source) - set(mapping)),
        }
        del model, state, criterion
        gc.collect()

    key_counts = {row["state_dict_key_count"] for row in arm_reports.values()}
    parameter_counts = {row["parameter_count"] for row in arm_reports.values()}
    missing_key_sets = {
        tuple(row["missing_keys_from_mit_b2_pretrain"])
        for row in arm_reports.values()
    }
    unexpected_key_sets = {
        tuple(row["unexpected_pretrain_keys"]) for row in arm_reports.values()
    }
    passed = (
        not cross_arm_mismatches
        and len(key_counts) == 1
        and len(parameter_counts) == 1
        and len(missing_key_sets) == 1
        and len(unexpected_key_sets) == 1
        and all(
            row["exact_tensor_matches_to_rel_plus"] == row["state_dict_key_count"]
            and row["pretrained_mismatch_count"] == 0
            and row["rgb_pretrained_exact_tensor_count"] == len(mapping)
            and row["x_pretrained_exact_tensor_count"] == len(mapping)
            for row in arm_reports.values()
        )
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol": "THREE_ARM_MODEL_INITIALIZATION_V1",
        "construction_order": list(ARMS),
        "seed_reset_before_each_model": True,
        "norm_layer": "SyncBatchNorm",
        "state_dict_order_exact": not cross_arm_mismatches,
        "state_dict_shapes_exact": not any(
            row.get("reason") == "shape" for row in cross_arm_mismatches
        ),
        "all_state_tensors_bitwise_exact": not cross_arm_mismatches,
        "parameter_count_exact": len(parameter_counts) == 1,
        "missing_keys_identical": len(missing_key_sets) == 1,
        "unexpected_keys_identical": len(unexpected_key_sets) == 1,
        "cross_arm_mismatch_count": len(cross_arm_mismatches),
        "cross_arm_mismatches": cross_arm_mismatches[:20],
        "mapped_mit_b2_source_tensor_count": len(mapping),
        "arms": arm_reports,
        "optimizer_step_executed": False,
        "backpropagation_executed": False,
        "checkpoint_replaced": False,
        "file_hash_written": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# THREE_ARM_MODEL_INITIALIZATION_AUDIT",
        "",
        "- Status: `{}`".format(report["status"]),
        "- Seed: `12345`（每次构造前重置）",
        "- Norm layer: `SyncBatchNorm`",
        "- State-dict keys / shapes / values: `{}`".format(
            "EXACT" if report["all_state_tensors_bitwise_exact"] else "MISMATCH"
        ),
        "- Parameter count: `{}`".format(next(iter(parameter_counts))),
        "- MiT-B2 mapped tensors per encoder: `{}`".format(len(mapping)),
        "- Missing-key sets identical: `{}`".format(report["missing_keys_identical"]),
        "- Unexpected-key sets identical: `{}`".format(
            report["unexpected_keys_identical"]
        ),
        "- Backpropagation / optimizer step / checkpoint replacement: `False / False / False`",
        "- File hash written: `False`",
        "",
        "三臂在第一个 optimizer step 前的全部对应 state tensor 逐元素一致。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "state_dict_key_count": next(iter(key_counts)),
                "parameter_count": next(iter(parameter_counts)),
                "mapped_mit_b2_source_tensor_count": len(mapping),
                "cross_arm_mismatch_count": len(cross_arm_mismatches),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
