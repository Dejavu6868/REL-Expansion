#!/usr/bin/env python3
"""Compare two REL+ trajectory packets with exact tensor equality."""

import argparse
import json
from pathlib import Path

import torch


def _parameter_group(name):
    if name.startswith("decode_head."):
        return "decoder"
    if ".FRMs." in name or ".FFMs." in name:
        return "fusion"
    if "backbone.extra_" in name:
        return "x_encoder"
    if name.startswith("backbone."):
        return "rgb_encoder"
    return "other"


def _compare_tensor(label, left, right, mismatches):
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        mismatches.append({"path": label, "reason": "not_tensor"})
        return False
    if left.dtype != right.dtype:
        mismatches.append(
            {
                "path": label,
                "reason": "dtype",
                "left": str(left.dtype),
                "right": str(right.dtype),
            }
        )
        return False
    if tuple(left.shape) != tuple(right.shape):
        mismatches.append(
            {
                "path": label,
                "reason": "shape",
                "left": list(left.shape),
                "right": list(right.shape),
            }
        )
        return False
    if torch.equal(left, right):
        return True
    difference = (left.to(torch.float64) - right.to(torch.float64)).abs()
    mismatches.append(
        {
            "path": label,
            "reason": "value",
            "different_element_count": int(torch.count_nonzero(difference).item()),
            "max_abs_difference": float(difference.max().item()),
        }
    )
    return False


def _compare_mapping(label, left, right, mismatches):
    left_keys = list(left)
    right_keys = list(right)
    if left_keys != right_keys:
        mismatches.append(
            {
                "path": label,
                "reason": "ordered_keys",
                "left_only": sorted(set(left_keys) - set(right_keys)),
                "right_only": sorted(set(right_keys) - set(left_keys)),
            }
        )
        return 0
    matches = 0
    for name in left_keys:
        matches += int(
            _compare_tensor(
                "{}.{}".format(label, name), left[name], right[name], mismatches
            )
        )
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    old = torch.load(str(args.old.resolve()), map_location="cpu")
    new = torch.load(str(args.new.resolve()), map_location="cpu")
    mismatches = []
    scalar_fields = (
        "protocol",
        "steps",
        "seed",
        "epoch",
        "batch_size",
        "num_workers",
        "sample_ids",
        "losses",
        "learning_rates",
        "nan_replacement_count",
        "torch_version",
        "cudnn_version",
        "deterministic_audit",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
    )
    for field in scalar_fields:
        if old.get(field) != new.get(field):
            mismatches.append({"path": field, "reason": "value_sequence"})

    first_scalar_fields = ("sample_id",)
    for field in first_scalar_fields:
        if old["first"].get(field) != new["first"].get(field):
            mismatches.append(
                {"path": "first.{}".format(field), "reason": "value"}
            )
    first_tensor_fields = (
        "worker_id",
        "rgb",
        "modal_x",
        "label",
        "logits",
        "focal_loss_map",
    )
    first_tensor_matches = sum(
        int(
            _compare_tensor(
                "first.{}".format(field),
                old["first"][field],
                new["first"][field],
                mismatches,
            )
        )
        for field in first_tensor_fields
    )
    gradient_matches = _compare_mapping(
        "first.gradients",
        old["first"]["gradients"],
        new["first"]["gradients"],
        mismatches,
    )
    first_step_parameter_matches = _compare_mapping(
        "first.model_after_adamw_step",
        old["first"]["model_after_adamw_step"],
        new["first"]["model_after_adamw_step"],
        mismatches,
    )
    final_parameter_matches = _compare_mapping(
        "final_model", old["final_model"], new["final_model"], mismatches
    )

    gradient_groups = {}
    for group in ("rgb_encoder", "x_encoder", "fusion", "decoder", "other"):
        names = [
            name
            for name in old["first"]["gradients"]
            if _parameter_group(name) == group
        ]
        gradient_groups[group] = {
            "tensor_count": len(names),
            "all_exact": all(
                torch.equal(
                    old["first"]["gradients"][name],
                    new["first"]["gradients"][name],
                )
                for name in names
            ),
        }
    required_groups = ("rgb_encoder", "x_encoder", "fusion", "decoder")
    group_gate = all(
        gradient_groups[group]["tensor_count"] > 0
        and gradient_groups[group]["all_exact"]
        for group in required_groups
    )
    status = "PASS" if not mismatches and group_gate else "FAIL"
    report = {
        "status": status,
        "protocol": "RELPLUS_OLD_VS_THREE_ARM_200_STEP_ATOL0_V1",
        "old_code_root": old["code_root"],
        "new_code_root": new["code_root"],
        "device": old["device"],
        "steps": old["steps"],
        "single_gpu": True,
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
        "sample_id_sequence_exact": old["sample_ids"] == new["sample_ids"],
        "loss_sequence_exact": old["losses"] == new["losses"],
        "learning_rate_sequence_exact": (
            old["learning_rates"] == new["learning_rates"]
        ),
        "first_tensor_field_count": len(first_tensor_fields),
        "first_tensor_exact_match_count": first_tensor_matches,
        "gradient_tensor_count": len(old["first"]["gradients"]),
        "gradient_exact_match_count": gradient_matches,
        "gradient_groups": gradient_groups,
        "first_step_model_tensor_count": len(
            old["first"]["model_after_adamw_step"]
        ),
        "first_step_model_exact_match_count": first_step_parameter_matches,
        "final_model_tensor_count": len(old["final_model"]),
        "final_model_exact_match_count": final_parameter_matches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
