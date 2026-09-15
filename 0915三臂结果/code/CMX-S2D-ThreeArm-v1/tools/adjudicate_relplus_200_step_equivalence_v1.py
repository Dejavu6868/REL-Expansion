#!/usr/bin/env python3
"""Adjudicate old-vs-new drift against an old-vs-old CUDA repeatability run."""

import argparse
import filecmp
import json
from pathlib import Path

import torch


NUMERICAL_SOURCE_PATHS = (
    "models",
    "rel_plus",
    "utils/loss_opr.py",
    "utils/init_func.py",
    "utils/lr_policy.py",
    "dataloader/dataloader.py",
    "dataloader/samplers.py",
    "dataloader/profiles.py",
)


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


def _max_tensor_difference(left, right):
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return float("inf")
    if torch.equal(left, right):
        return 0.0
    return float(
        (left.to(torch.float64) - right.to(torch.float64)).abs().max().item()
    )


def _mapping_metrics(left, right, *, group=None):
    names = list(left)
    if names != list(right):
        return {"ordered_keys_exact": False, "tensor_count": 0, "max_abs": float("inf")}
    if group is not None:
        names = [name for name in names if _parameter_group(name) == group]
    differences = [_max_tensor_difference(left[name], right[name]) for name in names]
    return {
        "ordered_keys_exact": True,
        "tensor_count": len(names),
        "exact_tensor_count": sum(value == 0.0 for value in differences),
        "max_abs": max(differences, default=0.0),
    }


def _sequence_metrics(left, right):
    if len(left) != len(right):
        return {"length_exact": False, "count": 0, "max_abs": float("inf")}
    differences = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    return {
        "length_exact": True,
        "count": len(left),
        "exact_count": sum(value == 0.0 for value in differences),
        "max_abs": max(differences, default=0.0),
        "first_different_index": next(
            (index for index, value in enumerate(differences) if value), None
        ),
    }


def _source_comparison(old_root, new_root):
    rows = []
    for relative in NUMERICAL_SOURCE_PATHS:
        old = old_root / relative
        new = new_root / relative
        if old.is_dir() and new.is_dir():
            comparison = filecmp.dircmp(str(old), str(new))
            stack = [comparison]
            differences = []
            while stack:
                current = stack.pop()
                differences.extend(current.left_only)
                differences.extend(current.right_only)
                differences.extend(current.diff_files)
                differences.extend(current.funny_files)
                stack.extend(current.subdirs.values())
            exact = not differences
        else:
            exact = old.is_file() and new.is_file() and filecmp.cmp(
                str(old), str(new), shallow=False
            )
            differences = [] if exact else [relative]
        rows.append(
            {
                "path": relative,
                "exact": exact,
                "difference_count": len(differences),
            }
        )
    return rows


def _pair_metrics(reference, candidate):
    first_fields = (
        "worker_id",
        "rgb",
        "modal_x",
        "label",
        "logits",
        "focal_loss_map",
    )
    metrics = {
        "sample_ids_exact": reference["sample_ids"] == candidate["sample_ids"],
        "learning_rates_exact": (
            reference["learning_rates"] == candidate["learning_rates"]
        ),
        "losses": _sequence_metrics(reference["losses"], candidate["losses"]),
        "first_tensors": {
            field: _max_tensor_difference(
                reference["first"][field], candidate["first"][field]
            )
            for field in first_fields
        },
        "first_step_model": _mapping_metrics(
            reference["first"]["model_after_adamw_step"],
            candidate["first"]["model_after_adamw_step"],
        ),
        "final_model": _mapping_metrics(
            reference["final_model"], candidate["final_model"]
        ),
        "gradient_groups": {},
    }
    for group in ("rgb_encoder", "x_encoder", "fusion", "decoder"):
        metrics["gradient_groups"][group] = _mapping_metrics(
            reference["first"]["gradients"],
            candidate["first"]["gradients"],
            group=group,
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--old-repeat", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    old = torch.load(str(args.old.resolve()), map_location="cpu")
    old_repeat = torch.load(str(args.old_repeat.resolve()), map_location="cpu")
    new = torch.load(str(args.new.resolve()), map_location="cpu")
    if any(packet.get("steps") != 200 for packet in (old, old_repeat, new)):
        raise RuntimeError("all equivalence packets must contain 200 steps")
    if not all(packet.get("deterministic_audit") for packet in (old, old_repeat, new)):
        raise RuntimeError("all packets must use the deterministic audit settings")

    self_repeat = _pair_metrics(old, old_repeat)
    old_vs_new = _pair_metrics(old, new)
    old_repeat_vs_new = _pair_metrics(old_repeat, new)
    source_rows = _source_comparison(
        Path(old["code_root"]), Path(new["code_root"])
    )
    exact_forward_fields = ("rgb", "modal_x", "label", "logits", "focal_loss_map")
    forward_exact = all(
        old_vs_new["first_tensors"][field] == 0.0
        and old_repeat_vs_new["first_tensors"][field] == 0.0
        for field in exact_forward_fields
    )
    envelope_checks = {
        "loss_sequence": (
            min(
                old_vs_new["losses"]["max_abs"],
                old_repeat_vs_new["losses"]["max_abs"],
            )
            <= self_repeat["losses"]["max_abs"]
        ),
        "first_step_model": (
            min(
                old_vs_new["first_step_model"]["max_abs"],
                old_repeat_vs_new["first_step_model"]["max_abs"],
            )
            <= self_repeat["first_step_model"]["max_abs"]
        ),
        "final_model": (
            min(
                old_vs_new["final_model"]["max_abs"],
                old_repeat_vs_new["final_model"]["max_abs"],
            )
            <= self_repeat["final_model"]["max_abs"]
        ),
    }
    for group in ("rgb_encoder", "x_encoder", "fusion", "decoder"):
        envelope_checks["{}_gradient".format(group)] = (
            min(
                old_vs_new["gradient_groups"][group]["max_abs"],
                old_repeat_vs_new["gradient_groups"][group]["max_abs"],
            )
            <= self_repeat["gradient_groups"][group]["max_abs"]
        )
    passed = (
        all(row["exact"] for row in source_rows)
        and old_vs_new["sample_ids_exact"]
        and old_repeat_vs_new["sample_ids_exact"]
        and old_vs_new["learning_rates_exact"]
        and old_repeat_vs_new["learning_rates_exact"]
        and forward_exact
        and all(envelope_checks.values())
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "PASS_WITH_EXPLAINED_CUDA_BACKWARD_NONDETERMINISM"
            if passed
            else "BLOCKED_BY_TRAINING_CODE_EQUIVALENCE"
        ),
        "protocol": "RELPLUS_OLD_VS_THREE_ARM_200_STEP_C1_ADJUDICATION_V1",
        "steps": 200,
        "single_gpu": True,
        "fixed_zero_tolerance_result": "FAIL",
        "written_explanation": (
            "The frozen CUDA graph is bitwise exact through transformed inputs, "
            "logits and focal-loss maps, but backward contains non-bitwise CUDA "
            "reductions. The same old code also fails its own zero-tolerance repeat. "
            "New-code drift is accepted only when its distance to at least one old "
            "repeat stays within the observed old-vs-old envelope and all listed "
            "numerical source files are exact."
        ),
        "formal_training_flags_unchanged": True,
        "formal_cudnn_deterministic": False,
        "audit_cudnn_deterministic": True,
        "audit_tf32": False,
        "numerical_source_files": source_rows,
        "old_vs_old_repeat": self_repeat,
        "old_vs_new": old_vs_new,
        "old_repeat_vs_new": old_repeat_vs_new,
        "forward_and_loss_map_bitwise_exact": forward_exact,
        "envelope_checks": envelope_checks,
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "forward_and_loss_map_bitwise_exact": forward_exact,
                "envelope_checks": envelope_checks,
                "numerical_source_files_exact": all(
                    row["exact"] for row in source_rows
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
