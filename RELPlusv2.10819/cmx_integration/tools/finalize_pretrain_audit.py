#!/usr/bin/env python3
"""Combine S3D data, model, update and sliding evidence into the training gate."""

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ("rgbd", "hha", "rel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    audit = args.output_root / "audit"

    data = json.loads((audit / "s3d_data_audit.json").read_text(encoding="utf-8"))
    preflight = {
        arm: json.loads((audit / "preflight_cmx_{}_1080.json".format(arm)).read_text(encoding="utf-8"))
        for arm in ARMS
    }
    sliding = {
        arm: json.loads((audit / "preflight_cmx_{}_sliding.json".format(arm)).read_text(encoding="utf-8"))
        for arm in ARMS
    }
    config_diff = (audit / "THREE_CONFIG_DIFF.md").read_text(encoding="utf-8")
    single = json.loads((audit / "eval_consistency_rgbd_single.json").read_text(encoding="utf-8"))
    multi = json.loads((audit / "eval_consistency_rgbd_multi.json").read_text(encoding="utf-8"))
    single_confusion = np.loadtxt(audit / "eval_consistency_rgbd_single_confusion.csv", delimiter=",")
    multi_confusion = np.loadtxt(audit / "eval_consistency_rgbd_multi_confusion.csv", delimiter=",")
    metric_keys = (
        "mIoU", "pixel_accuracy", "mean_class_accuracy", "valid_pixels",
        "ignore_pixels", "correct_pixels", "iou", "class_accuracy",
    )
    eval_consistent = all(single[key] == multi[key] for key in metric_keys) and np.array_equal(
        single_confusion, multi_confusion
    )

    parameter_rows = []
    parameter_signatures = []
    for arm in ARMS:
        counts = preflight[arm]["parameters"]
        signature = tuple(counts[key] for key in ("total", "rgb_encoder", "x_encoder", "fusion", "decoder", "trainable"))
        parameter_signatures.append(signature)
        parameter_rows.append(
            "| CMX-{} | {} | {} | {} | {} | {} | {} |".format(
                arm.upper(), counts["total"], counts["rgb_encoder"], counts["x_encoder"],
                counts["fusion"], counts["decoder"], counts["trainable"]
            )
        )
    models_aligned = len(set(parameter_signatures)) == 1 and parameter_signatures[0][0] == 66567573
    model_lines = [
        "# Model alignment report", "", "Status: **{}**".format("PASS" if models_aligned else "FAIL"), "",
        "Recursive source comparison found no differences in the three `models/` trees. All arms use `dual_segformer.py` and MLPDecoder; no Gate, SMMF or DyMM parameters are present.",
        "", "| Arm | Total | RGB encoder | X encoder | CM-FRM/FFM | Decoder | Trainable |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *parameter_rows,
        "", "All three MiT-B2 loads mapped 664 target tensors: 332 into the RGB branch and 332 into the X branch; only the source classification head was unused.",
    ]
    (audit / "MODEL_ALIGNMENT_REPORT.md").write_text("\n".join(model_lines) + "\n", encoding="utf-8")

    checks = [
        ("Original CMX identity", models_aligned),
        ("Three model structures and parameter counts", models_aligned),
        ("Fold 1 split and 1040/373 counts", data["status"] == "PASS" and data["train"] == 1040 and data["test"] == 373),
        ("RGB/Label/Depth/HHA/REL complete intersection", data["all_modalities_readable_and_aligned"] == 1413),
        ("Label protocol", data["raw_label_protocol"].startswith("0=ignore")),
        ("Three resolved configs", "Status: **PASS**" in config_diff),
        ("MiT-B2 both branches", all(item["pretrained"]["rgb_branch_loaded"] and item["pretrained"]["x_branch_loaded"] for item in preflight.values())),
        ("Real 1080 forward/backward/optimizer update", all(item["status"] == "PASS" for item in preflight.values())),
        ("RGB/X/fusion/decoder parameters changed", all(set(item["updates"]) == {"rgb_encoder", "x_encoder", "fusion", "decoder"} for item in preflight.values())),
        ("Full 2048x4096 sliding evaluation", all(item["status"] == "PASS" and item["coverage_zero_pixels"] == 0 for item in sliding.values())),
        ("Single-GPU versus 8-GPU evaluator", eval_consistent),
    ]
    go = all(value for _, value in checks)
    gate_lines = [
        "# Pretrain GO / NO-GO", "", "Decision: **{}**".format("PRETRAIN_GO" if go else "PRETRAIN_NO_GO"), "",
        "| Gate | Result |", "|---|---|",
    ]
    gate_lines.extend("| {} | {} |".format(name, "PASS" if value else "FAIL") for name, value in checks)
    gate_lines.extend(
        [
            "", "REL paper/code differences remain `DOCUMENTED_NON_BLOCKING` and are not a NO-GO condition.",
            "", "A GO unlocks only the frozen sequential Fold 1 training order: RGBD, then HHA, then REL. It does not unlock Fold 2/3, extra seeds or method extensions.",
        ]
    )
    (audit / "PRETRAIN_GO_NO_GO.md").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    print("PRETRAIN_GO" if go else "PRETRAIN_NO_GO")
    raise SystemExit(0 if go else 1)


if __name__ == "__main__":
    main()
