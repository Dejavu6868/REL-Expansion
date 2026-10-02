#!/usr/bin/env python3
"""Compatibility entry for the 1--3 sample evaluator plumbing smoke only."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _unbatch(batch):
    sample = {"fn": batch["fn"][0]}
    for key in ("data", "modal_x", "label", "modal_x_valid_mask"):
        value = batch[key][0]
        sample[key] = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    return sample


def _load_checkpoint(network, checkpoint):
    payload = torch.load(str(checkpoint), map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    incompatible = network.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    return {
        "path": str(checkpoint),
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-module",
        default=(
            "configs.stanford2d3d_s2d."
            "cmx_mit_b2_rel_plus_v2_2_pilot"
        ),
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 3:
        raise ValueError("plumbing smoke limit must be between 1 and 3")

    selected = importlib.import_module(args.config_module)
    config = selected.config
    sys.modules["config"] = selected

    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.data_setting import build_data_setting
    from dataloader.dataloader import ValPre
    from engine.relplus_evaluator import (
        evaluate_prepared_sample,
        prepare_eval_sample,
        save_prediction_pair,
    )
    from models.builder import EncoderDecoder

    setting = build_data_setting(config, split="val")
    dataset = RGBXDataset(
        setting, "val", ValPre(x_mode=setting["x_mode"])
    )
    dataset._file_names = dataset._file_names[: args.limit]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    network = EncoderDecoder(
        cfg=config, criterion=None, norm_layer=nn.BatchNorm2d
    )
    checkpoint = {"path": None, "epoch": None, "mode": "random_initialization"}
    if args.checkpoint is not None:
        checkpoint.update(_load_checkpoint(network, args.checkpoint))
        checkpoint["mode"] = "checkpoint"
    network.to(args.device)

    hist = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    samples = []
    for batch in loader:
        prepared = prepare_eval_sample(_unbatch(batch), config)
        result = evaluate_prepared_sample(
            network,
            prepared,
            class_num=config.num_classes,
            ignore_index=config.background,
            device=args.device,
        )
        hist += result["hist"]
        saved = None
        if args.predictions is not None:
            paths = save_prediction_pair(
                result["prediction"],
                prepared.sample_id,
                args.predictions,
                RGBXDataset.get_class_colors(),
            )
            saved = {name: str(path) for name, path in paths.items()}
        samples.append(
            {
                "sample_id": prepared.sample_id,
                "prediction_shape": list(result["prediction"].shape),
                "hist_shape": list(result["hist"].shape),
                "logits_finite": result["logits_finite"],
                "diagnostic_mask_passed_to_model": False,
                "saved": saved,
            }
        )

    report = {
        "status": "PASS",
        "claim": "evaluator plumbing smoke PASS",
        "scientific_metric_reported": False,
        "protocol_id": config.protocol_id,
        "representation_protocol_id": config.representation_protocol_id,
        "x_mode": config.x_mode,
        "channel_order": list(config.channel_order),
        "eval_mode": "full_image_480",
        "eval_flip": config.eval_flip,
        "eval_scales": list(config.eval_scale_array),
        "align_corners": config.eval_align_corners,
        "ignore_index": config.background,
        "class_count": config.num_classes,
        "checkpoint": checkpoint,
        "processed_samples": len(samples),
        "confusion_shape": list(hist.shape),
        "confusion_total_valid_pixels": int(hist.sum()),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
