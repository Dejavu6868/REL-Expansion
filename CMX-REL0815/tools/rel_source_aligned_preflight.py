import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import SourceAlignedRELTrainPre
from models.builder import EncoderDecoder
from rel_source_aligned.adapters.stanford2d3d_perspective_adapter import (
    PerspectiveInputAdapter,
)


def _dataset_setting(dataset_root):
    return {
        "rgb_root": str(dataset_root / "RGB"),
        "rgb_format": ".png",
        "gt_root": str(dataset_root / "Label"),
        "gt_format": ".png",
        "transform_gt": True,
        "x_root": str(dataset_root / "HHA"),
        "x_format": ".png",
        "x_single_channel": False,
        "x_mode": "rel_source_aligned",
        "rel_impl": "official_source",
        "x_online_relplus": False,
        "depth_root": str(dataset_root / "Depth16"),
        "depth_format": ".png",
        "pose_root": str(dataset_root / "Pose"),
        "pose_format": ".json",
        "train_source": str(dataset_root / "train.txt"),
        "eval_source": str(dataset_root / "test.txt"),
        "class_names": [
            "beam",
            "board",
            "bookcase",
            "ceiling",
            "chair",
            "clutter",
            "column",
            "door",
            "floor",
            "sofa",
            "table",
            "wall",
            "window",
        ],
    }


def _model_config():
    return SimpleNamespace(
        backbone="mit_b2",
        decoder="MLPDecoder",
        num_classes=13,
        decoder_embed_dim=512,
        pretrained_model=None,
        bn_eps=1e-3,
        bn_momentum=0.1,
    )


def run_one_batch_preflight(repo_root, authority_root, dataset_root, device="cuda:0"):
    del repo_root
    authority_root = Path(authority_root)
    dataset_root = Path(dataset_root)
    if not authority_root.is_dir():
        raise FileNotFoundError("authority source root not found: {}".format(authority_root))
    if not dataset_root.is_dir():
        raise FileNotFoundError("dataset root not found: {}".format(dataset_root))
    if not device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("bounded CMX preflight requires an explicit available CUDA device")

    torch.manual_seed(12345)
    np.random.seed(12345)
    target_device = torch.device(device)
    torch.cuda.set_device(target_device)

    adapter = PerspectiveInputAdapter(authority_root)
    preprocess = SourceAlignedRELTrainPre(
        np.array([0.485, 0.456, 0.406]),
        np.array([0.229, 0.224, 0.225]),
        adapter=adapter,
        target_size=(480, 480),
        scale_array=[1.0],
        horizontal_flip=False,
    )
    dataset = RGBXDataset(_dataset_setting(dataset_root), "train", preprocess)
    item = dataset[0]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    rgb = batch["data"].to(target_device, non_blocking=False)
    modal_x = batch["modal_x"].to(target_device, non_blocking=False)
    label = batch["label"].to(target_device, non_blocking=False)
    model = EncoderDecoder(
        cfg=_model_config(),
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        norm_layer=nn.BatchNorm2d,
    ).to(target_device)
    model.eval()

    rgb_inputs = []
    x_inputs = []

    def capture_rgb(_module, arguments):
        rgb_inputs.append(arguments[0].detach().cpu())

    def capture_x(_module, arguments):
        x_inputs.append(arguments[0].detach().cpu())

    rgb_hook = model.backbone.patch_embed1.proj.register_forward_pre_hook(capture_rgb)
    x_hook = model.backbone.extra_patch_embed1.proj.register_forward_pre_hook(capture_x)
    try:
        with torch.no_grad():
            logits_a = model(rgb, modal_x)
            modal_x_b = torch.flip(modal_x, dims=[-1])
            logits_b = model(rgb, modal_x_b)
            loss = functional.cross_entropy(logits_a, label.long(), ignore_index=255)
    finally:
        rgb_hook.remove()
        x_hook.remove()

    report = {
        "sample_id": str(item["fn"]),
        "dataset_item_pass": tuple(item["modal_x"].shape) == (3, 480, 480),
        "collation_pass": tuple(batch["modal_x"].shape) == (1, 3, 480, 480),
        "forward_pass": tuple(logits_a.shape) == (1, 13, 480, 480),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "loss_value": float(loss.detach().cpu().item()),
        "x_input_changed": not torch.equal(x_inputs[0], x_inputs[1]),
        "logits_changed": not torch.equal(logits_a.detach().cpu(), logits_b.detach().cpu()),
        "rgb_input_unchanged": torch.equal(rgb_inputs[0], rgb_inputs[1]),
        "rel_generation_order": list(preprocess.last_stage_order),
        "rel_impl": preprocess.last_impl,
        "model_architecture": "CMX dual MiT-B2 + MLPDecoder",
        "model_weights": "single fixed random initialization; no checkpoint loaded",
        "backward_called": False,
        "optimizer_step_called": False,
        "checkpoint_written": False,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_one_batch_preflight(
        args.repo_root, args.authority_root, args.dataset_root, args.device
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
