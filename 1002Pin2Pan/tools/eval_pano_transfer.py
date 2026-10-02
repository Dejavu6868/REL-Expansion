#!/usr/bin/env python3
"""Evaluate a frozen S2D (pinhole) CMX checkpoint on Stanford2D3D panoramas.

Source-only pinhole-to-panorama transfer: the network, checkpoint and input
normalisation are the frozen training ones; only the input images change.

* RGB is read with the frozen loader's own call (``cv2.imread(path,
  cv2.COLOR_BGR2RGB)``), so the channel order matches training.
* The REL+ arm's X input on the panorama is the original ERP ``getREL``,
  computed on the depth resized (nearest) to the evaluation size, mirroring
  the perspective cache, which was generated at the 480x480 model size.
  ``tools/cross_projection.py`` checks that the two code paths agree.
* Labels come from ``pano/semantic`` through ``semantic_labels.json`` and are
  resized with nearest neighbour; ``<UNK>`` becomes ignore 255.
* Inference is one whole-panorama pass with circular padding on the left and
  right edges, so the 180-degree seam sees context from both sides.

HHA and raw-depth arms are refused: their frozen inputs are defined only for
pinhole cameras, and no ERP version exists in this repository.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "0927调参结果" / "selected_recipe" / "code" / "training" / "source"
)
EXPECTED_CLASSES = [
    "<UNK>", "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
]
SUPPORTED_X_MODE = "rel_plus_v2_1"


def label_lookup(semantic_labels_path):
    """Instance index -> class id 0..13, as in tools/prepare_cmx_rel_smoke_data.py."""
    labels = json.loads(Path(semantic_labels_path).read_text(encoding="utf-8"))
    classes = []
    class_ids = []
    for label in labels:
        class_name = label.split("_", 1)[0]
        if class_name not in classes:
            classes.append(class_name)
        class_ids.append(classes.index(class_name))
    if classes != EXPECTED_CLASSES:
        raise ValueError("Unexpected Stanford2D3D semantic class order")
    return np.asarray(class_ids, dtype=np.uint8)


def decode_semantic(semantic_bgr, lookup):
    """RGB-encoded instance index -> model label 0..12, ignore 255."""
    indices = (
        semantic_bgr[:, :, 0].astype(np.uint32)
        + semantic_bgr[:, :, 1].astype(np.uint32) * 256
        + semantic_bgr[:, :, 2].astype(np.uint32) * 65536
    )
    stored = np.zeros(indices.shape, dtype=np.uint8)
    known = indices < len(lookup)
    stored[known] = lookup[indices[known]]
    label = stored.astype(np.int16) - 1
    label[stored == 0] = 255
    return label.astype(np.uint8)


def list_panoramas(stanford_root, areas):
    samples = []
    for area in areas:
        for rgb_path in sorted((stanford_root / area / "pano" / "rgb").glob("*_rgb.png")):
            stem = rgb_path.name[: -len("_rgb.png")]
            depth_path = rgb_path.parents[1] / "depth" / (stem + "_depth.png")
            semantic_path = rgb_path.parents[1] / "semantic" / (stem + "_semantic.png")
            for path in (depth_path, semantic_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            samples.append(
                {
                    "sample_id": "{}/{}".format(area, stem),
                    "rgb": rgb_path,
                    "depth": depth_path,
                    "semantic": semantic_path,
                }
            )
    if not samples:
        raise FileNotFoundError("no panoramas found under {}".format(stanford_root))
    return samples


def load_sample(sample, size, lookup, frozen):
    height, width = size
    rgb = frozen["open_image"](str(sample["rgb"]), frozen["rgb_flag"])
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("{} is not a 3-channel image".format(sample["rgb"]))
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    raw_depth = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
    if raw_depth is None or raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise ValueError("{} is not a uint16 depth map".format(sample["depth"]))
    raw_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m = (raw_depth + np.uint16(1)).astype(np.float32) / 512.0  # rel.getImage
    modal_x = frozen["getREL"](depth_m)

    semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
    if semantic is None:
        raise FileNotFoundError(sample["semantic"])
    semantic = cv2.resize(semantic, (width, height), interpolation=cv2.INTER_NEAREST)
    return rgb, modal_x, decode_semantic(semantic, lookup)


def predict_with_wrap(network, rgb, modal_x, wrap_pad, device):
    """Whole-image logits with circular padding along the panorama width."""
    import torch
    import torch.nn.functional as F

    rgb = rgb.to(device)
    modal_x = modal_x.to(device)
    if wrap_pad > 0:
        rgb = F.pad(rgb, (wrap_pad, wrap_pad, 0, 0), mode="circular")
        modal_x = F.pad(modal_x, (wrap_pad, wrap_pad, 0, 0), mode="circular")
    with torch.no_grad():
        logits = network(rgb, modal_x)
    if wrap_pad > 0:
        logits = logits[:, :, :, wrap_pad:-wrap_pad]
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("non-finite logits")
    return logits


def import_frozen(source_root):
    source_root = Path(source_root).resolve()
    sys.dont_write_bytecode = True  # keep the frozen archive's file set unchanged
    sys.path.insert(0, str(source_root))
    from dataloader.RGBXDataset import RGBXDataset
    from engine.relplus_evaluator import _normalized_bchw, metrics_from_confusion
    from models.builder import EncoderDecoder
    from third_party.rel_original.rel import getREL
    from tools.eval_rel_plus_v2_3_full import load_checkpoint_once
    from utils.metric import hist_info

    return {
        "open_image": RGBXDataset._open_image,
        "rgb_flag": cv2.COLOR_BGR2RGB,  # the frozen loader's imread flag
        "normalized_bchw": _normalized_bchw,
        "metrics_from_confusion": metrics_from_confusion,
        "EncoderDecoder": EncoderDecoder,
        "getREL": getREL,
        "load_checkpoint_once": load_checkpoint_once,
        "hist_info": hist_info,
    }


def load_config(path):
    from easydict import EasyDict

    config = EasyDict(json.loads(Path(path).read_text(encoding="utf-8")))
    if config.x_mode != SUPPORTED_X_MODE:
        raise ValueError(
            "x_mode {} has no panorama input: only {} (ERP getREL) is supported; "
            "HHA and raw depth are defined for pinhole cameras only".format(
                config.x_mode, SUPPORTED_X_MODE
            )
        )
    config.norm_mean = np.asarray(config.norm_mean, dtype=np.float64)
    config.norm_std = np.asarray(config.norm_std, dtype=np.float64)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stanford-root", required=True, type=Path)
    parser.add_argument("--semantic-labels", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="resolved arm config JSON")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-epoch", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--areas", nargs="+", default=["area_5a", "area_5b"])
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--wrap-pad", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = import_frozen(args.source_root)
    config = load_config(args.config)
    lookup = label_lookup(args.semantic_labels)
    samples = list_panoramas(args.stanford_root, args.areas)[: args.limit]

    device = torch.device(args.device)
    network = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    epoch = frozen["load_checkpoint_once"](
        network, args.checkpoint, expected_epoch=args.expected_epoch
    )
    network.to(device)
    network.eval()

    started = time.time()
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    for index, sample in enumerate(samples, start=1):
        rgb, modal_x, label = load_sample(sample, (args.height, args.width), lookup, frozen)
        logits = predict_with_wrap(
            network,
            frozen["normalized_bchw"](rgb, config.norm_mean, config.norm_std),
            frozen["normalized_bchw"](modal_x, config.norm_mean, config.norm_std),
            args.wrap_pad,
            device,
        )
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        hist, _, _ = frozen["hist_info"](config.num_classes, prediction, label)
        confusion += hist.astype(np.int64)
        if index % 20 == 0 or index == len(samples):
            print("{}/{} panoramas".format(index, len(samples)), flush=True)

    metrics = frozen["metrics_from_confusion"](confusion)
    report = {
        "status": "SMOKE" if args.limit else "COMPLETED",
        "setting": "source-only pinhole-to-panorama transfer",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "config": str(args.config),
        "x_mode": config.x_mode,
        "panorama_x_input": "ERP getREL on nearest-resized depth",
        "areas": args.areas,
        "eval_size": [args.height, args.width],
        "wrap_pad": args.wrap_pad,
        "sample_count": len(samples),
        "seconds": time.time() - started,
        "metrics": metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "samples.txt").write_text(
        "".join(sample["sample_id"] + "\n" for sample in samples), encoding="utf-8"
    )
    np.savetxt(args.output / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    with (args.output / "per_class_iou.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "IoU_percent"])
        for class_id, (name, iou) in enumerate(
            zip(config.class_names, metrics["per_class_iou_percent"])
        ):
            writer.writerow([class_id, name, iou])
    print("mIoU {:.4f}% over {} panoramas".format(metrics["mIoU_percent"], len(samples)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
