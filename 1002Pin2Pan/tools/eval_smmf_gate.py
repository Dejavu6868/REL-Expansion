#!/usr/bin/env python3
"""Evaluate the trained gated SMMF checkpoint with SMMF's window layout, or with its polar rows at lower resolution.

Everything that touches the model runs inside the original Pano_MA_Seg code (``--smmf-root``):
its dataset reader, model builder, strict checkpoint loader, gate, and SMMF window fusion
(``Evaluator.scale_process_rgbX_smmf``: cyclic padding, exp(logits) summed over windows, the
two padded ends added). This script sets the evaluation fields of its config, calls that fusion
once per zone, and scores the result.

* Default: SMMF's own pass. Every window row runs on the native panorama.
* ``--polar-scale S``: every row except the centred middle one runs on the panorama resized by
  S (RGB, X and latitude with INTER_LINEAR, as the original multi-scale path), with window size,
  cyclic padding and row starts scaled by S; the fused map is resized back. Those rows give the
  prediction only outside the middle row. The middle row runs native and is the only source
  inside it. ``--middle-height H`` makes the middle windows H rows tall, centred.
* ``--whole-scale S``: every row on the panorama resized by S (the control).

The gate is unchanged: each window picks its own branch. A forward hook on ``backbone.Gate``
counts the choices per zone and row. Scores are SMMF's own 13-class metric (its
``compute_score``), plus per-band confusions for the 13-class argmax and for the argmax over
the 8 SPan8 classes. The original forward appends gate_selection.txt and gate_weights.txt to
the working directory, so inference runs from ``--output``.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from eval_pano_transfer import EIGHT_CLASS_NAMES, elevation_band_confusions

CLASS_NAMES = ["beam", "board", "bookcase", "ceiling", "chair", "clutter", "column", "door",
               "floor", "sofa", "table", "wall", "window"]
CROP = (1080, 1080)
GATE_BRANCHES = 5
# model-100.pth of log_rebut/log_Stanford2D3DPano_rle_gate_mit_b2 (internal epoch 175).
EXPECTED_SHA256 = "a2b13d1a5a0036fe4371092af56c6989ab0efedf10d3b923dbcbfd62a7326ff6"
REPORTED_MIOU_PERCENT = 67.367  # val_2026_01_28_00_57_48.log
# Fields of the original config that the reader, builder and evaluator read; set explicitly
# because the current config.py selects Matterport.
EVAL_CONFIG = {
    "dataset_name": "Stanford2D3DPano",
    "num_classes": 13,
    "class_names": CLASS_NAMES,
    "x_name": "rle",
    "x_is_single_channel": False,
    "x_selected_channel": [],
    "in_chans": 3,
    "in_chans_x": 3,
    "backbone": "gate_mit_b2",
    "decoder": "MLPDecoder",
    "decoder_embed_dim": 512,
    "using_gate": True,
    "gate_cfg": {"soft_gate": False, "temp": 1.0},
    "eval_gate_cfg": {"soft_gate": False, "temp": 1.0},
    "using_smmf": True,
    "eval_crop_size": list(CROP),
    "eval_scale_array": [1],
    "eval_flip": False,
    "eval_reprojection": False,
    "use_guided_rotation": False,
}
SOURCE_FILES = (
    "config.py", "engine/evaluator.py", "models/builder.py", "models/encoders/dual_gate_segformer.py",
    "models/net_utils.py", "models/decoders/MLPDecoder.py", "dataloader/RGBXDataset.py",
    "dataloader/dataloader.py", "utils/transforms.py", "utils/metric.py", "utils/pyt_utils.py",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def scaled(value, scale):
    out = value * scale
    if abs(out - round(out)) > 1e-9:
        raise ValueError("{} x {} is not a whole pixel".format(value, scale))
    return int(round(out))


def window_columns(width, crop_width, pad, stride_rate):
    """Left columns of one row's windows in panorama coordinates, as scale_process_rgbX_smmf places them."""
    stride = int(np.ceil(crop_width * stride_rate))
    padded = width + 2 * pad
    count = int(np.ceil((padded - crop_width) / stride)) + 1
    return [(min(k * stride + crop_width, padded) - crop_width - pad) % width for k in range(count)]


def zone(name, scale, height, width, crop, pad, rows, keep, stride_rate):
    size = (scaled(height, scale), scaled(width, scale))
    crop = tuple(scaled(length, scale) for length in crop)
    rows = [scaled(row, scale) for row in rows]
    if size[0] <= crop[0] or size[1] <= crop[1]:  # the original then runs one whole-image pass
        raise ValueError("{} zone: windows {} do not tile a {} panorama".format(name, crop, size))
    if any(row < 0 or row + crop[0] > size[0] for row in rows):
        raise ValueError("{} zone: a window row leaves the panorama".format(name))
    return {
        "name": name,
        "scale": scale,
        "input_size": list(size),
        "crop": crop,
        "pad": scaled(pad, scale),
        "rows": rows,
        "columns": window_columns(size[1], crop[1], scaled(pad, scale), stride_rate),
        "stride_rate": stride_rate,
        "keep": keep,
    }


def zone_plan(height, width, rows, pad, stride_rate, polar_scale=None, middle_height=None,
              whole_scale=None, crop=CROP):
    """The zones of one pipeline; each zone's ``keep`` marks the panorama rows it predicts."""
    if polar_scale and whole_scale:
        raise ValueError("--polar-scale and --whole-scale are exclusive")
    if middle_height is not None and not polar_scale:
        raise ValueError("--middle-height needs --polar-scale")
    for scale in (polar_scale, whole_scale):
        if scale is not None and not 0 < scale < 1:
            raise ValueError("scales must lie in (0, 1)")
    covered = np.zeros(height, dtype=np.int64)
    for row in rows:
        covered[row:row + crop[0]] += 1
    if not covered.all():
        raise ValueError("window rows {} leave panorama rows uncovered".format(rows))
    everything = np.ones(height, dtype=bool)
    if not polar_scale:
        name = "whole" if whole_scale else "smmf"
        return [zone(name, whole_scale or 1, height, width, crop, pad, rows, everything, stride_rate)]

    middle = [row for row in rows if 2 * row + crop[0] == height]
    if len(middle) != 1:
        raise ValueError("--polar-scale needs exactly one centred window row")
    middle_height = middle_height or crop[0]
    if not crop[0] <= middle_height < height or (height - middle_height) % 2:
        raise ValueError("--middle-height must be in [{}, {}) with an even remainder".format(crop[0], height))
    start = (height - middle_height) // 2
    inside = np.zeros(height, dtype=bool)
    inside[start:start + middle_height] = True
    polar_rows = [row for row in rows if row != middle[0]]
    covered[:] = 0
    for row in polar_rows:
        covered[row:row + crop[0]] += 1
    if not covered[~inside].all():
        raise ValueError("the polar rows do not cover the panorama outside the middle zone")
    return [
        zone("polar", polar_scale, height, width, crop, pad, polar_rows, ~inside, stride_rate),
        zone("middle", 1, height, width, (middle_height, crop[1]), pad, [start], inside, stride_rate),
    ]


def row_runs(mask):
    """[start, end) runs of True rows."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return edges.reshape(-1, 2).tolist()


def zone_scores(segmentor, config, zone_spec, image, modal_x, lat, device):
    """One call of the original SMMF fusion; exp(logit) sums at the panorama size."""
    height, width = image.shape[:2]
    scale = zone_spec["scale"]
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        x_mode = cv2.INTER_NEAREST if modal_x.ndim == 2 else cv2.INTER_LINEAR
        modal_x = cv2.resize(modal_x, None, fx=scale, fy=scale, interpolation=x_mode)
        lat = cv2.resize(lat, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    config.smmf_pad_h = zone_spec["pad"]
    config.window_positions = list(zone_spec["rows"])
    return segmentor.scale_process_rgbX_smmf(
        image, modal_x, lat, (height, width), zone_spec["crop"], zone_spec["stride_rate"], device
    )


def fused_scores(segmentor, config, plan, image, modal_x, lat, device, gate_log, gate_counts):
    """Each panorama row from its own zone; gate choices counted per zone and window row."""
    fused = None
    for zone_spec, counts in zip(plan, gate_counts):
        gate_log.clear()
        scores = zone_scores(segmentor, config, zone_spec, image, modal_x, lat, device)
        columns = len(zone_spec["columns"])
        if len(gate_log) != len(zone_spec["rows"]) * columns:
            raise RuntimeError("{} zone ran {} windows, expected {}".format(
                zone_spec["name"], len(gate_log), len(zone_spec["rows"]) * columns))
        for index, choice in enumerate(gate_log):
            counts[index // columns, choice] += 1
        if fused is None:
            fused = np.zeros_like(scores)
        keep = zone_spec["keep"]
        fused[keep] = scores[keep]
    return fused


def import_smmf(root):
    """The original modules, imported from their root (config.py reads the working directory)."""
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    cwd = os.getcwd()
    os.chdir(root)
    try:
        from config import config
        from dataloader.RGBXDataset import RGBXDataset
        from dataloader.dataloader import ValPre
        from engine.evaluator import Evaluator
        from models.builder import EncoderDecoder
        from utils.metric import compute_score, hist_info
        from utils.pyt_utils import load_model
    finally:
        os.chdir(cwd)
    return {
        "config": config, "RGBXDataset": RGBXDataset, "ValPre": ValPre, "Evaluator": Evaluator,
        "EncoderDecoder": EncoderDecoder, "compute_score": compute_score, "hist_info": hist_info,
        "load_model": load_model,
    }


def configure(config, stride_rate):
    for key, value in EVAL_CONFIG.items():
        setattr(config, key, value)
    config.eval_stride_rate = stride_rate
    if not (np.allclose(config.norm_mean, [0.485, 0.456, 0.406])
            and np.allclose(config.norm_std, [0.229, 0.224, 0.225])):
        raise ValueError("the original config's normalisation is not ImageNet's")


def build_network(smmf, stride_rate):
    import torch.nn as nn

    configure(smmf["config"], stride_rate)
    network = smmf["EncoderDecoder"](cfg=smmf["config"], criterion=None, norm_layer=nn.BatchNorm2d, mode="eval")
    network.eval()
    return network


def data_setting(root):
    return {
        "rgb_root": str(root / "image"), "rgb_format": "_rgb.png",
        "gt_root": str(root / "label"), "gt_format": "_semantic.png", "transform_gt": True,
        "x_root": str(root / "rle"), "x_format": "_hha.png", "x_single_channel": False,
        "regions_root": "", "regions_format": "", "lw_root": "", "lw_format": "",
        "train_source": str(root / "fold1_train.txt"), "eval_source": str(root / "fold1_test.txt"),
        "class_names": CLASS_NAMES,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smmf-root", required=True, type=Path, help="original Pano_MA_Seg folder")
    parser.add_argument("--data-root", required=True, type=Path, help="datasets/Stanford2D3DPano")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride-rate", type=float, default=2 / 3)
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 484, 968], help="window row starts")
    parser.add_argument("--pad", type=int, default=652, help="cyclic padding on each side")
    parser.add_argument("--polar-scale", type=float, default=None)
    parser.add_argument("--middle-height", type=int, default=None)
    parser.add_argument("--whole-scale", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("smmf_root", "data_root", "checkpoint", "output"):  # inference runs from --output
        setattr(args, name, getattr(args, name).resolve())

    plan = zone_plan(2048, 4096, args.rows, args.pad, args.stride_rate, args.polar_scale,
                     args.middle_height, args.whole_scale)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(output))
    output.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(args.checkpoint)
    if digest != args.expected_sha256:
        raise ValueError("checkpoint sha256 {} is not the expected {}".format(digest, args.expected_sha256))

    import torch

    smmf = import_smmf(args.smmf_root)
    config = smmf["config"]
    network = build_network(smmf, args.stride_rate)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    smmf["load_model"](network, checkpoint["model"])  # strict
    epoch = checkpoint.get("epoch")
    del checkpoint
    gate_log = []
    network.backbone.Gate.register_forward_hook(
        lambda module, inputs, weight: gate_log.append(int(weight.argmax(1)[0]))
    )

    dataset = smmf["RGBXDataset"](data_setting(args.data_root), "val", smmf["ValPre"]())
    dataset._file_names = dataset._file_names[: args.limit]
    segmentor = smmf["Evaluator"](dataset, config.num_classes, config.norm_mean, config.norm_std, network,
                                  config.eval_scale_array, config.eval_flip, [args.device])
    segmentor.val_func = network

    eight = np.asarray([CLASS_NAMES.index(name) for name in EIGHT_CLASS_NAMES])
    gate_counts = [np.zeros((len(z["rows"]), GATE_BRANCHES), dtype=np.int64) for z in plan]
    bands = {"13": None, "8pred": None}
    names = []
    started = time.time()
    os.chdir(output)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            label = sample["label"]
            if sample["data"].shape[:2] != (2048, 4096) or label.shape != (2048, 4096):
                raise ValueError("{} is not 2048x4096".format(sample["fn"]))
            scores = fused_scores(segmentor, config, plan, sample["data"], sample["modal_x"], sample["lat"],
                                  args.device, gate_log, gate_counts)
            for key, prediction in (("13", scores.argmax(2)), ("8pred", eight[scores[:, :, eight].argmax(2)])):
                confusion = elevation_band_confusions(config.num_classes, prediction, label, smmf["hist_info"])
                bands[key] = confusion if bands[key] is None else bands[key] + confusion
            names.append(sample["fn"])
            if index % 10 == 0 or index + 1 == len(dataset):
                print("{}/{} {} {:.0f}s".format(index + 1, len(dataset), sample["fn"], time.time() - started),
                      flush=True)

    np.save(output / "elevation_band_confusion.npy", bands["13"])
    np.save(output / "elevation_band_confusion_8pred.npy", bands["8pred"])
    confusion = bands["13"].sum(0)
    iou, mean_iou, _, freq_iou, _, mean_pixel_acc, pixel_acc = smmf["compute_score"](
        confusion.astype(np.float64), int(np.trace(confusion)), int(confusion.sum())
    )
    report = {
        "status": "SMOKE" if args.limit is not None else "COMPLETED",
        "pipeline": "polar" if args.polar_scale else ("whole" if args.whole_scale else "smmf"),
        "checkpoint": {"path": str(args.checkpoint), "sha256": digest, "epoch": epoch},
        "smmf_root": str(args.smmf_root),
        "source_sha256": {name: file_sha256(args.smmf_root / name) for name in SOURCE_FILES},
        "torch_version": torch.__version__,
        "config": dict({key: getattr(config, key) for key in EVAL_CONFIG}, eval_stride_rate=args.stride_rate),
        "data_root": str(args.data_root),
        "panoramas": len(names),
        "first_panoramas": names[:3],
        "panorama_size": [2048, 4096],
        "zones": [
            {
                "name": z["name"], "scale": z["scale"], "input_size": z["input_size"], "window_size": list(z["crop"]),
                "cyclic_pad": z["pad"], "window_rows": z["rows"], "window_columns": z["columns"],
                "predicts_rows": row_runs(z["keep"]),
                "gate_counts_per_window_row": counts.tolist(),
            }
            for z, counts in zip(plan, gate_counts)
        ],
        "fusion": "original scale_process_rgbX_smmf: sum of exp(logits), cyclic ends added",
        "mIoU_13_percent": 100 * float(mean_iou),
        "pixel_accuracy_percent": 100 * float(pixel_acc),
        "freq_iou_percent": 100 * float(freq_iou),
        "mean_pixel_accuracy_percent": 100 * float(mean_pixel_acc),
        "per_class_iou_percent": dict(zip(CLASS_NAMES, (100 * iou).round(3).tolist())),
        "reported_smmf_mIoU_13_percent": REPORTED_MIOU_PERCENT,
        "seconds": round(time.time() - started),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("mIoU (13 classes) {:.3f}%".format(report["mIoU_13_percent"]))


if __name__ == "__main__":
    main()
