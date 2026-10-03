#!/usr/bin/env python3
"""Crop-tiled panorama evaluation of a frozen S2D (pinhole) CMX checkpoint.

The whole-panorama evaluation (``eval_pano_transfer.py``) feeds the network
ERP inputs whose per-image normalisation spans 360 degrees. This tool instead
cuts each panorama into pinhole crops, builds every crop's input the
way the training cache was built, runs the network on each crop, and stitches
the crop probabilities back onto the ERP label grid. This changes projection,
normalisation support, gravity and multi-view fusion together; the score
difference does not identify the effect of normalisation alone.

* Default FOV: 62.47653165897473 degrees, the median from all 52,903 training
  camera intrinsics (the training range is 45 to 75 degrees).
* Crop layout: 8 crops at pitch 0 (every 45 degrees of yaw), 6 each at
  pitch +45 and -45 (every 60 degrees), and one each at pitch +80 and -80.
  These 22 crops cover every ERP pixel at the default FOV.
* Each crop is rendered at the S2D native 1080x1080 from the native ERP:
  RGB bilinear (then resized like the panorama eval), z-depth from the
  nearest ERP range sample.
* REL+: depth and K go through the frozen canonical recipe (nearest resize to
  the model size, scaled K), then ``generate_rel_plus_v2_1`` with the crop's
  known rotation, i.e. ground-truth gravity and per-crop ReD/height
  normalisation, as in training.
* HHA: Depth2HHA on the crop with the recipe and channel order of a
  ``check_hha_cache.py`` report (same rules as the whole-panorama eval).
* Stitching: each ERP pixel averages the softmax of every crop that sees it.
"""

import argparse
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import cross_projection as cp
import eval_pano_transfer as ev


CROP_FOV_DEG = 62.47653165897473  # audited S2D training median, see ../evidence/
CROP_NATIVE_SIZE = 1080  # Stanford2D3D S2D native pinhole resolution
CROP_LAYOUT = tuple((float(yaw), 0.0) for yaw in range(0, 360, 45)) + tuple(
    (float(yaw), pitch) for pitch in (45.0, -45.0) for yaw in range(0, 360, 60)
) + ((0.0, 80.0), (0.0, -80.0))  # avoid the frozen generator's 180-degree alignment singularity
CROP_BATCH = 8
_WORKER_STATE = {}


def erp_coordinates(directions, height, width):
    """Continuous ERP (row, column) per unit direction, ``getPointCloud_ERP`` convention."""
    phi = np.arctan2(directions[..., 0], directions[..., 1])
    theta = np.arcsin(np.clip(directions[..., 2], -1.0, 1.0))
    return (0.5 - theta / np.pi) * height, (phi + np.pi) / (2 * np.pi) * width


def sample_erp_rgb(rgb, directions):
    """Bilinear ERP RGB along ``directions``, wrapping across the 180-degree seam."""
    height, width = rgb.shape[:2]
    rows, columns = erp_coordinates(directions, height, width)
    wrapped = np.concatenate([rgb, rgb[:, :1]], axis=1)  # column W is column 0
    return cv2.remap(
        wrapped, columns.astype(np.float32), rows.astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def stitch_grids(eval_size, layout=CROP_LAYOUT, fov=CROP_FOV_DEG):
    """Per crop, where each ERP pixel lands (grid_sample units) and whether it is inside."""
    k_json = cp.crop_intrinsics(2.0, fov)  # normalised: the image spans [0, 2]
    directions = cp.erp_directions(*eval_size)
    grids, inside = [], []
    for yaw, pitch in layout:
        camera = directions @ cp.crop_camera_to_world(yaw, pitch)
        depth = camera[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = k_json[0, 0] * camera[..., 0] / depth + k_json[0, 2] - 1.0
            y = k_json[1, 1] * camera[..., 1] / depth + k_json[1, 2] - 1.0
        seen = (depth > 0) & (np.abs(x) <= 1 + 1e-6) & (np.abs(y) <= 1 + 1e-6)
        grids.append(np.where(seen[..., None], np.stack([x, y], axis=-1), 0.0).astype(np.float32))
        inside.append(seen)
    return np.asarray(grids), np.asarray(inside)


def crop_hha(raw, k_json, hha_recipe, model_size):
    """Depth2HHA on one Stanford-encoded crop, resized as the verified cache recipe."""
    import hha
    from check_hha_cache import NATIVE_RESIZES, decode, resize_native_hha
    from vendor.depth2hha.getHHA import getHHA

    variant = hha_recipe["variant"]
    if variant == "resize_then_hha":
        k_json = k_json * np.array([[model_size / raw.shape[1]], [model_size / raw.shape[0]], [1.0]])
        raw = cv2.resize(raw, (model_size, model_size), interpolation=cv2.INTER_NEAREST)
    elif variant not in NATIVE_RESIZES:
        raise ValueError("unsupported HHA resolution recipe: " + str(variant))
    depth, valid = decode(raw)
    depth = np.where(valid, depth, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        encoded = getHHA(hha._helper_k(k_json), depth, depth)
    if variant != "resize_then_hha":
        encoded = resize_native_hha(encoded, (model_size, model_size), variant)
    return np.ascontiguousarray(encoded[:, :, hha_recipe["channel_order"]])


def prepare_panorama(
    sample, frozen, x_mode, hha_recipe, lookup, eval_size, model_size,
    native_size=CROP_NATIVE_SIZE, layout=CROP_LAYOUT, fov=CROP_FOV_DEG,
):
    """RGB and X crops (model size, uint8) plus the ERP label at ``eval_size``."""
    rgb_erp = frozen["open_image"](str(sample["rgb"]), frozen["rgb_flag"])
    if rgb_erp.ndim != 3 or rgb_erp.shape[2] != 3:
        raise ValueError("{} is not a 3-channel image".format(sample["rgb"]))
    raw_erp = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
    if raw_erp is None or raw_erp.dtype != np.uint16 or raw_erp.ndim != 2:
        raise ValueError("{} is not a uint16 depth map".format(sample["depth"]))
    depth_erp = cp.decode_erp_depth(raw_erp)

    k_json = cp.crop_intrinsics(native_size, fov)
    rgb_crops, x_crops = [], []
    for yaw, pitch in layout:
        camera_to_world = cp.crop_camera_to_world(yaw, pitch)
        rays_camera, rays_world = cp.crop_rays(native_size, k_json, camera_to_world)
        rgb = sample_erp_rgb(rgb_erp, rays_world)
        rgb_crops.append(cv2.resize(rgb, (model_size, model_size), interpolation=cv2.INTER_LINEAR))
        rows, columns = cp.erp_indices_for_directions(rays_world, *depth_erp.shape)
        raw = cp.render_crop_raw_depth(depth_erp, rows, columns, rays_camera)
        if x_mode == "rel_plus_v2_1":
            camera = frozen["CameraGeometry"].from_json_k(
                k_json, (native_size, native_size), camera_to_world.T,
                sample_id="{}@yaw{:g}_pitch{:g}".format(sample["sample_id"], yaw, pitch),
            )
            shape = (model_size, model_size)
            x_crops.append(frozen["generate_rel_plus_v2_1"](
                frozen["resize_raw_depth_nearest"](raw, shape),
                frozen["resize_camera_geometry"](camera, shape),
            ))
        elif x_mode == "hha_frozen_cache":
            if hha_recipe is None:
                raise ValueError("HHA input requires a verified cache recipe")
            x_crops.append(crop_hha(raw, k_json, hha_recipe, model_size))
        else:
            raise ValueError("unsupported crop x_mode: " + str(x_mode))

    height, width = eval_size
    semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
    if semantic is None:
        raise FileNotFoundError(sample["semantic"])
    semantic = cv2.resize(semantic, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.asarray(rgb_crops), np.asarray(x_crops), ev.decode_semantic(semantic, lookup)


def _prepare_in_worker(sample):
    return prepare_panorama(sample, **_WORKER_STATE)


def predict_panorama(network, rgb_crops, x_crops, grids, inside, normalize, device):
    """Stitched ERP prediction: argmax of the summed softmax of the crops seeing each pixel."""
    import torch
    import torch.nn.functional as F

    total = None
    for start in range(0, len(rgb_crops), CROP_BATCH):
        batch = slice(start, start + CROP_BATCH)
        rgb = torch.cat([normalize(image) for image in rgb_crops[batch]]).to(device)
        modal_x = torch.cat([normalize(image) for image in x_crops[batch]]).to(device)
        with torch.no_grad():
            logits = network(rgb, modal_x)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite logits")
        sampled = F.grid_sample(
            logits.softmax(dim=1), grids[batch], mode="bilinear",
            padding_mode="border", align_corners=False,
        )
        sampled = (sampled * inside[batch, None]).sum(dim=0)
        total = sampled if total is None else total + sampled
    return total.argmax(dim=0)


def import_frozen_crops(source_root):
    frozen = ev.import_frozen(source_root)
    from rel_plus.camera import CameraGeometry, resize_camera_geometry
    from rel_plus.depth import resize_raw_depth_nearest
    from rel_plus.generator import generate_rel_plus_v2_1

    frozen.update(
        CameraGeometry=CameraGeometry,
        resize_camera_geometry=resize_camera_geometry,
        resize_raw_depth_nearest=resize_raw_depth_nearest,
        generate_rel_plus_v2_1=generate_rel_plus_v2_1,
    )
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stanford-root", required=True, type=Path)
    parser.add_argument("--semantic-labels", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="resolved arm config JSON")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-epoch", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=ev.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--areas", nargs="+", default=["area_5a", "area_5b"])
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--crop-fov-deg", type=float, default=CROP_FOV_DEG,
        help="pinhole horizontal/vertical FOV; default is the training median; full ERP coverage is required",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="CPU processes building crop inputs (0: in the main process)",
    )
    parser.add_argument("--hha-cache-report", type=Path, default=None)
    parser.add_argument(
        "--accept-hha-near-match", action="store_true",
        help="also accept a NEAR_MATCH cache report (recorded in metrics.json)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive; omit it for full evaluation")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if args.workers < 0:
        parser.error("--workers must be zero or positive")
    if not np.isfinite(args.crop_fov_deg) or not 0 < args.crop_fov_deg < 180:
        parser.error("--crop-fov-deg must be finite and between 0 and 180 degrees")

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = import_frozen_crops(args.source_root)
    config = ev.load_config(args.config)
    if config.image_height != config.image_width:
        raise ValueError("crop evaluation needs a square model input size")
    hha_recipe = None
    if config.x_mode == "hha_frozen_cache":
        hha_recipe = ev.hha_cache_recipe(
            args.hha_cache_report, config.x_root_folder, args.accept_hha_near_match
        )
    lookup = ev.label_lookup(args.semantic_labels)
    samples = ev.list_panoramas(args.stanford_root, args.areas)[: args.limit]
    eval_size = (args.height, args.width)
    grids, inside = stitch_grids(eval_size, fov=args.crop_fov_deg)
    crops_per_pixel = inside.sum(axis=0)
    if crops_per_pixel.min() == 0:
        raise ValueError("crop layout leaves ERP pixels uncovered")

    _WORKER_STATE.update(
        frozen=frozen, x_mode=config.x_mode, hha_recipe=hha_recipe, lookup=lookup,
        eval_size=eval_size, model_size=int(config.image_height), fov=args.crop_fov_deg,
    )
    # Fork the workers before CUDA is initialised; they only run numpy/OpenCV.
    pool = multiprocessing.get_context("fork").Pool(args.workers) if args.workers else None
    prepared = pool.imap(_prepare_in_worker, samples) if pool else map(_prepare_in_worker, samples)

    import torch
    import torch.nn as nn

    device = torch.device(args.device)
    network = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    epoch = frozen["load_checkpoint_once"](
        network, args.checkpoint, expected_epoch=args.expected_epoch
    )
    network.to(device)
    network.eval()
    grids_device = torch.from_numpy(grids).to(device)
    inside_device = torch.from_numpy(inside.astype(np.float32)).to(device)

    def normalize(image):
        return frozen["normalized_bchw"](image, config.norm_mean, config.norm_std)

    started = time.time()
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    for index, (rgb_crops, x_crops, label) in enumerate(prepared, start=1):
        prediction = predict_panorama(
            network, rgb_crops, x_crops, grids_device, inside_device, normalize, device
        ).cpu().numpy().astype(np.uint8)
        hist, _, _ = frozen["hist_info"](config.num_classes, prediction, label)
        confusion += hist.astype(np.int64)
        if index % 20 == 0 or index == len(samples):
            print("{}/{} panoramas".format(index, len(samples)), flush=True)
    if pool:
        pool.close()
        pool.join()

    if confusion.sum() == 0:
        raise ValueError("no valid semantic pixels were evaluated")
    metrics = frozen["metrics_from_confusion"](confusion)
    report = {
        "status": "SMOKE" if args.limit is not None else "COMPLETED",
        "setting": "source-only pinhole-to-panorama transfer, crop-tiled",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "config": str(args.config),
        "x_mode": config.x_mode,
        "crop_x_input": (
            "frozen generate_rel_plus_v2_1 per crop (canonical nearest resize, crop rotation as gravity)"
            if hha_recipe is None
            else "Depth2HHA per crop, {}, channel order {}".format(
                hha_recipe["variant"], hha_recipe["channel_order"]
            )
        ),
        "hha_cache_report": None if hha_recipe is None else str(args.hha_cache_report),
        "hha_recipe": hha_recipe,
        "crop_layout_yaw_pitch_deg": [list(crop) for crop in CROP_LAYOUT],
        "crop_fov_deg": args.crop_fov_deg,
        "crop_native_size": CROP_NATIVE_SIZE,
        "crop_model_size": int(config.image_height),
        "crops_per_erp_pixel": [int(crops_per_pixel.min()), int(crops_per_pixel.max())],
        "areas": args.areas,
        "eval_size": list(eval_size),
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
