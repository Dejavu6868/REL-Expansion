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
  These 22 crops cover every ERP pixel at the default FOV. ``--layout level``
  keeps only the 8 pitch-0 crops (Depth2HHA's gravity estimate fails on the
  tilted ones) and scores only the pixels they see, which include the whole
  +-25 degree elevation band at the default FOV.
* Both evaluations also score each elevation band separately.
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
* ``--projection erp`` is the control for the level crops: the same 8 windows and the same
  stitching, but each tile is cut straight from the whole-panorama inputs (RGB and X built as
  ``eval_pano_transfer.py`` builds them) with no reprojection. The inputs are built at the ERP
  size on which a tile of the model size spans the crop FOV, so tiles and crops have the same
  window and pixel count. Only the projection, and the X normalisation scope that comes with
  the whole-panorama inputs, differ.
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
import relplus_variant as rv


CROP_FOV_DEG = 62.47653165897473  # audited S2D training median, see ../evidence/
CROP_NATIVE_SIZE = 1080  # Stanford2D3D S2D native pinhole resolution
CROP_LAYOUT = tuple((float(yaw), 0.0) for yaw in range(0, 360, 45)) + tuple(
    (float(yaw), pitch) for pitch in (45.0, -45.0) for yaw in range(0, 360, 60)
) + ((0.0, 80.0), (0.0, -80.0))  # avoid the frozen generator's 180-degree alignment singularity
# "level": the 8 pitch-0 crops only, where Depth2HHA's gravity estimate holds; ERP pixels no
# crop sees are left out of the score.
LAYOUTS = {"full": CROP_LAYOUT, "level": tuple(crop for crop in CROP_LAYOUT if crop[1] == 0.0)}
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


def erp_tile_source_size(model_size, fov=CROP_FOV_DEG):
    """ERP (height, width) on which ``model_size`` pixels span ``fov`` degrees; width a multiple of 4."""
    width = 4 * max(1, int(round(model_size * 360.0 / fov / 4)))
    return width // 2, width


def erp_tile_windows(layout, source_size, model_size):
    """Top-left (row, column) of each tile, centred on its level crop's optical axis."""
    height, width = source_size
    windows = []
    for yaw, pitch in layout:
        if pitch != 0.0:
            raise ValueError("ERP tiles are defined for the level layout only")
        _, column = erp_coordinates(cp.crop_camera_to_world(yaw, pitch)[:, 2], height, width)
        left = int(round(float(column) - (model_size - 1) / 2)) % width
        windows.append(((height - model_size) // 2, left))
    return windows


def erp_tile_grids(eval_size, windows, source_size, model_size):
    """Per tile, where each eval-grid ERP pixel lands (grid_sample units) and whether it is inside."""
    height, width = source_size
    rows, columns = erp_coordinates(cp.erp_directions(*eval_size), height, width)
    grids, inside = [], []
    for top, left in windows:
        x = np.mod(columns - left + 0.5, width) - 0.5  # wrap across the 180-degree seam
        y = rows - top
        seen = (x >= -0.5) & (x <= model_size - 0.5) & (y >= -0.5) & (y <= model_size - 0.5)
        grid = np.stack([(2 * x + 1) / model_size - 1, (2 * y + 1) / model_size - 1], axis=-1)
        grids.append(np.where(seen[..., None], grid, 0.0).astype(np.float32))
        inside.append(seen)
    return np.asarray(grids), np.asarray(inside)


def cut_erp_tiles(image, windows, model_size):
    """The ``model_size`` square of ``image`` at each window, wrapping across the seam."""
    return np.asarray([
        np.roll(image, -left, axis=1)[top:top + model_size, :model_size] for top, left in windows
    ])


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

    return np.asarray(rgb_crops), np.asarray(x_crops), eval_label(sample, lookup, eval_size)


def prepare_erp_tiles(sample, frozen, x_mode, hha_recipe, lookup, eval_size, model_size, windows, source_size):
    """RGB and X tiles cut from the whole-panorama inputs at ``source_size``, plus the eval label."""
    rgb, modal_x, _ = ev.load_sample(sample, source_size, lookup, frozen, x_mode, hha_recipe)
    return (
        cut_erp_tiles(rgb, windows, model_size), cut_erp_tiles(modal_x, windows, model_size),
        eval_label(sample, lookup, eval_size),
    )


def eval_label(sample, lookup, eval_size):
    height, width = eval_size
    semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
    if semantic is None:
        raise FileNotFoundError(sample["semantic"])
    semantic = cv2.resize(semantic, (width, height), interpolation=cv2.INTER_NEAREST)
    return ev.decode_semantic(semantic, lookup)


def _prepare_in_worker(sample):
    return _WORKER_STATE["prepare"](sample, **_WORKER_STATE["arguments"])


def predict_panorama(network, rgb_crops, x_crops, grids, inside, normalize, device, channels=None):
    """Stitched ERP prediction: argmax of the summed softmax of the crops seeing each pixel.

    With ``channels``, the softmax runs over those output channels only, so only they can win.
    """
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
        if channels is not None:
            logits = logits[:, channels]
        sampled = F.grid_sample(
            logits.softmax(dim=1), grids[batch], mode="bilinear",
            padding_mode="border", align_corners=False,
        )
        sampled = (sampled * inside[batch, None]).sum(dim=0)
        total = sampled if total is None else total + sampled
    prediction = total.argmax(dim=0)
    return prediction if channels is None else torch.as_tensor(channels, device=prediction.device)[prediction]


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
        help="pinhole horizontal/vertical FOV; default is the training median; the full layout must cover every ERP pixel",
    )
    parser.add_argument(
        "--layout", choices=sorted(LAYOUTS), default="full",
        help="full: 22 crops covering the sphere; level: pitch-0 crops only, unseen pixels unscored",
    )
    parser.add_argument(
        "--projection", choices=["pinhole", "erp"], default="pinhole",
        help="pinhole: rendered crops; erp: the level windows cut straight from the whole-panorama inputs (level layout only)",
    )
    parser.add_argument(
        "--eight-class-predictions", action="store_true",
        help="softmax and argmax over the 8 SPan8 classes only ({}); the network is unchanged".format(
            " ".join(ev.EIGHT_CLASS_NAMES)
        ),
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
    parser.add_argument(
        "--relplus-variant", choices=sorted(rv.VARIANT_ALPHA), default="v2_1",
        help="REL+ EGVIA encoding; must match the checkpoint's training cache (relplus_variant.py)",
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
    if args.projection == "erp" and args.layout != "level":
        parser.error("--projection erp needs --layout level")

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = import_frozen_crops(args.source_root)
    config = ev.load_config(args.config)
    if config.image_height != config.image_width:
        raise ValueError("crop evaluation needs a square model input size")
    relplus_variant = rv.select_for_eval(args.relplus_variant, config, frozen)
    hha_recipe = None
    if config.x_mode == "hha_frozen_cache":
        hha_recipe = ev.hha_cache_recipe(
            args.hha_cache_report, config.x_root_folder, args.accept_hha_near_match
        )
    lookup = ev.label_lookup(args.semantic_labels)
    samples = ev.list_panoramas(args.stanford_root, args.areas)[: args.limit]
    eval_size = (args.height, args.width)
    model_size = int(config.image_height)
    layout = LAYOUTS[args.layout]
    arguments = dict(
        frozen=frozen, x_mode=config.x_mode, hha_recipe=hha_recipe, lookup=lookup,
        eval_size=eval_size, model_size=model_size,
    )
    if args.projection == "erp":
        source_size = erp_tile_source_size(model_size, args.crop_fov_deg)
        windows = erp_tile_windows(layout, source_size, model_size)
        grids, inside = erp_tile_grids(eval_size, windows, source_size, model_size)
        arguments.update(windows=windows, source_size=source_size)
        _WORKER_STATE.update(prepare=prepare_erp_tiles, arguments=arguments)
    else:
        grids, inside = stitch_grids(eval_size, layout=layout, fov=args.crop_fov_deg)
        arguments.update(fov=args.crop_fov_deg, layout=layout)
        _WORKER_STATE.update(prepare=prepare_panorama, arguments=arguments)
    crops_per_pixel = inside.sum(axis=0)
    seen = crops_per_pixel > 0
    if args.layout == "full" and not seen.all():
        raise ValueError("crop layout leaves ERP pixels uncovered")
    channels = ev.eight_class_channels(config.class_names) if args.eight_class_predictions else None
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
    band_confusion = np.zeros((len(ev.ELEVATION_BAND_EDGES_DEG) - 1,) + confusion.shape, dtype=np.int64)
    for index, (rgb_crops, x_crops, label) in enumerate(prepared, start=1):
        prediction = predict_panorama(
            network, rgb_crops, x_crops, grids_device, inside_device, normalize, device, channels
        ).cpu().numpy().astype(np.uint8)
        label = np.where(seen, label, 255).astype(np.uint8)
        hist, _, _ = frozen["hist_info"](config.num_classes, prediction, label)
        confusion += hist.astype(np.int64)
        band_confusion += ev.elevation_band_confusions(
            config.num_classes, prediction, label, frozen["hist_info"]
        )
        if index % 20 == 0 or index == len(samples):
            print("{}/{} panoramas".format(index, len(samples)), flush=True)
    if pool:
        pool.close()
        pool.join()

    if confusion.sum() == 0:
        raise ValueError("no valid semantic pixels were evaluated")
    metrics = frozen["metrics_from_confusion"](confusion)
    if args.projection == "erp":
        crop_x_input = "whole-panorama X at {}x{} as in eval_pano_transfer.py ({}), cut into tiles".format(
            source_size[0], source_size[1],
            rv.describe(relplus_variant) if hha_recipe is None else "ERP HHA, {}, channel order {}".format(
                hha_recipe["variant"], hha_recipe["channel_order"]
            ),
        )
    elif hha_recipe is None:
        crop_x_input = "frozen generate_rel_plus_v2_1 ({}) per crop (canonical nearest resize, crop rotation as gravity)".format(
            rv.describe(relplus_variant)
        )
    else:
        crop_x_input = "Depth2HHA per crop, {}, channel order {}".format(
            hha_recipe["variant"], hha_recipe["channel_order"]
        )
    report = {
        "status": "SMOKE" if args.limit is not None else "COMPLETED",
        "setting": "source-only pinhole-to-panorama transfer, {}".format(
            "ERP-tiled" if args.projection == "erp" else "crop-tiled"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "config": str(args.config),
        "x_mode": config.x_mode,
        "projection": args.projection,
        "erp_tile_source_size": list(source_size) if args.projection == "erp" else None,
        "erp_tile_fov_deg": model_size * 360.0 / source_size[1] if args.projection == "erp" else None,
        "prediction_classes": list(ev.EIGHT_CLASS_NAMES if channels else config.class_names),
        "crop_x_input": crop_x_input,
        "relplus_variant": relplus_variant,
        "hha_cache_report": None if hha_recipe is None else str(args.hha_cache_report),
        "hha_recipe": hha_recipe,
        "crop_layout": args.layout,
        "crop_layout_yaw_pitch_deg": [list(crop) for crop in layout],
        "scored_erp_pixel_fraction": float(seen.mean()),
        "crop_fov_deg": args.crop_fov_deg,
        "crop_native_size": CROP_NATIVE_SIZE,
        "crop_model_size": model_size,
        "crops_per_erp_pixel": [int(crops_per_pixel.min()), int(crops_per_pixel.max())],
        "areas": args.areas,
        "eval_size": list(eval_size),
        "sample_count": len(samples),
        "seconds": time.time() - started,
        "metrics": metrics,
        "elevation_bands": ev.write_elevation_bands(
            args.output, band_confusion, frozen["metrics_from_confusion"], config.class_names
        ),
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
