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
* ``--infer-scale`` runs that pass on RGB and X resized by the factor (wrap
  padding scaled with it) and resizes the logits back to the evaluation size
  before the argmax, so labels and scored pixels stay those of the default run.
* ``--attention-chunk-mb`` runs every MiT attention with its queries in chunks,
  so a large input (such as the native 2048x4096 panorama, ``--infer-scale 2``)
  never holds a whole query-by-key score matrix. Softmax is taken per query, so
  the result is the frozen forward's up to float rounding.
* ``--polar-strips MARGIN`` replaces the whole pass by three strips: the bands
  between +60 and -60 degrees at the evaluation size, and the two polar bands
  at half size from the panorama resized by 0.5 (as ``--infer-scale 0.5``).
  Each strip also gets MARGIN rows of context past +-60 degrees (at the
  evaluation size, rounded up to a multiple of 32 rows at the strip's scale),
  which are dropped before scoring.
* ``--erp-tiles M N`` runs the network on an M x N grid of overlapping
  windows cut from the evaluation-size panorama (RGB and the X encoded on the
  whole panorama), each ``--erp-tile-size`` pixels. Rows span the height
  exactly; columns start floor(i W / N) apart and wrap across the 180-degree seam. Each
  pixel takes the mean softmax of the tiles covering it.
* ``--polar-caps SIZE`` replaces the bands beyond +-60 degrees by two
  SIZE x SIZE tiles, 67.5 degrees wide, centred on the poles of the sphere
  turned 90 degrees about the x axis, where the poles lie on the equator and
  ERP barely stretches. RGB (bilinear) and the X encoded on the whole
  panorama (nearest) are resampled into each tile; the tile softmax is read
  back onto the polar rows. It works with the whole pass or ``--erp-tiles``.

* The HHA arm's X input is ``hha.erp_hha`` (Depth2HHA definitions on ERP
  geometry, range instead of z-depth for disparity). It runs only with a
  ``check_hha_cache.py`` report whose status is MATCH (or NEAR_MATCH with
  ``--accept-hha-near-match``), and uses that report's
  channel order and resolution recipe. This checks the sampled cache only;
  it does not remove the z-depth/range domain difference.

The raw-depth (RGBD) arm is refused: a Stanford pinhole z-depth byte has no
ERP equivalent.

An RGB-only checkpoint (``rgb_only.py``, config ``architecture`` set by
``train_rgb_only.py``) gets zeros as X, which its network ignores; it needs no
HHA cache report.
"""

import argparse
import csv
import functools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import relplus_variant as rv
import rgb_only as ro
from cross_projection import decode_erp_depth


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "0927调参结果" / "selected_recipe" / "code" / "training" / "source"
)
EXPECTED_CLASSES = [
    "<UNK>", "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
]
SUPPORTED_X_MODES = ("rel_plus_v2_1", "hha_frozen_cache")
# Horizon band +-25: fully seen by the level crops of eval_pano_crops.py at the training FOV.
ELEVATION_BAND_EDGES_DEG = (90, 60, 25, 0, -25, -60, -90)
# SPan8 benchmark classes (Trans4PASS+ stanford2d3d_pan8.py maps the other 5 to ignore).
EIGHT_CLASS_NAMES = ("ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window")
# --polar-strips: the bands beyond +-60 deg run at this scale; strip heights are multiples of the MiT stride.
POLAR_STRIP_ELEVATION_DEG = 60
POLAR_STRIP_SCALE = 0.5
STRIP_ROW_MULTIPLE = 32
# --erp-tiles: per-tile class probabilities are averaged over this many tiles per forward pass.
TILE_BATCH = 8
# --polar-caps: tile width; it covers the 30-degree caps with 3.75 degrees of context. After the
# turn the north pole sits at longitude -90 and the south pole at +90 (degrees, latitude 0).
POLAR_CAP_FOV_DEG = 67.5
POLAR_CAP_CENTRE_LONGITUDES_DEG = (-90.0, 90.0)


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
    if not areas or len(set(areas)) != len(areas):
        raise ValueError("requested areas must be nonempty and unique")
    samples = []
    for area in areas:
        rgb_folder = stanford_root / area / "pano" / "rgb"
        rgb_paths = sorted(rgb_folder.glob("*_rgb.png"))
        if not rgb_paths:
            raise FileNotFoundError("no panoramas found in requested area: {}".format(rgb_folder))
        for rgb_path in rgb_paths:
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
    return samples


def elevation_band_confusions(num_classes, prediction, label, hist_info):
    """One confusion matrix per elevation band (top edge included); row v is at 90 - 180 v / H degrees."""
    elevation = 90.0 - np.arange(label.shape[0]) * 180.0 / label.shape[0]
    edges = ELEVATION_BAND_EDGES_DEG
    return np.stack([
        hist_info(num_classes, prediction[rows], label[rows])[0]
        for rows in (
            (elevation <= top) & (elevation > bottom) for top, bottom in zip(edges[:-1], edges[1:])
        )
    ]).astype(np.int64)


def write_elevation_bands(output, band_confusion, metrics_from_confusion, class_names):
    """Save per-band confusions and metrics; return the per-band summary for metrics.json."""
    np.save(output / "elevation_band_confusion.npy", band_confusion)
    edges = ELEVATION_BAND_EDGES_DEG
    summary = []
    for top, bottom, confusion in zip(edges[:-1], edges[1:], band_confusion):
        metrics = metrics_from_confusion(confusion) if confusion.sum() else None
        summary.append({
            "elevation_deg": [top, bottom],
            "valid_pixels": int(confusion.sum()),
            "mIoU_percent": metrics and metrics["mIoU_percent"],
            "pixel_accuracy_percent": metrics and metrics["pixel_accuracy_percent"],
            "per_class_iou_percent": metrics and metrics["per_class_iou_percent"],
        })
    with (output / "elevation_band_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["elevation_top_deg", "elevation_bottom_deg", "valid_pixels", "mIoU_percent",
             "pixel_accuracy_percent"] + list(class_names)
        )
        for band in summary:
            writer.writerow(
                band["elevation_deg"] + [band["valid_pixels"], band["mIoU_percent"],
                band["pixel_accuracy_percent"]]
                + (band["per_class_iou_percent"] or [None] * len(class_names))
            )
    return summary


def load_sample(sample, size, lookup, frozen, x_mode, hha_recipe=None, label_size=None):
    """RGB and X at ``size``; labels at ``label_size`` (default ``size``)."""
    height, width = size
    rgb = frozen["open_image"](str(sample["rgb"]), frozen["rgb_flag"])
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("{} is not a 3-channel image".format(sample["rgb"]))
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    raw_depth = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
    if raw_depth is None or raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise ValueError("{} is not a uint16 depth map".format(sample["depth"]))
    if x_mode is None:  # RGB-only network: X is ignored
        modal_x = np.zeros_like(rgb)
    elif x_mode == "rel_plus_v2_1":
        resized = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        modal_x = frozen["getREL"](decode_erp_depth(resized))
    elif x_mode == "hha_frozen_cache":
        import hha

        if hha_recipe is None:
            raise ValueError("HHA input requires a verified cache recipe")
        from check_hha_cache import NATIVE_RESIZES, resize_native_hha

        variant = hha_recipe["variant"]
        if variant == "resize_then_hha":
            raw_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        elif variant not in NATIVE_RESIZES:
            raise ValueError("unsupported HHA resolution recipe: " + str(variant))
        modal_x, _ = hha.erp_hha(decode_erp_depth(raw_depth), frozen)
        if variant in NATIVE_RESIZES:
            modal_x = resize_native_hha(modal_x, (height, width), variant)
        modal_x = np.ascontiguousarray(modal_x[:, :, hha_recipe["channel_order"]])
    else:
        raise ValueError("unsupported panorama x_mode: " + str(x_mode))

    semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
    if semantic is None:
        raise FileNotFoundError(sample["semantic"])
    label_height, label_width = size if label_size is None else label_size
    semantic = cv2.resize(semantic, (label_width, label_height), interpolation=cv2.INTER_NEAREST)
    return rgb, modal_x, decode_semantic(semantic, lookup)


def predict_with_wrap(network, rgb, modal_x, wrap_pad, device, output_size=None):
    """Whole-image logits with circular padding along the panorama width.

    With ``output_size`` the cropped logits are resized (bilinear) to it.
    """
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
    if output_size is not None and tuple(logits.shape[2:]) != tuple(output_size):
        logits = F.interpolate(logits, size=tuple(output_size), mode="bilinear", align_corners=False)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("non-finite logits")
    return logits


def polar_band_rows(height):
    """End of the top band and start of the bottom band (beyond +-60 deg), as ``elevation_band_confusions``."""
    elevation = 90.0 - np.arange(height) * 180.0 / height
    return (
        int((elevation > POLAR_STRIP_ELEVATION_DEG).sum()),
        int(np.argmax(elevation <= -POLAR_STRIP_ELEVATION_DEG)),
    )


def polar_strip_rows(height, margin, scale=POLAR_STRIP_SCALE):
    """Row windows of the three-strip pass; the core split is the band rule of ``elevation_band_confusions``.

    ``top_end`` and ``bottom_start`` (evaluation rows) bound the middle core, ``middle`` is the
    middle strip in evaluation rows, ``top`` and ``bottom`` the polar strips in scaled rows.
    """
    top_end, bottom_start = polar_band_rows(height)
    scaled_height = height * scale
    if scaled_height != int(scaled_height):
        raise ValueError("--height times the polar scale must be a whole number of rows")
    scaled_height = int(scaled_height)

    def rounded(rows, limit):
        return min(limit, STRIP_ROW_MULTIPLE * int(np.ceil(rows / STRIP_ROW_MULTIPLE)))

    middle_need = bottom_start - top_end + 2 * margin
    middle_rows = rounded(middle_need, height)
    middle_start = top_end - margin - (middle_rows - middle_need) // 2
    middle_start = max(0, min(middle_start, height - middle_rows))
    top_rows = rounded(np.ceil((top_end + margin) * scale), scaled_height)
    bottom_rows = rounded(np.ceil((height - bottom_start + margin) * scale), scaled_height)
    for rows in (top_rows, bottom_rows):
        if rows / scale != int(rows / scale):
            raise ValueError("a polar strip does not map to whole evaluation rows")
    return {
        "top_end": top_end,
        "bottom_start": bottom_start,
        "middle": [middle_start, middle_start + middle_rows],
        "top": [0, top_rows],
        "bottom": [scaled_height - bottom_rows, scaled_height],
    }


def predict_polar_strips(network, full, polar, rows, wrap_pad, polar_wrap_pad, device):
    """Evaluation-size logits from the middle strip of ``full`` and the polar strips of ``polar``.

    ``full`` and ``polar`` are (rgb, modal_x) tensors at the evaluation size and at the polar
    scale. Each strip keeps only its own band's rows; the rest of it is context.
    """
    import torch

    height, width = full[0].shape[2:]
    scale = polar[0].shape[2] / height
    middle_start, middle_end = rows["middle"]
    middle = predict_with_wrap(
        network, full[0][:, :, middle_start:middle_end], full[1][:, :, middle_start:middle_end],
        wrap_pad, device,
    )
    top, bottom = (
        predict_with_wrap(
            network, polar[0][:, :, start:end], polar[1][:, :, start:end], polar_wrap_pad, device,
            output_size=(int(round((end - start) / scale)), width),
        )
        for start, end in (rows["top"], rows["bottom"])
    )
    bottom_offset = height - bottom.shape[2]
    return torch.cat([
        top[:, :, :rows["top_end"]],
        middle[:, :, rows["top_end"] - middle_start:rows["bottom_start"] - middle_start],
        bottom[:, :, rows["bottom_start"] - bottom_offset:],
    ], dim=2)


def erp_tile_windows(height, width, rows, columns, tile_height, tile_width):
    """(top, left) of each tile: rows span the height exactly, column i starts at ``floor(i * width / columns)``."""
    if rows < 1 or columns < 1:
        raise ValueError("the tile grid needs at least one row and one column")
    if rows == 1:
        if tile_height != height:
            raise ValueError("one tile row must be as tall as the panorama")
        row_stride = 0
    else:
        row_stride, remainder = divmod(height - tile_height, rows - 1)
        if remainder or not 0 < row_stride <= tile_height:
            raise ValueError("tile rows must cover the height with a whole, positive stride")
    if not -(-width // columns) <= tile_width <= width:
        raise ValueError("tile columns must cover the width")
    return [
        (row * row_stride, column * width // columns) for row in range(rows) for column in range(columns)
    ]


def class_probabilities(logits, channels=None):
    """Softmax over the classes; with channels, over those only (the rest get 0), as an 8-output model would."""
    import torch

    if channels is None:
        return logits.softmax(dim=1)
    masked = torch.full_like(logits, float("-inf"))
    masked[:, channels] = logits[:, channels]
    return masked.softmax(dim=1)


def predict_erp_tiles(network, rgb, modal_x, windows, tile_size, device, channels=None):
    """Mean class probabilities over the tiles covering each pixel; columns wrap at the seam."""
    import torch

    height, width = rgb.shape[2:]
    tile_height, tile_width = tile_size
    total, count = None, torch.zeros(1, 1, height, width, device=device)
    for start in range(0, len(windows), TILE_BATCH):
        batch = windows[start:start + TILE_BATCH]
        columns = [torch.arange(left, left + tile_width) % width for _, left in batch]
        tiles = [
            torch.cat([image[:, :, top:top + tile_height, cols] for (top, _), cols in zip(batch, columns)])
            for image in (rgb, modal_x)
        ]
        probabilities = class_probabilities(predict_with_wrap(network, *tiles, 0, device), channels)
        if total is None:
            total = torch.zeros((1, probabilities.shape[1], height, width), device=device)
        for tile, (top, _), cols in zip(probabilities, batch, columns):
            cols = cols.to(device)
            total[0, :, top:top + tile_height].index_add_(2, cols, tile)
            count[0, :, top:top + tile_height].index_add_(2, cols, torch.ones_like(tile[:1]))
    if not bool((count > 0).all()):
        raise ValueError("the tile grid leaves panorama pixels uncovered")
    return total / count


def unit_directions(longitude, latitude):
    """Unit vectors with z up and longitude 0 along x (angles in radians)."""
    return np.stack([
        np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)
    ], axis=-1)


def longitude_latitude(directions):
    return np.arctan2(directions[..., 1], directions[..., 0]), np.arcsin(np.clip(directions[..., 2], -1, 1))


def to_cap_frame(directions):
    """Turn the sphere 90 degrees about x, (x, y, z) -> (x, -z, y), so the poles land on the equator."""
    x, y, z = np.moveaxis(directions, -1, 0)
    return np.stack([x, -z, y], axis=-1)


def from_cap_frame(directions):
    x, y, z = np.moveaxis(directions, -1, 0)
    return np.stack([x, z, -y], axis=-1)


def polar_cap_grids(height, width, size, fov_deg=POLAR_CAP_FOV_DEG):
    """``grid_sample`` coordinates between the ERP and the two cap tiles (north, south).

    ``tiles`` (2, size, size, 2): where each tile pixel reads the ERP padded by one column on each
    side, so bilinear sampling wraps at the seam. ``north`` and ``south``: where each pixel of the
    polar rows (``polar_band_rows``) reads its tile. ERP row 0 is the zenith, column 0 starts at -pi.
    """
    top_end, bottom_start = polar_band_rows(height)
    half = np.radians(fov_deg) / 2
    steps = (np.arange(size) + 0.5) / size * 2 * half - half
    centres = np.radians(POLAR_CAP_CENTRE_LONGITUDES_DEG)
    tiles = []
    for centre in centres:
        cap_longitude, cap_latitude = np.meshgrid(centre + steps, -steps)
        longitude, latitude = longitude_latitude(from_cap_frame(unit_directions(cap_longitude, cap_latitude)))
        tiles.append(np.stack([longitude / np.pi * width / (width + 2), -latitude / (np.pi / 2)], axis=-1))
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    back = []
    for rows, centre in zip((slice(0, top_end), slice(bottom_start, height)), centres):
        cap_longitude, cap_latitude = longitude_latitude(
            to_cap_frame(unit_directions(longitude[rows], latitude[rows]))
        )
        back.append(np.stack([(cap_longitude - centre) / half, -cap_latitude / half], axis=-1))
    return {
        "top_end": top_end,
        "bottom_start": bottom_start,
        "tiles": np.stack(tiles).astype(np.float32),
        "north": back[0].astype(np.float32),
        "south": back[1].astype(np.float32),
    }


def sample_cap_tiles(rgb, modal_x, caps, device):
    """The two cap tiles: RGB bilinear, X nearest so depth edges are not blended."""
    import torch
    import torch.nn.functional as F

    grid = torch.as_tensor(caps["tiles"], device=device)
    return [
        F.grid_sample(
            F.pad(image.to(device), (1, 1, 0, 0), mode="circular").expand(2, -1, -1, -1), grid,
            mode=mode, padding_mode="border", align_corners=False,
        )
        for image, mode in ((rgb, "bilinear"), (modal_x, "nearest"))
    ]


def predict_polar_caps(network, rgb, modal_x, caps, device, channels=None):
    """Class probabilities of the north and south polar rows, read back from the cap tiles."""
    import torch
    import torch.nn.functional as F

    probabilities = class_probabilities(
        predict_with_wrap(network, *sample_cap_tiles(rgb, modal_x, caps, device), 0, device), channels
    )
    return [
        F.grid_sample(
            probabilities[index:index + 1], torch.as_tensor(caps[name], device=device)[None],
            mode="bilinear", align_corners=False,
        )
        for index, name in enumerate(("north", "south"))
    ]


def _chunked_attention_forward(module, budget_bytes, x, H, W):
    """The frozen MiT ``Attention.forward`` with the scores computed a chunk of queries at a time."""
    import torch

    B, N, C = x.shape
    heads = module.num_heads
    q = module.q(x).reshape(B, N, heads, C // heads).permute(0, 2, 1, 3)
    if module.sr_ratio > 1:
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = module.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        x_ = module.norm(x_)
        kv = module.kv(x_).reshape(B, -1, 2, heads, C // heads).permute(2, 0, 3, 1, 4)
    else:
        kv = module.kv(x).reshape(B, -1, 2, heads, C // heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    step = max(1, budget_bytes // (B * heads * k.shape[2] * q.element_size()))
    out = torch.empty_like(q)
    for start in range(0, N, step):
        attn = (q[:, :, start:start + step] @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attn = module.attn_drop(attn)
        out[:, :, start:start + step] = attn @ v
    x = out.transpose(1, 2).reshape(B, N, C)
    x = module.proj(x)
    x = module.proj_drop(x)
    return x


def chunk_attention(network, budget_mb):
    """Chunk every MiT attention so one score chunk holds at most ``budget_mb`` MiB."""
    from models.encoders.dual_segformer import Attention

    modules = [module for module in network.modules() if isinstance(module, Attention)]
    if not modules:
        raise ValueError("the network has no MiT attention to chunk")
    budget_bytes = int(budget_mb * 2 ** 20)
    for module in modules:
        module.forward = functools.partial(_chunked_attention_forward, module, budget_bytes)
    return len(modules)


def eight_class_channels(class_names):
    """Output channels of the SPan8 classes, looked up by name in the config's class order."""
    names = list(class_names)
    missing = [name for name in EIGHT_CLASS_NAMES if name not in names]
    if missing:
        raise ValueError("config class_names lack {}".format(", ".join(missing)))
    return [names.index(name) for name in EIGHT_CLASS_NAMES]


def predict_classes(logits, channels=None):
    """Per-pixel argmax; with channels, only those output channels can win."""
    import torch

    if channels is None:
        return logits.argmax(dim=1)
    channels = torch.as_tensor(channels, device=logits.device)
    return channels[logits[:, channels].argmax(dim=1)]


def import_frozen(source_root):
    source_root = Path(source_root).resolve()
    sys.dont_write_bytecode = True  # keep the frozen archive's file set unchanged
    sys.path.insert(0, str(source_root))
    from dataloader.RGBXDataset import RGBXDataset
    from engine.relplus_evaluator import _normalized_bchw, metrics_from_confusion
    from models.builder import EncoderDecoder
    from third_party.rel_original.rel import getREL
    from third_party.rel_original.rgbd_util import (
        computeNormalsSquareSupport_ERP,
        getPointCloud_ERP,
    )
    from tools.eval_rel_plus_v2_3_full import load_checkpoint_once
    from utils.metric import hist_info

    return {
        "open_image": RGBXDataset._open_image,
        "rgb_flag": cv2.COLOR_BGR2RGB,  # the frozen loader's imread flag
        "normalized_bchw": _normalized_bchw,
        "metrics_from_confusion": metrics_from_confusion,
        "EncoderDecoder": EncoderDecoder,
        "getREL": getREL,
        "getPointCloud_ERP": getPointCloud_ERP,
        "computeNormalsSquareSupport_ERP": computeNormalsSquareSupport_ERP,
        "load_checkpoint_once": load_checkpoint_once,
        "hist_info": hist_info,
    }


def load_config(path):
    from easydict import EasyDict

    config = EasyDict(json.loads(Path(path).read_text(encoding="utf-8")))
    if config.x_mode not in SUPPORTED_X_MODES:
        raise ValueError(
            "x_mode {} has no panorama input; supported: {}".format(
                config.x_mode, ", ".join(SUPPORTED_X_MODES)
            )
        )
    config.norm_mean = np.asarray(config.norm_mean, dtype=np.float64)
    config.norm_std = np.asarray(config.norm_std, dtype=np.float64)
    return config


def hha_cache_recipe(report_path, expected_hha_root=None, accept_near_match=False):
    """Validate the sampled cache evidence and preserve both recipe choices."""
    from check_hha_cache import CHANNEL_ORDERS, MATCH_RULE, summarize_results

    if report_path is None:
        raise ValueError("the HHA arm needs --hha-cache-report (tools/check_hha_cache.py)")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    accepted = ("MATCH", "NEAR_MATCH") if accept_near_match else ("MATCH",)
    if report.get("status") not in accepted:
        raise ValueError(
            "HHA cache report is {}: not accepted as the same modality as the "
            "cached HHA{}".format(
                report.get("status"),
                "" if accept_near_match else " (NEAR_MATCH needs --accept-hha-near-match)",
            )
        )
    if report.get("match_rule") != MATCH_RULE:
        raise ValueError("old HHA cache report: rerun tools/check_hha_cache.py")
    try:
        checked = summarize_results(report["samples"])
        if checked["status"] != report["status"] or checked["best"] != report["best"]:
            raise ValueError("inconsistent HHA cache report")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid HHA cache evidence; rerun tools/check_hha_cache.py") from error
    cache_root = report.get("hha_root")
    if expected_hha_root is not None and (
        not cache_root or Path(cache_root).resolve() != Path(expected_hha_root).resolve()
    ):
        raise ValueError("HHA report cache root does not match the checkpoint config")
    return {
        "cache_status": checked["status"],
        "variant": checked["best"]["variant"],
        "channel_order": CHANNEL_ORDERS[checked["best"]["channel_order"]],
        "sample_count": len(report["samples"]),
        "hha_root": cache_root,
    }


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
    parser.add_argument(
        "--infer-scale", type=float, default=1.0,
        help="run the network on the panorama resized by this factor; logits are resized "
        "back to --height x --width, where labels stay",
    )
    parser.add_argument(
        "--attention-chunk-mb", type=int, default=0,
        help="compute MiT attention scores in query chunks of at most this many MiB "
        "(same result; for inputs whose score matrix does not fit); 0 keeps the frozen forward",
    )
    parser.add_argument(
        "--polar-strips", type=int, default=None, metavar="MARGIN",
        help="three passes instead of one: +60..-60 deg at the evaluation size, the two polar "
        "bands at x{} size, each with MARGIN context rows past +-60 deg".format(POLAR_STRIP_SCALE),
    )
    parser.add_argument(
        "--erp-tiles", type=int, nargs=2, default=None, metavar=("M", "N"),
        help="M x N overlapping ERP tiles instead of one pass; mean softmax where tiles overlap",
    )
    parser.add_argument(
        "--erp-tile-size", type=int, nargs=2, default=(384, 384), metavar=("H", "W"),
        help="tile size in evaluation pixels (default 384 x 384, 67.5 degrees at 1024 x 2048)",
    )
    parser.add_argument(
        "--polar-caps", type=int, default=None, metavar="SIZE",
        help="replace the bands beyond +-60 deg by two SIZE x SIZE tiles, {} degrees wide, centred on "
        "the poles of the sphere turned 90 deg; 384 at 1024 x 2048 and 768 at 2048 x 4096 keep the "
        "panorama's pixel density".format(POLAR_CAP_FOV_DEG),
    )
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hha-cache-report", type=Path, default=None)
    parser.add_argument(
        "--accept-hha-near-match", action="store_true",
        help="also accept a NEAR_MATCH cache report (recorded in metrics.json)",
    )
    parser.add_argument(
        "--relplus-variant", choices=sorted(rv.VARIANT_ALPHA), default="v2_1",
        help="REL+ EGVIA encoding; must match the checkpoint's training cache (relplus_variant.py)",
    )
    parser.add_argument(
        "--eight-class-predictions", action="store_true",
        help="argmax over the 8 SPan8 classes only ({}); the network is unchanged".format(
            " ".join(EIGHT_CLASS_NAMES)
        ),
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive; omit it for full evaluation")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if not 0 <= args.wrap_pad <= args.width:
        parser.error("--wrap-pad must be between zero and --width")
    if not np.isfinite(args.infer_scale) or args.infer_scale <= 0:
        parser.error("--infer-scale must be finite and positive")
    if args.attention_chunk_mb < 0:
        parser.error("--attention-chunk-mb must be zero or positive")
    infer_size = (int(round(args.height * args.infer_scale)), int(round(args.width * args.infer_scale)))
    infer_wrap_pad = int(round(args.wrap_pad * args.infer_scale))
    strip_rows = None
    if args.polar_strips is not None:
        if args.polar_strips < 0:
            parser.error("--polar-strips must be zero or positive")
        if args.infer_scale != 1.0:
            parser.error("--polar-strips runs the middle strip at the evaluation size; keep --infer-scale 1")
        try:
            strip_rows = polar_strip_rows(args.height, args.polar_strips)
        except ValueError as error:
            parser.error(str(error))
        polar_size = (int(args.height * POLAR_STRIP_SCALE), int(round(args.width * POLAR_STRIP_SCALE)))
        polar_wrap_pad = int(round(args.wrap_pad * POLAR_STRIP_SCALE))
    tile_windows = None
    if args.erp_tiles is not None:
        if args.polar_strips is not None or args.infer_scale != 1.0:
            parser.error("--erp-tiles cuts the evaluation-size panorama; drop --polar-strips and --infer-scale")
        try:
            tile_windows = erp_tile_windows(args.height, args.width, *args.erp_tiles, *args.erp_tile_size)
        except ValueError as error:
            parser.error(str(error))
    caps = None
    if args.polar_caps is not None:
        if args.polar_caps <= 0 or args.polar_caps % STRIP_ROW_MULTIPLE:
            parser.error("--polar-caps must be a positive multiple of {}".format(STRIP_ROW_MULTIPLE))
        if args.polar_strips is not None or args.infer_scale != 1.0:
            parser.error("--polar-caps samples the evaluation-size panorama; drop --polar-strips and --infer-scale")
        caps = polar_cap_grids(args.height, args.width, args.polar_caps)

    import torch
    import torch.nn as nn

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = import_frozen(args.source_root)
    config = load_config(args.config)
    channels = eight_class_channels(config.class_names) if args.eight_class_predictions else None
    relplus_variant = rv.select_for_eval(args.relplus_variant, config, frozen)
    rgb_only = ro.is_rgb_only(config)
    hha_recipe = None
    if config.x_mode == "hha_frozen_cache" and not rgb_only:
        hha_recipe = hha_cache_recipe(
            args.hha_cache_report, config.x_root_folder, args.accept_hha_near_match
        )
    lookup = label_lookup(args.semantic_labels)
    samples = list_panoramas(args.stanford_root, args.areas)[: args.limit]

    device = torch.device(args.device)
    network_class = ro.model_class() if rgb_only else frozen["EncoderDecoder"]
    network = network_class(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    epoch = frozen["load_checkpoint_once"](
        network, args.checkpoint, expected_epoch=args.expected_epoch
    )
    chunked_attention_modules = 0
    if args.attention_chunk_mb:
        chunked_attention_modules = chunk_attention(network, args.attention_chunk_mb)
    network.to(device)
    network.eval()

    started = time.time()
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    band_confusion = np.zeros((len(ELEVATION_BAND_EDGES_DEG) - 1,) + confusion.shape, dtype=np.int64)
    for index, sample in enumerate(samples, start=1):
        rgb, modal_x, label = load_sample(
            sample, infer_size, lookup, frozen, None if rgb_only else config.x_mode, hha_recipe,
            label_size=(args.height, args.width),
        )
        inputs = [
            frozen["normalized_bchw"](image, config.norm_mean, config.norm_std) for image in (rgb, modal_x)
        ]
        if tile_windows is not None:  # mean probabilities; only their argmax is used below
            logits = predict_erp_tiles(network, *inputs, tile_windows, args.erp_tile_size, device, channels)
        elif strip_rows is None:
            logits = predict_with_wrap(
                network, *inputs, infer_wrap_pad, device, output_size=(args.height, args.width),
            )
        else:
            polar_rgb, polar_x, _ = load_sample(
                sample, polar_size, lookup, frozen, None if rgb_only else config.x_mode, hha_recipe,
            )
            polar_inputs = [
                frozen["normalized_bchw"](image, config.norm_mean, config.norm_std)
                for image in (polar_rgb, polar_x)
            ]
            logits = predict_polar_strips(
                network, inputs, polar_inputs, strip_rows, args.wrap_pad, polar_wrap_pad, device
            )
        if caps is not None:  # probabilities from here on; only their argmax is used below
            if tile_windows is None:
                logits = class_probabilities(logits, channels)
            logits[:, :, :caps["top_end"]], logits[:, :, caps["bottom_start"]:] = predict_polar_caps(
                network, *inputs, caps, device, channels
            )
        prediction = predict_classes(logits, channels)[0].cpu().numpy().astype(np.uint8)
        hist, _, _ = frozen["hist_info"](config.num_classes, prediction, label)
        confusion += hist.astype(np.int64)
        band_confusion += elevation_band_confusions(
            config.num_classes, prediction, label, frozen["hist_info"]
        )
        if index % 20 == 0 or index == len(samples):
            print("{}/{} panoramas".format(index, len(samples)), flush=True)

    if confusion.sum() == 0:
        raise ValueError("no valid semantic pixels were evaluated")
    metrics = frozen["metrics_from_confusion"](confusion)
    report = {
        "status": "SMOKE" if args.limit is not None else "COMPLETED",
        "setting": "source-only pinhole-to-panorama transfer",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "config": str(args.config),
        "x_mode": config.x_mode,
        "architecture": config.get("architecture"),
        "panorama_x_input": (
            "none: the RGB-only network ignores X" if rgb_only
            else "ERP getREL ({}) on nearest-resized depth".format(rv.describe(relplus_variant))
            if hha_recipe is None
            else "ERP HHA (hha.erp_hha), {}, channel order {}".format(
                hha_recipe["variant"], hha_recipe["channel_order"]
            )
        ),
        "relplus_variant": relplus_variant,
        "hha_cache_report": None if hha_recipe is None else str(args.hha_cache_report),
        "hha_recipe": hha_recipe,
        "areas": args.areas,
        "eval_size": [args.height, args.width],
        "wrap_pad": args.wrap_pad,
        "infer_scale": args.infer_scale,
        "infer_size": list(infer_size),
        "infer_wrap_pad": infer_wrap_pad,
        "attention_chunk_mb": args.attention_chunk_mb,
        "chunked_attention_modules": chunked_attention_modules,
        "polar_strips": None if strip_rows is None else {
            "margin_rows": args.polar_strips,
            "polar_scale": POLAR_STRIP_SCALE,
            "polar_infer_size": list(polar_size),
            "polar_wrap_pad": polar_wrap_pad,
            "rows": strip_rows,
        },
        "erp_tiles": None if tile_windows is None else {
            "grid": list(args.erp_tiles),
            "tile_size": list(args.erp_tile_size),
            "tile_origins": [list(window) for window in tile_windows],
        },
        "polar_caps": None if caps is None else {
            "tile_size": args.polar_caps,
            "fov_deg": POLAR_CAP_FOV_DEG,
            "turn": "90 deg about x: (x, y, z) -> (x, -z, y)",
            "replaced_rows": [caps["top_end"], caps["bottom_start"]],
        },
        "prediction_classes": list(EIGHT_CLASS_NAMES if channels else config.class_names),
        "sample_count": len(samples),
        "seconds": time.time() - started,
        "metrics": metrics,
        "elevation_bands": write_elevation_bands(
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
