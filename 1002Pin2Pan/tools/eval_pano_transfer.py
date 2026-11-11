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
* ``--erp-tile-pole-patches SIZE``, with a 3-row tile grid, replaces the top and
  bottom tile rows by one SIZE x SIZE pinhole view centred on each pole (looking
  straight up or down). Its inscribed circle is the cap beyond the middle tile row,
  which stays the only source inside it.
* ``--sphere-views SIZE`` replaces the whole pass by 22 SIZE x SIZE pinhole views
  on the sphere, with no roll, in ``eval_pano_crops.py``'s layout and field of view
  (8 at pitch 0, 6 at +-45, 1 at +-80; 62.48 degrees). RGB (bilinear) and the X
  encoded on the whole panorama (nearest) are sampled into each view; each pixel
  fuses every view that sees it. With ``--erp-tiles`` there is one view per tile
  instead, centred on the tile and as wide as it (SMMF's partition on the sphere);
  ``--sphere-views-middle-row`` turns only the middle tile row into views, and
  ``--sphere-views-poles`` replaces the top and bottom tile rows by one view per pole.
* ``--erp-gores N`` replaces the whole pass by N equal-area gores: each gore is the
  sinusoidal projection about its own central meridian (one on longitude 0), cut from
  the evaluation-size panorama, so every gore pixel covers the solid angle of an
  equatorial ERP pixel and the pixel pitch is the same across as down everywhere.
  RGB (bilinear) and the X encoded on the whole panorama (nearest) are sampled into each
  gore; each pixel fuses every gore that covers it. ``--erp-gore-fill petal`` blanks the
  pixels beyond half a gore (the gores then tile the sphere); the default ``full`` keeps
  the panorama content there, so gores overlap towards the poles.
  ``--erp-gore-hemispheres MARGIN`` cuts each gore at the equator into a north and a
  south petal reaching MARGIN degrees past it; ``--erp-gore-offset-set`` adds a second
  set of gores half a gore further round; ``--erp-gore-columns COLUMNS`` (fill ``full``
  only) makes every gore image wider than a gore at the equator, so neighbouring gores
  overlap there as well.
* ``--erp-tile-sliding-stride S`` places the tiles as CMX's own sliding-window
  evaluation does instead: every S pixels from the top-left corner, the last row and
  column moved back to the panorama edge, no wrap across the seam.
* ``--input-pool RGB X`` puts a fixed 2x2 pooling (avg, max or nearest) in front of
  the network and a 2x bilinear upsampling of its logits behind it, so each
  ``--erp-tiles`` window runs at half size and its logits return to the window's
  size before fusion. With SMMF's layout this is SMMF050 as one model-level change,
  except that X is pooled from the native encoding instead of re-encoded at half size.
  It also goes in front of ``--erp-tile-pole-patches`` (the patches are pooled too) and of
  ``--erp-tile-polar-scale 1`` (polar rows at the evaluation size, used only outside the middle row).
  With ``--erp-tile-polar-squeeze`` each polar row then runs as one window-sized input.
* ``--polar-caps SIZE`` replaces the bands beyond +-60 degrees by two
  SIZE x SIZE tiles, 67.5 degrees wide, centred on the poles of the sphere
  turned 90 degrees about the x axis, where the poles lie on the equator and
  ERP barely stretches. RGB (bilinear) and the X encoded on the whole
  panorama (nearest) are resampled into each tile; the tile softmax is read
  back onto the polar rows. It works with the whole pass or ``--erp-tiles``.
* ``--ignore-black-rgb`` drops labelled pixels whose native RGB is (0, 0, 0)
  from scoring, as Trans4PASS does; prediction is unchanged.

* ``--erp-tile-black-cut TOP BOTTOM`` (with ``--erp-tiles`` and ``--ignore-black-rgb``) starts the first
  tile row at row TOP and ends the last at row BOTTOM, so the tiles skip the black RGB caps. Rows outside
  get no prediction; every labelled pixel there must be black RGB (checked per panorama).

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

A gated REL+ checkpoint (``gated_relplus.py``, trained by ``train_gated_relplus.py``) runs
with the whole pass or ERP tiles. Every window gets its own gate choice, and metrics.json
counts the choices per window row (``gate_counts_per_window_row``).
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
import gated_relplus as gr
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
# --sphere-views: eval_pano_crops.py's CROP_FOV_DEG (median S2D training field of view) and CROP_LAYOUT,
# as (yaw, pitch) in degrees; yaw is the ERP longitude of the view centre.
# --sphere-views-poles: the views straight up and down, oriented as --erp-tile-pole-patches' patches.
POLE_VIEWS = ((-90.0, 90.0), (-90.0, -90.0))
SPHERE_VIEW_FOV_DEG = 62.47653165897473
SPHERE_VIEW_LAYOUT = tuple((float(yaw), 0.0) for yaw in range(0, 360, 45)) + tuple(
    (float(yaw), pitch) for pitch in (45.0, -45.0) for yaw in range(0, 360, 60)
) + ((0.0, 80.0), (0.0, -80.0))


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


def black_rgb_mask(rgb_path, size):
    """Pixels whose native RGB is (0, 0, 0), nearest-resized to ``size`` (height, width) like the labels.

    Trans4PASS's ``_color2id`` (stanford2d3d_pan.py) sets these labels to ignore on the native
    panorama, before its own nearest resize.
    """
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None or rgb.ndim != 3:
        raise ValueError("{} is not a 3-channel image".format(rgb_path))
    black = (rgb == 0).all(axis=2).astype(np.uint8)
    return cv2.resize(black, (size[1], size[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


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


def erp_tile_windows(height, width, rows, columns, tile_height, tile_width, column_stride=None, column_offset=0):
    """(top, left) of each tile: rows span the height exactly, column i starts at ``floor(i * width / columns)``.

    With ``column_stride``, column i starts at ``column_offset + i * column_stride`` wrapped by the width instead.
    """
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
    if column_stride is None:
        if not -(-width // columns) <= tile_width <= width:
            raise ValueError("tile columns must cover the width")
        lefts = [column * width // columns for column in range(columns)]
    else:
        if not 0 < column_stride <= tile_width <= width or (columns - 1) * column_stride + tile_width < width:
            raise ValueError("tile columns must cover the width")
        lefts = [(column_offset + column * column_stride) % width for column in range(columns)]
    return [(row * row_stride, left) for row in range(rows) for left in lefts]


def sliding_windows(height, width, tile_height, tile_width, stride):
    """(top, left) of each tile where CMX's sliding-window evaluation (``scale_process_rgbX``) puts it.

    Tiles start every ``stride`` pixels from the top-left corner, the last row and column moved back
    to end at the panorama edge. Nothing wraps across the seam.
    """
    if not 0 < stride <= min(tile_height, tile_width) or tile_height > height or tile_width > width:
        raise ValueError("sliding tiles need a stride between 1 and the tile size, and tiles inside the panorama")

    def starts(length, tile):
        return [min(index * stride + tile, length) - tile for index in range(-(-(length - tile) // stride) + 1)]

    return [(top, left) for top in starts(height, tile_height) for left in starts(width, tile_width)]


def gate_rows(windows, polar_tiles=None, pole_patches=None, sphere_views=None, gores=None):
    """The window row of each network input, in the order the predictors feed them (gated REL+ counts)."""
    if gores is not None:
        return list(gores["labels"])
    if sphere_views is not None and windows is None:
        return ["pitch {:g}".format(pitch) for _, pitch in sphere_views["layout"]]
    if sphere_views is not None and sphere_views["pole_views"]:  # the middle row's views, then one per pole
        middle = sorted({top for top, _ in windows})[1]
        return ["row {}".format(middle)] * (len(sphere_views["layout"]) - 2) + ["north view", "south view"]
    if sphere_views is not None and sphere_views["middle_row"] is not None:  # ERP tiles first, then the views
        middle = sphere_views["middle_row"]
        return ["row {}".format(top) for top, _ in windows if top != middle] + ["row {}".format(middle)] * len(
            sphere_views["layout"])
    if windows is None:
        return ["whole"]
    if pole_patches is not None:
        middle = pole_patches["middle_rows"][0]
        return ["north patch", "south patch"] + ["row {}".format(middle) for row, _ in windows if row == middle]
    if polar_tiles is None:
        return ["row {}".format(top) for top, _ in windows]
    top, middle, bottom = sorted({row for row, _ in windows})
    if polar_tiles.get("squeeze"):  # one tile-sized input per polar row
        polar = ["row {} (x{:g} squeezed)".format(row, polar_tiles["scale"]) for row in (top, bottom)]
    elif polar_tiles.get("strip_wrap_pad") is not None:  # one full-width strip per polar row
        polar = ["row {} (x{:g} strip)".format(row, polar_tiles["scale"]) for row in (top, bottom)]
    else:
        polar = ["row {} (x{:g})".format(row, polar_tiles["scale"]) for row, _ in windows if row in (top, bottom)]
    return polar + ["row {}".format(middle) for row, _ in windows if row == middle]


def class_probabilities(logits, channels=None):
    """Softmax over the classes; with channels, over those only (the rest get 0), as an 8-output model would."""
    import torch

    if channels is None:
        return logits.softmax(dim=1)
    masked = torch.full_like(logits, float("-inf"))
    masked[:, channels] = logits[:, channels]
    return masked.softmax(dim=1)


def predict_erp_tiles(network, rgb, modal_x, windows, tile_size, device, channels=None, exp_sum=False,
                      return_sums=False):
    """Mean class probabilities over the tiles covering each pixel; columns wrap at the seam.

    With ``exp_sum`` each tile contributes exp(logits): CMX's val_func_process_rgbX returns those, and SMMF sums them.
    With ``return_sums``: the per-pixel sum and tile count instead, without the coverage check.
    """
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
        logits = predict_with_wrap(network, *tiles, 0, device)
        probabilities = logits.exp() if exp_sum else class_probabilities(logits, channels)
        if total is None:
            total = torch.zeros((1, probabilities.shape[1], height, width), device=device)
        for tile, (top, _), cols in zip(probabilities, batch, columns):
            cols = cols.to(device)
            total[0, :, top:top + tile_height].index_add_(2, cols, tile)
            count[0, :, top:top + tile_height].index_add_(2, cols, torch.ones_like(tile[:1]))
    if return_sums:
        return total, count
    if not bool((count > 0).all()):
        raise ValueError("the tile grid leaves panorama pixels uncovered")
    return total / count


def predict_black_cut_tiles(network, rgb, modal_x, windows, tile_size, cut, device, channels=None, exp_sum=False):
    """``predict_erp_tiles`` with the first tile row starting at row ``cut[0]`` and the last ending at ``cut[1]``.

    Each tile row runs as its own group. Rows outside ``cut`` get no tile and come back as zeros; they must hold
    only pixels that are not scored (the black RGB caps under --ignore-black-rgb).
    """
    import torch

    height, width = rgb.shape[2:]
    tops = sorted({top for top, _ in windows})
    total = count = None
    for top in tops:
        start = max(top, cut[0]) if top == tops[0] else top
        end = min(top + tile_size[0], cut[1]) if top == tops[-1] else top + tile_size[0]
        row_total, row_count = predict_erp_tiles(
            network, rgb[:, :, start:end], modal_x[:, :, start:end], [(0, left) for row, left in windows if row == top],
            (end - start, tile_size[1]), device, channels, exp_sum, return_sums=True,
        )
        if total is None:
            total = torch.zeros((1, row_total.shape[1], height, width), device=device)
            count = torch.zeros((1, 1, height, width), device=device)
        total[:, :, start:end] += row_total
        count[:, :, start:end] += row_count
    if not bool((count[:, :, cut[0]:cut[1]] > 0).all()):
        raise ValueError("the cut tile grid leaves pixels inside the cut uncovered")
    return total / count.clamp(min=1)


def check_black_cut_unscored(label, cut, sample_name):
    """Every pixel outside rows ``cut`` must be unscored (255) once black RGB is ignored."""
    outside = np.concatenate([label[:cut[0]].ravel(), label[cut[1]:].ravel()])
    scored = int((outside != 255).sum())
    if scored:
        raise ValueError("{}: {} labelled pixels outside the black cut are not black RGB".format(sample_name, scored))


def predict_polar_scaled_tiles(network, full, polar, windows, tile_size, middle_rows, device, channels=None,
                               exp_sum=False, strip_wrap_pad=None, squeeze=False):
    """Evaluation-size tile fusion with the first and last tile rows run on the smaller ``polar`` inputs.

    ``full`` and ``polar`` are (rgb, modal_x) at the evaluation size and at the polar scale. The rows
    ``middle_rows`` take only the middle tile row, cut ``middle_rows`` tall from ``full`` at the same columns;
    every other row takes only the polar tile rows, fused at the polar size and resized back (bilinear).
    With ``strip_wrap_pad`` each polar tile row runs as one full-width strip (the same rows, circular
    padding of ``strip_wrap_pad`` columns) instead of its tiles. With ``squeeze`` each polar tile row, full
    width, is resized (area) to one polar tile and run without padding (the seam at its left and right edges);
    its logits are resized back to the row (bilinear).
    """
    import torch

    height, width = full[0].shape[2:]
    scale = polar[0].shape[2] / height
    top, middle, bottom = sorted({row for row, _ in windows})
    polar_windows = [(round(row * scale), round(left * scale)) for row, left in windows if row in (top, bottom)]
    polar_size = tuple(int(round(length * scale)) for length in tile_size)
    if strip_wrap_pad is None and not squeeze:
        fused = predict_erp_tiles(network, *polar, polar_windows, polar_size, device, channels, exp_sum)
    else:
        fused = count = None
        for row in sorted({row for row, _ in polar_windows}):
            strip = [image[:, :, row:row + polar_size[0]] for image in polar]
            if squeeze:
                strip = [torch.nn.functional.interpolate(image.to(device), size=polar_size, mode="area")
                         for image in strip]
            logits = predict_with_wrap(
                network, *strip, 0 if squeeze else strip_wrap_pad, device, (polar_size[0], polar[0].shape[3])
            )
            probabilities = logits.exp() if exp_sum else class_probabilities(logits, channels)
            if fused is None:
                fused = torch.zeros((1, probabilities.shape[1]) + tuple(polar[0].shape[2:]), device=device)
                count = torch.zeros((1, 1) + tuple(polar[0].shape[2:]), device=device)
            fused[:, :, row:row + polar_size[0]] += probabilities
            count[:, :, row:row + polar_size[0]] += 1
        if not bool((count > 0).all()):
            raise ValueError("the polar strips leave panorama rows uncovered")
        fused = fused / count
    fused = torch.nn.functional.interpolate(fused, size=(height, width), mode="bilinear", align_corners=False)
    start, end = middle_rows
    middle_windows = [(0, left) for row, left in windows if row == middle]
    fused[:, :, start:end] = predict_erp_tiles(
        network, full[0][:, :, start:end], full[1][:, :, start:end], middle_windows,
        (end - start, tile_size[1]), device, channels, exp_sum,
    )
    return fused


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
    """The two cap tiles: RGB bilinear, X nearest so depth edges are not blended.

    With ``masks`` (gore petals) the pixels outside the mask are zeroed after sampling.
    """
    import torch
    import torch.nn.functional as F

    grid = torch.as_tensor(caps["tiles"], device=device)
    mask = None if "masks" not in caps else torch.as_tensor(caps["masks"], device=device)[:, None]
    sampled = [
        F.grid_sample(
            F.pad(image.to(device), (1, 1, 0, 0), mode="circular").expand(len(grid), -1, -1, -1), grid,
            mode=mode, padding_mode="border", align_corners=False,
        )
        for image, mode in ((rgb, "bilinear"), (modal_x, "nearest"))
    ]
    return sampled if mask is None else [image * mask.to(image.dtype) for image in sampled]


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


def pole_patch_grids(height, width, size, middle_rows):
    """``grid_sample`` coordinates between the ERP and one pinhole patch centred on each pole (north, south).

    The north patch looks straight up and the south one straight down, with image x along +x and the
    same handedness as the ERP image (not mirrored against it). Each patch's inscribed circle is the cap beyond ``middle_rows`` (rows above
    its first and below its last). ``tiles`` (2, size, size, 2): where each patch pixel reads the ERP
    padded by one column on each side (as ``sample_cap_tiles`` expects). ``north`` and ``south``: where
    each ERP pixel above and below ``middle_rows`` reads its patch.
    """
    start, end = middle_rows
    if not 0 < start < end < height or start != height - end:
        raise ValueError("the middle rows must be centred on the equator")
    radius = start * np.pi / height  # angle from the pole to the top edge of the middle rows
    half = np.tan(radius)
    steps = ((np.arange(size) + 0.5) / size * 2 - 1) * half
    a, b = np.meshgrid(steps, steps)  # image x to the right, image y down
    one = np.ones_like(a)
    tiles = []
    for direction in (np.stack([a, -b, one], axis=-1), np.stack([a, b, -one], axis=-1)):
        longitude, latitude = longitude_latitude(direction / np.linalg.norm(direction, axis=-1, keepdims=True))
        tiles.append(np.stack([longitude / np.pi * width / (width + 2), -latitude / (np.pi / 2)], axis=-1))
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    north = unit_directions(longitude[:start], latitude[:start])
    south = unit_directions(longitude[end:], latitude[end:])
    focal = size / 2 / half
    return {
        "size": size,
        "middle_rows": (start, end),
        "cap_radius_deg": float(np.degrees(radius)),
        "focal_px": float(focal),
        "tiles": np.stack(tiles).astype(np.float32),
        "north": (np.stack([north[..., 0], -north[..., 1]], axis=-1) / north[..., 2:] / half).astype(np.float32),
        "south": (np.stack([-south[..., 0], -south[..., 1]], axis=-1) / south[..., 2:] / half).astype(np.float32),
    }


def predict_pole_patches(network, full, windows, tile_size, middle_rows, patches, device, channels=None,
                         exp_sum=False):
    """The middle rows from the middle tile row only, as P47; every other row read back from its pole patch."""
    import torch
    import torch.nn.functional as F

    height, width = full[0].shape[2:]
    logits = predict_with_wrap(network, *sample_cap_tiles(*full, patches, device), 0, device)
    probabilities = logits.exp() if exp_sum else class_probabilities(logits, channels)
    start, end = middle_rows
    fused = torch.zeros((1, probabilities.shape[1], height, width), device=device)
    for index, (name, rows) in enumerate((("north", slice(0, start)), ("south", slice(end, height)))):
        fused[:, :, rows] = F.grid_sample(
            probabilities[index:index + 1], torch.as_tensor(patches[name], device=device)[None],
            mode="bilinear", align_corners=False,
        )
    middle_windows = [(0, left) for row, left in windows if row == start]
    fused[:, :, start:end] = predict_erp_tiles(
        network, full[0][:, :, start:end], full[1][:, :, start:end], middle_windows,
        (end - start, tile_size[1]), device, channels, exp_sum,
    )
    return fused


def sphere_view_grids(height, width, size, layout=SPHERE_VIEW_LAYOUT, fov_deg=SPHERE_VIEW_FOV_DEG, require_full=True):
    """``grid_sample`` coordinates between the ERP and square pinhole views on the sphere.

    View i looks at longitude ``layout[i][0]`` and latitude ``layout[i][1]`` (degrees) with no roll: image x
    along +longitude (the ERP's handedness, not mirrored), image y down. ``tiles`` (views, size, size, 2):
    where each view pixel reads the ERP padded by one column on each side (as ``sample_cap_tiles`` expects).
    ``back``: per view, the flat indices of the ERP pixels it sees and where each reads the view.
    """
    half = np.tan(np.radians(fov_deg) / 2)
    steps = ((np.arange(size) + 0.5) / size * 2 - 1) * half
    a, b = np.meshgrid(steps, steps)
    longitude, latitude = np.meshgrid(
        (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi,
        np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi,
    )
    erp = unit_directions(longitude, latitude).reshape(-1, 3)
    tiles, back = [], []
    seen = np.zeros(height * width, dtype=np.int64)
    for yaw, pitch in layout:
        forward = unit_directions(np.radians(yaw), np.radians(pitch))
        right = np.array([-np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0.0])
        down = np.cross(right, forward)
        rays = forward + a[..., None] * right + b[..., None] * down
        view_longitude, view_latitude = longitude_latitude(rays / np.linalg.norm(rays, axis=-1, keepdims=True))
        tiles.append(np.stack([view_longitude / np.pi * width / (width + 2), -view_latitude / (np.pi / 2)], axis=-1))
        depth = erp @ forward
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = erp @ right / depth / half, erp @ down / depth / half
        inside = np.flatnonzero((depth > 0) & (np.abs(x) <= 1) & (np.abs(y) <= 1))
        back.append((inside, np.stack([x[inside], y[inside]], axis=-1).astype(np.float32)))
        seen[inside] += 1
    if require_full and not seen.all():
        raise ValueError("the sphere views leave panorama pixels unseen")
    return {
        "size": size,
        "fov_deg": fov_deg,
        "layout": tuple(layout),
        "focal_px": float(size / 2 / half),
        "tiles": np.stack(tiles).astype(np.float32),
        "back": back,
        "views_per_pixel": {"min": int(seen.min()), "max": int(seen.max()), "mean": float(seen.mean())},
    }


def tile_view_layout(height, width, windows, tile_size):
    """(yaw, pitch) in degrees of a view centred on each tile, and the tile's width in degrees.

    A 2:1 ERP has as many pixels per degree down as across, so a square tile spans the same angle both ways.
    """
    tile_height, tile_width = tile_size
    layout = tuple(
        ((left + tile_width / 2) / width * 360 - 180, 90 - (top + tile_height / 2) / height * 180)
        for top, left in windows
    )
    return layout, tile_width / width * 360


def predict_sphere_views(network, full, views, device, channels=None, exp_sum=False, return_sums=False):
    """Each panorama pixel: the mean probabilities (or exp(logits) with ``exp_sum``) of the views that see it.

    With ``return_sums``: the per-pixel sum and view count instead.
    """
    import torch
    import torch.nn.functional as F

    height, width = full[0].shape[2:]
    total = count = None
    for start in range(0, len(views["tiles"]), TILE_BATCH):
        batch = {key: views[key][start:start + TILE_BATCH] for key in ("tiles", "masks") if key in views}
        logits = predict_with_wrap(network, *sample_cap_tiles(*full, batch, device), 0, device)
        probabilities = logits.exp() if exp_sum else class_probabilities(logits, channels)
        if total is None:
            total = torch.zeros((probabilities.shape[1], height * width), device=device)
            count = torch.zeros(height * width, device=device)
        for view, (inside, where) in zip(probabilities, views["back"][start:start + TILE_BATCH]):
            inside = torch.as_tensor(inside, device=device)
            sampled = F.grid_sample(view[None], torch.as_tensor(where, device=device)[None, None],
                                    mode="bilinear", padding_mode="border", align_corners=False)
            total.index_add_(1, inside, sampled[0, :, 0])
            count.index_add_(0, inside, torch.ones(len(inside), device=device))
    total, count = total.reshape(1, -1, height, width), count.reshape(1, 1, height, width)
    return (total, count) if return_sums else total / count


def predict_tiles_and_views(network, full, windows, tile_size, views, device, channels=None, exp_sum=False):
    """ERP ``windows`` plus sphere ``views``: each pixel takes the mean over every tile and view that sees it
    (of exp(logits) with ``exp_sum``), as SMMF fuses its overlapping tiles."""
    tile_total, tile_count = predict_erp_tiles(network, *full, windows, tile_size, device, channels, exp_sum,
                                               return_sums=True)
    view_total, view_count = predict_sphere_views(network, full, views, device, channels, exp_sum, return_sums=True)
    count = tile_count + view_count
    if not bool((count > 0).all()):
        raise ValueError("the tiles and sphere views leave panorama pixels unseen")
    return (tile_total + view_total) / count


# --erp-gores: each gore is the sinusoidal (Sanson-Flamsteed, equal-area) projection about its own central
# meridian, so every gore pixel covers the solid angle of an equatorial ERP pixel. One gore is centred on the
# ERP's central meridian (longitude 0).
GORE_FILLS = ("full", "petal")
GORE_CONTEXT_COLUMNS = 2  # columns past the gore on each side that the network sees but that are never read back


def gore_centres(count, offset_set=False):
    """Central meridians (radians in [-pi, pi)) of ``count`` gores a full turn apart, starting at longitude 0;
    with ``offset_set`` the set half a gore further is appended."""
    step = 2 * np.pi / count
    centres = np.arange(count) * step
    if offset_set:
        centres = np.concatenate([centres, centres + step / 2])
    return (centres + np.pi) % (2 * np.pi) - np.pi


def gore_rectangle(height, width, count, hemisphere_margin_rows=None, columns=None):
    """(rows, columns) of one gore image: the ERP rows it spans (all of them, or a hemisphere plus the margin)
    and the smallest even width that holds a gore at the equator, plus the context columns on each side.
    ``columns`` asks for a wider image than that, context columns included."""
    smallest = -(-width // count)
    smallest += smallest % 2 + 2 * GORE_CONTEXT_COLUMNS
    if columns is None:
        columns = smallest
    elif not smallest <= columns <= width:
        raise ValueError("gore columns must be between {} (a gore at the equator plus its context columns) and "
                         "the panorama width".format(smallest))
    rows = height if hemisphere_margin_rows is None else height // 2 + hemisphere_margin_rows
    return rows, columns


def gore_grids(height, width, count, hemisphere_margin_rows=None, fill="full", offset_set=False,
               require_full=True, columns=None):
    """``grid_sample`` coordinates between the ERP and equal-area gores.

    Gore pixel (v, u) keeps the latitude of its ERP row and sits at the longitude offset
    ``(u + 0.5 - columns / 2) / cos(latitude)`` ERP pixels from the gore's central meridian, so a row of the gore
    reads the ERP row compressed by cos(latitude): every gore pixel covers one equatorial pixel's solid angle,
    and the pixel pitch is the same across as down everywhere. With ``fill`` "petal" the pixels beyond half a
    gore (|offset| > pi / count) are blanked (zero after normalisation) and never read back; with "full" they
    keep the ERP content, so gores overlap towards the poles. The outermost ``GORE_CONTEXT_COLUMNS`` columns on
    each side are context only and never read back. With ``hemisphere_margin_rows`` each gore is cut
    at the equator into a north and a south petal, each reaching that many rows past the equator. With
    ``offset_set`` a second set of gores, half a gore further round, is added. With ``columns`` (fill "full"
    only) every gore image is that wide instead of the smallest width that holds a gore at the equator: it shows
    more longitude at every latitude, so neighbouring gores overlap at the equator as well. Gores are fed in the
    order of ``labels``: all north petals (or whole gores), then all south petals.
    ``tiles`` (gores, rows, columns, 2): where each gore pixel reads the ERP padded by one column on each side
    (as ``sample_cap_tiles`` expects); ``masks``: the pixels kept; ``back``: per gore, the flat indices of the
    ERP pixels it covers and where each reads the gore.
    """
    if fill not in GORE_FILLS:
        raise ValueError("gore fill must be one of {}".format(", ".join(GORE_FILLS)))
    if count < 2:
        raise ValueError("at least 2 gores are needed")
    if hemisphere_margin_rows is not None and (height % 2 or not 0 <= hemisphere_margin_rows < height // 2):
        raise ValueError("hemisphere petals need an even panorama height and a margin below half of it")
    if columns is not None and fill == "petal":
        raise ValueError("wider gores need the fill 'full': 'petal' blanks everything beyond half a gore")
    rows, columns = gore_rectangle(height, width, count, hemisphere_margin_rows, columns)
    half_gore = np.pi / count
    px_per_rad = width / (2 * np.pi)
    centres = gore_centres(count, offset_set)
    if hemisphere_margin_rows is None:
        petals = [(0, "gore")]
    else:
        petals = [(0, "north petal"), (height - rows, "south petal")]
    gores = [(centre, start, label) for start, label in petals for centre in centres]
    erp_longitude = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    erp_latitude = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    offsets = (np.arange(columns) + 0.5 - columns / 2) / px_per_rad  # longitude offset times cos(latitude)
    tiles, masks, back = [], [], []
    seen = np.zeros(height * width, dtype=np.int64)
    for centre, start, _ in gores:
        latitude = erp_latitude[start:start + rows]
        delta = offsets[None, :] / np.cos(latitude)[:, None]  # (rows, columns), longitude offset per gore pixel
        longitude = (centre + delta + np.pi) % (2 * np.pi) - np.pi
        tiles.append(np.stack([
            longitude / np.pi * width / (width + 2),
            np.broadcast_to(-latitude[:, None] / (np.pi / 2), delta.shape),
        ], axis=-1))
        if fill == "petal":  # keep 3 gore pixels past the petal edge, so neither the bilinear read-back nor a 2x2
            # input pooling with its 2x upsampling ever mixes a blank pixel into a pixel that is read back
            masks.append(np.abs(delta) <= half_gore + 3 / (np.cos(latitude)[:, None] * px_per_rad))
        else:
            masks.append(np.ones(delta.shape, dtype=bool))
        delta_back = (erp_longitude[None, :] - centre + np.pi) % (2 * np.pi) - np.pi  # (1, width)
        x = delta_back * np.cos(latitude)[:, None] * px_per_rad + columns / 2  # (rows, width), gore column
        covered = (x >= GORE_CONTEXT_COLUMNS) & (x <= columns - GORE_CONTEXT_COLUMNS)
        if fill == "petal":
            covered &= np.abs(delta_back) <= half_gore
        r, c = np.nonzero(covered)
        flat = (r + start) * width + c
        back.append((flat, np.stack([x[r, c] / columns * 2 - 1, (r + 0.5) / rows * 2 - 1], axis=-1).astype(
            np.float32)))
        seen[flat] += 1
    if require_full and not seen.all():
        raise ValueError("the gores leave panorama pixels uncovered")
    return {
        "count": count,
        "offset_set": offset_set,
        "fill": fill,
        "centres_deg": [float(np.degrees(centre)) for centre in centres],
        "rows": rows,
        "columns": columns,
        "hemisphere_margin_rows": hemisphere_margin_rows,
        "labels": [label for _, _, label in gores],
        "starts": [int(start) for _, start, _ in gores],
        "tiles": np.stack(tiles).astype(np.float32),
        "masks": np.stack(masks),
        "back": back,
        "gores_per_pixel": {"min": int(seen.min()), "max": int(seen.max()), "mean": float(seen.mean())},
    }


def predict_gores(network, full, gores, device, channels=None, exp_sum=False):
    """Each panorama pixel: the mean probabilities (or exp(logits) with ``exp_sum``) of the gores covering it,
    read back bilinear. A petal's blanked pixels are never read back."""
    return predict_sphere_views(network, full, gores, device, channels, exp_sum)


INPUT_POOL_MODES = ("avg", "max", "nearest")


def pool_half(image, mode):
    """A fixed 2x2 pooling of a (B, C, H, W) tensor. "avg" equals bilinear resizing by exactly 0.5 (and
    cv2.INTER_LINEAR); "nearest" keeps the top-left pixel of each block, as cv2.INTER_NEAREST does."""
    import torch.nn.functional as F

    if mode == "avg":
        return F.avg_pool2d(image, 2)
    if mode == "max":
        return F.max_pool2d(image, 2)
    if mode == "nearest":
        return image[:, :, ::2, ::2]
    raise ValueError("unknown pooling mode: " + str(mode))


def input_pool_network(network, rgb_mode, x_mode):
    """``network`` behind a 2x2 pooling of RGB and X and in front of a 2x bilinear upsampling of its logits.

    Other forward arguments pass through unchanged. ``gate_log`` is the wrapped network's (gated REL+).
    """
    import torch
    import torch.nn.functional as F

    class InputPool(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network

        @property
        def gate_log(self):
            return self.network.gate_log

        def forward(self, rgb, modal_x, *rest):
            if rgb.shape[2] % 2 or rgb.shape[3] % 2:
                raise ValueError("2x2 pooling needs even input sizes, got {}".format(tuple(rgb.shape[2:])))
            logits = self.network(pool_half(rgb, rgb_mode), pool_half(modal_x, x_mode), *rest)
            return F.interpolate(logits, size=rgb.shape[2:], mode="bilinear", align_corners=False)

    return InputPool()


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
        help="tile size in network-input pixels, i.e. after --infer-scale "
        "(default 384 x 384, 67.5 degrees at 1024 x 2048)",
    )
    parser.add_argument(
        "--erp-tile-column-stride", type=int, default=None, metavar="S",
        help="start column i at O + i * S (wrapping) instead of floor(i * W / N); 720 with --erp-tiles 3 7 "
        "and 1080 x 1080 tiles at 2048 x 4096 is REL-SF4PASS's SMMF stride",
    )
    parser.add_argument(
        "--erp-tile-column-offset", type=int, default=0, metavar="O",
        help="first column start for --erp-tile-column-stride; -652 matches SMMF's 652 px cyclic padding",
    )
    parser.add_argument(
        "--erp-tile-sliding-stride", type=int, default=None, metavar="S",
        help="instead of --erp-tiles, place --erp-tile-size tiles as CMX's sliding-window evaluation does: every "
        "S pixels from the top-left corner, the last row and column moved back to the panorama edge, no wrap "
        "across the seam. 720 with 1080 x 1080 tiles at 2048 x 4096 is REL-SF4PASS's region stride",
    )
    parser.add_argument(
        "--erp-tile-exp-sum", action="store_true",
        help="fuse overlapping tiles by summing exp(logits), as SMMF sums CMX's val_func_process_rgbX "
        "outputs, instead of averaging the softmax",
    )
    parser.add_argument(
        "--erp-tile-polar-scale", type=float, default=None, metavar="S",
        help="with a 3-row tile grid, cut the top and bottom rows from the panorama resized by S (same place, "
        "S times smaller) and use them only outside the middle row; the middle row stays at the evaluation "
        "size. 0.5 with SMMF's 3 x 7 layout halves it beyond +-47.5 deg; 1 keeps every row at the evaluation size",
    )
    parser.add_argument(
        "--erp-tile-middle-height", type=int, default=None, metavar="H",
        help="with --erp-tile-polar-scale, make the middle-row tiles H tall, centred on the equator "
        "(default the tile height); 1366 at 2048 rows reaches +-60 deg",
    )
    parser.add_argument(
        "--erp-tile-polar-strips", action="store_true",
        help="with --erp-tile-polar-scale, run each polar tile row as one full-width strip (same rows, "
        "circular padding of --wrap-pad times the polar scale) instead of its tiles",
    )
    parser.add_argument(
        "--erp-tile-polar-squeeze", action="store_true",
        help="with --erp-tile-polar-scale, run each polar tile row (full width) as one input resized (area) to "
        "the polar tile size, without padding; its logits are resized back to the row (bilinear). With SMMF's "
        "layout, --erp-tile-polar-scale 1 and --input-pool avg avg: pooled SMMF's middle row plus one 540 x 540 "
        "input per polar row",
    )
    parser.add_argument(
        "--erp-tile-pole-patches", type=int, default=None, metavar="SIZE",
        help="with a 3-row tile grid, replace the top and bottom tile rows by one SIZE x SIZE pinhole view centred "
        "on each pole, whose inscribed circle is the cap beyond the middle tile row; the middle row stays the only "
        "source inside it. With SMMF's 3 x 7 layout the caps are beyond +-47.5 deg",
    )
    parser.add_argument(
        "--sphere-views", type=int, default=None, metavar="SIZE",
        help="instead of one pass, {} SIZE x SIZE pinhole views on the sphere, {:.2f} deg wide, with no roll: "
        "8 at pitch 0, 6 at +-45 and 1 at +-80 (eval_pano_crops.py's layout), cut from the evaluation-size "
        "panorama; each pixel fuses every view that sees it. 480 is the training crop size. With --erp-tiles, "
        "one view per tile instead, centred on the tile and as wide as it (square tiles)".format(
            len(SPHERE_VIEW_LAYOUT), SPHERE_VIEW_FOV_DEG),
    )
    parser.add_argument(
        "--sphere-views-middle-row", action="store_true",
        help="with --sphere-views and a 3-row --erp-tiles grid, only the middle tile row becomes sphere views; "
        "the top and bottom rows stay ERP tiles, and each pixel fuses every tile and view that sees it",
    )
    parser.add_argument(
        "--sphere-views-poles", action="store_true",
        help="with --sphere-views and a 3-row --erp-tiles grid, one view per middle-row tile plus one view straight "
        "up and one straight down, all as wide as a tile; each pixel fuses every view that sees it",
    )
    parser.add_argument(
        "--erp-gores", type=int, default=None, metavar="N",
        help="instead of one pass, N equal-area gores (the sinusoidal projection about each gore's central "
        "meridian, one of them on longitude 0) cut from the evaluation-size panorama: a gore row is the ERP row "
        "compressed by cos(latitude), so every gore pixel covers an equatorial pixel's solid angle; each pixel "
        "fuses every gore that covers it. 7 gores on 2048 x 4096 are 2048 x 590 each (586 plus 2 context columns on "
        "each side)",
    )
    parser.add_argument(
        "--erp-gore-hemispheres", type=float, default=None, metavar="MARGIN_DEG",
        help="with --erp-gores, cut each gore at the equator into a north and a south petal, each reaching "
        "MARGIN_DEG degrees past the equator (SMMF's polar windows reach 4.9); the two petals of a gore are "
        "separate network inputs",
    )
    parser.add_argument(
        "--erp-gore-fill", choices=GORE_FILLS, default="full",
        help="with --erp-gores: 'full' (default) keeps the panorama content beyond half a gore, so gores overlap "
        "towards the poles; 'petal' blanks it (zero after normalisation) and reads nothing back from there, so "
        "the gores tile the sphere",
    )
    parser.add_argument(
        "--erp-gore-offset-set", action="store_true",
        help="with --erp-gores, also run a second set of gores whose central meridians sit half a gore further "
        "round, so every pixel is covered by both sets",
    )
    parser.add_argument(
        "--erp-gore-columns", type=int, default=None, metavar="COLUMNS",
        help="with --erp-gores and the fill 'full', make every gore image COLUMNS wide (context columns included) "
        "instead of the smallest width that holds a gore at the equator (590 for 7 gores on 4096): a wider gore "
        "shows more longitude at every latitude, so neighbouring gores overlap at the equator as well; 826 shows "
        "72.6 degrees there instead of 51.9",
    )
    parser.add_argument(
        "--input-pool", nargs=2, default=None, metavar=("RGB", "X"),
        help="with --erp-tiles (also with --sphere-views) or --erp-gores, pool RGB and X 2x2 ({}) in front of the "
        "network and upsample its logits 2x (bilinear) behind it; bilinear resizing by 0.5 is the same as avg".format(
            ", ".join(INPUT_POOL_MODES)),
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
    parser.add_argument(
        "--ignore-black-rgb", action="store_true",
        help="score without labelled pixels whose native RGB is black, as Trans4PASS does",
    )
    parser.add_argument(
        "--erp-tile-black-cut", type=int, nargs=2, default=None, metavar=("TOP", "BOTTOM"),
        help="with --erp-tiles and --ignore-black-rgb, start the first tile row at row TOP and end the last at row "
        "BOTTOM (evaluation size), so the tiles skip the black RGB caps; rows outside get no prediction and every "
        "labelled pixel there must be black RGB. 248 1800 keeps every valid pixel of Stanford2D3D's 2048 x 4096 "
        "panoramas (black beyond +-62.8..66.3 deg)",
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
    if args.erp_tile_column_stride is not None and args.erp_tiles is None:
        parser.error("--erp-tile-column-stride needs --erp-tiles")
    if args.erp_tile_column_offset and args.erp_tile_column_stride is None:
        parser.error("--erp-tile-column-offset needs --erp-tile-column-stride")
    if (args.erp_tile_exp_sum and args.erp_tiles is None and args.erp_tile_sliding_stride is None
            and args.sphere_views is None and args.erp_gores is None):
        parser.error("--erp-tile-exp-sum needs --erp-tiles, --erp-tile-sliding-stride, --sphere-views or "
                     "--erp-gores")
    if args.erp_tile_sliding_stride is not None:
        if args.erp_tiles is not None:
            parser.error("--erp-tile-sliding-stride places the tiles itself; drop --erp-tiles")
        if args.polar_strips is not None:
            parser.error("--erp-tile-sliding-stride replaces the whole pass; drop --polar-strips")
        try:  # tiles are cut from the network-input panorama (evaluation size times --infer-scale)
            tile_windows = sliding_windows(*infer_size, *args.erp_tile_size, args.erp_tile_sliding_stride)
        except ValueError as error:
            parser.error(str(error))
    if args.erp_tiles is not None:
        if args.polar_strips is not None:
            parser.error("--erp-tiles replaces the whole pass; drop --polar-strips")
        try:  # tiles are cut from the network-input panorama (evaluation size times --infer-scale)
            tile_windows = erp_tile_windows(
                *infer_size, *args.erp_tiles, *args.erp_tile_size,
                args.erp_tile_column_stride, args.erp_tile_column_offset,
            )
        except ValueError as error:
            parser.error(str(error))
    polar_tiles = None
    if args.erp_tile_middle_height is not None and args.erp_tile_polar_scale is None:
        parser.error("--erp-tile-middle-height needs --erp-tile-polar-scale")
    if args.erp_tile_polar_strips and args.erp_tile_polar_scale is None:
        parser.error("--erp-tile-polar-strips needs --erp-tile-polar-scale")
    if args.erp_tile_polar_squeeze and (args.erp_tile_polar_scale is None or args.erp_tile_polar_strips):
        parser.error("--erp-tile-polar-squeeze needs --erp-tile-polar-scale, without --erp-tile-polar-strips")
    if args.erp_tile_polar_scale is not None:
        scale = args.erp_tile_polar_scale
        if args.erp_tiles is None or args.erp_tiles[0] != 3:
            parser.error("--erp-tile-polar-scale needs --erp-tiles with 3 rows")
        if args.infer_scale != 1.0 or args.polar_caps is not None:
            parser.error("--erp-tile-polar-scale keeps the middle row at the evaluation size; "
                         "drop --infer-scale and --polar-caps")
        if not 0 < scale <= 1:
            parser.error("--erp-tile-polar-scale must be above 0 and at most 1")
        if 2 * args.erp_tile_size[0] < args.height:
            parser.error("--erp-tile-polar-scale needs the top and bottom tile rows to meet")
        middle_height = args.erp_tile_size[0] if args.erp_tile_middle_height is None else args.erp_tile_middle_height
        if not args.erp_tile_size[0] <= middle_height <= args.height or (args.height - middle_height) % 2:
            parser.error("--erp-tile-middle-height must be between the tile height and --height, "
                         "and leave an even number of rows")
        tops = (tile_windows[0][0], tile_windows[-1][0])
        lengths = [args.height, args.width, *args.erp_tile_size] + [
            value for window in tile_windows if window[0] in tops for value in window
        ]
        if any(length * scale != int(length * scale) for length in lengths):
            parser.error("--erp-tile-polar-scale must map the panorama, the tile size and the "
                         "polar tile origins to whole pixels")
        start = (args.height - middle_height) // 2
        polar_tiles = {
            "scale": scale,
            "size": (int(args.height * scale), int(args.width * scale)),
            "middle_rows": (start, start + middle_height),
            "strip_wrap_pad": None,
            "squeeze": args.erp_tile_polar_squeeze,
        }
        if args.erp_tile_polar_strips:
            if args.wrap_pad * scale != int(args.wrap_pad * scale):
                parser.error("--erp-tile-polar-strips needs --wrap-pad times the polar scale to be whole")
            polar_tiles["strip_wrap_pad"] = int(args.wrap_pad * scale)
    pole_patches = None
    if args.erp_tile_pole_patches is not None:
        if args.erp_tiles is None or args.erp_tiles[0] != 3:
            parser.error("--erp-tile-pole-patches needs --erp-tiles with 3 rows")
        if args.erp_tile_polar_scale is not None or args.polar_caps is not None or args.infer_scale != 1.0:
            parser.error("--erp-tile-pole-patches replaces the polar tile rows at the evaluation size; "
                         "drop --erp-tile-polar-scale, --polar-caps and --infer-scale")
        if args.erp_tile_pole_patches <= 0:
            parser.error("--erp-tile-pole-patches must be positive")
        start, remainder = divmod(args.height - args.erp_tile_size[0], 2)
        if remainder or start not in {top for top, _ in tile_windows}:
            parser.error("--erp-tile-pole-patches needs a tile row centred on the equator")
        pole_patches = pole_patch_grids(
            args.height, args.width, args.erp_tile_pole_patches, (start, start + args.erp_tile_size[0])
        )
    caps = None
    if args.polar_caps is not None:
        if args.polar_caps <= 0 or args.polar_caps % STRIP_ROW_MULTIPLE:
            parser.error("--polar-caps must be a positive multiple of {}".format(STRIP_ROW_MULTIPLE))
        if args.polar_strips is not None or args.infer_scale != 1.0:
            parser.error("--polar-caps samples the evaluation-size panorama; drop --polar-strips and --infer-scale")
        caps = polar_cap_grids(args.height, args.width, args.polar_caps)
    sphere_views = None
    if (args.sphere_views_middle_row or args.sphere_views_poles) and args.sphere_views is None:
        parser.error("--sphere-views-middle-row and --sphere-views-poles need --sphere-views")
    if args.sphere_views is not None:
        if args.sphere_views <= 0:
            parser.error("--sphere-views must be positive")
        if (args.erp_tile_sliding_stride is not None or polar_tiles is not None or pole_patches is not None
                or strip_rows is not None or caps is not None or args.infer_scale != 1.0):
            parser.error("--sphere-views replaces the whole pass, or the --erp-tiles windows, at the evaluation "
                         "size; drop --erp-tile-sliding-stride, --erp-tile-polar-scale, --erp-tile-pole-patches, "
                         "--polar-strips, --polar-caps and --infer-scale")
        if (args.sphere_views_middle_row or args.sphere_views_poles) and (
                args.erp_tiles is None or args.erp_tiles[0] != 3):
            parser.error("--sphere-views-middle-row and --sphere-views-poles need --erp-tiles with 3 rows")
        if args.sphere_views_middle_row and args.sphere_views_poles:
            parser.error("--sphere-views-poles replaces the polar tile rows that --sphere-views-middle-row keeps; "
                         "use one of them")
        if tile_windows is None:
            sphere_views = sphere_view_grids(args.height, args.width, args.sphere_views)
            sphere_views["middle_row"], sphere_views["pole_views"] = None, False
        else:  # one view per tile (or per middle-row tile), as wide as the tile
            if args.erp_tile_size[0] * args.width != 2 * args.erp_tile_size[1] * args.height or (
                    2 * args.erp_tile_size[1] >= args.width):
                parser.error("--sphere-views with --erp-tiles needs tiles that span the same angle across and "
                             "down, less than 180 degrees")
            centre = (args.height - args.erp_tile_size[0]) // 2  # the middle tile row
            middle = centre if args.sphere_views_middle_row else None
            as_views = [window for window in tile_windows
                        if not (args.sphere_views_middle_row or args.sphere_views_poles) or window[0] == centre]
            layout, fov_deg = tile_view_layout(args.height, args.width, as_views, args.erp_tile_size)
            if args.sphere_views_poles:
                layout += POLE_VIEWS
            try:
                sphere_views = sphere_view_grids(args.height, args.width, args.sphere_views, layout, fov_deg,
                                                 require_full=middle is None)
            except ValueError as error:
                parser.error(str(error))
            sphere_views["middle_row"], sphere_views["pole_views"] = middle, args.sphere_views_poles
    gores = None
    if args.erp_gores is None and (args.erp_gore_hemispheres is not None or args.erp_gore_offset_set
                                   or args.erp_gore_fill != "full" or args.erp_gore_columns is not None):
        parser.error("--erp-gore-hemispheres, --erp-gore-fill, --erp-gore-offset-set and --erp-gore-columns need "
                     "--erp-gores")
    if args.erp_gores is not None:
        if args.erp_gores < 2:
            parser.error("--erp-gores needs at least 2 gores")
        if (args.erp_tiles is not None or args.erp_tile_sliding_stride is not None or sphere_views is not None
                or strip_rows is not None or caps is not None or args.infer_scale != 1.0):
            parser.error("--erp-gores replaces the whole pass at the evaluation size; drop --erp-tiles, "
                         "--erp-tile-sliding-stride, --sphere-views, --polar-strips, --polar-caps and --infer-scale")
        margin_rows = None
        if args.erp_gore_hemispheres is not None:
            if not 0 <= args.erp_gore_hemispheres < 90:
                parser.error("--erp-gore-hemispheres must be at least 0 and below 90 degrees")
            if args.height % 2:
                parser.error("--erp-gore-hemispheres needs an even --height")
            margin_rows = int(round(args.erp_gore_hemispheres / 180 * args.height))
        try:
            gores = gore_grids(args.height, args.width, args.erp_gores, margin_rows, args.erp_gore_fill,
                               args.erp_gore_offset_set, columns=args.erp_gore_columns)
        except ValueError as error:
            parser.error(str(error))
    black_cut = None
    if args.erp_tile_black_cut is not None:
        if (args.erp_tiles is None or args.erp_tiles[0] < 2 or polar_tiles is not None or pole_patches is not None
                or sphere_views is not None or gores is not None or caps is not None or args.infer_scale != 1.0):
            parser.error("--erp-tile-black-cut needs plain --erp-tiles with at least 2 rows at the evaluation size; "
                         "drop --erp-tile-polar-scale, --erp-tile-pole-patches, --sphere-views, --erp-gores, "
                         "--polar-caps and --infer-scale")
        if not args.ignore_black_rgb:
            parser.error("--erp-tile-black-cut leaves rows without a prediction; add --ignore-black-rgb")
        tops = sorted({top for top, _ in tile_windows})
        top_cut, bottom_cut = args.erp_tile_black_cut
        if not (0 < top_cut < tops[0] + args.erp_tile_size[0] and tops[-1] < bottom_cut < args.height
                and top_cut < bottom_cut):
            parser.error("--erp-tile-black-cut must cut inside the first tile row (TOP) and the last (BOTTOM)")
        black_cut = (top_cut, bottom_cut)
    if args.input_pool is not None:
        if any(mode not in INPUT_POOL_MODES for mode in args.input_pool):
            parser.error("--input-pool modes are {} (bilinear by 0.5 is avg)".format(", ".join(INPUT_POOL_MODES)))
        if (args.erp_tiles is None and gores is None or args.infer_scale != 1.0 or caps is not None
                or polar_tiles is not None
                and (polar_tiles["scale"] != 1 or polar_tiles["strip_wrap_pad"] is not None)):
            parser.error("--input-pool goes in front of --erp-tiles at the evaluation size, also with "
                         "--erp-tile-pole-patches, --erp-tile-polar-scale 1 or --sphere-views, or in front of "
                         "--erp-gores; drop --infer-scale, --polar-caps, --erp-tile-polar-strips and polar scales "
                         "below 1")
        lengths = [] if tile_windows is None else list(args.erp_tile_size) + [v for window in tile_windows
                                                                                 for v in window]
        lengths += [] if pole_patches is None else [pole_patches["size"]]
        lengths += [] if polar_tiles is None else list(polar_tiles["middle_rows"])
        lengths += [] if sphere_views is None else [sphere_views["size"]]
        lengths += [] if gores is None else [gores["rows"], gores["columns"]]
        lengths += [] if black_cut is None else list(black_cut)
        if any(value % 2 for value in lengths):
            parser.error("--input-pool needs even tile sizes, tile origins, middle rows, pole patch sizes, "
                         "sphere view sizes and gore sizes, so pooling each input equals cutting it from the "
                         "pooled panorama")

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
    gated = gr.is_gated(config)
    if gated and (strip_rows is not None or caps is not None):
        raise ValueError("gated REL+ is evaluated with the whole pass or ERP tiles only")
    hha_recipe = None
    if config.x_mode == "hha_frozen_cache" and not rgb_only:
        hha_recipe = hha_cache_recipe(
            args.hha_cache_report, config.x_root_folder, args.accept_hha_near_match
        )
    lookup = label_lookup(args.semantic_labels)
    samples = list_panoramas(args.stanford_root, args.areas)[: args.limit]

    device = torch.device(args.device)
    network_class = ro.model_class() if rgb_only else gr.model_class() if gated else frozen["EncoderDecoder"]
    network = network_class(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    epoch = frozen["load_checkpoint_once"](
        network, args.checkpoint, expected_epoch=args.expected_epoch
    )
    chunked_attention_modules = 0
    if args.attention_chunk_mb:
        chunked_attention_modules = chunk_attention(network, args.attention_chunk_mb)
    if args.input_pool is not None:
        network = input_pool_network(network, *args.input_pool)
    network.to(device)
    network.eval()

    started = time.time()
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    band_confusion = np.zeros((len(ELEVATION_BAND_EDGES_DEG) - 1,) + confusion.shape, dtype=np.int64)
    rows = gate_rows(tile_windows, polar_tiles, pole_patches, sphere_views, gores)
    gate_counts = {row: [0] * gr.BRANCHES for row in rows} if gated else None
    for index, sample in enumerate(samples, start=1):
        rgb, modal_x, label = load_sample(
            sample, infer_size, lookup, frozen, None if rgb_only else config.x_mode, hha_recipe,
            label_size=(args.height, args.width),
        )
        if args.ignore_black_rgb:
            label[black_rgb_mask(sample["rgb"], (args.height, args.width))] = 255
        if black_cut is not None:
            check_black_cut_unscored(label, black_cut, sample["sample_id"])
        inputs = [
            frozen["normalized_bchw"](image, config.norm_mean, config.norm_std) for image in (rgb, modal_x)
        ]
        if sphere_views is not None and sphere_views["middle_row"] is not None:  # only the argmax is used below
            logits = predict_tiles_and_views(
                network, inputs, [window for window in tile_windows if window[0] != sphere_views["middle_row"]],
                args.erp_tile_size, sphere_views, device, channels, args.erp_tile_exp_sum,
            )
        elif sphere_views is not None:  # only the argmax is used below
            logits = predict_sphere_views(network, inputs, sphere_views, device, channels, args.erp_tile_exp_sum)
        elif gores is not None:  # only the argmax is used below
            logits = predict_gores(network, inputs, gores, device, channels, args.erp_tile_exp_sum)
        elif pole_patches is not None:  # only the argmax is used below
            logits = predict_pole_patches(
                network, inputs, tile_windows, args.erp_tile_size, pole_patches["middle_rows"], pole_patches,
                device, channels, args.erp_tile_exp_sum,
            )
        elif polar_tiles is not None:  # mean probabilities; only their argmax is used below
            polar_rgb, polar_x, _ = load_sample(
                sample, polar_tiles["size"], lookup, frozen, None if rgb_only else config.x_mode, hha_recipe,
            )
            polar_inputs = [
                frozen["normalized_bchw"](image, config.norm_mean, config.norm_std) for image in (polar_rgb, polar_x)
            ]
            logits = predict_polar_scaled_tiles(
                network, inputs, polar_inputs, tile_windows, args.erp_tile_size, polar_tiles["middle_rows"],
                device, channels, args.erp_tile_exp_sum, polar_tiles["strip_wrap_pad"], polar_tiles["squeeze"],
            )
        elif black_cut is not None:  # mean probabilities inside the cut; only their argmax is used below
            logits = predict_black_cut_tiles(
                network, *inputs, tile_windows, args.erp_tile_size, black_cut, device, channels, args.erp_tile_exp_sum
            )
        elif tile_windows is not None:  # mean probabilities; only their argmax is used below
            logits = predict_erp_tiles(
                network, *inputs, tile_windows, args.erp_tile_size, device, channels, args.erp_tile_exp_sum
            )
            if tuple(infer_size) != (args.height, args.width):  # fused map to label size, as the whole pass does
                logits = torch.nn.functional.interpolate(
                    logits, size=(args.height, args.width), mode="bilinear", align_corners=False
                )
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
        if gated:
            if len(network.gate_log) != len(rows):
                raise RuntimeError("{} gate choices for {} network inputs".format(len(network.gate_log), len(rows)))
            for row, choice in zip(rows, network.gate_log):
                gate_counts[row][choice] += 1
            network.gate_log.clear()
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
        "ignore_black_rgb": args.ignore_black_rgb,
        "black_cut": None if black_cut is None else {
            "rows": list(black_cut),
            "latitudes_deg": [90 - row / args.height * 180 for row in black_cut],
            "first_row_tile_size": [min(tile_windows[0][0] + args.erp_tile_size[0], black_cut[1]) - black_cut[0],
                                    args.erp_tile_size[1]],
            "last_row_tile_size": [black_cut[1] - max(tile_windows[-1][0], black_cut[0]), args.erp_tile_size[1]],
            "rule": "first tile row starts at rows[0], last ends at rows[1]; rows outside get no prediction and every "
                    "labelled pixel there must be black RGB (checked per panorama); each tile row runs as its own group",
        },
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
            "grid": None if args.erp_tiles is None else list(args.erp_tiles),
            "sliding_stride": args.erp_tile_sliding_stride,
            "tile_size": list(args.erp_tile_size),
            "column_stride": args.erp_tile_column_stride,
            "column_offset": args.erp_tile_column_offset,
            "fusion": "sum of exp(logits)" if args.erp_tile_exp_sum else "mean softmax",
            "tile_origins": [list(window) for window in tile_windows],
            "origins_in": "network-input pixels ({} x {})".format(*infer_size),
            "polar_rows": None if polar_tiles is None else {
                "scale": polar_tiles["scale"],
                "input_size": list(polar_tiles["size"]),
                "network_tile_size": [int(length * polar_tiles["scale"]) for length in args.erp_tile_size],
                "middle_rows": list(polar_tiles["middle_rows"]),
                "middle_tile_size": [polar_tiles["middle_rows"][1] - polar_tiles["middle_rows"][0],
                                     args.erp_tile_size[1]],
                "rule": "middle_rows from the middle tile row only; all other rows from the top and bottom "
                        "tile rows only, fused at input_size and resized back (bilinear)",
                "strips": None if polar_tiles["strip_wrap_pad"] is None else {
                    "wrap_pad": polar_tiles["strip_wrap_pad"],
                    "network_input_size": [int(args.erp_tile_size[0] * polar_tiles["scale"]),
                                           polar_tiles["size"][1] + 2 * polar_tiles["strip_wrap_pad"]],
                    "rule": "each polar tile row runs as one full-width strip with circular padding",
                },
                "squeeze": None if not polar_tiles["squeeze"] else {
                    "row_size": [int(args.erp_tile_size[0] * polar_tiles["scale"]), polar_tiles["size"][1]],
                    "network_input_size": [int(length * polar_tiles["scale"]) for length in args.erp_tile_size],
                    "rule": "each polar tile row, full width, resized (area) to one tile and run without padding; "
                            "logits resized back to the row (bilinear)",
                },
            },
        },
        "pole_patches": None if pole_patches is None else {
            "size": [pole_patches["size"]] * 2,
            "projection": "pinhole (gnomonic), centred on each pole, looking straight up / down",
            "cap_radius_deg": pole_patches["cap_radius_deg"],
            "boundary_latitude_deg": 90 - pole_patches["cap_radius_deg"],
            "focal_px": pole_patches["focal_px"],
            "centre_px_per_deg": pole_patches["focal_px"] * np.pi / 180,
            "middle_rows": list(pole_patches["middle_rows"]),
            "rule": "middle_rows from the middle tile row only; all other rows read back (bilinear) from the "
                    "patch of their pole; RGB sampled bilinear, X nearest",
        },
        "sphere_views": None if sphere_views is None else {
            "size": [sphere_views["size"]] * 2,
            "projection": "pinhole (gnomonic), no roll, image x along +longitude",
            "fov_deg": sphere_views["fov_deg"],
            "focal_px": sphere_views["focal_px"],
            "centre_px_per_deg": sphere_views["focal_px"] * np.pi / 180,
            "layout_yaw_pitch_deg": [list(view) for view in sphere_views["layout"]],
            "views_per_pixel": sphere_views["views_per_pixel"],
            "fusion": "sum of exp(logits)" if args.erp_tile_exp_sum else "mean softmax",
            "centred_on_erp_tiles": tile_windows is not None,
            "middle_tile_row_only": sphere_views["middle_row"] is not None,
            "pole_views": sphere_views["pole_views"],
            "rule": ("the middle tile row as views, the other tile rows as ERP tiles; each pixel fuses every tile and "
                     "view that sees it" if sphere_views["middle_row"] is not None else
                     "each pixel fuses every view that sees it")
                    + ", read back bilinear; RGB sampled bilinear, X nearest",
        },
        "erp_gores": None if gores is None else {
            "count": gores["count"],
            "offset_set": gores["offset_set"],
            "projection": "sinusoidal (Sanson-Flamsteed, equal-area) about each gore's central meridian",
            "centre_longitudes_deg": gores["centres_deg"],
            "fill": gores["fill"],
            "hemispheres": None if gores["hemisphere_margin_rows"] is None else {
                "margin_deg": args.erp_gore_hemispheres,
                "margin_rows": gores["hemisphere_margin_rows"],
            },
            "gore_size": [gores["rows"], gores["columns"]],
            "smallest_columns": gore_rectangle(args.height, args.width, gores["count"])[1],
            "context_columns_each_side": GORE_CONTEXT_COLUMNS,
            "gore_row_starts": gores["starts"],
            "network_inputs": len(gores["labels"]),
            "px_per_deg": [args.height / 180, args.width / 360],
            "gores_per_pixel": gores["gores_per_pixel"],
            "fusion": "sum of exp(logits)" if args.erp_tile_exp_sum else "mean softmax",
            "rule": "each pixel fuses every gore that covers it, read back bilinear; RGB sampled bilinear, "
                    "X nearest" + ("; pixels beyond half a gore are blanked and never read back"
                                   if gores["fill"] == "petal" else ""),
        },
        "input_pool": None if args.input_pool is None else {
            "rgb": args.input_pool[0],
            "x": args.input_pool[1],
            "factor": 2,
            "network_tile_size": None if tile_windows is None else [length // 2 for length in args.erp_tile_size],
            "network_pole_patch_size": None if pole_patches is None else [pole_patches["size"] // 2] * 2,
            "network_sphere_view_size": None if sphere_views is None else [sphere_views["size"] // 2] * 2,
            "network_gore_size": None if gores is None else [gores["rows"] // 2, gores["columns"] // 2],
            "logits": "upsampled 2x (bilinear) to the tile size before fusion",
        },
        "polar_caps": None if caps is None else {
            "tile_size": args.polar_caps,
            "fov_deg": POLAR_CAP_FOV_DEG,
            "turn": "90 deg about x: (x, y, z) -> (x, -z, y)",
            "replaced_rows": [caps["top_end"], caps["bottom_start"]],
        },
        "gate_counts_per_window_row": gate_counts,
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
