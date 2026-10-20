#!/usr/bin/env python3
"""Attention mass against query-key angle and 3D distance: level pinhole crops against the whole panorama.

For each panorama, the frozen network runs once on the whole-panorama inputs (built as
``eval_pano_transfer.py`` builds them, with its wrap padding) and once on each of the 8 level
crops (built as ``eval_pano_crops.py --layout level`` builds them, at the training median FOV).
For MiT stages 1 and 2 of each stream, the tool records how each query's attention, averaged
over heads, spreads over geodesic angle and camera-centred 3D distance to its keys. Each
profile is the mean over queries and blocks.

* Queries: tokens within +-25 degrees of elevation (the band the level crops see whole) with
  depth, on every ``--query-step``-th token row and column.
* A token's direction and 3D point are those of the input pixel at its centre, and a
  spatial-reduction key's those of the centre pixel of its cell (``eval_pano_scope.py`` token
  geometry). Crop points use the nearest native ERP range along the crop ray, as the crop
  depth does. The last distance bin also holds keys beyond its upper edge.
* Mass on keys without depth goes to ``no_depth_mass`` instead of the distance bins.
* ``mass_beyond_cap`` is the mass beyond the 40.6-degree default radius of
  ``eval_pano_scope.py --attention-cap-deg``: what that cap removes.
* Panoramas: ``--panoramas`` of them, evenly spaced through the sorted list of the areas.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import cross_projection as cp
import eval_pano_crops as ec
import eval_pano_scope as es
import eval_pano_transfer as ev
import relplus_variant as rv
import rgb_only as ro


ANGLE_EDGES_DEG = np.arange(0.0, 180.1, 5.0)
DISTANCE_EDGES_M = np.arange(0.0, 12.01, 0.25)
QUERY_BAND_DEG = 25.0
STAGES = (1, 2)


def with_wrap(array, wrap_pad):
    """``array`` with ``wrap_pad`` columns of circular padding on each side, as the network input."""
    return np.concatenate([array[:, -wrap_pad:], array, array[:, :wrap_pad]], axis=1) if wrap_pad else array


def points_from_range(directions, range_m):
    """Camera-centred 3D points in metres; NaN where there is no depth."""
    return np.where((range_m > 0)[..., None], directions * range_m[..., None], np.nan)


class Profile:
    """Attention mass per angle and distance bin, summed over queries, per (presentation, stream, stage)."""

    def __init__(self, query_step, cap_deg=es.TRAINING_CORNER_DEG):
        self.query_step = query_step
        self.cap_deg = cap_deg
        self.totals = {}
        self.inputs = None
        self.tokens = {}

    def set_input(self, presentation, directions, points):
        """Per-pixel directions and points of the next network input (one image)."""
        self.inputs = (presentation, directions, points)
        self.tokens = {}

    def _tokens(self, H, W, sr):
        if (H, W, sr) not in self.tokens:
            _, directions, points = self.inputs
            self.tokens[(H, W, sr)] = (es.token_centres(directions, H, W, sr), es.token_centres(points, H, W, sr))
        return self.tokens[(H, W, sr)]

    def __call__(self, stream, stage, probabilities, H, W, sr):
        if stage not in STAGES:
            return
        import torch

        if probabilities.shape[0] != 1:
            raise ValueError("the profile takes one image per forward pass")
        query_directions, query_points = self._tokens(H, W, 1)
        key_directions, key_points = self._tokens(H, W, sr)
        on_grid = ((np.arange(H)[:, None] % self.query_step == 0)
                   & (np.arange(W)[None, :] % self.query_step == 0)).reshape(-1)
        elevation = np.degrees(np.arcsin(np.clip(query_directions[:, 2], -1, 1)))
        chosen = np.nonzero(on_grid & (np.abs(elevation) <= QUERY_BAND_DEG)
                            & np.isfinite(query_points).all(axis=1))[0]
        if not chosen.size:
            return
        device = probabilities.device
        mass = probabilities[0, :, torch.as_tensor(chosen, device=device)].mean(dim=0).double()  # (queries, keys)

        def tensor(array):
            return torch.from_numpy(np.ascontiguousarray(array)).to(device, torch.float64)

        angle = torch.rad2deg(torch.acos((tensor(query_directions[chosen]) @ tensor(key_directions).T).clamp(-1, 1)))
        with_depth = torch.from_numpy(np.isfinite(key_points).all(axis=1)).to(device)
        distance = torch.cdist(tensor(query_points[chosen]), tensor(np.nan_to_num(key_points)))

        def histogram(values, edges, weights):
            bins = torch.bucketize(values.reshape(-1), tensor(edges[1:-1]))
            return torch.zeros(len(edges) - 1, dtype=torch.float64, device=device).index_add_(
                0, bins, weights.reshape(-1))

        known = mass[:, with_depth]
        total = self.totals.setdefault((self.inputs[0], stream, stage), {
            "queries": 0, "angle_mass": 0.0, "distance_mass": 0.0, "no_depth_mass": 0.0,
            "angle_sum": 0.0, "distance_sum": 0.0, "mass_beyond_cap": 0.0,
        })
        total["queries"] += int(chosen.size)
        total["angle_mass"] += histogram(angle, ANGLE_EDGES_DEG, mass).cpu().numpy()
        total["distance_mass"] += histogram(distance[:, with_depth], DISTANCE_EDGES_M, known).cpu().numpy()
        total["no_depth_mass"] += float(mass[:, ~with_depth].sum())
        total["angle_sum"] += float((mass * angle).sum())
        total["distance_sum"] += float((known * distance[:, with_depth]).sum())
        total["mass_beyond_cap"] += float(mass[angle > self.cap_deg].sum())

    def summary(self):
        rows = []
        for (presentation, stream, stage), total in sorted(self.totals.items()):
            queries = total["queries"]
            known = float(np.sum(total["distance_mass"]))
            rows.append({
                "presentation": presentation,
                "stream": stream,
                "stage": stage,
                "queries": queries,
                "angle_mass": list(np.asarray(total["angle_mass"]) / queries),
                "distance_mass": list(np.asarray(total["distance_mass"]) / queries),
                "no_depth_mass": total["no_depth_mass"] / queries,
                "mean_angle_deg": total["angle_sum"] / queries,
                "mean_distance_m": total["distance_sum"] / known if known else None,
                "mass_beyond_cap": total["mass_beyond_cap"] / queries,
            })
        return rows


def spaced(samples, count):
    """``count`` samples evenly spaced through the list (all of them if there are fewer)."""
    if count >= len(samples):
        return list(samples)
    return [samples[index] for index in np.linspace(0, len(samples) - 1, count).round().astype(int)]


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
    parser.add_argument("--panoramas", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--wrap-pad", type=int, default=128)
    parser.add_argument("--query-step", type=int, default=4, help="query every n-th token row and column")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hha-cache-report", type=Path, default=None)
    parser.add_argument(
        "--accept-hha-near-match", action="store_true",
        help="also accept a NEAR_MATCH cache report (recorded in the output)",
    )
    parser.add_argument(
        "--relplus-variant", choices=sorted(rv.VARIANT_ALPHA), default="v2_1",
        help="REL+ EGVIA encoding; must match the checkpoint's training cache (relplus_variant.py)",
    )
    args = parser.parse_args()
    if args.panoramas <= 0 or args.query_step <= 0:
        parser.error("--panoramas and --query-step must be positive")

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = ec.import_frozen_crops(args.source_root)
    config = ev.load_config(args.config)
    if ro.is_rgb_only(config):
        raise ValueError("the profile compares the CMX arms; the crop path has no RGB-only input")
    if config.image_height != config.image_width:
        raise ValueError("crop inputs need a square model input size")
    relplus_variant = rv.select_for_eval(args.relplus_variant, config, frozen)
    hha_recipe = None
    if config.x_mode == "hha_frozen_cache":
        hha_recipe = ev.hha_cache_recipe(args.hha_cache_report, config.x_root_folder, args.accept_hha_near_match)
    lookup = ev.label_lookup(args.semantic_labels)
    samples = spaced(ev.list_panoramas(args.stanford_root, args.areas), args.panoramas)
    eval_size = (args.height, args.width)
    model_size = int(config.image_height)
    layout = ec.LAYOUTS["level"]

    import torch
    import torch.nn as nn

    device = torch.device(args.device)
    network = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    epoch = frozen["load_checkpoint_once"](network, args.checkpoint, expected_epoch=args.expected_epoch)
    network.to(device)
    network.eval()
    profile = Profile(args.query_step)
    es.patch_attention(network, record=profile)

    def normalize(image):
        return frozen["normalized_bchw"](image, config.norm_mean, config.norm_std)

    erp_directions = es.padded_erp_directions(args.height, args.width, args.wrap_pad)
    crop_k = cp.crop_intrinsics(model_size, ec.CROP_FOV_DEG)
    crop_rays = [cp.crop_rays(model_size, crop_k, cp.crop_camera_to_world(yaw, pitch))[1] for yaw, pitch in layout]
    started = time.time()
    for index, sample in enumerate(samples, start=1):
        raw = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError("{} is not a uint16 depth map".format(sample["depth"]))
        resized = cp.decode_erp_depth(cv2.resize(raw, (args.width, args.height), interpolation=cv2.INTER_NEAREST))
        profile.set_input("erp", erp_directions, with_wrap(points_from_range(
            cp.erp_directions(*eval_size), resized), args.wrap_pad))
        rgb, modal_x, _ = ev.load_sample(sample, eval_size, lookup, frozen, config.x_mode, hha_recipe)
        ev.predict_with_wrap(network, normalize(rgb), normalize(modal_x), args.wrap_pad, device)

        native = cp.decode_erp_depth(raw)
        rgb_crops, x_crops, _ = ec.prepare_panorama(
            sample, frozen, config.x_mode, hha_recipe, lookup, eval_size, model_size, layout=layout,
        )
        for rays, rgb_crop, x_crop in zip(crop_rays, rgb_crops, x_crops):
            rows, columns = cp.erp_indices_for_directions(rays, *native.shape)
            profile.set_input("crop", rays, points_from_range(rays, native[rows, columns]))
            with torch.no_grad():
                network(normalize(rgb_crop).to(device), normalize(x_crop).to(device))
        print("{}/{} panoramas".format(index, len(samples)), flush=True)

    report = {
        "status": "COMPLETED",
        "setting": "attention profile, level pinhole crops against the whole panorama",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "config": str(args.config),
        "x_mode": config.x_mode,
        "relplus_variant": relplus_variant,
        "hha_cache_report": None if hha_recipe is None else str(args.hha_cache_report),
        "hha_recipe": hha_recipe,
        "areas": args.areas,
        "samples": [sample["sample_id"] for sample in samples],
        "eval_size": list(eval_size),
        "wrap_pad": args.wrap_pad,
        "crop_fov_deg": ec.CROP_FOV_DEG,
        "crop_model_size": model_size,
        "crop_layout_yaw_pitch_deg": [list(crop) for crop in layout],
        "stages": list(STAGES),
        "query_band_deg": QUERY_BAND_DEG,
        "query_step": args.query_step,
        "angle_bin_edges_deg": list(ANGLE_EDGES_DEG),
        "distance_bin_edges_m": list(DISTANCE_EDGES_M),
        "reference_cap_deg": profile.cap_deg,
        "seconds": time.time() - started,
        "profiles": profile.summary(),
    }
    (args.output / "attention_profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
