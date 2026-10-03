#!/usr/bin/env python3
"""Audit which viewing directions the S2D pinhole training images cover.

Reads every pose in the frozen manifest with the frozen loader and reports,
per split: camera pitch and roll (from the same world-down gravity REL+ uses),
how many images sit near each pitch of the crop layout in
``eval_pano_crops.py``, and the share of image pixels per ERP elevation band
(rays sampled on a 32x32 grid per image). The last item can be read directly
against the per-band scores of the panorama evaluations.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import cross_projection as cp
import eval_pano_crops as ec
import eval_pano_transfer as ev

NATIVE_SHAPE = (1080, 1080)
RAY_GRID = 32
PITCH_BIN_EDGES_DEG = np.arange(-90, 91, 15)
PERCENTILES = {"min": 0, "p5": 5, "p25": 25, "median": 50, "p75": 75, "p95": 95, "max": 100}


def view_angles(camera, gravity_in_camera):
    """Pitch (positive looking up) and roll in degrees from the camera's world-down."""
    gravity = gravity_in_camera(camera.R_world_to_camera)
    pitch = np.degrees(np.arcsin(np.clip(-gravity[2], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(-gravity[0], gravity[1]))
    return float(pitch), float(roll)


def ray_elevations(camera):
    """Elevation in degrees of rays through a RAY_GRID x RAY_GRID grid of pixel centres."""
    height, width = camera.intrinsics_shape
    k_json = camera.K_json
    u = (np.arange(RAY_GRID) + 0.5) * width / RAY_GRID
    v = (np.arange(RAY_GRID) + 0.5) * height / RAY_GRID
    u, v = np.meshgrid(u, v)
    rays = np.stack([(u - k_json[0, 2]) / k_json[0, 0], (v - k_json[1, 2]) / k_json[1, 1], np.ones_like(u)], -1)
    world = rays.reshape(-1, 3) @ camera.R_world_to_camera  # rows: R^T @ ray
    return np.degrees(np.arcsin(world[:, 2] / np.linalg.norm(world, axis=1)))


def band_shares(elevations):
    edges = ev.ELEVATION_BAND_EDGES_DEG
    return [
        float(np.mean((elevations <= top) & (elevations > bottom)))
        for top, bottom in zip(edges[:-1], edges[1:])
    ]


def summarize_split(pitches, rolls, elevations):
    pitches = np.asarray(pitches)
    crop_pitches = sorted({pitch for _, pitch in ec.CROP_LAYOUT})
    return {
        "count": int(pitches.size),
        "pitch_deg_percentiles": {k: float(np.percentile(pitches, q)) for k, q in PERCENTILES.items()},
        "abs_roll_deg_percentiles": {
            k: float(np.percentile(np.abs(rolls), q)) for k, q in PERCENTILES.items()
        },
        "pitch_histogram": {
            "edges_deg": PITCH_BIN_EDGES_DEG.tolist(),
            "counts": np.histogram(pitches, PITCH_BIN_EDGES_DEG)[0].tolist(),
        },
        "fraction_within_10_deg_of_crop_pitch": {
            "{:g}".format(p): float(np.mean(np.abs(pitches - p) <= 10)) for p in crop_pitches
        },
        "pixel_share_by_elevation_band": band_shares(np.concatenate(elevations)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="full_manifest.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=cp.DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()

    cp.import_frozen_source(args.source_root)
    from rel_plus.camera import gravity_in_camera, load_stanford_s2d_camera_geometry

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("no samples in manifest")
    started = time.time()
    splits = {}
    for row in rows:
        camera = load_stanford_s2d_camera_geometry(row["camera_metadata_path"], NATIVE_SHAPE)
        pitch, roll = view_angles(camera, gravity_in_camera)
        split = splits.setdefault(row["split"], ([], [], []))
        split[0].append(pitch)
        split[1].append(roll)
        split[2].append(ray_elevations(camera))

    erp_rows = 90.0 - np.arange(1024) * 180.0 / 1024
    report = {
        "manifest": str(args.manifest.resolve()),
        "pitch_convention": "degrees above the horizon of the optical axis, from rel_plus.camera.gravity_in_camera",
        "elevation_bands_deg": [
            list(band) for band in zip(ev.ELEVATION_BAND_EDGES_DEG[:-1], ev.ELEVATION_BAND_EDGES_DEG[1:])
        ],
        "erp_grid_pixel_share_by_elevation_band": band_shares(erp_rows),
        "rays_per_image": RAY_GRID * RAY_GRID,
        "splits": {name: summarize_split(*values) for name, values in sorted(splits.items())},
        "seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, summary in report["splits"].items():
        print(name, summary["count"], "pitch", summary["pitch_deg_percentiles"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
