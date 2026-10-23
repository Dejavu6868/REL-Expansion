#!/usr/bin/env python3
"""How redundant are the S2D pinhole views, and what would a 15% drop remove?

CPU only, no images are read. For one split of the frozen manifest it reports:

- pitch, roll and horizontal FOV of every view (pitch and roll as in
  ``audit_training_views.py``);
- views per camera location (the ``camera_<uuid>_`` prefix of the pose file);
- per location, how many views see each direction of the sphere
  (``--directions`` near-uniform directions), by elevation band;
- what three ways of dropping ``--drop-fraction`` of the views do to that
  coverage: the steepest |pitch| first, uniformly at random (``--random-trials``
  draws), and a greedy rule that removes the view whose loss pushes the fewest
  directions below ``--greedy-k`` views.

The decision rule is fixed here, before the run (see ``decide``).
"""

import argparse
import csv
import heapq
import json
import re
import time
from pathlib import Path

import numpy as np

import audit_training_views as audit
import cross_projection as cp
import eval_pano_transfer as ev

LOCATION_PATTERN = re.compile(r"^camera_([0-9a-f]+)_")
COVERAGE_LEVELS = (1, 2, 3, 5)
PITCH_BIN_EDGES_DEG = np.arange(-30, 31, 5)
# Pre-set rule. Coverage-aware selection has room only if random dropping makes
# real holes (directions a location no longer sees at all) and the greedy rule
# avoids most of them.
HOLE_SHARE_THRESHOLD = 0.01
GREEDY_KEEPS_AT_LEAST = 0.5


def sphere_directions(count):
    """Fibonacci-lattice unit vectors, z up: equal solid angle per direction."""
    index = np.arange(count) + 0.5
    z = 1.0 - 2.0 * index / count
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * index
    radius = np.sqrt(1.0 - z * z)
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=1)


def location_key(pose_path):
    match = LOCATION_PATTERN.match(Path(pose_path).name)
    if match is None:
        raise ValueError("pose file name has no camera_<uuid>_ prefix: {}".format(pose_path))
    return match.group(1)


def visible(camera, directions):
    """Which world directions fall inside the image (JSON half-pixel K)."""
    height, width = camera.intrinsics_shape
    k_json = camera.K_json
    in_camera = directions @ camera.R_world_to_camera.T
    depth = in_camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k_json[0, 0] * in_camera[:, 0] / depth + k_json[0, 2]
        v = k_json[1, 1] * in_camera[:, 1] / depth + k_json[1, 2]
    return (depth > 0) & (u >= 0) & (u <= width) & (v >= 0) & (v <= height)


def horizontal_fov_deg(camera):
    return float(np.degrees(2.0 * np.arctan(camera.intrinsics_shape[1] / 2.0 / camera.K_json[0, 0])))


def percentiles(values):
    return {k: float(np.percentile(values, q)) for k, q in audit.PERCENTILES.items()}


def band_masks(directions):
    elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
    edges = ev.ELEVATION_BAND_EDGES_DEG
    return [(elevation <= top) & (elevation > bottom) for top, bottom in zip(edges[:-1], edges[1:])]


def coverage_summary(multiplicity, bands):
    """Solid-angle shares averaged over locations; multiplicity is locations x directions."""
    def shares(columns):
        part = multiplicity[:, columns]
        return {
            "mean_views_per_direction": float(part.mean()),
            "share_seen_by_at_least": {
                str(level): float((part >= level).mean()) for level in COVERAGE_LEVELS
            },
        }

    return {
        "whole_sphere": shares(slice(None)),
        "by_elevation_band": [shares(mask) for mask in bands],
    }


def lost_shares(before, after, bands):
    """Share of the directions seen by >= k views that drop below k, overall and per band."""
    result = {}
    for level in COVERAGE_LEVELS:
        had = before >= level
        lost = had & (after < level)
        per_band = []
        for mask in bands:
            total = had[:, mask].sum()
            per_band.append(float(lost[:, mask].sum() / total) if total else None)
        result[str(level)] = {
            "whole_sphere": float(lost.sum() / max(had.sum(), 1)),
            "by_elevation_band": per_band,
        }
    return result


def multiplicity_after(masks, groups, keep):
    return np.stack([masks[rows][keep[rows]].sum(axis=0) for rows in groups])


def greedy_drop(masks, groups, count, k):
    """Remove ``count`` views, each time the one pushing the fewest directions below k.

    Ties go to the view whose directions are seen most often. Scores only change
    inside the location a view was removed from, so stale heap entries are
    skipped by a per-location version number.
    """
    keep = np.ones(len(masks), dtype=bool)
    counts = [masks[rows].sum(axis=0) for rows in groups]
    version = [0] * len(groups)
    heap = []

    def push(location):
        for row in groups[location]:
            if keep[row]:
                seen = masks[row]
                critical = int(np.count_nonzero(seen & (counts[location] <= k)))
                heapq.heappush(heap, (critical, -int(counts[location][seen].sum()), row, location, version[location]))

    for location in range(len(groups)):
        push(location)
    dropped = 0
    while dropped < count:
        _, _, row, location, stamp = heapq.heappop(heap)
        if stamp != version[location] or not keep[row]:
            continue
        keep[row] = False
        counts[location] = counts[location] - masks[row]
        version[location] += 1
        dropped += 1
        push(location)
    return keep


def read_split(manifest, split):
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError("no {} samples in manifest".format(split))
    return rows


def load_views(rows, directions):
    """Pitch, roll, FOV and visible directions per view, views grouped by camera location.

    Needs ``cp.import_frozen_source`` first.
    """
    from rel_plus.camera import gravity_in_camera, load_stanford_s2d_camera_geometry

    pitches, rolls, fovs, masks, keys, centres = [], [], [], [], [], {}
    for row in rows:
        camera = load_stanford_s2d_camera_geometry(row["camera_metadata_path"], audit.NATIVE_SHAPE)
        pitch, roll = audit.view_angles(camera, gravity_in_camera)
        pitches.append(pitch)
        rolls.append(roll)
        fovs.append(horizontal_fov_deg(camera))
        masks.append(visible(camera, directions))
        key = location_key(row["camera_metadata_path"])
        keys.append(key)
        centres.setdefault(key, []).append(-camera.R_world_to_camera.T @ camera.t_world_to_camera)
    order = sorted(set(keys))
    index = {key: i for i, key in enumerate(order)}
    groups = [[] for _ in order]
    for row, key in enumerate(keys):
        groups[index[key]].append(row)
    groups = [np.asarray(rows_) for rows_ in groups]
    centre_spread = [
        float(np.max(np.linalg.norm(np.asarray(points) - points[0], axis=1))) for points in centres.values()
    ]
    return np.asarray(pitches), rolls, fovs, np.stack(masks), groups, centre_spread


def decide(random_holes, greedy_holes):
    if random_holes < HOLE_SHARE_THRESHOLD:
        return "REDUNDANT_RANDOM_SUFFICES"
    if greedy_holes <= (1.0 - GREEDY_KEEPS_AT_LEAST) * random_holes:
        return "COVERAGE_RULE_HAS_ROOM"
    return "COVERAGE_RULE_NO_BETTER"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="full_manifest.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=cp.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--drop-fraction", type=float, default=0.15)
    parser.add_argument("--directions", type=int, default=4096)
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("--greedy-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cp.import_frozen_source(args.source_root)
    rows = read_split(args.manifest, args.split)
    started = time.time()
    directions = sphere_directions(args.directions)
    bands = band_masks(directions)
    pitches, rolls, fovs, masks, groups, centre_spread = load_views(rows, directions)

    drop_count = int(round(args.drop_fraction * len(rows)))
    all_kept = np.ones(len(rows), dtype=bool)
    before = multiplicity_after(masks, groups, all_kept)

    steep_first = np.argsort(-np.abs(pitches), kind="stable")
    keep_pitch = all_kept.copy()
    keep_pitch[steep_first[:drop_count]] = False

    rng = np.random.default_rng(args.seed)
    random_keeps = []
    for _ in range(args.random_trials):
        keep_random = all_kept.copy()
        keep_random[rng.choice(len(rows), drop_count, replace=False)] = False
        random_keeps.append(keep_random)

    keep_greedy = greedy_drop(masks, groups, drop_count, args.greedy_k)

    def method_report(keep):
        after = multiplicity_after(masks, groups, keep)
        return {
            "dropped": int((~keep).sum()),
            "min_views_left_at_a_location": int(min(keep[rows_].sum() for rows_ in groups)),
            "max_abs_pitch_kept_deg": float(np.abs(pitches[keep]).max()),
            "coverage": coverage_summary(after, bands),
            "lost_share_by_level": lost_shares(before, after, bands),
        }

    random_reports = [method_report(keep) for keep in random_keeps]
    random_holes = [report["lost_share_by_level"]["1"]["whole_sphere"] for report in random_reports]
    greedy_report = method_report(keep_greedy)
    pitch_report = method_report(keep_pitch)

    report = {
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "views": len(rows),
        "pitch_convention": "degrees above the horizon of the optical axis, from rel_plus.camera.gravity_in_camera",
        "pitch_deg_percentiles": percentiles(pitches),
        "pitch_histogram": {
            "edges_deg": PITCH_BIN_EDGES_DEG.tolist(),
            "counts": np.histogram(pitches, PITCH_BIN_EDGES_DEG)[0].tolist(),
            "outside_edges": int(np.sum((pitches < PITCH_BIN_EDGES_DEG[0]) | (pitches > PITCH_BIN_EDGES_DEG[-1]))),
        },
        "abs_roll_deg_percentiles": percentiles(np.abs(rolls)),
        "hfov_deg_percentiles": percentiles(fovs),
        "locations": len(groups),
        "views_per_location_percentiles": percentiles([len(rows_) for rows_ in groups]),
        "camera_centre_spread_m_percentiles": percentiles(centre_spread),
        "directions": args.directions,
        "elevation_bands_deg": [
            list(band) for band in zip(ev.ELEVATION_BAND_EDGES_DEG[:-1], ev.ELEVATION_BAND_EDGES_DEG[1:])
        ],
        "coverage_all_views": coverage_summary(before, bands),
        "drop_fraction": args.drop_fraction,
        "drop_count": drop_count,
        "steepest_pitch_first": pitch_report,
        "random": {
            "trials": args.random_trials,
            "seed": args.seed,
            "hole_share_mean": float(np.mean(random_holes)),
            "hole_share_range": [float(np.min(random_holes)), float(np.max(random_holes))],
            "draws": random_reports,
        },
        "greedy_coverage": dict(greedy_report, k=args.greedy_k),
        "rule": {
            "hole": "a direction a location saw with at least 1 view and sees with none after the drop",
            "hole_share_threshold": HOLE_SHARE_THRESHOLD,
            "greedy_keeps_at_least": GREEDY_KEEPS_AT_LEAST,
            "verdict": decide(float(np.mean(random_holes)), greedy_report["lost_share_by_level"]["1"]["whole_sphere"]),
        },
        "seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.split, len(rows), "views,", len(groups), "locations; verdict", report["rule"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
