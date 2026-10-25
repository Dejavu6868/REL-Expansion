#!/usr/bin/env python3
"""Render steep-pitch pinhole crops from the training-area panoramas. CPU only.

Each crop is a square gnomonic view of one Stanford2D3D panorama, pitched
``--min-pitch``..``--max-pitch`` degrees up or down (alternating), with a uniform
yaw and a horizontal FOV drawn uniformly from ``--fov-range`` (the training views
span 45-75 degrees). ``--total-crops`` crops are spread evenly over the panoramas
of ``--areas``; area 5 is refused.

The output mirrors ``Stanford2D3D_480`` so the frozen cache tools read it as is:

- ``RGB/<id>.png`` and ``Label/<id>.png`` at 480x480 (RGB bilinear, stored label
  0..13 with 0 unknown, nearest);
- ``Depth16/<id>.png`` at 1080x1080, z-depth * 512, 65535 where the panorama has
  no depth;
- ``Pose/<id>.json`` with ``camera_k_matrix`` (1080, half-pixel), a world-to-camera
  ``camera_rt_matrix`` and ``camera_location`` 0. The world frame is the panorama's
  own frame (z up), i.e. the panorama is taken as level; ``check_polar_crops.py
  reproduce`` measures how far that is from the dataset's gravity.

Sample IDs are ``<area>/camera_<uuid>_<room>_frame_<1000 + k>``, so the frozen
manifest pattern and the per-location grouping still apply. ``manifest.csv`` has
the frozen ``build_full_manifest.py`` columns (split ``train``) for
``relplus_variant_cache.py generate --manifest``; ``train.txt`` lists the IDs and
``crops.json`` the pose, hole share and class pixels of every crop.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import re
from pathlib import Path

import cv2
import numpy as np

import cross_projection as cp
import eval_pano_transfer as ev


TRAIN_AREAS = ("area_1", "area_2", "area_3", "area_4", "area_6")
PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"  # what the frozen generator accepts
PANO_PATTERN = re.compile(r"^camera_([0-9a-f]+)_(.+)_frame_equirectangular_domain$")
DEPTH_SIZE = 1080
IMAGE_SIZE = 480
FRAME_OFFSET = 1000
MISSING_DEPTH = 65535
NUM_STORED_CLASSES = 14
# The frozen generator refuses gravity within 0.001 deg of straight ahead (rel_plus/geometry.py),
# i.e. a camera looking exactly down; 89.5 keeps every crop well clear of it.
MAX_PITCH_DEG = 89.5


def stored_labels(semantic_bgr, lookup):
    """RGB-encoded instance index -> stored label 0..13 (0 unknown), as in the S2D Label PNGs."""
    indices = (
        semantic_bgr[:, :, 0].astype(np.uint32)
        + semantic_bgr[:, :, 1].astype(np.uint32) * 256
        + semantic_bgr[:, :, 2].astype(np.uint32) * 65536
    )
    stored = np.zeros(indices.shape, dtype=np.uint8)
    known = indices < len(lookup)
    stored[known] = lookup[indices[known]]
    return stored


def sample_rgb(rgb_erp, directions):
    """Bilinear ERP sample at unit directions, wrapping in longitude."""
    height, width = rgb_erp.shape[:2]
    phi = np.arctan2(directions[..., 0], directions[..., 1])
    theta = np.arcsin(np.clip(directions[..., 2], -1.0, 1.0))
    # Same pixel convention as ``cp.erp_directions``: column u sits at phi = u / width * 2pi - pi.
    map_x = ((phi + np.pi) / (2 * np.pi) * width).astype(np.float32)
    map_y = ((0.5 - theta / np.pi) * height).astype(np.float32)
    padded = np.concatenate([rgb_erp[:, -1:], rgb_erp, rgb_erp[:, :1]], axis=1)
    return cv2.remap(padded, map_x + 1, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def render(rgb_erp, label_erp, range_m, camera_to_world, fov):
    """RGB and stored label at 480, Stanford Depth16 at 1080, plus the 1080 K."""
    k_depth = cp.crop_intrinsics(DEPTH_SIZE, fov)
    rays_camera, rays_world = cp.crop_rays(DEPTH_SIZE, k_depth, camera_to_world)
    rows, columns = cp.erp_indices_for_directions(rays_world, *range_m.shape)
    depth = cp.render_crop_raw_depth(range_m, rows, columns, rays_camera)
    depth[depth == 0] = MISSING_DEPTH

    _, small_world = cp.crop_rays(IMAGE_SIZE, cp.crop_intrinsics(IMAGE_SIZE, fov), camera_to_world)
    rgb = sample_rgb(rgb_erp, small_world)
    rows, columns = cp.erp_indices_for_directions(small_world, *label_erp.shape)
    return rgb, label_erp[rows, columns], depth, k_depth


def pose_json(k_depth, camera_to_world, provenance):
    rotation = camera_to_world.T  # world-to-camera; the camera sits at the world origin
    return dict(
        provenance,
        camera_k_matrix=k_depth.tolist(),
        camera_rt_matrix=np.concatenate([rotation, np.zeros((3, 1))], axis=1).tolist(),
        camera_location=[0.0, 0.0, 0.0],
    )


def crop_plan(panoramas, total, seed, min_pitch, max_pitch, fov_range):
    """Per panorama, its crops' (index, yaw, pitch, fov); deterministic per sample ID."""
    base, extra = divmod(total, len(panoramas))
    plans = []
    for index, sample in enumerate(panoramas):
        count = base + (index < extra)
        digest = hashlib.sha256("{}:{}".format(seed, sample["sample_id"]).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        start = int(rng.integers(2))  # which sign comes first, so odd counts balance overall
        crops = []
        for k in range(count):
            sign = 1.0 if (k + start) % 2 == 0 else -1.0
            crops.append((
                k,
                float(rng.uniform(0.0, 360.0)),
                sign * float(rng.uniform(min_pitch, max_pitch)),
                float(rng.uniform(*fov_range)),
            ))
        plans.append((sample, crops))
    return plans


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError("failed to write {}".format(path))


def make_panorama(job):
    sample, crops, output_root, lookup = job
    area, stem = sample["sample_id"].split("/", 1)
    match = PANO_PATTERN.match(stem)
    if match is None:
        raise ValueError("unexpected panorama name: {}".format(stem))
    camera, room = match.groups()
    rgb_erp = cv2.imread(str(sample["rgb"]), cv2.IMREAD_COLOR)  # BGR on disk and in memory
    raw = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
    semantic = cv2.imread(str(sample["semantic"]), cv2.IMREAD_COLOR)
    if rgb_erp is None or semantic is None or raw is None or raw.dtype != np.uint16:
        raise ValueError("cannot read panorama {}".format(sample["sample_id"]))
    range_m = cp.decode_erp_depth(raw)
    label_erp = stored_labels(semantic, lookup)
    if rgb_erp.shape[:2] != raw.shape:
        rgb_erp = cv2.resize(rgb_erp, raw.shape[::-1], interpolation=cv2.INTER_AREA)
    if label_erp.shape != raw.shape:
        label_erp = cv2.resize(label_erp, raw.shape[::-1], interpolation=cv2.INTER_NEAREST)

    rows = []
    for k, yaw, pitch, fov in crops:
        frame = FRAME_OFFSET + k
        sample_id = "{}/camera_{}_{}_frame_{}".format(area, camera, room, frame)
        camera_to_world = cp.crop_camera_to_world(yaw, pitch)
        rgb, label, depth, k_depth = render(rgb_erp, label_erp, range_m, camera_to_world, fov)
        paths = {
            "rgb_path": output_root / "RGB" / (sample_id + ".png"),
            "label_path": output_root / "Label" / (sample_id + ".png"),
            "depth_path": output_root / "Depth16" / (sample_id + ".png"),
            "camera_metadata_path": output_root / "Pose" / (sample_id + ".json"),
        }
        write_png(paths["rgb_path"], rgb)
        write_png(paths["label_path"], label)
        write_png(paths["depth_path"], depth)
        provenance = {"polar_crop": {
            "panorama": sample["sample_id"], "yaw_deg": yaw, "pitch_deg": pitch, "hfov_deg": fov,
            "world_frame": "panorama (taken as level)",
        }}
        paths["camera_metadata_path"].parent.mkdir(parents=True, exist_ok=True)
        paths["camera_metadata_path"].write_text(
            json.dumps(pose_json(k_depth, camera_to_world, provenance), indent=2) + "\n", encoding="utf-8"
        )
        rows.append({
            "manifest": dict(
                sample_id=sample_id, split="train", area=area,
                area_group=area, room=room, camera=camera, frame=frame,
                **{key: str(value) for key, value in paths.items()},
                dataset_profile="stanford2d3d_s2d", protocol_id=PROTOCOL_ID,
            ),
            "crop": {
                "sample_id": sample_id, "panorama": sample["sample_id"],
                "yaw_deg": yaw, "pitch_deg": pitch, "hfov_deg": fov,
                "missing_depth_share": float(np.mean(depth == MISSING_DEPTH)),
                "unlabelled_share": float(np.mean(label == 0)),
                "black_rgb_share": float(np.mean((rgb == 0).all(axis=2))),
                "stored_class_pixels": np.bincount(label.ravel(), minlength=NUM_STORED_CLASSES).tolist(),
            },
        })
    return rows


def summarize(crops):
    def part(items):
        if not items:
            return None
        pixels = np.sum([c["stored_class_pixels"] for c in items], axis=0)
        return {
            "crops": len(items),
            "pitch_deg_range": [min(c["pitch_deg"] for c in items), max(c["pitch_deg"] for c in items)],
            "missing_depth_share_mean": float(np.mean([c["missing_depth_share"] for c in items])),
            "unlabelled_share_mean": float(np.mean([c["unlabelled_share"] for c in items])),
            "black_rgb_share_mean": float(np.mean([c["black_rgb_share"] for c in items])),
            "class_pixel_share": {
                name: float(pixels[index] / pixels.sum())
                for index, name in enumerate(["unknown"] + list(ev.EXPECTED_CLASSES[1:]))
            },
        }

    return {
        "all": part(crops),
        "up": part([c for c in crops if c["pitch_deg"] > 0]),
        "down": part([c for c in crops if c["pitch_deg"] < 0]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stanford-root", required=True, type=Path, help="holds area_*/pano/{rgb,depth,semantic}")
    parser.add_argument("--semantic-labels", required=True, type=Path, help="assets/semantic_labels.json")
    parser.add_argument("--output-root", required=True, type=Path, help="new, empty folder")
    parser.add_argument("--areas", nargs="+", default=list(TRAIN_AREAS))
    parser.add_argument("--total-crops", type=int, default=7935, help="the 15%% the greedy subset drops")
    parser.add_argument("--min-pitch", type=float, default=55.0)
    parser.add_argument("--max-pitch", type=float, default=MAX_PITCH_DEG)
    parser.add_argument("--fov-range", type=float, nargs=2, default=(45.0, 75.0))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit-panoramas", type=int, default=None, help="first N panoramas only (checks)")
    parser.add_argument("--manifest", type=Path, default=None, help="frozen full_manifest.csv: refuse ID clashes")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    bad = sorted(set(args.areas) - set(TRAIN_AREAS))
    if bad:
        parser.error("only training areas {} are allowed, got {}".format(", ".join(TRAIN_AREAS), ", ".join(bad)))
    if not 0.0 <= args.min_pitch <= args.max_pitch <= MAX_PITCH_DEG:
        parser.error("need 0 <= --min-pitch <= --max-pitch <= {:g}".format(MAX_PITCH_DEG))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output_root))

    panoramas = ev.list_panoramas(args.stanford_root, args.areas)[: args.limit_panoramas]
    if args.total_crops < 1:
        parser.error("--total-crops must be positive")
    plans = crop_plan(panoramas, args.total_crops, args.seed, args.min_pitch, args.max_pitch, args.fov_range)
    lookup = ev.label_lookup(args.semantic_labels)
    jobs = [(sample, crops, args.output_root, lookup) for sample, crops in plans if crops]
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(make_panorama, jobs, chunksize=1)
    else:
        results = [make_panorama(job) for job in jobs]
    rows = [row for result in results for row in result]

    ids = [row["manifest"]["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("two crops share a sample ID")
    if args.manifest is not None:
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            clashes = set(ids) & {row["sample_id"] for row in csv.DictReader(handle)}
        if clashes:
            raise RuntimeError("{} crop IDs are already in {}".format(len(clashes), args.manifest))
    with (args.output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]["manifest"]))
        writer.writeheader()
        writer.writerows(row["manifest"] for row in rows)
    (args.output_root / "train.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    crops = [row["crop"] for row in rows]
    report = {
        "settings": {
            "stanford_root": str(args.stanford_root.resolve()), "areas": args.areas,
            "panoramas": len(panoramas), "total_crops": len(crops), "seed": args.seed,
            "pitch_abs_deg": [args.min_pitch, args.max_pitch], "hfov_deg": list(args.fov_range),
            "image_size": IMAGE_SIZE, "depth_size": DEPTH_SIZE,
            "world_frame": "panorama frame taken as level",
        },
        "summary": summarize(crops),
        "crops": crops,
    }
    (args.output_root / "crops.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"crops": len(crops), "output": str(args.output_root.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
