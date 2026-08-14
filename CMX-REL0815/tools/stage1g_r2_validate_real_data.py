#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import Imath
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
from PIL import Image, ImageDraw


AREAS = ("area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6")
SEED = 20260805
DATA_ROOT = Path("/data/zhuzhaoziao/datasets/Stanford2D3D/with_xyz")
REPO_ROOT = Path("/home/zhuzhaoziao/rel_exp/cmx_rel+")
R1_ROOT = Path("/data/zhuzhaoziao/cmx/outputs/stage1g_r1_production_repair_20260805_114851")
REFERENCE_PATH = Path(
    "/data/zhuzhaoziao/cmx/outputs/s1g_native_relplus_channels_20260805T084027Z/"
    "code/stage1g_reference_rel.py"
)
RGB_RE = re.compile(
    r"^camera_([0-9a-f]{32})_(.+)_frame_([0-9]+)_domain_rgb[.]png$"
)
GEOMETRY_P95_TOLERANCE_M = 1.0 / 512.0


class Blocked(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_rgb_name(name: str) -> tuple[str, str, int]:
    match = RGB_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid RGB filename: {name}")
    return match.group(1), match.group(2), int(match.group(3))


def enumerate_complete_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for area in AREAS:
        rgb_dir = DATA_ROOT / area / "data/rgb"
        if not rgb_dir.is_dir():
            raise Blocked(f"missing RGB directory: {rgb_dir}")
        for rgb_path in sorted(rgb_dir.glob("camera_*_domain_rgb.png")):
            try:
                camera_id, room, frame_number = parse_rgb_name(rgb_path.name)
            except ValueError:
                continue
            stem = f"camera_{camera_id}_{room}_frame_{frame_number}_domain"
            paths = {
                "rgb_path": rgb_path,
                "depth_path": DATA_ROOT / area / "data/depth" / f"{stem}_depth.png",
                "pose_path": DATA_ROOT / area / "data/pose" / f"{stem}_pose.json",
                "global_xyz_path": DATA_ROOT / area / "data/global_xyz" / f"{stem}_global_xyz.exr",
            }
            if not all(path.is_file() for path in paths.values()):
                continue
            try:
                pose = json.loads(paths["pose_path"].read_text(encoding="utf-8"))
            except Exception:
                continue
            if pose.get("camera_uuid") != camera_id or int(pose.get("frame_num", -1)) != frame_number:
                continue
            candidates.append(
                {
                    "area": area,
                    "room": room,
                    "camera_id": camera_id,
                    "frame_id": f"{area}:{camera_id}:{frame_number}",
                    "frame_number": frame_number,
                    **{key: str(value) for key, value in paths.items()},
                }
            )
    return candidates


def select_four_per_area(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for area_index, area in enumerate(AREAS):
        area_rows = sorted(
            (dict(row) for row in candidates if row["area"] == area),
            key=lambda row: (
                row["camera_id"], row["room"],
                int(row.get("frame_number", row.get("frame_id", 0))),
            ),
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in area_rows:
            grouped[row["camera_id"]].append(row)
        if len(grouped) < 4:
            raise Blocked(f"{area} has only {len(grouped)} complete distinct cameras")
        rng = random.Random(seed + 1009 * area_index)
        camera_ids = sorted(grouped)
        rng.shuffle(camera_ids)
        one_per_camera: list[dict[str, Any]] = []
        for camera_id in camera_ids:
            rows = grouped[camera_id]
            one_per_camera.append(dict(rows[rng.randrange(len(rows))]))
        chosen: list[dict[str, Any]] = []
        used_rooms: set[str] = set()
        for row in one_per_camera:
            if row["room"] not in used_rooms:
                chosen.append(row)
                used_rooms.add(row["room"])
            if len(chosen) == 4:
                break
        if len(chosen) < 4:
            used_cameras = {row["camera_id"] for row in chosen}
            for row in one_per_camera:
                if row["camera_id"] not in used_cameras:
                    chosen.append(row)
                    used_cameras.add(row["camera_id"])
                if len(chosen) == 4:
                    break
        if len(chosen) != 4 or len({row["camera_id"] for row in chosen}) != 4:
            raise Blocked(f"unable to select four distinct cameras in {area}")
        selected.extend(chosen)
    return selected


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def append_log(root: Path, message: str) -> None:
    with (root / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_reference():
    spec = importlib.util.spec_from_file_location("stage1g_r2_reference_rel", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise Blocked(f"cannot load reference: {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_xyz(path: Path) -> np.ndarray:
    source = OpenEXR.InputFile(str(path))
    try:
        window = source.header()["dataWindow"]
        width = int(window.max.x - window.min.x + 1)
        height = int(window.max.y - window.min.y + 1)
        payload = source.channels(["R", "G", "B"], Imath.PixelType(Imath.PixelType.FLOAT))
    finally:
        source.close()
    arrays = [np.frombuffer(value, dtype=np.float32).reshape(height, width) for value in payload]
    return np.stack(arrays, axis=-1).astype(np.float64, copy=False)


def load_frame(row: dict[str, str]) -> dict[str, Any]:
    rgb = np.asarray(Image.open(row["rgb_path"]).convert("RGB"))
    raw_depth = cv2.imread(row["depth_path"], cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise Blocked(f"cannot read depth: {row['depth_path']}")
    xyz_world = decode_xyz(Path(row["global_xyz_path"]))
    pose = json.loads(Path(row["pose_path"]).read_text(encoding="utf-8"))
    k = np.asarray(pose["camera_k_matrix"], dtype=np.float64)
    rt = np.asarray(pose["camera_rt_matrix"], dtype=np.float64)
    if rgb.shape != (1080, 1080, 3) or raw_depth.shape != (1080, 1080):
        raise Blocked(f"native RGB/depth shape mismatch for {row['frame_id']}")
    if xyz_world.shape != (1080, 1080, 3) or k.shape != (3, 3) or rt.shape != (3, 4):
        raise Blocked(f"native XYZ/K/pose shape mismatch for {row['frame_id']}")
    if raw_depth.dtype != np.uint16 or not np.isfinite(k).all() or not np.isfinite(rt).all():
        raise Blocked(f"native dtype/nonfinite metadata mismatch for {row['frame_id']}")
    depth = raw_depth.astype(np.float64) / 512.0
    depth_valid = (raw_depth != 65535) & (raw_depth > 0) & np.isfinite(depth)
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    points_camera = np.stack(
        (
            (columns + 0.5 - k[0, 2]) * depth / k[0, 0],
            (rows + 0.5 - k[1, 2]) * depth / k[1, 1],
            depth,
        ),
        axis=-1,
    )
    return {
        "rgb": rgb,
        "raw_depth": raw_depth,
        "depth": depth,
        "depth_valid": depth_valid,
        "xyz_world": xyz_world,
        "k": k,
        "rotation_w2c": rt[:, :3],
        "translation_w2c": rt[:, 3],
        "points_camera": points_camera,
    }


def stable_joint_indices(mask: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(ys) < count:
        raise Blocked(f"only {len(ys)} joint-valid pixels")
    indices = np.linspace(0, len(ys) - 1, count, dtype=np.int64)
    return ys[indices], xs[indices]


def roundtrip_png(
    rel: np.ndarray, valid: np.ndarray, directory: Path, stem: str
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    rel_path = directory / f"{stem}_rel.png"
    mask_path = directory / f"{stem}_valid.png"
    Image.fromarray(np.asarray(rel, dtype=np.uint8)).save(rel_path)
    Image.fromarray(np.asarray(valid, dtype=np.uint8) * 255).save(mask_path)
    loaded_rel = np.asarray(Image.open(rel_path).convert("RGB"))
    loaded_valid = np.asarray(Image.open(mask_path).convert("L")) > 0
    rel_mismatch = int(np.count_nonzero(loaded_rel != rel))
    mask_mismatch = int(np.count_nonzero(loaded_valid != valid))
    return {
        "rel_path": str(rel_path),
        "mask_path": str(mask_path),
        "rel_mismatch_count": rel_mismatch,
        "mask_mismatch_count": mask_mismatch,
        "shape_preserved": loaded_rel.shape == rel.shape and loaded_valid.shape == valid.shape,
        "dtype_preserved": loaded_rel.dtype == np.uint8,
        "channel_order_preserved": rel_mismatch == 0,
        "invalid_triplet_preserved": bool(
            np.all(loaded_rel[~loaded_valid] == np.array([255, 255, 255], dtype=np.uint8))
        ),
    }


def histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    total = int(histogram.sum())
    if total == 0:
        return math.nan
    target = quantile * (total - 1)
    cumulative = np.cumsum(histogram)
    return float(np.searchsorted(cumulative, target + 1, side="left"))


def channel_stats(values: np.ndarray) -> dict[str, Any]:
    histogram = np.bincount(values.astype(np.uint8), minlength=256)
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(values.max()),
        "value_0_ratio": float(np.mean(values == 0)),
        "value_255_ratio": float(np.mean(values == 255)),
        "histogram": histogram,
    }


def choose_probe_pixels(valid: np.ndarray, depth: np.ndarray) -> list[tuple[int, int]]:
    targets = [
        (540, 541), (270, 270), (270, 540), (270, 810), (540, 270),
        (540, 810), (810, 270), (810, 540), (810, 810),
    ]
    chosen: list[tuple[int, int]] = []
    for target_y, target_x in targets:
        found = None
        for radius in range(0, 121):
            y0, y1 = max(0, target_y - radius), min(1080, target_y + radius + 1)
            x0, x1 = max(0, target_x - radius), min(1080, target_x + radius + 1)
            ys, xs = np.nonzero(valid[y0:y1, x0:x1])
            if len(ys):
                distances = (ys + y0 - target_y) ** 2 + (xs + x0 - target_x) ** 2
                order = np.lexsort((xs, ys, distances))
                found = (int(ys[order[0]] + y0), int(xs[order[0]] + x0))
                break
        if found is not None and found not in chosen:
            chosen.append(found)
    sampled_valid = valid[::8, ::8]
    sampled_depth = depth[::8, ::8]
    ys, xs = np.nonzero(sampled_valid)
    if len(ys):
        values = sampled_depth[ys, xs]
        for index in (int(np.argmin(values)), int(np.argmax(values))):
            point = (int(ys[index] * 8), int(xs[index] * 8))
            if point not in chosen:
                chosen.append(point)
    gradient = cv2.magnitude(
        cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0),
        cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1),
    )
    gradient[~valid] = -1.0
    edge = tuple(int(value) for value in np.unravel_index(int(np.argmax(gradient)), gradient.shape))
    if edge not in chosen:
        chosen.append(edge)
    if len(chosen) < 12:
        ys, xs = np.nonzero(valid[::16, ::16])
        for y, x in zip(ys, xs):
            point = (int(y * 16), int(x * 16))
            if point not in chosen:
                chosen.append(point)
            if len(chosen) == 12:
                break
    return chosen[:12]


def save_frame_visualization(
    path: Path,
    title: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    rel: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(rgb); axes[0].set_title("RGB")
    depth_show = np.where(depth > 0, depth, np.nan)
    axes[1].imshow(depth_show, cmap="magma"); axes[1].set_title("Z depth (m)")
    normal_show = np.clip((normals + 1.0) / 2.0, 0.0, 1.0)
    normal_show[~valid] = 0.0
    axes[2].imshow(normal_show); axes[2].set_title("camera normal")
    axes[3].imshow(valid, cmap="gray", vmin=0, vmax=1); axes[3].set_title("rel_valid")
    for index, name in enumerate(("ReD", "EGVIA", "LOA"), start=4):
        axes[index].imshow(rel[..., index - 4], cmap="viridis", vmin=0, vmax=255)
        axes[index].set_title(f"{name} [0,255]")
    overlay = rgb.copy()
    overlay[~valid] = np.array([255, 0, 0], dtype=np.uint8)
    axes[7].imshow(overlay); axes[7].set_title("invalid overlay (red)")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def create_contact_sheets(root: Path, frame_paths: list[Path], rows: list[dict[str, str]]) -> None:
    thumbnails = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((480, 240))
        thumbnails.append(image.copy())
    sheet = Image.new("RGB", (480 * 4, 240 * 6), "white")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % 4) * 480, (index // 4) * 240))
    sheet.save(root / "visualizations/all_frames_contact_sheet.png")
    area_images = []
    for area in AREAS:
        index = next(i for i, row in enumerate(rows) if row["area"] == area)
        image = thumbnails[index].copy()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 150, 24), fill="white")
        draw.text((5, 5), area, fill="black")
        area_images.append(image)
    area_sheet = Image.new("RGB", (480 * 2, 240 * 3), "white")
    for index, image in enumerate(area_images):
        area_sheet.paste(image, ((index % 2) * 480, (index // 2) * 240))
    area_sheet.save(root / "visualizations/area_summary.png")


def select_phase(root: Path) -> None:
    if not root.is_dir():
        raise Blocked(f"output root does not exist: {root}")
    r1 = json.loads((R1_ROOT / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    if r1.get("final_status") != "PASS_PRODUCTION_ENCODER_SYNTHETIC_CONFORMANCE":
        raise Blocked("R1 PASS prerequisite is absent")
    candidates = enumerate_complete_candidates()
    selected = select_four_per_area(candidates, SEED)
    for index, row in enumerate(selected):
        row["selection_index"] = index
        row["selection_seed"] = SEED
    manifest_fields = [
        "selection_index", "selection_seed", "area", "room", "camera_id", "frame_id",
        "frame_number", "rgb_path", "depth_path", "pose_path", "global_xyz_path",
    ]
    write_csv(root / "stage1g_r2_sample_manifest.csv", manifest_fields, selected)
    counts = {area: len({row["camera_id"] for row in selected if row["area"] == area}) for area in AREAS}
    if any(value != 4 for value in counts.values()):
        raise Blocked(f"selection quota failure: {counts}")
    environment = f"""# Environment and entrypoints

- Python: `{sys.executable}` / `{platform.python_version()}`
- NumPy: `{np.__version__}`; OpenCV: `{cv2.__version__}`
- Production: `{REPO_ROOT / 'relplus/representation.py'}::encode_relplus_channels`
- Shared normal estimator: `{REPO_ROOT / 'relplus/representation.py'}::estimate_rel_normals`
- Independent reference: `{REFERENCE_PATH}::generate_rel`
- R1 prerequisite: `{R1_ROOT}` = `PASS_PRODUCTION_ENCODER_SYNTHETIC_CONFORMANCE`
- Data: `{DATA_ROOT}`
- Frozen areas: `{', '.join(AREAS)}`; seed: `{SEED}`
- Native contract: 1080x1080, depth/512 Z-depth, pixel center 0.5, W2C pose.
"""
    (root / "environment_and_entrypoints.md").write_text(environment, encoding="utf-8")
    append_log(root, f"sample_manifest_saved candidates={len(candidates)} selected=24 counts={counts}")


def validate_phase(root: Path) -> None:
    manifest_path = root / "stage1g_r2_sample_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24 or any(sum(row["area"] == area for row in rows) != 4 for area in AREAS):
        raise Blocked("sample manifest is not the frozen six-area 24-frame set")
    if any(len({row["camera_id"] for row in rows if row["area"] == area}) != 4 for area in AREAS):
        raise Blocked("sample manifest repeats a camera within an area")

    reference = load_reference()
    from relplus.representation import encode_relplus_channels, estimate_rel_normals

    input_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    statistics_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    roundtrip_rows: list[dict[str, Any]] = []
    frame_visual_paths: list[Path] = []
    area_histograms = {area: [np.zeros(256, dtype=np.int64) for _ in range(3)] for area in AREAS}
    area_valid = defaultdict(int)
    area_pixels = defaultdict(int)

    for frame_index, row in enumerate(rows):
        append_log(root, f"frame_start index={frame_index} id={row['frame_id']}")
        data = load_frame(row)
        depth_valid_count = int(np.count_nonzero(data["depth_valid"]))
        if depth_valid_count == 0:
            raise Blocked(f"no valid depth: {row['frame_id']}")
        rotation = data["rotation_w2c"]
        orth_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        if orth_error > 1e-5 or abs(determinant - 1.0) > 1e-5:
            raise Blocked(f"invalid W2C rotation: {row['frame_id']}")
        input_rows.append(
            {
                "area": row["area"], "room": row["room"], "camera_id": row["camera_id"],
                "frame_id": row["frame_id"], "rgb_shape": "1080x1080x3",
                "depth_shape": "1080x1080", "global_xyz_shape": "1080x1080x3",
                "depth_dtype": str(data["raw_depth"].dtype), "k_readable": True,
                "pose_readable": True, "depth_valid_count": depth_valid_count,
                "depth_valid_ratio": depth_valid_count / data["depth_valid"].size,
            }
        )

        xyz_valid = np.isfinite(data["xyz_world"]).all(axis=-1) & ~np.all(data["xyz_world"] == 0.0, axis=-1)
        joint = data["depth_valid"] & xyz_valid
        ys, xs = stable_joint_indices(joint, 512)
        xyz_camera = data["xyz_world"][ys, xs] @ rotation.T + data["translation_w2c"]
        point_camera = data["points_camera"][ys, xs]
        component_error = np.abs(xyz_camera - point_camera)
        p95 = np.quantile(component_error, 0.95, axis=0)
        geometry_pass = bool(np.all(p95 <= GEOMETRY_P95_TOLERANCE_M))
        geometry_rows.append(
            {
                "area": row["area"], "frame_id": row["frame_id"], "probe_count": 512,
                "median_abs_x_m": float(np.median(component_error[:, 0])),
                "median_abs_y_m": float(np.median(component_error[:, 1])),
                "median_abs_z_m": float(np.median(component_error[:, 2])),
                "p95_abs_x_m": float(p95[0]), "p95_abs_y_m": float(p95[1]),
                "p95_abs_z_m": float(p95[2]),
                "max_euclidean_m": float(np.max(np.linalg.norm(component_error, axis=1))),
                "p95_tolerance_m": GEOMETRY_P95_TOLERANCE_M,
                "geometry_wiring_pass": geometry_pass,
            }
        )
        if not geometry_pass:
            raise Blocked(f"geometry wiring exceeded frozen component P95 tolerance: {row['frame_id']}")

        normals_camera, normal_valid = estimate_rel_normals(
            data["points_camera"], data["depth_valid"], radius=3
        )
        gravity_camera = rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        ref = reference.generate_rel(
            np.ascontiguousarray(data["points_camera"], dtype=np.float64),
            np.ascontiguousarray(normals_camera, dtype=np.float64),
            np.ascontiguousarray(gravity_camera, dtype=np.float64),
            np.ascontiguousarray(data["depth_valid"], dtype=bool),
            np.ascontiguousarray(normal_valid, dtype=bool),
        )
        align_rotation = reference.shortest_arc_rotation_to_down(gravity_camera)
        points_aligned = np.ascontiguousarray(data["points_camera"] @ align_rotation.T)
        normals_aligned = np.ascontiguousarray(normals_camera @ align_rotation.T)
        production_rel, production_aux = encode_relplus_channels(
            points_aligned, normals_aligned, data["depth_valid"] & normal_valid
        )
        production_valid = np.asarray(production_aux["valid"], dtype=bool)
        reference_rel = ref.rel_uint8
        reference_valid = ref.rel_valid
        differences = np.abs(production_rel.astype(np.int16) - reference_rel.astype(np.int16))
        mismatch_by_channel = [int(np.count_nonzero(differences[..., index])) for index in range(3)]
        mask_mismatch = int(np.count_nonzero(production_valid != reference_valid))
        comparison_rows.append(
            {
                "area": row["area"], "room": row["room"], "camera_id": row["camera_id"],
                "frame_id": row["frame_id"], "red_mismatch_count": mismatch_by_channel[0],
                "egvia_mismatch_count": mismatch_by_channel[1], "loa_mismatch_count": mismatch_by_channel[2],
                "total_mismatch_count": sum(mismatch_by_channel),
                "max_uint8_difference": int(differences.max()),
                "mean_absolute_difference": float(differences.mean()),
                "mask_mismatch_count": mask_mismatch,
            }
        )
        mismatch_locations = np.argwhere(np.any(differences != 0, axis=-1) | (production_valid != reference_valid))
        for y, x in mismatch_locations[:20]:
            mismatch_rows.append(
                {
                    "area": row["area"], "frame_id": row["frame_id"], "x": int(x), "y": int(y),
                    "reference_valid": bool(reference_valid[y, x]),
                    "production_valid": bool(production_valid[y, x]),
                    "reference_red": int(reference_rel[y, x, 0]), "production_red": int(production_rel[y, x, 0]),
                    "reference_egvia": int(reference_rel[y, x, 1]), "production_egvia": int(production_rel[y, x, 1]),
                    "reference_loa": int(reference_rel[y, x, 2]), "production_loa": int(production_rel[y, x, 2]),
                    "reference_red_float": float(ref.diagnostics.ReD_01[y, x]),
                    "production_red_float": float(production_aux["red_01"][y, x]),
                    "reference_egvia_float": float(ref.diagnostics.EGVIA_01[y, x]),
                    "production_egvia_float": float(production_aux["egvia_01"][y, x]),
                    "reference_loa_float": float(ref.diagnostics.LOA_01[y, x]),
                    "production_loa_float": float(production_aux["loa_01"][y, x]),
                }
            )

        valid_count = int(np.count_nonzero(reference_valid))
        area_valid[row["area"]] += valid_count
        area_pixels[row["area"]] += reference_valid.size
        float_arrays = [
            ref.diagnostics.ReD_01, ref.diagnostics.H_01, ref.diagnostics.A_01,
            ref.diagnostics.EGVIA_01, ref.diagnostics.LOA_01,
            production_aux["red_01"], production_aux["height_01"], production_aux["angle_01"],
            production_aux["egvia_01"], production_aux["loa_01"],
        ]
        nonfinite_count = sum(int(np.count_nonzero(~np.isfinite(value[reference_valid]))) for value in float_arrays)
        for channel_index, channel_name in enumerate(("ReD", "EGVIA", "LOA")):
            values = production_rel[..., channel_index][reference_valid]
            stats = channel_stats(values)
            area_histograms[row["area"]][channel_index] += stats.pop("histogram")
            statistics_rows.append(
                {
                    "area": row["area"], "room": row["room"], "frame_id": row["frame_id"],
                    "channel": channel_name, "valid_pixel_count": valid_count,
                    "valid_ratio": valid_count / reference_valid.size,
                    **stats,
                }
            )
        red = np.hypot(points_aligned[..., 0], points_aligned[..., 1])
        rho_invalid = data["depth_valid"] & normal_valid & ~(red > 1e-12)
        invalid_rows.append(
            {
                "area": row["area"], "frame_id": row["frame_id"],
                "total_pixel_count": reference_valid.size,
                "depth_invalid_count": int(np.count_nonzero(~data["depth_valid"])),
                "depth_valid_ratio": depth_valid_count / reference_valid.size,
                "normal_invalid_count_with_valid_depth": int(np.count_nonzero(data["depth_valid"] & ~normal_valid)),
                "normal_valid_ratio": int(np.count_nonzero(normal_valid)) / reference_valid.size,
                "rho_invalid_count": int(np.count_nonzero(rho_invalid)),
                "rho_invalid_ratio": int(np.count_nonzero(rho_invalid)) / reference_valid.size,
                "rel_valid_count": valid_count, "rel_valid_ratio": valid_count / reference_valid.size,
                "non_finite_valid_float_count": nonfinite_count,
            }
        )

        for probe_index, (y, x) in enumerate(choose_probe_pixels(reference_valid, data["depth"])):
            probe_rows.append(
                {
                    "area": row["area"], "room": row["room"], "frame_id": row["frame_id"],
                    "probe_index": probe_index, "x": x, "y": y, "depth": float(data["depth"][y, x]),
                    "camera_xyz_x": float(data["points_camera"][y, x, 0]),
                    "camera_xyz_y": float(data["points_camera"][y, x, 1]),
                    "camera_xyz_z": float(data["points_camera"][y, x, 2]),
                    "normal_x": float(normals_camera[y, x, 0]), "normal_y": float(normals_camera[y, x, 1]),
                    "normal_z": float(normals_camera[y, x, 2]),
                    "gravity_x": float(gravity_camera[0]), "gravity_y": float(gravity_camera[1]),
                    "gravity_z": float(gravity_camera[2]),
                    "red_float": float(ref.diagnostics.ReD_01[y, x]),
                    "egvia_float": float(ref.diagnostics.EGVIA_01[y, x]),
                    "loa_float": float(ref.diagnostics.LOA_01[y, x]),
                    "reference_red_uint8": int(reference_rel[y, x, 0]),
                    "reference_egvia_uint8": int(reference_rel[y, x, 1]),
                    "reference_loa_uint8": int(reference_rel[y, x, 2]),
                    "production_red_uint8": int(production_rel[y, x, 0]),
                    "production_egvia_uint8": int(production_rel[y, x, 1]),
                    "production_loa_uint8": int(production_rel[y, x, 2]),
                    "rel_valid": bool(reference_valid[y, x]),
                }
            )

        roundtrip = roundtrip_png(production_rel, production_valid, root / "roundtrip", f"frame_{frame_index:02d}")
        roundtrip_rows.append({"area": row["area"], "frame_id": row["frame_id"], **roundtrip})
        visual_path = root / "visualizations" / f"frame_{frame_index:02d}_{row['area']}.png"
        save_frame_visualization(
            visual_path, row["frame_id"], data["rgb"], data["depth"],
            normals_camera, reference_valid, production_rel,
        )
        frame_visual_paths.append(visual_path)
        append_log(root, f"frame_complete index={frame_index} mismatch={sum(mismatch_by_channel)} mask={mask_mismatch}")

    area_rows: list[dict[str, Any]] = []
    for area in AREAS:
        for channel_index, channel_name in enumerate(("ReD", "EGVIA", "LOA")):
            hist = area_histograms[area][channel_index]
            total = int(hist.sum())
            values = np.arange(256, dtype=np.float64)
            area_rows.append(
                {
                    "area": area, "channel": channel_name, "valid_pixel_count": total,
                    "valid_ratio": area_valid[area] / area_pixels[area],
                    "min": int(np.flatnonzero(hist)[0]), "median": histogram_quantile(hist, 0.5),
                    "mean": float(np.sum(hist * values) / total), "p95": histogram_quantile(hist, 0.95),
                    "max": int(np.flatnonzero(hist)[-1]), "value_0_ratio": float(hist[0] / total),
                    "value_255_ratio": float(hist[255] / total),
                }
            )

    create_contact_sheets(root, frame_visual_paths, rows)
    write_csv(root / "frame_input_summary.csv", list(input_rows[0]), input_rows)
    write_csv(root / "geometry_wiring_summary.csv", list(geometry_rows[0]), geometry_rows)
    comparison_fields = list(comparison_rows[0])
    write_csv(root / "paired_comparison_summary.csv", comparison_fields, comparison_rows)
    mismatch_fields = [
        "area", "frame_id", "x", "y", "reference_valid", "production_valid",
        "reference_red", "production_red", "reference_egvia", "production_egvia",
        "reference_loa", "production_loa", "reference_red_float", "production_red_float",
        "reference_egvia_float", "production_egvia_float", "reference_loa_float", "production_loa_float",
    ]
    write_csv(root / "paired_mismatch_examples.csv", mismatch_fields, mismatch_rows)
    write_csv(root / "rel_channel_statistics.csv", list(statistics_rows[0]), statistics_rows)
    write_csv(root / "area_channel_summary.csv", list(area_rows[0]), area_rows)
    write_csv(root / "invalid_reason_summary.csv", list(invalid_rows[0]), invalid_rows)
    write_csv(root / "pixel_probe_results.csv", list(probe_rows[0]), probe_rows)
    write_csv(root / "roundtrip_results.csv", list(roundtrip_rows[0]), roundtrip_rows)

    mismatch_total = sum(int(row["total_mismatch_count"]) for row in comparison_rows)
    mask_mismatch_total = sum(int(row["mask_mismatch_count"]) for row in comparison_rows)
    max_difference = max(int(row["max_uint8_difference"]) for row in comparison_rows)
    nonfinite_total = sum(int(row["non_finite_valid_float_count"]) for row in invalid_rows)
    roundtrip_mismatch = sum(
        int(row["rel_mismatch_count"]) + int(row["mask_mismatch_count"])
        for row in roundtrip_rows
    )
    constant_channel_frames = sum(
        int(row["min"] == row["max"]) for row in statistics_rows
    )
    extreme_saturation_frames = sum(
        int(max(float(row["value_0_ratio"]), float(row["value_255_ratio"])) > 0.98)
        for row in statistics_rows
    )
    numeric_pass = (
        mismatch_total == 0 and mask_mismatch_total == 0 and max_difference == 0
        and nonfinite_total == 0 and roundtrip_mismatch == 0
        and constant_channel_frames == 0 and extreme_saturation_frames == 0
        and all(row["geometry_wiring_pass"] for row in geometry_rows)
    )
    result = {
        "stage": "Stage1G-R2", "area_scope": list(AREAS), "frames_total": 24,
        "frames_completed": 24, "reference_production_uint8_mismatch_total": mismatch_total,
        "reference_production_mask_mismatch_total": mask_mismatch_total,
        "max_uint8_difference": max_difference,
        "non_finite_valid_float_count": nonfinite_total,
        "constant_channel_frame_count": constant_channel_frames,
        "extreme_saturation_channel_frame_count": extreme_saturation_frames,
        "roundtrip_mismatch_total": roundtrip_mismatch,
        "visualizations_generated": len(frame_visual_paths),
        "visual_review_status": "PENDING_CODEX_REVIEW",
        "numeric_validation_pass": numeric_pass,
        "stage1g_r2_status": "INCONCLUSIVE_PENDING_VISUAL_REVIEW" if numeric_pass else "FAIL_STAGE1G_R2_REAL_DATA_RELPLUS_VALIDATION",
        "stage1g_closed": False, "stage1h_authorized": False, "training_authorized": False,
    }
    (root / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    append_log(root, f"numeric_complete status={result['stage1g_r2_status']} mismatch={mismatch_total} roundtrip={roundtrip_mismatch}")


def finalize_phase(root: Path, visual_status: str) -> None:
    result_path = root / "FINAL_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if visual_status not in {"PASS", "FAIL", "PENDING_HUMAN_REVIEW"}:
        raise ValueError("invalid visual status")
    result["visual_review_status"] = {
        "PASS": "PASS_NO_SYSTEMATIC_VISUAL_ANOMALY",
        "FAIL": "FAIL_SYSTEMATIC_VISUAL_ANOMALY",
        "PENDING_HUMAN_REVIEW": "PENDING_HUMAN_VISUAL_CONFIRMATION",
    }[visual_status]
    if not result.get("numeric_validation_pass"):
        result["stage1g_r2_status"] = "FAIL_STAGE1G_R2_REAL_DATA_RELPLUS_VALIDATION"
    elif visual_status == "PASS":
        result["stage1g_r2_status"] = "PASS_STAGE1G_R2_REAL_DATA_RELPLUS_VALIDATION"
        result["stage1g_closed"] = True
        result["stage1h_authorized"] = True
    elif visual_status == "FAIL":
        result["stage1g_r2_status"] = "FAIL_STAGE1G_R2_REAL_DATA_RELPLUS_VALIDATION"
    else:
        result["stage1g_r2_status"] = "INCONCLUSIVE_PENDING_HUMAN_VISUAL_REVIEW"
    result["training_authorized"] = False
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    append_log(root, f"finalized visual={visual_status} status={result['stage1g_r2_status']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("select", "validate", "finalize"))
    parser.add_argument("output_root")
    parser.add_argument("--visual-status", default="PENDING_HUMAN_REVIEW")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    if root.parent != Path("/data/zhuzhaoziao/cmx/outputs") or not root.name.startswith("stage1g_r2_realdata_validation_"):
        raise SystemExit("invalid output root")
    try:
        if args.phase == "select":
            select_phase(root)
        elif args.phase == "validate":
            validate_phase(root)
        else:
            finalize_phase(root, args.visual_status)
    except Exception as error:
        append_log(root, f"ERROR {type(error).__name__}: {error}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
