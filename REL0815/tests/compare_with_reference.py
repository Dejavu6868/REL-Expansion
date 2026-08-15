#!/usr/bin/env python3
"""Compare the public REL call path with the independent reproduction."""

import argparse
import csv
import gc
import importlib.util
import sys
import tempfile
import time
import types
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rel_original import rel as reproduced_rel
from rel_original import rgbd_util as reproduced_rgbd


STAGES = (
    "D",
    "missingMask",
    "ERP_point_cloud",
    "normal",
    "gravity_direction",
    "rotation",
    "rotation_matrix",
    "pcRot",
    "NRot",
    "angle",
    "HA",
    "RD",
    "final_REL",
    "saved_reloaded_PNG",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference(reference_root, compatibility_hha):
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = [str(reference_root / "utils")]
    sys.modules["utils"] = utils_package

    load_module("utils.hha_util", compatibility_hha)
    rgbd_module = load_module(
        "utils.rgbd_util", reference_root / "utils" / "rgbd_util.py"
    )
    rel_module = load_module("reference_getREL", reference_root / "getREL.py")
    return rel_module, rgbd_module


def save_stage(stage_dir, name, value):
    np.save(stage_dir / f"{name}.npy", np.asarray(value))


def save_derived_stages(stage_dir, shape, alpha, lam):
    h, w = shape
    normal = np.load(stage_dir / "NRot.npy", mmap_mode="r")
    pc_rot = np.load(stage_dir / "pcRot.npy", mmap_mode="r")

    u = np.arange(w)
    phi = (u / w) * 2 * np.pi - np.pi
    cos_theta = np.cos(phi)[np.newaxis, :]
    sin_theta = np.sin(phi)[np.newaxis, :]

    hcos = normal[:, :, 0] * cos_theta - normal[:, :, 1] * sin_theta
    hcos = np.nan_to_num(hcos, nan=0)
    hcos = np.clip(hcos, -1.0, 1.0)
    HA = (np.arccos(hcos) * 180 / np.pi).astype(np.uint8)
    save_stage(stage_dir, "HA", HA)
    del hcos, HA

    RD = np.hypot(pc_rot[:, :, 0], pc_rot[:, :, 1])
    RD_min = RD.min()
    RD_max = RD.max()
    if RD_max > RD_min:
        RD = (RD - RD_min) * 255.0 / (RD_max - RD_min)
    RD = np.clip(RD, 0, 255).astype(np.uint8)
    save_stage(stage_dir, "RD", RD)
    del RD

    h_val = pc_rot[:, :, 2]
    hmin = np.percentile(h_val, 1)
    hmax = np.percentile(h_val, 99)
    if hmax > hmin:
        h_val = (h_val - hmin) * 255.0 / (hmax - hmin)
    h_val = np.clip(h_val, 0, 255).astype(np.float32)

    N_z = -normal[:, :, 2]
    N_z = np.clip(N_z, -1.0, 1.0)
    angle = (np.arccos(N_z, dtype=np.float32) / np.pi) * 255.0
    angle = np.clip(angle, 0, 255).astype(np.float32)
    angle_threshold = alpha * 255.0 / 180.0
    is_horizontal = (angle <= angle_threshold) | (
        angle >= 255.0 - angle_threshold
    )
    angle[~is_horizontal] = lam * angle[~is_horizontal] + (
        1 - lam
    ) * h_val[~is_horizontal]
    save_stage(stage_dir, "angle", angle)


def run_traced(rel_module, rgbd_module, depth_path, stage_dir, png_path, alpha, lam):
    depth = rel_module.getImage(str(depth_path), "Stanford2D3DPano")
    save_stage(stage_dir, "D", depth)
    save_stage(stage_dir, "missingMask", depth == 0)

    originals = {
        "point_cloud": rgbd_module.getPointCloud_ERP,
        "normal": rgbd_module.computeNormalsSquareSupport_ERP,
        "gravity": rgbd_module.getGDir,
        "rotation_matrix": rgbd_module.getRMatrix,
        "rotate": rgbd_module.rotatePC,
        "process": rel_module.processDepthImage_ERP,
    }
    rotate_calls = {"count": 0}

    def traced_point_cloud(*args, **kwargs):
        value = originals["point_cloud"](*args, **kwargs)
        save_stage(stage_dir, "ERP_point_cloud", value)
        return value

    def traced_normal(*args, **kwargs):
        value, offset = originals["normal"](*args, **kwargs)
        save_stage(stage_dir, "normal", value)
        return value, offset

    def traced_gravity(*args, **kwargs):
        value = originals["gravity"](*args, **kwargs)
        save_stage(stage_dir, "gravity_direction", value)
        return value

    def traced_rotation_matrix(*args, **kwargs):
        value = originals["rotation_matrix"](*args, **kwargs)
        save_stage(stage_dir, "rotation_matrix", value)
        return value

    def traced_rotate(*args, **kwargs):
        value = originals["rotate"](*args, **kwargs)
        name = "NRot" if rotate_calls["count"] == 0 else "pcRot"
        rotate_calls["count"] += 1
        save_stage(stage_dir, name, value)
        return value

    def traced_process(*args, **kwargs):
        pc_rot, normal_rot, rotation = originals["process"](*args, **kwargs)
        save_stage(stage_dir, "rotation", rotation)
        return pc_rot, normal_rot, rotation

    rgbd_module.getPointCloud_ERP = traced_point_cloud
    rgbd_module.computeNormalsSquareSupport_ERP = traced_normal
    rgbd_module.getGDir = traced_gravity
    rgbd_module.getRMatrix = traced_rotation_matrix
    rgbd_module.rotatePC = traced_rotate
    rel_module.processDepthImage_ERP = traced_process
    try:
        rel = rel_module.getREL(depth, alpha=alpha, lam=lam)
    finally:
        rgbd_module.getPointCloud_ERP = originals["point_cloud"]
        rgbd_module.computeNormalsSquareSupport_ERP = originals["normal"]
        rgbd_module.getGDir = originals["gravity"]
        rgbd_module.getRMatrix = originals["rotation_matrix"]
        rgbd_module.rotatePC = originals["rotate"]
        rel_module.processDepthImage_ERP = originals["process"]

    save_stage(stage_dir, "final_REL", rel)
    if not cv2.imwrite(str(png_path), rel):
        raise RuntimeError(f"Failed to save {png_path}")
    reloaded = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    save_stage(stage_dir, "saved_reloaded_PNG", reloaded)
    save_derived_stages(stage_dir, depth.shape, alpha, lam)
    del depth, rel, reloaded
    gc.collect()


def arrays_equal_exact(left_path, right_path, rows_per_chunk=32):
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.ndim == 0:
        return bool(np.array_equal(left, right, equal_nan=True))

    floating = np.issubdtype(left.dtype, np.floating)
    for start in range(0, left.shape[0], rows_per_chunk):
        left_chunk = left[start : start + rows_per_chunk]
        right_chunk = right[start : start + rows_per_chunk]
        if floating:
            same = (left_chunk == right_chunk) | (
                np.isnan(left_chunk) & np.isnan(right_chunk)
            )
            if not bool(np.all(same)):
                return False
        elif not bool(np.array_equal(left_chunk, right_chunk)):
            return False
    return True


def final_difference(left_path, right_path):
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
    mismatch_pixels = 0
    channel_max = np.zeros(3, dtype=np.int64)
    for start in range(0, left.shape[0], 64):
        diff = np.abs(
            left[start : start + 64].astype(np.int16)
            - right[start : start + 64].astype(np.int16)
        )
        mismatch_pixels += int(np.count_nonzero(np.any(diff != 0, axis=2)))
        channel_max = np.maximum(channel_max, diff.max(axis=(0, 1)))
    return mismatch_pixels, channel_max.tolist()


def render_panel(values, title, actual_min, actual_max, color_map=None):
    scaled = np.clip(values, 0, 255).astype(np.uint8)
    scaled = cv2.resize(scaled, (800, 400), interpolation=cv2.INTER_AREA)
    if color_map is None:
        image = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.applyColorMap(scaled, color_map)

    canvas = np.full((472, 800, 3), 255, dtype=np.uint8)
    canvas[72:, :, :] = image
    cv2.putText(
        canvas,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"actual min/max={actual_min:.3f}/{actual_max:.3f}",
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


def create_visualization(sample, reference_png, reproduced_png, output_path):
    depth = reproduced_rel.getImage(
        sample["absolute_depth_path"], "Stanford2D3DPano"
    )
    reference = cv2.imread(str(reference_png), cv2.IMREAD_UNCHANGED)
    reproduced = cv2.imread(str(reproduced_png), cv2.IMREAD_UNCHANGED)
    difference = np.max(
        np.abs(reference.astype(np.int16) - reproduced.astype(np.int16)), axis=2
    ).astype(np.uint8)
    sample_id = sample["sample_id"]

    depth_display = np.clip(depth / 20.0 * 255.0, 0, 255)
    panels = [
        render_panel(
            depth_display,
            f"{sample_id} original D | fixed 0..20",
            float(depth.min()),
            float(depth.max()),
            cv2.COLORMAP_TURBO,
        )
    ]
    names = ("angle/EGVIA", "HA/LOA", "RD/ReD")
    for channel, name in enumerate(names):
        equal = bool(np.array_equal(reference[:, :, channel], reproduced[:, :, channel]))
        panels.append(
            render_panel(
                reference[:, :, channel],
                f"{sample_id} reference c{channel} {name} | 0..255 | equal={equal}",
                float(reference[:, :, channel].min()),
                float(reference[:, :, channel].max()),
            )
        )
    for channel, name in enumerate(names):
        equal = bool(np.array_equal(reference[:, :, channel], reproduced[:, :, channel]))
        panels.append(
            render_panel(
                reproduced[:, :, channel],
                f"{sample_id} reproduced c{channel} {name} | 0..255 | equal={equal}",
                float(reproduced[:, :, channel].min()),
                float(reproduced[:, :, channel].max()),
            )
        )
    panels.append(
        render_panel(
            difference,
            f"{sample_id} max absolute channel difference | fixed 0..255",
            float(difference.min()),
            float(difference.max()),
            cv2.COLORMAP_HOT,
        )
    )

    montage = np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:])))
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Failed to save {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--compatibility-hha", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=45)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    if args.limit:
        samples = samples[: args.limit]

    reference_rel, reference_rgbd = load_reference(
        args.reference_root, args.compatibility_hha
    )
    reference_png_dir = args.output_root / "reference_png"
    reproduced_png_dir = args.output_root / "reproduced_png"
    visualization_dir = args.output_root / "visualizations"
    reference_png_dir.mkdir(exist_ok=True)
    reproduced_png_dir.mkdir(exist_ok=True)
    visualization_dir.mkdir(exist_ok=True)

    comparison_path = args.output_root / "reference_comparison.csv"
    stage_columns = [f"{stage}_equal" for stage in STAGES]
    fieldnames = [
        "sample_id",
        "shape",
        "dtype",
        *stage_columns,
        "all_intermediates_equal",
        "final_rel_equal",
        "final_mismatch_pixels",
        "channel0_max_abs_diff",
        "channel1_max_abs_diff",
        "channel2_max_abs_diff",
        "png_reloaded_equal",
        "png_mismatch_pixels",
        "first_difference_stage",
        "elapsed_seconds",
    ]

    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample in enumerate(samples):
            started = time.time()
            sample_id = sample["sample_id"]
            print(f"START {sample_id}", flush=True)
            with tempfile.TemporaryDirectory(
                prefix=f"trace_{sample_id}_", dir=args.output_root
            ) as temporary:
                temporary = Path(temporary)
                reference_stages = temporary / "reference"
                reproduced_stages = temporary / "reproduced"
                reference_stages.mkdir()
                reproduced_stages.mkdir()
                reference_png = reference_png_dir / f"{sample_id}.png"
                reproduced_png = reproduced_png_dir / f"{sample_id}.png"

                run_traced(
                    reference_rel,
                    reference_rgbd,
                    sample["absolute_depth_path"],
                    reference_stages,
                    reference_png,
                    args.alpha,
                    args.lam,
                )
                run_traced(
                    reproduced_rel,
                    reproduced_rgbd,
                    sample["absolute_depth_path"],
                    reproduced_stages,
                    reproduced_png,
                    args.alpha,
                    args.lam,
                )

                equality = {
                    stage: arrays_equal_exact(
                        reference_stages / f"{stage}.npy",
                        reproduced_stages / f"{stage}.npy",
                    )
                    for stage in STAGES
                }
                first_difference = next(
                    (stage for stage in STAGES if not equality[stage]), "NONE"
                )
                mismatch_pixels, channel_max = final_difference(
                    reference_stages / "final_REL.npy",
                    reproduced_stages / "final_REL.npy",
                )
                png_mismatch_pixels, _ = final_difference(
                    reference_stages / "saved_reloaded_PNG.npy",
                    reproduced_stages / "saved_reloaded_PNG.npy",
                )
                final_array = np.load(reference_stages / "final_REL.npy", mmap_mode="r")
                elapsed = time.time() - started
                row = {
                    "sample_id": sample_id,
                    "shape": "x".join(str(value) for value in final_array.shape),
                    "dtype": str(final_array.dtype),
                    **{f"{stage}_equal": equality[stage] for stage in STAGES},
                    "all_intermediates_equal": all(
                        equality[stage]
                        for stage in STAGES
                        if stage not in ("final_REL", "saved_reloaded_PNG")
                    ),
                    "final_rel_equal": equality["final_REL"],
                    "final_mismatch_pixels": mismatch_pixels,
                    "channel0_max_abs_diff": channel_max[0],
                    "channel1_max_abs_diff": channel_max[1],
                    "channel2_max_abs_diff": channel_max[2],
                    "png_reloaded_equal": equality["saved_reloaded_PNG"],
                    "png_mismatch_pixels": png_mismatch_pixels,
                    "first_difference_stage": first_difference,
                    "elapsed_seconds": f"{elapsed:.2f}",
                }
                writer.writerow(row)
                handle.flush()

            if index < 5:
                create_visualization(
                    sample,
                    reference_png,
                    reproduced_png,
                    visualization_dir / f"{sample_id}_comparison.png",
                )
            print(
                f"PASS {sample_id} final={row['final_rel_equal']} "
                f"png={row['png_reloaded_equal']} first_difference={first_difference} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )

    print(f"comparison_csv={comparison_path}", flush=True)


if __name__ == "__main__":
    main()
