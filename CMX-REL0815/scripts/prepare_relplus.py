#!/usr/bin/env python3
"""Audit Stanford2D3D and precompute calibrated REL+ into a run-local cache."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np

from relplus.geometry import load_camera_metadata
from relplus.io import (
    read_relplus_png,
    read_split,
    resolve_sample_paths,
    validate_disjoint_splits,
    write_relplus_png,
)
from relplus.representation import CHANNEL_ORDER, compute_relplus, decode_stanford_depth
from relplus.spec import RELPLUS_SPEC, RELPLUS_SPEC_SHA256, canonical_spec_json


GENERATOR_FILES = (
    "relplus/__init__.py",
    "relplus/geometry.py",
    "relplus/io.py",
    "relplus/representation.py",
    "relplus/spec.py",
    "scripts/prepare_relplus.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_identity(repo_root):
    files = {
        relative: sha256_file(os.path.join(repo_root, relative))
        for relative in GENERATOR_FILES
    }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n"
    return files, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _histograms(image, valid):
    return [
        np.bincount(image[..., channel][valid], minlength=256).astype(np.int64).tolist()
        for channel in range(3)
    ]


def _prepare_one(task):
    (
        sample_id,
        dataset_root,
        cache_root,
        output_shape,
        overwrite,
        representation_spec_sha256,
        representation_generator_sha256,
        representation_generator_bundle_sha256,
    ) = task
    cv2.setNumThreads(1)
    output_path = os.path.join(cache_root, sample_id + ".png")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            "refusing an unverified existing REL+ cache entry: {}; use --overwrite".format(
                output_path
            )
        )

    paths = resolve_sample_paths(dataset_root, sample_id)
    raw = cv2.imread(paths["depth"], cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        raise ValueError("expected uint16 depth: {}".format(paths["depth"]))
    camera = load_camera_metadata(paths["pose"])
    depth, valid_depth = decode_stanford_depth(raw)
    rel_native, auxiliary = compute_relplus(depth, valid_depth, camera)

    output_height, output_width = output_shape
    rel_output = cv2.resize(
        rel_native, (output_width, output_height), interpolation=cv2.INTER_LINEAR
    )
    valid_output = cv2.resize(
        auxiliary["valid"].astype(np.uint8),
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    rel_output[~valid_output] = 255
    rel_output = np.ascontiguousarray(rel_output, dtype=np.uint8)
    write_relplus_png(output_path, rel_output)
    return {
        "sample_id": sample_id,
        "status": "generated",
        "sha256": sha256_file(output_path),
        "valid_pixels": int(valid_output.sum()),
        "invalid_pixels": int((~valid_output).sum()),
        "histograms": _histograms(rel_output, valid_output),
        "pose_center_residual": camera.center_residual,
        "representation_version": RELPLUS_SPEC["representation_version"],
        "representation_spec_sha256": representation_spec_sha256,
        "representation_generator_sha256": representation_generator_sha256,
        "representation_generator_bundle_sha256": representation_generator_bundle_sha256,
    }


def _audit_samples(dataset_root, train_ids, test_ids, selected_ids):
    rows = []
    pose_hashes = set()
    for sample_id in selected_ids:
        paths = resolve_sample_paths(dataset_root, sample_id)
        missing = [name for name, path in paths.items() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("{} missing {}".format(sample_id, missing))
        rgb = cv2.imread(paths["rgb"], cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(paths["depth"], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(paths["label"], cv2.IMREAD_GRAYSCALE)
        camera = load_camera_metadata(paths["pose"])
        depth, valid = decode_stanford_depth(depth_raw)
        pose_hash = sha256_file(paths["pose"])
        pose_hashes.add(pose_hash)
        valid_values = depth[valid]
        rows.append(
            {
                "sample_id": sample_id,
                "split": "train" if sample_id in train_ids else "test",
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth_raw.shape),
                "depth_dtype": str(depth_raw.dtype),
                "label_shape": list(label.shape),
                "depth_min_m": float(valid_values.min()),
                "depth_median_m": float(np.median(valid_values)),
                "depth_p99_m": float(np.percentile(valid_values, 99)),
                "depth_max_m": float(valid_values.max()),
                "invalid_ratio": float(1.0 - valid.mean()),
                "fx": float(camera.k[0, 0]),
                "fy": float(camera.k[1, 1]),
                "cx": float(camera.k[0, 2]),
                "cy": float(camera.k[1, 2]),
                "pose_center_residual": camera.center_residual,
                "pose_sha256": pose_hash,
            }
        )
    if len(pose_hashes) != len(selected_ids):
        raise ValueError("sample audit found reused pose JSON content")
    return rows


def _colorize_label(label):
    palette = np.array(
        [
            [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255],
            [255, 0, 255], [255, 255, 0], [128, 64, 255], [255, 128, 64],
            [64, 255, 128], [128, 255, 64], [64, 128, 255], [255, 64, 128],
            [128, 128, 128],
        ],
        dtype=np.uint8,
    )
    output = np.zeros(label.shape + (3,), dtype=np.uint8)
    valid = (label >= 1) & (label <= 13)
    output[valid] = palette[label[valid] - 1]
    output[label == 0] = [255, 255, 255]
    return output


def _depth_visual(depth_raw):
    depth, valid = decode_stanford_depth(depth_raw)
    output = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [1, 99])
        if high > low:
            output[valid] = np.clip((depth[valid] - low) * 255.0 / (high - low), 0, 255)
    return cv2.applyColorMap(output, cv2.COLORMAP_TURBO)


def _panel(image, title):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def _write_visualizations(dataset_root, cache_root, visual_root, selected_ids):
    visual_root = Path(visual_root)
    visual_root.mkdir(parents=True, exist_ok=True)
    for index, sample_id in enumerate(selected_ids):
        paths = resolve_sample_paths(dataset_root, sample_id)
        rgb = cv2.imread(paths["rgb"], cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(paths["depth"], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(paths["label"], cv2.IMREAD_GRAYSCALE)
        rel = read_relplus_png(os.path.join(cache_root, sample_id + ".png"))
        group = visual_root / ("{:02d}_{}".format(index, sample_id.replace("/", "__")))
        group.mkdir(parents=True, exist_ok=True)
        depth_vis = cv2.resize(_depth_visual(depth_raw), (480, 480), interpolation=cv2.INTER_LINEAR)
        label_vis = _colorize_label(label)
        channel_images = [cv2.applyColorMap(rel[..., i], cv2.COLORMAP_TURBO) for i in range(3)]
        named = [
            ("rgb", rgb),
            ("depth", depth_vis),
            ("ReD", channel_images[0]),
            ("EGVIA", channel_images[1]),
            ("LOA", channel_images[2]),
            ("label", label_vis),
        ]
        for name, image in named:
            cv2.imwrite(str(group / (name + ".png")), image)
        montage = np.vstack(
            [
                np.hstack([_panel(image, name) for name, image in named[:3]]),
                np.hstack([_panel(image, name) for name, image in named[3:]]),
            ]
        )
        cv2.imwrite(str(group / "montage.png"), montage)


def _histogram_summary(histograms, valid_count, invalid_count):
    output = {}
    for channel, name in enumerate(CHANNEL_ORDER):
        histogram = np.asarray(histograms[channel], dtype=np.int64)
        indices = np.arange(256, dtype=np.float64)
        count = int(histogram.sum())
        mean = float((histogram * indices).sum() / count) if count else 0.0
        variance = float((histogram * (indices - mean) ** 2).sum() / count) if count else 0.0
        cumulative = np.cumsum(histogram)

        def percentile(percent):
            if count == 0:
                return 0
            target = percent / 100.0 * max(count - 1, 0)
            return int(np.searchsorted(cumulative, target + 1, side="left"))

        nonzero = np.flatnonzero(histogram)
        output[name] = {
            "min": int(nonzero[0]) if nonzero.size else 0,
            "max": int(nonzero[-1]) if nonzero.size else 0,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "percentiles": {str(p): percentile(p) for p in (1, 5, 50, 95, 99)},
            "valid_pixels": count,
            "nan_count": 0,
            "inf_count": 0,
        }
    total = valid_count + invalid_count
    output["invalid_pixels"] = int(invalid_count)
    output["invalid_ratio"] = float(invalid_count / total) if total else 0.0
    output["channel_order"] = list(CHANNEL_ORDER)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visual-count", type=int, default=16)
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    run_dir = os.path.abspath(args.run_dir)
    cache_root = os.path.join(run_dir, "relplus_cache")
    reports_root = os.path.join(run_dir, "data_reports")
    visual_root = os.path.join(run_dir, "visualizations", "relplus_sanity")
    if os.path.islink(cache_root):
        raise ValueError("run-local REL+ cache must not be a symlink: {}".format(cache_root))
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(reports_root, exist_ok=True)

    spec_path = os.path.join(reports_root, "relplus_representation_spec.json")
    with open(spec_path, "w", encoding="utf-8") as handle:
        handle.write(canonical_spec_json())
    if sha256_file(spec_path) != RELPLUS_SPEC_SHA256:
        raise RuntimeError("REL+ representation spec serialization hash mismatch")
    generator_sha256 = sha256_file(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generator_files_sha256, generator_bundle_sha256 = generator_identity(repo_root)

    train_path = os.path.join(dataset_root, "train.txt")
    test_path = os.path.join(dataset_root, "test.txt")
    train_ids = read_split(train_path)
    test_ids = read_split(test_path)
    validate_disjoint_splits(train_ids, test_ids)
    all_ids = train_ids + test_ids
    if args.max_samples:
        all_ids = all_ids[: args.max_samples]

    available_ids = set(all_ids)
    smoke_ids = [sample_id for sample_id in train_ids if sample_id in available_ids][:16]
    if not smoke_ids:
        raise ValueError("no generated training samples are available for smoke")
    with open(os.path.join(reports_root, "smoke_split.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(smoke_ids) + "\n")

    rng = random.Random(12345)
    half = args.visual_count // 2
    selected = rng.sample(train_ids, half) + rng.sample(test_ids, args.visual_count - half)
    if args.max_samples:
        selected = all_ids[: min(args.visual_count, len(all_ids))]
    audit_rows = _audit_samples(dataset_root, set(train_ids), set(test_ids), selected)
    audit = {
        "dataset_root": dataset_root,
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "overlap_count": 0,
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "selected_count": len(selected),
        "smoke_sample_count": len(smoke_ids),
        "smoke_split": os.path.join(reports_root, "smoke_split.txt"),
        "selected_samples": audit_rows,
        "depth_definition": "camera-z; metres=(uint16+1)/512; 65535 invalid",
        "pose_definition": (
            "row-major world-to-camera [R|t], "
            "p_rel=p_camera@R=p_world-C; t/C provenance-only"
        ),
        "representation_semantics": RELPLUS_SPEC["representation_semantics"],
        "representation_version": RELPLUS_SPEC["representation_version"],
        "point_frame": RELPLUS_SPEC["point_frame"],
        "translation_in_red_loa": RELPLUS_SPEC["translation_in_red_loa"],
        "representation_spec_sha256": RELPLUS_SPEC_SHA256,
        "representation_generator_sha256": generator_sha256,
        "representation_generator_files_sha256": generator_files_sha256,
        "representation_generator_bundle_sha256": generator_bundle_sha256,
        "native_depth_shape": [1080, 1080],
        "cache_shape": [480, 480, 3],
    }
    with open(os.path.join(reports_root, "data_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    with open(os.path.join(reports_root, "sample_audit.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_rows[0].keys())
        writer.writeheader()
        writer.writerows(audit_rows)

    tasks = [
        (
            sample_id,
            dataset_root,
            cache_root,
            (480, 480),
            args.overwrite,
            RELPLUS_SPEC_SHA256,
            generator_sha256,
            generator_bundle_sha256,
        )
        for sample_id in all_ids
    ]
    manifest_path = os.path.join(reports_root, "cache_manifest.jsonl")
    aggregate_histograms = np.zeros((3, 256), dtype=np.int64)
    valid_count = 0
    invalid_count = 0
    generated = 0
    skipped = 0
    start = time.time()
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, result in enumerate(executor.map(_prepare_one, tasks, chunksize=1), 1):
                aggregate_histograms += np.asarray(result.pop("histograms"), dtype=np.int64)
                valid_count += result["valid_pixels"]
                invalid_count += result["invalid_pixels"]
                generated += result["status"] == "generated"
                skipped += result["status"] == "skipped"
                manifest.write(json.dumps(result, sort_keys=True) + "\n")
                if index % 100 == 0 or index == len(tasks):
                    elapsed = time.time() - start
                    print(
                        "prepare {}/{} generated={} skipped={} rate={:.2f} samples/s".format(
                            index, len(tasks), generated, skipped, index / max(elapsed, 1e-6)
                        ),
                        flush=True,
                    )

    summary = _histogram_summary(aggregate_histograms, valid_count, invalid_count)
    summary.update(
        {
            "sample_count": len(tasks),
            "generated_count": generated,
            "skipped_count": skipped,
            "elapsed_seconds": time.time() - start,
            "workers": args.workers,
            "normal_radius_native_pixels": 3,
            "alpha_degrees": 45.0,
            "lambda": 0.5,
            "representation_semantics": RELPLUS_SPEC["representation_semantics"],
            "representation_version": RELPLUS_SPEC["representation_version"],
            "point_frame": RELPLUS_SPEC["point_frame"],
            "translation_in_red_loa": RELPLUS_SPEC["translation_in_red_loa"],
            "representation_spec_sha256": RELPLUS_SPEC_SHA256,
            "representation_generator_sha256": generator_sha256,
            "representation_generator_files_sha256": generator_files_sha256,
            "representation_generator_bundle_sha256": generator_bundle_sha256,
        }
    )
    with open(os.path.join(reports_root, "relplus_statistics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    _write_visualizations(dataset_root, cache_root, visual_root, selected)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
