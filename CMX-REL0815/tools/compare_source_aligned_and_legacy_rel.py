import argparse
import csv
import gc
import json
from pathlib import Path

import cv2
import numpy as np

from rel_source_aligned.adapters.stanford2d3d_perspective_adapter import (
    PerspectiveInputAdapter,
)
from relplus.geometry import load_camera_metadata
from relplus.pipeline import generate_relplus_from_depth, generate_relplus_from_depth_local


SEMANTIC_CHANNELS = ("ReD", "EGVIA", "LOA")
SOURCE_INDEX = {"EGVIA": 0, "LOA": 1, "ReD": 2}
LEGACY_INDEX = {"ReD": 0, "EGVIA": 1, "LOA": 2}


def _read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _values(array, mask):
    return np.asarray(array)[np.asarray(mask, dtype=bool)].astype(np.float64)


def _stats(array, mask):
    values = _values(array, mask)
    if values.size == 0:
        raise ValueError("numeric comparison mask is empty")
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _correlation(left, right):
    if left.size < 2:
        return float("nan")
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std == 0.0 or right_std == 0.0:
        return 1.0 if np.array_equal(left, right) else float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _comparison_row(sample_id, comparator, channel, source, legacy, source_valid, legacy_valid):
    common = source_valid & legacy_valid
    source_values = _values(source, common)
    legacy_values = _values(legacy, common)
    difference = np.abs(source_values - legacy_values)
    source_stats = _stats(source, source_valid)
    legacy_stats = _stats(legacy, legacy_valid)
    return {
        "sample_id": sample_id,
        "comparator": comparator,
        "semantic_channel": channel,
        "source_array_index": SOURCE_INDEX[channel],
        "legacy_array_index": LEGACY_INDEX[channel],
        "source_valid_ratio": float(source_valid.mean()),
        "legacy_valid_ratio": float(legacy_valid.mean()),
        "common_valid_ratio": float(common.mean()),
        "source_min": source_stats["min"],
        "source_max": source_stats["max"],
        "source_mean": source_stats["mean"],
        "source_std": source_stats["std"],
        "legacy_min": legacy_stats["min"],
        "legacy_max": legacy_stats["max"],
        "legacy_mean": legacy_stats["mean"],
        "legacy_std": legacy_stats["std"],
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
        "correlation": _correlation(source_values, legacy_values),
    }


def _safe_id(sample_id):
    return sample_id.replace("/", "__")


def _resize_gray(image, size=(480, 480)):
    return cv2.resize(np.asarray(image, dtype=np.uint8), size, interpolation=cv2.INTER_NEAREST)


def _panel(image, label, panel_size=240):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((panel_size + 30, panel_size, 3), dtype=np.uint8)
    canvas[:panel_size] = image
    cv2.putText(
        canvas,
        label,
        (5, panel_size + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _write_visuals(output_root, sample_id, rgb, source, local, pose, source_valid, local_valid, pose_valid):
    sample_root = output_root / "visualizations" / _safe_id(sample_id)
    sample_root.mkdir(parents=True, exist_ok=True)
    source_semantic = {name: source[:, :, SOURCE_INDEX[name]] for name in SEMANTIC_CHANNELS}
    local_semantic = {name: local[:, :, LEGACY_INDEX[name]] for name in SEMANTIC_CHANNELS}
    pose_semantic = {name: pose[:, :, LEGACY_INDEX[name]] for name in SEMANTIC_CHANNELS}
    panels = [_panel(rgb, "RGB")]

    for prefix, arrays in (
        ("source_aligned", source_semantic),
        ("legacy_local", local_semantic),
        ("legacy_pose", pose_semantic),
    ):
        for channel in SEMANTIC_CHANNELS:
            image = _resize_gray(arrays[channel])
            cv2.imwrite(str(sample_root / (prefix + "_" + channel.lower() + ".png")), image)
            panels.append(_panel(image, prefix + " " + channel))

    for prefix, arrays, valid in (
        ("absdiff_source_vs_local", local_semantic, source_valid & local_valid),
        ("absdiff_source_vs_pose", pose_semantic, source_valid & pose_valid),
    ):
        for channel in SEMANTIC_CHANNELS:
            difference = np.abs(
                source_semantic[channel].astype(np.int16) - arrays[channel].astype(np.int16)
            ).astype(np.uint8)
            difference[~valid] = 0
            image = _resize_gray(difference)
            cv2.imwrite(str(sample_root / (prefix + "_" + channel.lower() + ".png")), image)
            panels.append(_panel(image, prefix.replace("absdiff_", "diff ") + " " + channel))

    while len(panels) % 4:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.concatenate(panels[index : index + 4], axis=1) for index in range(0, len(panels), 4)]
    cv2.imwrite(str(sample_root / "comparison_sheet.png"), np.concatenate(rows, axis=0))


def compare(dataset_root, authority_root, manifest_path, output_root):
    manifest = _read_manifest(manifest_path)
    if not 12 <= len(manifest) <= 20:
        raise ValueError("frozen manifest must contain 12 to 20 samples")
    adapter = PerspectiveInputAdapter(authority_root)
    rows = []
    gravity_rows = []
    for index, record in enumerate(manifest, start=1):
        sample_id = record["sample_id"]
        depth_path = dataset_root / "Depth16" / (sample_id + ".png")
        pose_path = dataset_root / "Pose" / (sample_id + ".json")
        rgb_path = dataset_root / "RGB" / (sample_id + ".png")
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if raw_depth is None or rgb is None:
            raise FileNotFoundError("review sample input missing: {}".format(sample_id))
        camera = load_camera_metadata(str(pose_path))
        source_result = adapter.encode(raw_depth, camera.k)
        source_inputs = adapter.last_inputs

        depth = raw_depth.astype(np.float64) / 512.0
        depth_valid = (raw_depth > 0) & (raw_depth != 65535) & np.isfinite(depth)
        depth[~depth_valid] = 0.0
        legacy_local, local_valid, local_aux = generate_relplus_from_depth_local(
            depth, depth_valid, camera.k, normal_radius=3
        )
        legacy_pose, pose_valid, pose_aux = generate_relplus_from_depth(
            depth, depth_valid, camera.k, camera.r_world_to_camera, normal_radius=3
        )

        for comparator, legacy, legacy_valid in (
            ("legacy_custom_rel_like_local", legacy_local, local_valid),
            ("legacy_custom_rel_like_pose", legacy_pose, pose_valid),
        ):
            for channel in SEMANTIC_CHANNELS:
                rows.append(
                    _comparison_row(
                        sample_id,
                        comparator,
                        channel,
                        source_result.rel[:, :, SOURCE_INDEX[channel]],
                        legacy[:, :, LEGACY_INDEX[channel]],
                        source_result.valid_mask,
                        legacy_valid,
                    )
                )

        gravity_rows.append(
            {
                "sample_id": sample_id,
                "source_gravity_0": float(source_inputs.gravity_direction[0]),
                "source_gravity_1": float(source_inputs.gravity_direction[1]),
                "source_gravity_2": float(source_inputs.gravity_direction[2]),
                "legacy_local_gravity_0": float(local_aux["gravity_down_camera"][0]),
                "legacy_local_gravity_1": float(local_aux["gravity_down_camera"][1]),
                "legacy_local_gravity_2": float(local_aux["gravity_down_camera"][2]),
                "legacy_pose_gravity_0": float(pose_aux["gravity_down_camera"][0]),
                "legacy_pose_gravity_1": float(pose_aux["gravity_down_camera"][1]),
                "legacy_pose_gravity_2": float(pose_aux["gravity_down_camera"][2]),
            }
        )
        _write_visuals(
            output_root,
            sample_id,
            rgb,
            source_result.rel,
            legacy_local,
            legacy_pose,
            source_result.valid_mask,
            local_valid,
            pose_valid,
        )
        print("COMPARED={}/{} {}".format(index, len(manifest), sample_id), flush=True)
        del source_result, legacy_local, legacy_pose, local_aux, pose_aux
        gc.collect()

    summary_path = output_root / "new_vs_legacy_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gravity_path = output_root / "gravity_comparison.csv"
    with gravity_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gravity_rows[0]))
        writer.writeheader()
        writer.writerows(gravity_rows)
    metadata = {
        "sample_count": len(manifest),
        "comparators": [
            "legacy_custom_rel_like_local",
            "legacy_custom_rel_like_pose",
        ],
        "source_actual_array_order": [
            "EGVIA_source_code",
            "LOA_source_code",
            "ReD_source_code",
        ],
        "legacy_array_order": ["ReD", "EGVIA", "LOA"],
        "visual_range": "fixed uint8 0-255; no per-image stretching",
        "difference_mask": "common valid pixels; invalid pixels shown as 0 in difference images",
        "training_run": False,
    }
    (output_root / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args.dataset_root, args.authority_root, args.manifest, args.output_root)
    print("NUMERIC_ROW_COUNT={}".format(len(rows)))


if __name__ == "__main__":
    main()
