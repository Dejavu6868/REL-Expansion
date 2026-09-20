#!/usr/bin/env python3
"""Select and generate the fixed 36-sample REL+ v2.1 pilot cache."""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_ROOT), str(TOOLS_ROOT)]

from rel_plus.constants import (
    REL_PLUS_V2_1_INVALID_POLICY,
    REL_PLUS_V2_1_PROTOCOL_ID,
)
from rel_plus.generator import generate_rel_plus_v2_1
from rel_plus.integration.cmx_preprocess import (
    CMX_NORM_MEAN,
    CMX_NORM_STD,
    analyze_invalid_interpolation,
    apply_cmx_compatible_preprocess,
    sample_spatial_transform,
)
from rel_plus.profiles import STANFORD_S2D_PROFILE
from rel_plus.stanford_s2d import load_canonical_frame
from rel_plus.storage import save_rel_plus_png
from visualize_rel_plus import save_review_bundle


AREA_GROUPS = ("area_1", "area_2", "area_3", "area_4", "area_5", "area_6")


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row, field):
    value = row.get(field, "")
    return float(value) if value not in ("", None, "UNAVAILABLE") else 0.0


def _pick(candidates, selected, used_rooms, used_cameras, role, key, reverse):
    remaining = [row for row in candidates if row["sample_id"] not in selected]
    ranked = sorted(
        remaining,
        key=lambda row: (key(row), row["sample_id"]),
        reverse=reverse,
    )
    if not ranked:
        raise RuntimeError("not enough unique samples for role {}".format(role))
    diverse = [
        row
        for row in ranked
        if row.get("room") not in used_rooms and row.get("camera") not in used_cameras
    ]
    choice = (diverse or ranked)[0]
    selected[choice["sample_id"]] = (choice, role)
    used_rooms.add(choice.get("room"))
    used_cameras.add(choice.get("camera"))


def select_pilot(rows):
    if any(row["status"] != "PASS" for row in rows):
        raise ValueError("pilot selection is blocked by preflight failures")
    selected_rows = []
    for group_index, area_group in enumerate(AREA_GROUPS):
        candidates = [row for row in rows if row["area_group"] == area_group]
        if not candidates:
            raise ValueError("no preflight rows for {}".format(area_group))
        selected = {}
        rooms = set()
        cameras = set()
        roles = [
            ("tilt_high", lambda row: _float(row, "gravity_alignment_angle_deg"), True),
            ("invalid_high", lambda row: _float(row, "depth_invalid_ratio"), True),
            ("normal_quality_low", lambda row: _float(row, "normal_quality_ratio"), False),
            (
                "semantic_sparse",
                lambda row: _float(row, "floor_ratio") + _float(row, "ceiling_ratio"),
                False,
            ),
        ]
        if group_index % 2 == 0:
            roles.extend(
                [
                    ("tilt_low", lambda row: _float(row, "gravity_alignment_angle_deg"), False),
                    ("floor_visible", lambda row: _float(row, "floor_ratio"), True),
                ]
            )
        else:
            roles.extend(
                [
                    ("normal_quality_high", lambda row: _float(row, "normal_quality_ratio"), True),
                    ("ceiling_visible", lambda row: _float(row, "ceiling_ratio"), True),
                ]
            )
        for role, key, reverse in roles:
            _pick(candidates, selected, rooms, cameras, role, key, reverse)

        group_rows = list(selected.values())
        if area_group == "area_5":
            covered = {row["area"] for row, _ in group_rows}
            for required_area in ("area_5a", "area_5b"):
                if required_area not in covered:
                    replacement = next(
                        row
                        for row in sorted(candidates, key=lambda item: item["sample_id"])
                        if row["area"] == required_area
                        and row["sample_id"] not in selected
                    )
                    removed, _ = group_rows[-1]
                    del selected[removed["sample_id"]]
                    selected[replacement["sample_id"]] = (
                        replacement,
                        "area5_subarea_coverage",
                    )
                    group_rows = list(selected.values())
                    covered = {row["area"] for row, _ in group_rows}

        if len(selected) != 6:
            raise RuntimeError("{} pilot selection did not produce six rows".format(area_group))
        for row, role in selected.values():
            enriched = dict(row)
            enriched["selection_role"] = role
            enriched["selection_area_group"] = area_group
            selected_rows.append(enriched)

    if len(selected_rows) != 36:
        raise RuntimeError("pilot selection must contain exactly 36 rows")
    return selected_rows


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _depth_view(raw_depth):
    valid = (raw_depth != 0) & (raw_depth != 65535)
    view = np.zeros(raw_depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = raw_depth[valid].astype(np.float32)
        low, high = np.quantile(values, [0.01, 0.99])
        if high > low:
            view[valid] = np.clip((values - low) * 255.0 / (high - low), 0, 255)
    return cv2.applyColorMap(view, cv2.COLORMAP_TURBO)


def _labelled_panel(image, text):
    panel = cv2.resize(image, (240, 240), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (0, 0), (240, 26), (0, 0, 0), thickness=-1)
    cv2.putText(
        panel,
        text,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _label_view(label, *, stored_ids=False):
    values = np.asarray(label)
    display_ids = values.astype(np.int16)
    invalid = values == 255
    if stored_ids:
        invalid |= values == 0
        display_ids -= 1
    indexed = (((display_ids + 1).astype(np.uint16) * 17) % 256).astype(np.uint8)
    view = cv2.applyColorMap(indexed, cv2.COLORMAP_HSV)
    view[invalid] = 0
    return view


def _pilot_montage(
    rgb,
    raw_depth,
    label,
    rel_plus,
    transformed_rel,
    transformed_label,
    transformed_valid,
):
    mask = np.repeat((transformed_valid.astype(np.uint8) * 255)[..., None], 3, axis=2)
    return np.hstack(
        [
            _labelled_panel(rgb, "RGB: CMX cv2 bytes"),
            _labelled_panel(_depth_view(raw_depth), "canonical z-depth"),
            _labelled_panel(_label_view(label, stored_ids=True), "stored label"),
            _labelled_panel(rel_plus[:, :, ::-1], "REL+ display only"),
            _labelled_panel(transformed_rel[:, :, ::-1], "transformed REL+"),
            _labelled_panel(
                _label_view(transformed_label), "transformed model label"
            ),
            _labelled_panel(mask, "nearest valid mask"),
        ]
    )


def generate_pilot(rows, output_root):
    output_root = Path(output_root).resolve()
    generated = []
    errors = []
    for index, row in enumerate(rows):
        sample_id = row["sample_id"]
        try:
            raw_depth, camera, source_shape = load_canonical_frame(
                row["depth_path"],
                row["camera_metadata_path"],
                dataset_profile=STANFORD_S2D_PROFILE,
            )
            rgb = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
            label = cv2.imread(row["label_path"], cv2.IMREAD_UNCHANGED)
            if (
                rgb is None
                or rgb.dtype != np.uint8
                or rgb.shape != raw_depth.shape + (3,)
                or label is None
                or label.shape != raw_depth.shape
            ):
                raise ValueError("pilot RGB/label/canonical depth contract failed")
            rel_plus, debug = generate_rel_plus_v2_1(
                raw_depth, camera, return_debug=True
            )
            valid = np.asarray(debug["depth_valid"], dtype=bool)
            model_label = label.astype(np.int16) - 1
            model_label[label == 0] = 255
            model_label = model_label.astype(np.uint8)
            relative = Path(sample_id + ".png")
            rel_path = output_root / "RELPlus" / relative
            valid_path = output_root / "ValidMask" / relative
            rel_path.parent.mkdir(parents=True, exist_ok=True)
            valid_path.parent.mkdir(parents=True, exist_ok=True)
            save_rel_plus_png(rel_path, rel_plus)
            if not cv2.imwrite(str(valid_path), valid.astype(np.uint8) * 255):
                raise OSError("failed to save pilot valid mask")

            rng = np.random.default_rng(12345 + index)
            transform = sample_spatial_transform(
                raw_depth.shape, (0.75, 1.0, 1.25), (480, 480), rng
            )
            preprocessed = apply_cmx_compatible_preprocess(
                rgb, rel_plus, model_label, valid, transform
            )
            transformed_rel = np.rint(
                (
                    preprocessed.modal_x.transpose(1, 2, 0) * CMX_NORM_STD
                    + CMX_NORM_MEAN
                )
                * 255.0
            ).clip(0, 255).astype(np.uint8)
            diagnostic = analyze_invalid_interpolation(rel_plus, valid, transform)

            review_dir = output_root / "review" / row["area"] / Path(sample_id).name
            review_dir.mkdir(parents=True, exist_ok=True)
            save_review_bundle(review_dir, rgb, debug)
            transformed_rel_path = review_dir / "transformed_rel_plus.png"
            transformed_mask_path = review_dir / "transformed_valid_mask.png"
            montage_path = review_dir / "pilot_montage.png"
            cv2.imwrite(str(transformed_rel_path), transformed_rel)
            cv2.imwrite(
                str(transformed_mask_path),
                preprocessed.modal_x_valid_mask.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                str(montage_path),
                _pilot_montage(
                    rgb,
                    raw_depth,
                    label,
                    rel_plus,
                    transformed_rel,
                    preprocessed.label,
                    preprocessed.modal_x_valid_mask,
                ),
            )
            model_stats = {
                "rgb_dtype": str(preprocessed.rgb.dtype),
                "rgb_shape": list(preprocessed.rgb.shape),
                "rgb_channel_min": [
                    float(preprocessed.rgb[channel].min()) for channel in range(3)
                ],
                "rgb_channel_max": [
                    float(preprocessed.rgb[channel].max()) for channel in range(3)
                ],
                "modal_x_dtype": str(preprocessed.modal_x.dtype),
                "modal_x_shape": list(preprocessed.modal_x.shape),
                "modal_x_channel_min": [
                    float(preprocessed.modal_x[channel].min()) for channel in range(3)
                ],
                "modal_x_channel_max": [
                    float(preprocessed.modal_x[channel].max()) for channel in range(3)
                ],
                "label_dtype": str(preprocessed.label.dtype),
                "label_shape": list(preprocessed.label.shape),
                "label_ignore_pixel_count": int(
                    np.count_nonzero(preprocessed.label == 255)
                ),
            }
            summary_path = review_dir / "debug_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "protocol_id": REL_PLUS_V2_1_PROTOCOL_ID,
                        "invalid_policy": REL_PLUS_V2_1_INVALID_POLICY,
                        "source_shape": list(source_shape),
                        "canonical_shape": list(raw_depth.shape),
                        "channel_order": ["EGVIA", "LOA", "ReD"],
                        "depth_invalid_ratio": float(1.0 - np.mean(valid)),
                        "normal_nonfinite_ratio": debug["normal_invalid_ratio"],
                        "zero_normal_ratio": debug["zero_normal_ratio"],
                        "low_support_ratio": debug["low_support_ratio"],
                        "normal_quality_ratio": debug["normal_quality_ratio"],
                        "transform": asdict(transform),
                        "invalid_interpolation": diagnostic,
                        "model_input": model_stats,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_row = dict(row)
            output_row.update(
                {
                    "protocol_id": REL_PLUS_V2_1_PROTOCOL_ID,
                    "rel_plus_path": str(rel_path),
                    "valid_mask_path": str(valid_path),
                    "review_directory": str(review_dir),
                    "pilot_montage_path": str(montage_path),
                    "debug_summary_path": str(summary_path),
                    "generation_status": "PASS",
                }
            )
            generated.append(output_row)
        except Exception as error:
            errors.append(
                {
                    "sample_id": sample_id,
                    "status": "FAIL",
                    "error": repr(error),
                }
            )
    if generated:
        _write_csv(output_root / "pilot_manifest.csv", generated)
        sample_lines = "\n".join(row["sample_id"] for row in generated) + "\n"
        (output_root / "train.txt").write_text(sample_lines, encoding="utf-8")
        (output_root / "test.txt").write_text(sample_lines, encoding="utf-8")
    if errors:
        _write_csv(output_root / "pilot_errors.csv", errors)
    summary = {
        "status": "PASS" if len(generated) == 36 and not errors else "FAIL",
        "protocol_id": REL_PLUS_V2_1_PROTOCOL_ID,
        "requested_count": len(rows),
        "generated_count": len(generated),
        "error_count": len(errors),
        "area_group_counts": {
            group: sum(row["selection_area_group"] == group for row in generated)
            for group in AREA_GROUPS
        },
        "full_cache_generated": False,
    }
    (output_root / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    rows = _read_csv(args.preflight)
    selected = select_pilot(rows)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "selected_pilot_manifest.csv", selected)
    summary = generate_pilot(selected, output_root)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
