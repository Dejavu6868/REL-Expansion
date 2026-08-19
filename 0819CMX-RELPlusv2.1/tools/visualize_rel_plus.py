#!/usr/bin/env python3
"""Small OpenCV-only visualization helpers for REL+ review artifacts."""

from pathlib import Path

import cv2
import numpy as np


PANEL_SIZE = (480, 480)


def depth_to_bgr(depth_m, valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    shown = np.zeros(depth.shape, dtype=np.uint8)
    values = depth[valid]
    if values.size:
        low, high = np.percentile(values, [1.0, 99.0])
        if high > low:
            shown = np.clip((depth - low) * 255.0 / (high - low), 0, 255).astype(
                np.uint8
            )
    colour = cv2.applyColorMap(shown, cv2.COLORMAP_MAGMA)
    colour[~valid] = 0
    return colour


def normal_to_bgr(normals, valid_mask):
    normal_rgb = np.clip(
        (np.nan_to_num(normals, nan=0.0) + 1.0) * 127.5, 0, 255
    ).astype(np.uint8)
    normal_rgb[~np.asarray(valid_mask, dtype=bool)] = 0
    return cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)


def scalar_to_bgr(channel, valid_mask):
    colour = cv2.applyColorMap(np.asarray(channel, dtype=np.uint8), cv2.COLORMAP_VIRIDIS)
    colour[~np.asarray(valid_mask, dtype=bool)] = 0
    return colour


def labelled_panel(image_bgr, label):
    image = cv2.resize(image_bgr, PANEL_SIZE, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PANEL_SIZE[1] + 38, PANEL_SIZE[0], 3), dtype=np.uint8)
    canvas[38:] = image
    cv2.putText(
        canvas,
        label,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def build_montage(panels, columns=3):
    labelled = [labelled_panel(image, label) for label, image in panels]
    blank = np.zeros_like(labelled[0])
    while len(labelled) % columns:
        labelled.append(blank.copy())
    rows = [
        cv2.hconcat(labelled[index : index + columns])
        for index in range(0, len(labelled), columns)
    ]
    return cv2.vconcat(rows)


def save_review_bundle(output_dir, rgb_bgr, debug, rgb_label="RGB"):
    """Save numeric review views and a clearly display-only montage."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    valid = np.asarray(debug["encoding_valid_mask"], dtype=bool)
    rel_plus = np.asarray(debug["rel_plus"], dtype=np.uint8)
    depth_view = depth_to_bgr(debug["depth_m"], debug["depth_m"] > 0)
    valid_view = cv2.cvtColor(valid.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    camera_normal_view = normal_to_bgr(debug["normals_camera"], valid)
    quality_view = cv2.cvtColor(
        np.asarray(debug["normal_quality_mask"], dtype=np.uint8) * 255,
        cv2.COLOR_GRAY2BGR,
    )
    aligned_normal_view = normal_to_bgr(debug["normals_aligned"], valid)
    red_view = scalar_to_bgr(rel_plus[:, :, 2], valid)
    egvia_view = scalar_to_bgr(rel_plus[:, :, 0], valid)
    loa_view = scalar_to_bgr(rel_plus[:, :, 1], valid)
    # Writing stored bytes directly makes an ordinary RGB viewer display
    # [ReD, LOA, EGVIA]. This is explicitly display-only, never model input.
    rel_display = rel_plus.copy()
    rel_display[~valid] = 255

    files = {
        "rgb.png": rgb_bgr,
        "raw_depth_visualization.png": depth_view,
        "valid_mask.png": valid_view,
        "normal_camera.png": camera_normal_view,
        "normal_quality_mask.png": quality_view,
        "normal_gravity_aligned.png": aligned_normal_view,
        "red.png": rel_plus[:, :, 2],
        "egvia.png": rel_plus[:, :, 0],
        "loa.png": rel_plus[:, :, 1],
        "rel_plus_display_only.png": rel_display,
    }
    for name, value in files.items():
        if not cv2.imwrite(str(destination / name), value):
            raise OSError("failed to write review image: {}".format(destination / name))

    panels = [
        (rgb_label, rgb_bgr),
        ("raw z-depth (m)", depth_view),
        ("REL valid mask", valid_view),
        ("camera-space normal", camera_normal_view),
        ("normal quality (diagnostic only)", quality_view),
        ("gravity-aligned normal", aligned_normal_view),
        ("ReD = stored channel 2", red_view),
        ("EGVIA = stored channel 0", egvia_view),
        ("LOA degrees = stored channel 1", loa_view),
        ("DISPLAY ONLY RGB=[ReD,LOA,EGVIA]", rel_display),
    ]
    montage = build_montage(panels)
    montage_path = destination / "montage.png"
    if not cv2.imwrite(str(montage_path), montage):
        raise OSError("failed to write montage: {}".format(montage_path))
    return montage_path


def save_contact_sheet(montage_paths, labels, output_path):
    panels = []
    for path, label in zip(montage_paths, labels):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError("failed to read montage: {}".format(path))
        panels.append((label, image))
    sheet = build_montage(panels, columns=2)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError("failed to write contact sheet: {}".format(output_path))
