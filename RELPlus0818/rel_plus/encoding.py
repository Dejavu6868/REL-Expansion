"""REL channel encoding aligned to CMX-REL/.../rel_original/rel.py::getREL."""

import numpy as np


def perspective_tangent_field(points, epsilon=1e-12):
    """Return unit horizontal azimuth tangents [ry, -rx, 0]."""
    point_field = np.asarray(points)
    if point_field.ndim != 3 or point_field.shape[2] != 3:
        raise ValueError("points must have shape HxWx3")
    radius = np.hypot(point_field[:, :, 0], point_field[:, :, 1])
    tangent = np.zeros(point_field.shape, dtype=np.float64)
    non_singular = radius > epsilon
    tangent[:, :, 0][non_singular] = (
        point_field[:, :, 1][non_singular] / radius[non_singular]
    )
    tangent[:, :, 1][non_singular] = (
        -point_field[:, :, 0][non_singular] / radius[non_singular]
    )
    return tangent, radius


def erp_tangent_field(height, width):
    """Original ERP tangent [cos(phi), -sin(phi), 0]."""
    columns = np.arange(width)
    phi = (columns / width) * 2.0 * np.pi - np.pi
    tangent_row = np.stack(
        [np.cos(phi), -np.sin(phi), np.zeros_like(phi)], axis=1
    )
    tangent = np.broadcast_to(
        tangent_row[np.newaxis, :, :], (height, width, 3)
    ).copy()
    radius = np.ones((height, width), dtype=np.float64)
    return tangent, radius


def encode_rel_channels(
    points_aligned,
    normals_aligned,
    valid_mask,
    tangent=None,
    alpha=45.0,
    lam=0.5,
):
    """Encode uint8 [EGVIA, LOA, ReD] with original REL numeric order."""
    points = np.asarray(points_aligned)
    normals = np.asarray(normals_aligned)
    valid = np.asarray(valid_mask, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points_aligned must have shape HxWx3")
    if normals.shape != points.shape or valid.shape != points.shape[:2]:
        raise ValueError("normals_aligned and valid_mask must match points_aligned")
    if tangent is None:
        tangent_field, horizontal_radius = perspective_tangent_field(points)
    else:
        tangent_field = np.asarray(tangent)
        if tangent_field.shape != points.shape:
            raise ValueError("tangent must match points_aligned")
        horizontal_radius = np.hypot(points[:, :, 0], points[:, :, 1])

    hcos = (
        normals[:, :, 0] * tangent_field[:, :, 0]
        + normals[:, :, 1] * tangent_field[:, :, 1]
        + normals[:, :, 2] * tangent_field[:, :, 2]
    )
    hcos = np.nan_to_num(hcos, nan=0.0)
    hcos = np.clip(hcos, -1.0, 1.0)
    loa_encoded = (np.arccos(hcos) * 180.0 / np.pi).astype(np.uint8)

    red_raw = np.hypot(points[:, :, 0], points[:, :, 1])
    red_min = red_raw.min()
    red_max = red_raw.max()
    red_scaled = red_raw.copy()
    if red_max > red_min:
        red_scaled = (red_scaled - red_min) * 255.0 / (red_max - red_min)
    red_encoded = np.clip(red_scaled, 0, 255).astype(np.uint8)

    height_raw = points[:, :, 2]
    height_min = np.percentile(height_raw, 1)
    height_max = np.percentile(height_raw, 99)
    height_normalized = height_raw.copy()
    if height_max > height_min:
        height_normalized = (
            (height_normalized - height_min)
            * 255.0
            / (height_max - height_min)
        )
    height_normalized = np.clip(height_normalized, 0, 255).astype(np.float32)

    normal_z = -normals[:, :, 2]
    normal_z = np.clip(normal_z, -1.0, 1.0)
    egvia_angle = (np.arccos(normal_z, dtype=np.float32) / np.pi) * 255.0
    egvia_angle = np.clip(egvia_angle, 0, 255).astype(np.float32)
    egvia_angle_before_blend = egvia_angle.copy()
    angle_threshold = alpha * 255.0 / 180.0
    is_horizontal = (egvia_angle <= angle_threshold) | (
        egvia_angle >= 255.0 - angle_threshold
    )
    egvia_angle[~is_horizontal] = (
        lam * egvia_angle[~is_horizontal]
        + (1.0 - lam) * height_normalized[~is_horizontal]
    )

    encoded = np.stack([egvia_angle, loa_encoded, red_encoded], axis=2).astype(
        np.float32
    )
    encoded = np.nan_to_num(encoded, nan=255.0)
    encoded[~valid, :] = 255.0
    encoded = np.clip(encoded, 0, 255).astype(np.uint8)
    debug = {
        "red_raw": red_raw,
        "red_encoded": red_encoded,
        "height_raw": height_raw,
        "height_normalized": height_normalized,
        "egvia_angle_before_blend": egvia_angle_before_blend,
        "egvia_encoded": encoded[:, :, 0],
        "horizontal_radius": horizontal_radius,
        "tangent": tangent_field,
        "hcos": hcos,
        "loa_encoded": loa_encoded,
    }
    return encoded, debug
