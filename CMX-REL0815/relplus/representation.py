"""Frozen REL mathematics adapted to calibrated perspective RGB-D frames."""

import cv2
import numpy as np

from .geometry import backproject_z_depth, camera_to_world, rotate_camera_vectors_to_world


GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)
CHANNEL_ORDER = ("ReD", "EGVIA", "LOA")
DEFAULT_ALPHA_DEGREES = 45.0
DEFAULT_LAMBDA = 0.5
DEFAULT_NORMAL_RADIUS = 3


def decode_stanford_depth(raw_depth, scale=512.0, invalid_value=65535):
    """Decode documented camera-z depth in metres.

    The source HHA preparation used uint16 wraparound for ``65535 + 1``.  This
    implementation makes that sentinel handling explicit before converting to
    float, while retaining the valid ``(raw + 1) / 512`` encoding.
    """

    raw = np.asarray(raw_depth)
    if raw.ndim != 2:
        raise ValueError("raw depth must be a 2D array")
    valid = raw != invalid_value
    depth = np.zeros(raw.shape, dtype=np.float64)
    depth[valid] = (raw[valid].astype(np.float64) + 1.0) / float(scale)
    valid &= np.isfinite(depth) & (depth > 0.0)
    depth[~valid] = 0.0
    return depth, valid


def _box_sum(values, radius):
    size = 2 * int(radius) + 1
    return cv2.boxFilter(
        np.asarray(values, dtype=np.float64),
        ddepth=-1,
        ksize=(size, size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )


def estimate_rel_normals(points_camera, valid, radius=DEFAULT_NORMAL_RADIUS):
    """REL square-support algebraic plane normal estimator.

    This is the perspective counterpart of the local REL/HHA plane fit.  The
    radius-3 support is the local implementation's perspective default.
    """

    points = np.asarray(points_camera, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if points.shape != valid.shape + (3,):
        raise ValueError("points and valid shapes do not match")
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    safe = valid & np.isfinite(points).all(axis=-1) & (z > 0.0)

    x_over_z = np.zeros_like(z)
    y_over_z = np.zeros_like(z)
    one_over_z = np.zeros_like(z)
    x_over_z[safe] = x[safe] / z[safe]
    y_over_z[safe] = y[safe] / z[safe]
    one_over_z[safe] = 1.0 / z[safe]
    one = safe.astype(np.float64)

    raw = [
        x_over_z * x_over_z,
        x_over_z * y_over_z,
        x_over_z,
        y_over_z * y_over_z,
        y_over_z,
        one,
        x_over_z * one_over_z,
        y_over_z * one_over_z,
        one_over_z,
    ]
    summed = [_box_sum(component, radius) for component in raw]
    a, b, c, d, e, f_value, bx, by, bz = summed

    inv0 = d * f_value - e * e
    inv1 = -b * f_value + c * e
    inv2 = b * e - c * d
    inv3 = a * f_value - c * c
    inv4 = -a * e + b * c
    inv5 = a * d - b * b
    determinant = a * inv0 + b * inv1 + c * inv2

    normals = np.stack(
        [
            inv0 * bx + inv1 * by + inv2 * bz,
            inv1 * bx + inv3 * by + inv4 * bz,
            inv2 * bx + inv4 * by + inv5 * bz,
        ],
        axis=-1,
    )
    norm = np.linalg.norm(normals, axis=-1)
    support = _box_sum(one, radius)
    normal_valid = (
        safe
        & (support >= 3.0)
        & np.isfinite(determinant)
        & (np.abs(determinant) > 1e-12)
        & np.isfinite(norm)
        & (norm > 1e-12)
    )
    normals[normal_valid] /= norm[normal_valid, None]

    sign_z = np.sign(normals[..., 2])
    sign_z[sign_z == 0.0] = 1.0
    normals *= sign_z[..., None]
    facing = np.sign(np.sum(normals * points, axis=-1))
    facing[~np.isfinite(facing) | (facing == 0.0)] = 1.0
    normals *= facing[..., None]
    normals[~normal_valid] = 0.0
    return normals, normal_valid


def _scale_valid(values, valid):
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    out = np.zeros(values.shape, dtype=np.float64)
    if not np.any(valid):
        return out
    low = float(np.min(values[valid]))
    high = float(np.max(values[valid]))
    if high > low:
        out[valid] = (values[valid] - low) / (high - low)
    return out


def _quantize_unit_interval(values):
    values = np.asarray(values, dtype=np.float64)
    return np.floor(255.0 * np.clip(values, 0.0, 1.0) + 0.5).astype(np.uint8)


def encode_relplus_channels(
    points_rel,
    normals_world,
    valid,
    alpha_degrees=DEFAULT_ALPHA_DEGREES,
    lambda_angle=DEFAULT_LAMBDA,
):
    """Encode frozen ``[ReD, EGVIA, LOA]`` uint8 channels."""

    points = np.asarray(points_rel, dtype=np.float64)
    normals = np.asarray(normals_world, dtype=np.float64)
    valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(points).all(axis=-1)
        & np.isfinite(normals).all(axis=-1)
    )
    if points.shape != normals.shape or points.shape != valid.shape + (3,):
        raise ValueError("points, normals, and valid shapes do not match")
    if not 0.0 <= lambda_angle <= 1.0:
        raise ValueError("lambda_angle must be in [0, 1]")
    if not 0.0 < alpha_degrees < 90.0:
        raise ValueError("alpha_degrees must be in (0, 90)")

    normal_norm = np.linalg.norm(normals, axis=-1)
    unit_normals = np.zeros_like(normals)
    usable_normal = valid & np.isfinite(normal_norm) & (normal_norm > 1e-12)
    unit_normals[usable_normal] = normals[usable_normal] / normal_norm[usable_normal, None]
    red = np.hypot(points[..., 0], points[..., 1])
    valid = usable_normal & np.isfinite(red) & (red > 1e-12)

    red_01 = _scale_valid(red, valid)

    z = points[..., 2]
    height = np.zeros_like(z)
    if np.any(valid):
        height[valid] = z[valid] - float(np.min(z[valid]))
    height_01 = _scale_valid(height, valid)

    dot_gravity = np.sum(unit_normals * GRAVITY_WORLD, axis=-1)
    angle = np.arccos(np.clip(dot_gravity, -1.0, 1.0))
    angle_01 = angle / np.pi
    egvia_01 = np.zeros_like(angle)
    alpha = np.deg2rad(float(alpha_degrees))
    horizontal = valid & ((angle < alpha) | (angle > np.pi - alpha))
    egvia_01[valid & ~horizontal] = angle_01[valid & ~horizontal]
    egvia_01[horizontal] = (
        float(lambda_angle) * angle_01[horizontal]
        + (1.0 - float(lambda_angle)) * height_01[horizontal]
    )

    radial_unit = np.zeros_like(points)
    radial_unit[valid, 0] = points[valid, 0] / red[valid]
    radial_unit[valid, 1] = points[valid, 1] / red[valid]
    tangent = np.zeros_like(points)
    tangent[valid] = np.cross(GRAVITY_WORLD, radial_unit[valid])
    orientation_dot = np.sum(unit_normals * tangent, axis=-1)
    loa_angle = np.arccos(np.clip(orientation_dot, -1.0, 1.0))
    loa_01 = loa_angle / np.pi

    relplus = _quantize_unit_interval(np.stack([red_01, egvia_01, loa_01], axis=-1))
    relplus[~valid] = 255
    auxiliary = {
        "red": np.where(valid, red, 0.0),
        "red_01": np.where(valid, red_01, 0.0),
        "height": np.where(valid, height, 0.0),
        "height_01": np.where(valid, height_01, 0.0),
        "angle_degrees": np.where(valid, np.rad2deg(angle), 0.0),
        "angle_01": np.where(valid, angle_01, 0.0),
        "egvia_01": np.where(valid, egvia_01, 0.0),
        "loa_degrees": np.where(valid, np.rad2deg(loa_angle), 0.0),
        "loa_01": np.where(valid, loa_01, 0.0),
        "horizontal_mask": horizontal,
        "valid": valid,
        "valid_count": int(np.count_nonzero(valid)),
    }
    return relplus, auxiliary


def compute_relplus(depth_metres, valid_depth, camera, normal_radius=DEFAULT_NORMAL_RADIUS):
    """Compute camera-centred REL-default in gravity-aligned world axes."""

    depth = np.asarray(depth_metres, dtype=np.float64)
    valid_depth = np.asarray(valid_depth, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    points_camera = backproject_z_depth(depth, camera.k, pixel_origin=1.0)
    normals_camera, normal_valid = estimate_rel_normals(
        points_camera, valid_depth, radius=normal_radius
    )
    points_rel = rotate_camera_vectors_to_world(points_camera, camera.r_world_to_camera)
    points_world = camera_to_world(
        points_camera, camera.r_world_to_camera, camera.camera_center_world
    )
    normals_world = rotate_camera_vectors_to_world(normals_camera, camera.r_world_to_camera)
    relplus, auxiliary = encode_relplus_channels(
        points_rel, normals_world, valid_depth & normal_valid
    )
    auxiliary.update(
        {
            "points_camera": points_camera,
            "points_rel": points_rel,
            "points_world": points_world,
            "normals_camera": normals_camera,
            "normals_world": normals_world,
            "normal_valid": normal_valid,
        }
    )
    return relplus, auxiliary
