"""Geometry-aware REL+ preparation for the real CMX input pipeline."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import backproject_z_depth
from .representation import encode_relplus_channels, estimate_rel_normals


_EPS = 1.0e-12
_GRAVITY_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class SpatialTransformParameters:
    resize_height: int
    resize_width: int
    crop_y: int
    crop_x: int
    crop_height: int
    crop_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    flip: bool = False


def update_intrinsics(k, source_shape, parameters):
    """Apply resize, crop, and padding to K under the 0.5 pixel-center contract."""

    intrinsics = np.asarray(k, dtype=np.float64).copy()
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("K must be a finite 3x3 matrix")
    source_height, source_width = source_shape
    sx = float(parameters.resize_width) / float(source_width)
    sy = float(parameters.resize_height) / float(source_height)
    intrinsics[0, :] *= sx
    intrinsics[1, :] *= sy
    intrinsics[0, 2] += -float(parameters.crop_x) + float(parameters.pad_left)
    intrinsics[1, 2] += -float(parameters.crop_y) + float(parameters.pad_top)
    if parameters.flip:
        raise ValueError("horizontal flip is disabled for the four-arm geometry policy")
    return intrinsics


def transform_depth_geometry(depth, valid, k, parameters):
    """Transform Z-depth and validity with the exact parameters shared by RGB/label."""

    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("depth and valid must be matching 2D arrays")
    if parameters.flip:
        raise ValueError("horizontal flip is disabled for the four-arm geometry policy")
    resized_depth = cv2.resize(
        depth,
        (parameters.resize_width, parameters.resize_height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_valid = cv2.resize(
        valid.astype(np.uint8),
        (parameters.resize_width, parameters.resize_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    y0, x0 = parameters.crop_y, parameters.crop_x
    cropped_depth = resized_depth[
        y0 : y0 + parameters.crop_height,
        x0 : x0 + parameters.crop_width,
    ]
    cropped_valid = resized_valid[
        y0 : y0 + parameters.crop_height,
        x0 : x0 + parameters.crop_width,
    ]
    transformed_depth = cv2.copyMakeBorder(
        cropped_depth,
        parameters.pad_top,
        parameters.pad_bottom,
        parameters.pad_left,
        parameters.pad_right,
        cv2.BORDER_CONSTANT,
        value=0.0,
    )
    transformed_valid = cv2.copyMakeBorder(
        cropped_valid.astype(np.uint8),
        parameters.pad_top,
        parameters.pad_bottom,
        parameters.pad_left,
        parameters.pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    ).astype(bool)
    transformed_depth[~transformed_valid] = 0.0
    transformed_k = update_intrinsics(k, depth.shape, parameters)
    return transformed_depth, transformed_valid, transformed_k


def gravity_to_down_rotation(gravity_down_camera):
    gravity = np.asarray(gravity_down_camera, dtype=np.float64)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity must be a finite 3-vector")
    norm = float(np.linalg.norm(gravity))
    if norm <= _EPS:
        raise ValueError("gravity norm is zero")
    gravity = gravity / norm
    cross = np.cross(gravity, _GRAVITY_DOWN)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(gravity, _GRAVITY_DOWN), -1.0, 1.0))
    if sine <= _EPS:
        if cosine >= 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(gravity)))]
        axis = np.cross(gravity, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def estimate_gravity_down_camera(
    normals_camera,
    normal_valid,
    initial_gravity=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    angle_thresholds=(45.0, 15.0),
    iterations=(5, 5),
):
    """REL-default EstGravity using floor/wall normal sets in camera coordinates."""
    normals = np.asarray(normals_camera, dtype=np.float64)
    valid = np.asarray(normal_valid, dtype=bool) & np.isfinite(normals).all(axis=-1)
    samples = normals[valid]
    lengths = np.linalg.norm(samples, axis=1)
    samples = samples[np.isfinite(lengths) & (lengths > _EPS)]
    if len(samples) < 10:
        raise ValueError("EstGravity requires at least 10 finite normals")
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    gravity = np.asarray(initial_gravity, dtype=np.float64).reshape(3)
    gravity /= np.linalg.norm(gravity)
    for angle_degrees, count in zip(angle_thresholds, iterations):
        threshold = np.radians(float(angle_degrees))
        for _ in range(int(count)):
            similarity = samples @ gravity
            floor = np.abs(similarity) > np.cos(threshold)
            wall = np.abs(similarity) < np.sin(threshold)
            if int(floor.sum()) < 5 or int(wall.sum()) < 5:
                break
            matrix = samples[wall].T @ samples[wall] - samples[floor].T @ samples[floor]
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            updated = eigenvectors[:, int(np.argmin(eigenvalues))]
            if np.dot(gravity, updated) < 0.0:
                updated = -updated
            updated /= np.linalg.norm(updated)
            if np.linalg.norm(gravity - updated) < 1.0e-3:
                gravity = updated
                break
            gravity = updated
    if not np.isfinite(gravity).all() or abs(float(np.linalg.norm(gravity)) - 1.0) > 1.0e-10:
        raise ValueError("EstGravity produced a non-finite or non-unit direction")
    return np.ascontiguousarray(gravity)


def _depth_geometry(depth, valid, k, normal_radius):
    points_camera = backproject_z_depth(depth, k, pixel_origin=0.5)
    normals_camera, normal_valid = estimate_rel_normals(points_camera, valid, radius=normal_radius)
    return points_camera, normals_camera, np.asarray(valid, dtype=bool) & normal_valid


def _encode_with_gravity(points_camera, normals_camera, rel_valid, gravity_down_camera):
    """Single downstream encoder shared by Local and Pose gravity."""
    align = gravity_to_down_rotation(gravity_down_camera)
    points_aligned = np.ascontiguousarray(points_camera @ align.T)
    normals_aligned = np.ascontiguousarray(normals_camera @ align.T)
    relplus, auxiliary = encode_relplus_channels(points_aligned, normals_aligned, rel_valid)
    auxiliary = dict(auxiliary)
    auxiliary["gravity_down_camera"] = np.asarray(gravity_down_camera, dtype=np.float64)
    return relplus, np.asarray(auxiliary["valid"], dtype=bool), auxiliary


def generate_relplus_from_depth(depth, valid, k, r_world_to_camera, normal_radius=3):
    """Generate PoseGravity REL+ from transformed Z-depth and updated K."""
    points_camera, normals_camera, rel_valid = _depth_geometry(depth, valid, k, normal_radius)
    rotation = np.asarray(r_world_to_camera, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("world-to-camera rotation must be finite 3x3")
    gravity_camera = rotation @ _GRAVITY_DOWN
    return _encode_with_gravity(points_camera, normals_camera, rel_valid, gravity_camera)


def generate_relplus_from_depth_local(depth, valid, k, normal_radius=3):
    """Generate Local REL+ without accepting or reading a pose-derived rotation."""
    points_camera, normals_camera, rel_valid = _depth_geometry(depth, valid, k, normal_radius)
    gravity_camera = estimate_gravity_down_camera(normals_camera, rel_valid)
    return _encode_with_gravity(points_camera, normals_camera, rel_valid, gravity_camera)
