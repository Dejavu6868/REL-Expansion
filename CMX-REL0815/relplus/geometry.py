"""Camera geometry with the Stanford2D3D pose convention made explicit."""

from dataclasses import dataclass
import json

import numpy as np


@dataclass(frozen=True)
class CameraMetadata:
    """Per-frame camera metadata.

    ``r_world_to_camera`` and ``t_world_to_camera`` satisfy
    ``p_camera = R @ p_world + t`` for column vectors.  The corresponding
    row-vector camera-to-world transform is ``p_world = p_camera @ R + C``.
    """

    k: np.ndarray
    r_world_to_camera: np.ndarray
    t_world_to_camera: np.ndarray
    camera_center_world: np.ndarray
    center_residual: float


def _as_finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError("{} must have shape {}, got {}".format(name, shape, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("{} contains NaN or Inf".format(name))
    return array


def load_camera_metadata(path, center_tolerance=1e-4, rotation_tolerance=1e-5):
    """Load and validate a real Stanford2D3D world-to-camera pose JSON.

    No identity-pose fallback or convention guessing is permitted.
    """

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"camera_k_matrix", "camera_rt_matrix", "camera_location"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("pose metadata missing keys: {}".format(", ".join(missing)))

    k = _as_finite_array(payload["camera_k_matrix"], (3, 3), "camera_k_matrix")
    rt = _as_finite_array(payload["camera_rt_matrix"], (3, 4), "camera_rt_matrix")
    center = _as_finite_array(payload["camera_location"], (3,), "camera_location")
    r_wc = rt[:, :3]
    t_wc = rt[:, 3]

    orthogonality_error = float(np.max(np.abs(r_wc @ r_wc.T - np.eye(3))))
    determinant = float(np.linalg.det(r_wc))
    if orthogonality_error > rotation_tolerance or abs(determinant - 1.0) > rotation_tolerance:
        raise ValueError(
            "camera rotation is invalid: orthogonality_error={}, determinant={}".format(
                orthogonality_error, determinant
            )
        )

    recovered_center = -r_wc.T @ t_wc
    center_residual = float(np.linalg.norm(recovered_center - center))
    if center_residual > center_tolerance:
        raise ValueError(
            "camera_rt_matrix is not the validated world-to-camera [R|t] convention "
            "for camera_location: residual={}".format(center_residual)
        )
    if k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[2, 2] - 1.0) > 1e-8:
        raise ValueError("camera intrinsics are invalid")

    return CameraMetadata(k, r_wc, t_wc, center, center_residual)


def backproject_z_depth(depth, k, pixel_origin=1.0):
    """Backproject camera-z depth using the dataset's 1-based HHA convention."""

    depth = np.asarray(depth, dtype=np.float64)
    k = _as_finite_array(k, (3, 3), "k")
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    height, width = depth.shape
    u = np.arange(width, dtype=np.float64) + float(pixel_origin)
    v = np.arange(height, dtype=np.float64) + float(pixel_origin)
    uu, vv = np.meshgrid(u, v)
    z = depth
    x = (uu - k[0, 2]) * z / k[0, 0]
    y = (vv - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=-1)


def camera_to_world(points_camera, r_world_to_camera, camera_center_world):
    points = np.asarray(points_camera, dtype=np.float64)
    rotation = _as_finite_array(r_world_to_camera, (3, 3), "r_world_to_camera")
    center = _as_finite_array(camera_center_world, (3,), "camera_center_world")
    return points @ rotation + center


def world_to_camera(points_world, r_world_to_camera, camera_center_world):
    points = np.asarray(points_world, dtype=np.float64)
    rotation = _as_finite_array(r_world_to_camera, (3, 3), "r_world_to_camera")
    center = _as_finite_array(camera_center_world, (3,), "camera_center_world")
    return (points - center) @ rotation.T


def rotate_camera_vectors_to_world(vectors_camera, r_world_to_camera):
    vectors = np.asarray(vectors_camera, dtype=np.float64)
    rotation = _as_finite_array(r_world_to_camera, (3, 3), "r_world_to_camera")
    return vectors @ rotation


def resize_intrinsics(k, source_shape, destination_shape):
    """Update K for a pure resize under the inherited 1-based convention."""

    k = _as_finite_array(k, (3, 3), "k").copy()
    source_height, source_width = source_shape
    destination_height, destination_width = destination_shape
    sx = float(destination_width) / float(source_width)
    sy = float(destination_height) / float(source_height)
    transform = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    return transform @ k


def crop_intrinsics(k, left, top):
    k = _as_finite_array(k, (3, 3), "k").copy()
    k[0, 2] -= float(left)
    k[1, 2] -= float(top)
    return k


def pad_intrinsics(k, left, top):
    k = _as_finite_array(k, (3, 3), "k").copy()
    k[0, 2] += float(left)
    k[1, 2] += float(top)
    return k


def horizontal_flip_intrinsics(k, width):
    """Update K for ``u' = width + 1 - u`` in 1-based pixel coordinates."""

    k = _as_finite_array(k, (3, 3), "k").copy()
    k[0, 0] *= -1.0
    k[0, 2] = float(width) + 1.0 - k[0, 2]
    return k
