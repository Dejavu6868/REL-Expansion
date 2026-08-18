"""Stanford2D3D perspective-camera geometry for REL+ v1."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _as_finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError("{} must have shape {}, got {}".format(name, shape, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return array.copy()


@dataclass(frozen=True)
class CameraGeometry:
    """Camera intrinsics and an explicit world-to-camera rigid transform."""

    K_json: np.ndarray
    R_world_to_camera: np.ndarray
    t_world_to_camera: Optional[np.ndarray] = None

    def __post_init__(self):
        k_matrix = _as_finite_array(self.K_json, (3, 3), "K_json")
        rotation = _as_finite_array(
            self.R_world_to_camera, (3, 3), "R_world_to_camera"
        )
        translation = self.t_world_to_camera
        if translation is not None:
            translation = _as_finite_array(
                translation, (3,), "t_world_to_camera"
            )

        if k_matrix[0, 0] <= 0.0 or k_matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if not np.allclose(k_matrix[2], [0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("K_json must have final row [0, 0, 1]")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("R_world_to_camera must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("R_world_to_camera must have determinant +1")

        object.__setattr__(self, "K_json", k_matrix)
        object.__setattr__(self, "R_world_to_camera", rotation)
        object.__setattr__(self, "t_world_to_camera", translation)

    @classmethod
    def from_json_k(cls, k_matrix, rotation, translation=None):
        """Build from Stanford JSON intrinsics (zero-based pixel-centre convention)."""
        return cls(k_matrix, rotation, translation)

    @classmethod
    def from_zero_based_center_k(cls, k_matrix, rotation, translation=None):
        """Convert K whose principal point is expressed at zero-based centres."""
        k_json = _as_finite_array(k_matrix, (3, 3), "k_matrix")
        k_json[0, 2] += 0.5
        k_json[1, 2] += 0.5
        return cls.from_json_k(k_json, rotation, translation)


def load_stanford_s2d_camera_geometry(path):
    """Load and validate Stanford2D3D S2D JSON as an explicit W2C pose."""
    pose_path = Path(path)
    with pose_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    try:
        k_matrix = payload["camera_k_matrix"]
        rt_matrix = _as_finite_array(
            payload["camera_rt_matrix"], (3, 4), "camera_rt_matrix"
        )
        camera_location = _as_finite_array(
            payload["camera_location"], (3,), "camera_location"
        )
    except KeyError as error:
        raise ValueError("missing camera metadata key: {}".format(error.args[0]))

    camera = CameraGeometry.from_json_k(
        k_matrix, rt_matrix[:, :3], rt_matrix[:, 3]
    )
    recovered_location = -camera.R_world_to_camera.T @ camera.t_world_to_camera
    if not np.allclose(recovered_location, camera_location, rtol=0.0, atol=1e-4):
        raise ValueError(
            "camera_rt_matrix is not the expected world-to-camera transform: "
            "-R.T @ t does not match camera_location"
        )
    return camera


def backproject_z_depth(depth_m, valid_mask, k_matrix):
    """Backproject metric z-depth at JSON pixel centres (u+0.5, v+0.5)."""
    depth = np.asarray(depth_m)
    valid = np.asarray(valid_mask, dtype=bool)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("depth_m and valid_mask must be matching 2D arrays")
    k_json = _as_finite_array(k_matrix, (3, 3), "k_matrix")

    rows, columns = np.indices(depth.shape, dtype=np.float64)
    z_value = depth.astype(np.float64, copy=False)
    x_value = (columns + 0.5 - k_json[0, 2]) * z_value / k_json[0, 0]
    y_value = (rows + 0.5 - k_json[1, 2]) * z_value / k_json[1, 1]
    points = np.stack([x_value, y_value, z_value], axis=-1)
    points[~valid] = 0.0
    return points


def json_k_to_rel_helper_k(k_matrix):
    """Adapt JSON K once for the original helper's one-based meshgrid."""
    helper_k = _as_finite_array(k_matrix, (3, 3), "k_matrix")
    helper_k[0, 2] += 0.5
    helper_k[1, 2] += 0.5
    return helper_k


def resize_camera_geometry(camera, source_shape, destination_shape):
    """Scale both intrinsic rows for a pure image resize."""
    source_height, source_width = source_shape
    destination_height, destination_width = destination_shape
    if min(source_height, source_width, destination_height, destination_width) <= 0:
        raise ValueError("source and destination shapes must be positive")
    scaled_k = camera.K_json.copy()
    scaled_k[0, :] *= float(destination_width) / float(source_width)
    scaled_k[1, :] *= float(destination_height) / float(source_height)
    return CameraGeometry.from_json_k(
        scaled_k, camera.R_world_to_camera, camera.t_world_to_camera
    )


def gravity_in_camera(rotation_world_to_camera):
    """Transform frozen world-down [0, 0, -1] into the camera frame."""
    rotation = _as_finite_array(
        rotation_world_to_camera, (3, 3), "rotation_world_to_camera"
    )
    gravity = rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return gravity / np.linalg.norm(gravity)
