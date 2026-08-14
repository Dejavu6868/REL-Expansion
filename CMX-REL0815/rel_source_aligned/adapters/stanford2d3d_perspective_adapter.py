from dataclasses import dataclass

import numpy as np

from rel_source_aligned.reference.official_rel_core import SourceExactRELCore
from rel_source_aligned.reference.reference_loader import load_official_rel_module


@dataclass(frozen=True)
class PerspectiveRELInputs:
    decoded_depth: np.ndarray
    valid_mask: np.ndarray
    points_aligned: np.ndarray
    normals_aligned: np.ndarray
    azimuth: np.ndarray
    gravity_direction: np.ndarray


class PerspectiveInputAdapter:
    """Adapt S2D perspective depth/K to the author core's aligned XYZ convention."""

    def __init__(self, authority_root):
        self.authority_root = str(authority_root)
        self.reference = load_official_rel_module(authority_root)
        self.core = SourceExactRELCore()
        self.last_inputs = None

    @staticmethod
    def decode_depth(raw_depth):
        raw_depth = np.asarray(raw_depth)
        if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
            raise ValueError("Stanford2D3D perspective depth must be a uint16 image")
        valid = (raw_depth > 0) & (raw_depth != 65535)
        depth = raw_depth.astype(np.float64) / 512.0
        depth[~valid] = 0.0
        return depth, valid

    @staticmethod
    def validate_intrinsics(camera_matrix):
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            raise ValueError("camera_k_matrix must be finite 3x3")
        if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        return camera_matrix

    def adapt(self, raw_depth, camera_matrix):
        depth, valid = self.decode_depth(raw_depth)
        camera_matrix = self.validate_intrinsics(camera_matrix)
        missing = ~valid
        points, _, gravity, _, points_rotated, normals_rotated = (
            self.reference.processDepthImage(depth * 100, missing, camera_matrix)
        )

        # Perspective source geometry uses +y as gravity. The REL ERP core uses
        # -z as gravity, so this is a coordinate permutation, not a new formula.
        points_aligned = np.stack(
            [points_rotated[:, :, 0], points_rotated[:, :, 2], -points_rotated[:, :, 1]],
            axis=2,
        )
        normals_aligned = np.stack(
            [normals_rotated[:, :, 0], normals_rotated[:, :, 2], -normals_rotated[:, :, 1]],
            axis=2,
        )
        azimuth = np.arctan2(points[:, :, 0], points[:, :, 2])
        inputs = PerspectiveRELInputs(
            decoded_depth=depth,
            valid_mask=valid,
            points_aligned=points_aligned,
            normals_aligned=normals_aligned,
            azimuth=azimuth,
            gravity_direction=np.asarray(gravity),
        )
        self.last_inputs = inputs
        return inputs

    def encode(self, raw_depth, camera_matrix, alpha=45, lam=0.5):
        inputs = self.adapt(raw_depth, camera_matrix)
        return self.core.encode(
            inputs.points_aligned,
            inputs.normals_aligned,
            inputs.azimuth,
            ~inputs.valid_mask,
            alpha=alpha,
            lam=lam,
        )

