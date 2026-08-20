"""Gravity alignment with an explicit anti-parallel failure contract."""

import numpy as np

from .constants import GRAVITY_ANTIPARALLEL_THRESHOLD_DEGREES
from .source_helpers import get_r_matrix, rotate_pc


class GravityAlignmentSingularity(ValueError):
    """Raised when source-equivalent 180-degree alignment has no unique yaw."""


def align_points_and_normals_to_gravity(
    points, normals, gravity_camera, *, sample_id="<unspecified>"
):
    """Apply source ``getRMatrix(target, source)`` then ``rotatePC(..., R.T)``."""
    target_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gravity = np.asarray(gravity_camera, dtype=np.float64)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity_camera must be one finite 3-vector")
    norm = float(np.linalg.norm(gravity))
    if norm == 0.0:
        raise ValueError("gravity_camera must be nonzero")
    gravity = gravity / norm
    dot_value = float(np.clip(np.dot(target_down, gravity), -1.0, 1.0))
    angle_degrees = float(np.degrees(np.arccos(dot_value)))
    if angle_degrees >= GRAVITY_ANTIPARALLEL_THRESHOLD_DEGREES:
        raise GravityAlignmentSingularity(
            "sample_id={} gravity={} angle_deg={:.9f}: gravity alignment is "
            "anti-parallel and has no unique 180-degree yaw axis".format(
                sample_id, gravity.tolist(), angle_degrees
            )
        )
    source_rotation = get_r_matrix(target_down.T, gravity)
    alignment = source_rotation.T
    return (
        rotate_pc(np.asarray(points), alignment),
        rotate_pc(np.asarray(normals), alignment),
        alignment,
    )
