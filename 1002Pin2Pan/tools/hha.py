"""Pinhole HHA (Depth2HHA-python, as referenced by CMX) and its ERP counterpart.

The ERP version keeps every Depth2HHA definition that does not depend on the
camera model: normals from a radius-3 window, gravity from radius-10 normals
with ``getYDir`` (thresholds 45/15 degrees, 5+5 iterations), the angle channel
``angle + 128 - 90``, the height channel ``h - yMin`` with the ``yMin > -90 ->
-130`` rule, and the disparity channel ``31000 / max(depth_cm, 100)``.
Points and normals come from the ERP helpers of the original REL source. The
one definitional change: ERP has no z-depth, so disparity uses the range
(distance along the ray) instead.
"""

import sys
from pathlib import Path

import numpy as np

PIN2PAN_ROOT = Path(__file__).resolve().parents[1]
if str(PIN2PAN_ROOT) not in sys.path:
    sys.path.insert(0, str(PIN2PAN_ROOT))

from vendor.depth2hha.getHHA import getHHA  # noqa: E402
from vendor.depth2hha.utils.rgbd_util import processDepthImage  # noqa: E402
from vendor.depth2hha.utils.util import getYDir  # noqa: E402

ANGLE_THRESHOLDS = np.array([45, 15])
GRAVITY_ITERATIONS = np.array([5, 5])


def _helper_k(k_json):
    """Depth2HHA uses one-based pixel indices, like the REL helper."""
    helper = np.asarray(k_json, dtype=np.float64).copy()
    helper[0, 2] += 0.5
    helper[1, 2] += 0.5
    return helper


def _angle_degrees(normals, direction):
    cosine = np.clip(np.sum(normals * direction, axis=2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    angle[np.isnan(angle)] = 180
    return angle


def pinhole_hha(depth_m, valid_mask, k_json):
    """Unmodified Depth2HHA ``getHHA`` plus its intermediate geometry."""
    depth = np.where(valid_mask, depth_m, 0.0).astype(np.float64)
    helper = _helper_k(k_json)
    with np.errstate(divide="ignore", invalid="ignore"):
        encoded = getHHA(helper, depth, depth)
        points, normals, gravity, _, rotated, _ = processDepthImage(
            depth * 100, depth == 0, helper
        )
    return encoded, {
        "angle_deg": _angle_degrees(normals, gravity),
        "height_cm": -rotated[:, :, 1],
        "disparity_depth_cm": points[:, :, 2],
        "gravity_camera": np.asarray(gravity, dtype=np.float64),
        "normals": normals,
    }


def erp_hha(depth_erp_m, frozen):
    """HHA for an ERP panorama; ``frozen`` supplies the original ERP helpers."""
    depth = np.asarray(depth_erp_m, dtype=np.float64)
    missing = depth == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        points = frozen["getPointCloud_ERP"](depth * 100)
        normals, _ = frozen["computeNormalsSquareSupport_ERP"](
            points, missing, 3, np.ones(depth.shape)
        )
        coarse, _ = frozen["computeNormalsSquareSupport_ERP"](
            points, missing, 10, np.ones(depth.shape)
        )
        gravity = getYDir(
            coarse, ANGLE_THRESHOLDS, GRAVITY_ITERATIONS, np.array([0.0, 0.0, -1.0])
        )
    gravity = np.real(gravity) / np.linalg.norm(np.real(gravity))
    angle = _angle_degrees(normals, gravity)
    height_cm = -np.sum(points * gravity, axis=2)
    height = height_cm.copy()
    y_min = np.percentile(height, 0)
    if y_min > -90:
        y_min = -130
    height = height - y_min
    range_cm = np.linalg.norm(points, axis=2)

    encoded = np.zeros(depth.shape + (3,))
    encoded[:, :, 2] = 31000 / np.maximum(range_cm, 100)
    encoded[:, :, 1] = height
    encoded[:, :, 0] = angle + 128 - 90
    encoded = np.rint(encoded)
    encoded[encoded > 255] = 255
    return encoded.astype(np.uint8), {
        "angle_deg": angle,
        "height_cm": height_cm,
        "disparity_depth_cm": range_cm,
        "gravity_erp": gravity,
        "normals": normals,
    }
