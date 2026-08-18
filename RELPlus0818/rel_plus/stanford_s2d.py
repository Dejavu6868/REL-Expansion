"""Production-facing Stanford2D3D S2D frame adapter."""

from pathlib import Path

import cv2
import numpy as np

from .camera import load_stanford_s2d_camera_geometry, resize_camera_geometry
from .depth import resize_raw_depth_nearest


def load_canonical_frame(depth_path, camera_json_path, canonical_shape=(480, 480)):
    """Load native Depth16/pose and apply the frozen canonical-only resize."""
    source = Path(depth_path)
    raw_depth = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise OSError("failed to read raw depth: {}".format(source))
    if raw_depth.ndim != 2 or raw_depth.dtype != np.uint16:
        raise ValueError("Stanford2D3D Depth16 must be a 2D uint16 PNG")
    camera = load_stanford_s2d_camera_geometry(camera_json_path)
    source_shape = raw_depth.shape
    if tuple(source_shape) != tuple(canonical_shape):
        raw_depth = resize_raw_depth_nearest(raw_depth, canonical_shape)
        camera = resize_camera_geometry(camera, source_shape, canonical_shape)
    return raw_depth, camera, source_shape
