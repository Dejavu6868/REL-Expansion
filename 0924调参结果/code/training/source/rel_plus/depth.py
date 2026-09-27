"""Stanford2D3D S2D depth decoding and canonical resizing."""

import cv2
import numpy as np


def decode_stanford_s2d_depth(raw_depth):
    """Decode uint16 z-depth using raw/512; 0 and 65535 are invalid."""
    raw = np.asarray(raw_depth)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("raw_depth must be a 2D uint16 array")
    valid = (raw != 0) & (raw != np.iinfo(np.uint16).max)
    depth_m = raw.astype(np.float32) / np.float32(512.0)
    depth_m[~valid] = np.float32(0.0)
    return depth_m, valid


def resize_raw_depth_nearest(raw_depth, destination_shape):
    """Resize encoded depth without interpolating depth or sentinel values."""
    raw = np.asarray(raw_depth)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("raw_depth must be a 2D uint16 array")
    destination_height, destination_width = destination_shape
    if destination_height <= 0 or destination_width <= 0:
        raise ValueError("destination_shape must be positive")
    return cv2.resize(
        raw,
        (int(destination_width), int(destination_height)),
        interpolation=cv2.INTER_NEAREST,
    )
