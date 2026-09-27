"""Explicit OpenCV storage for the semantic channel order."""

from pathlib import Path

import cv2
import numpy as np


def _validate_rel_plus_array(array):
    rel_plus = np.asarray(array)
    if rel_plus.ndim != 3 or rel_plus.shape[2] != 3 or rel_plus.dtype != np.uint8:
        raise ValueError("REL+ image must be an HxWx3 uint8 array")
    return rel_plus


def save_rel_plus_png(path, array):
    """Write channels as stored, with no RGB/BGR conversion."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rel_plus = _validate_rel_plus_array(array)
    if not cv2.imwrite(str(destination), rel_plus):
        raise OSError("failed to write REL+ PNG: {}".format(destination))


def load_rel_plus_png(path):
    """Read channels as stored, with no RGB/BGR conversion."""
    source = Path(path)
    array = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise OSError("failed to read REL+ PNG: {}".format(source))
    return _validate_rel_plus_array(array)
