"""Normal-quality diagnostics that never alter formal REL+ v2 bytes."""

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import REL_PLUS_V2_MIN_NORMAL_SUPPORT


@dataclass(frozen=True)
class NormalDiagnostics:
    finite_mask: np.ndarray
    nonzero_mask: np.ndarray
    support_count: np.ndarray
    quality_mask: np.ndarray

    def ratios(self, depth_valid):
        valid = np.asarray(depth_valid, dtype=bool)
        denominator = max(1, int(np.count_nonzero(valid)))
        return {
            "normal_invalid_ratio": float(
                np.count_nonzero(valid & ~self.finite_mask) / denominator
            ),
            "zero_normal_ratio": float(
                np.count_nonzero(valid & self.finite_mask & ~self.nonzero_mask)
                / denominator
            ),
            "low_support_ratio": float(
                np.count_nonzero(
                    valid & (self.support_count < REL_PLUS_V2_MIN_NORMAL_SUPPORT)
                )
                / denominator
            ),
            "normal_quality_ratio": float(
                np.count_nonzero(valid & self.quality_mask) / denominator
            ),
        }


def build_normal_diagnostics(normals, depth_valid, radius):
    normal_field = np.asarray(normals)
    valid = np.asarray(depth_valid, dtype=bool)
    if normal_field.shape != valid.shape + (3,):
        raise ValueError("normals and depth_valid shapes do not match")
    if radius <= 0:
        raise ValueError("radius must be positive")
    finite = np.all(np.isfinite(normal_field), axis=2)
    nonzero = finite & (np.linalg.norm(np.nan_to_num(normal_field), axis=2) > 0.0)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    support = cv2.filter2D(
        valid.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    support = np.rint(support).astype(np.int16)
    quality = (
        valid
        & finite
        & nonzero
        & (support >= REL_PLUS_V2_MIN_NORMAL_SUPPORT)
    )
    return NormalDiagnostics(finite, nonzero, support, quality)
