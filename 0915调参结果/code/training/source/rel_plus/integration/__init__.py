"""Training-preparation adapters; this package does not start training."""

from .cmx_preprocess import (
    PreprocessedBatch,
    SpatialTransform,
    analyze_invalid_interpolation,
    apply_cmx_compatible_preprocess,
    sample_spatial_transform,
)

__all__ = [
    "PreprocessedBatch",
    "SpatialTransform",
    "analyze_invalid_interpolation",
    "apply_cmx_compatible_preprocess",
    "sample_spatial_transform",
]
