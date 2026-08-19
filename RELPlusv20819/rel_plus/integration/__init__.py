"""Training-preparation adapters; this package does not start training."""

from .cmx_preprocess import (
    PreprocessedBatch,
    SpatialTransform,
    apply_cmx_compatible_preprocess,
    sample_spatial_transform,
)

__all__ = [
    "PreprocessedBatch",
    "SpatialTransform",
    "apply_cmx_compatible_preprocess",
    "sample_spatial_transform",
]
