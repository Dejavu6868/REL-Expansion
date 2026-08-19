"""CMX-compatible post-generation spatial preprocessing for REL+ v2."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..policy import validate_rel_plus_augmentation_policy
from ..storage import load_rel_plus_png


CMX_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
CMX_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)


@dataclass(frozen=True)
class SpatialTransform:
    scale: float
    crop_top: int
    crop_left: int
    output_height: int
    output_width: int

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        if min(self.crop_top, self.crop_left) < 0:
            raise ValueError("crop offsets must be nonnegative")
        if min(self.output_height, self.output_width) <= 0:
            raise ValueError("output dimensions must be positive")


@dataclass(frozen=True)
class PreprocessedBatch:
    rgb_chw: np.ndarray
    rel_plus_chw: np.ndarray
    label: np.ndarray
    transform: SpatialTransform


def sample_spatial_transform(input_shape, scales, output_shape, rng):
    """Sample one transform for RGB, REL+ and label using one RNG."""
    if not scales:
        raise ValueError("scales must not be empty")
    scale = float(rng.choice(np.asarray(scales, dtype=np.float64)))
    scaled_height = max(1, int(int(input_shape[0]) * scale))
    scaled_width = max(1, int(int(input_shape[1]) * scale))
    output_height, output_width = (int(output_shape[0]), int(output_shape[1]))
    # Match the audited CMX generate_random_crop_pos upper bound, including
    # its one-pixel-short crop possibility followed by centred padding.
    max_top = scaled_height - output_height + 1 if scaled_height > output_height else 0
    max_left = scaled_width - output_width + 1 if scaled_width > output_width else 0
    crop_top = int(rng.integers(0, max_top + 1)) if max_top else 0
    crop_left = int(rng.integers(0, max_left + 1)) if max_left else 0
    return SpatialTransform(
        scale, crop_top, crop_left, output_height, output_width
    )


def _normalize(image, mean, std):
    return (image.astype(np.float64) / 255.0 - mean) / std


def _crop_and_center_pad(image, transform, pad_value):
    height, width = image.shape[:2]
    if transform.crop_top >= height or transform.crop_left >= width:
        raise ValueError("crop origin lies outside the scaled image")
    crop = image[
        transform.crop_top : transform.crop_top + transform.output_height,
        transform.crop_left : transform.crop_left + transform.output_width,
        ...,
    ]
    pad_height = max(0, transform.output_height - crop.shape[0])
    pad_width = max(0, transform.output_width - crop.shape[1])
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    return cv2.copyMakeBorder(
        crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
    )


def apply_cmx_compatible_preprocess(
    rgb,
    rel_plus,
    label,
    transform,
    *,
    norm_mean=CMX_NORM_MEAN,
    norm_std=CMX_NORM_STD,
    photometric_augmentation=None,
    horizontal_flip=False,
    vertical_flip=False,
    arbitrary_rotation=False,
    perspective_warp=False
):
    """Apply one post-generation transform and return CMX-like model inputs."""
    validate_rel_plus_augmentation_policy(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        arbitrary_rotation=arbitrary_rotation,
        perspective_warp=perspective_warp,
    )
    rgb_array = np.asarray(rgb)
    rel_array = np.asarray(rel_plus)
    label_array = np.asarray(label)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if rel_array.shape != rgb_array.shape or rel_array.dtype != np.uint8:
        raise ValueError("rel_plus must be an HxWx3 uint8 array matching rgb")
    if label_array.shape != rgb_array.shape[:2]:
        raise ValueError("label must be an HxW array matching rgb")
    scaled_height = max(1, int(rgb_array.shape[0] * transform.scale))
    scaled_width = max(1, int(rgb_array.shape[1] * transform.scale))
    size = (scaled_width, scaled_height)
    rgb_scaled = cv2.resize(rgb_array, size, interpolation=cv2.INTER_LINEAR)
    rel_scaled = cv2.resize(rel_array, size, interpolation=cv2.INTER_LINEAR)
    label_scaled = cv2.resize(label_array, size, interpolation=cv2.INTER_NEAREST)

    if photometric_augmentation is not None:
        rgb_scaled = np.asarray(photometric_augmentation(rgb_scaled))
        if rgb_scaled.shape != rel_scaled.shape:
            raise ValueError("photometric_augmentation changed RGB shape")

    mean = np.asarray(norm_mean, dtype=np.float64)
    std = np.asarray(norm_std, dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0.0):
        raise ValueError("norm_mean and norm_std must be three-vectors with positive std")
    rgb_normalized = _normalize(rgb_scaled, mean, std)
    rel_normalized = _normalize(rel_scaled, mean, std)
    rgb_output = _crop_and_center_pad(rgb_normalized, transform, 0.0)
    rel_output = _crop_and_center_pad(rel_normalized, transform, 0.0)
    label_output = _crop_and_center_pad(label_scaled, transform, 255)
    return PreprocessedBatch(
        rgb_output.transpose(2, 0, 1),
        rel_output.transpose(2, 0, 1),
        label_output,
        transform,
    )


def load_and_preprocess_for_cmx(rgb, rel_plus_path, label, transform, **kwargs):
    """Compatibility loader that preserves stored [EGVIA, LOA, ReD] bytes."""
    rel_plus = load_rel_plus_png(rel_plus_path)
    return apply_cmx_compatible_preprocess(
        rgb, rel_plus, label, transform, **kwargs
    )
