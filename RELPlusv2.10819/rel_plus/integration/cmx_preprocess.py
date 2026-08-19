"""CMX-compatible post-generation preprocessing for REL+ v2.1."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..policy import validate_rel_plus_augmentation_policy
from ..storage import load_rel_plus_png


CMX_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CMX_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SpatialTransform:
    source_height: int
    source_width: int
    scale: float
    scaled_height: int
    scaled_width: int
    crop_top: int
    crop_left: int
    output_height: int
    output_width: int

    def __post_init__(self):
        integer_fields = (
            "source_height",
            "source_width",
            "scaled_height",
            "scaled_width",
            "crop_top",
            "crop_left",
            "output_height",
            "output_width",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError("{} must be an integer".format(name))
            object.__setattr__(self, name, int(value))
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        if min(
            self.source_height,
            self.source_width,
            self.scaled_height,
            self.scaled_width,
            self.output_height,
            self.output_width,
        ) <= 0:
            raise ValueError("source, scaled and output dimensions must be positive")
        if min(self.crop_top, self.crop_left) < 0:
            raise ValueError("crop offsets must be nonnegative")


@dataclass(frozen=True)
class PreprocessedBatch:
    rgb: np.ndarray
    modal_x: np.ndarray
    label: np.ndarray
    modal_x_valid_mask: np.ndarray
    transform: SpatialTransform

    @property
    def rgb_chw(self):
        return self.rgb

    @property
    def rel_plus_chw(self):
        return self.modal_x


def sample_spatial_transform(input_shape, scales, output_shape, rng):
    """Sample exactly one scale/crop/pad transform for all four arrays."""
    if len(tuple(input_shape)) != 2 or min(int(value) for value in input_shape) <= 0:
        raise ValueError("input_shape must contain positive height and width")
    if not scales:
        raise ValueError("scales must not be empty")
    source_height, source_width = (int(input_shape[0]), int(input_shape[1]))
    scale = float(rng.choice(np.asarray(scales, dtype=np.float64)))
    scaled_height = max(1, int(source_height * scale))
    scaled_width = max(1, int(source_width * scale))
    output_height, output_width = (int(output_shape[0]), int(output_shape[1]))
    max_top = scaled_height - output_height + 1 if scaled_height > output_height else 0
    max_left = scaled_width - output_width + 1 if scaled_width > output_width else 0
    draw_integer = rng.integers if hasattr(rng, "integers") else rng.randint
    crop_top = int(draw_integer(0, max_top + 1)) if max_top else 0
    crop_left = int(draw_integer(0, max_left + 1)) if max_left else 0
    return SpatialTransform(
        source_height,
        source_width,
        scale,
        scaled_height,
        scaled_width,
        crop_top,
        crop_left,
        output_height,
        output_width,
    )


def _validate_source_shapes(rgb, rel_plus, label, valid_mask, transform):
    source_shape = (transform.source_height, transform.source_width)
    if rgb.shape[:2] != source_shape:
        raise ValueError("rgb source shape does not match SpatialTransform")
    if rel_plus.shape[:2] != source_shape:
        raise ValueError("rel_plus source shape does not match SpatialTransform")
    if label.shape != source_shape:
        raise ValueError("label source shape does not match SpatialTransform")
    if valid_mask.shape != source_shape:
        raise ValueError("valid_mask source shape does not match SpatialTransform")


def _validate_input_contract(rgb, rel_plus, label, valid_mask):
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 in [0,255]")
    if rel_plus.ndim != 3 or rel_plus.shape[2] != 3:
        raise ValueError("rel_plus must have shape HxWx3")
    if rel_plus.dtype != np.uint8:
        raise ValueError("rel_plus must be uint8 in [0,255]")
    if label.ndim != 2 or not np.issubdtype(label.dtype, np.integer):
        raise ValueError("label must be an HxW integer array")
    if valid_mask.ndim != 2 or valid_mask.dtype != np.bool_:
        raise ValueError("rel_plus_valid_mask must be an HxW bool array")


def _resize(array, transform, interpolation):
    resized = cv2.resize(
        array,
        (transform.scaled_width, transform.scaled_height),
        interpolation=interpolation,
    )
    if resized.shape[:2] != (transform.scaled_height, transform.scaled_width):
        raise ValueError("actual resize shape does not match SpatialTransform scaled shape")
    return resized


def _normalize(image, mean, std):
    return (image.astype(np.float32) / np.float32(255.0) - mean) / std


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
    output = cv2.copyMakeBorder(
        crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
    )
    if output.shape[:2] != (transform.output_height, transform.output_width):
        raise ValueError("crop/pad output shape does not match SpatialTransform")
    return output


def _validated_norm(mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0.0):
        raise ValueError("norm_mean and norm_std must be three-vectors with positive std")
    return mean, std


def apply_cmx_compatible_preprocess(
    rgb,
    rel_plus,
    label,
    rel_plus_valid_mask,
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
    """Return float32 CHW model arrays plus a nearest diagnostic valid mask."""
    validate_rel_plus_augmentation_policy(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        arbitrary_rotation=arbitrary_rotation,
        perspective_warp=perspective_warp,
    )
    rgb_array = np.asarray(rgb)
    rel_array = np.asarray(rel_plus)
    label_array = np.asarray(label)
    valid_array = np.asarray(rel_plus_valid_mask)
    _validate_input_contract(rgb_array, rel_array, label_array, valid_array)
    _validate_source_shapes(rgb_array, rel_array, label_array, valid_array, transform)
    expected_scaled = (
        max(1, int(transform.source_height * transform.scale)),
        max(1, int(transform.source_width * transform.scale)),
    )
    if expected_scaled != (transform.scaled_height, transform.scaled_width):
        raise ValueError(
            "SpatialTransform scaled shape {} does not match source shape and scale {}".format(
                (transform.scaled_height, transform.scaled_width), expected_scaled
            )
        )

    rgb_scaled = _resize(rgb_array, transform, cv2.INTER_LINEAR)
    rel_scaled = _resize(rel_array, transform, cv2.INTER_LINEAR)
    label_scaled = _resize(label_array, transform, cv2.INTER_NEAREST)
    valid_scaled = _resize(
        valid_array.astype(np.uint8), transform, cv2.INTER_NEAREST
    ).astype(bool)

    if photometric_augmentation is not None:
        augmented = np.asarray(photometric_augmentation(rgb_scaled.copy()))
        if augmented.shape != rgb_scaled.shape:
            raise ValueError("photometric_augmentation changed RGB shape")
        if augmented.dtype != np.uint8:
            raise ValueError("photometric_augmentation dtype must be uint8")
        rgb_scaled = augmented

    mean, std = _validated_norm(norm_mean, norm_std)
    rgb_output = _crop_and_center_pad(_normalize(rgb_scaled, mean, std), transform, 0.0)
    rel_output = _crop_and_center_pad(_normalize(rel_scaled, mean, std), transform, 0.0)
    label_output = _crop_and_center_pad(label_scaled, transform, 255)
    valid_output = _crop_and_center_pad(
        valid_scaled.astype(np.uint8), transform, 0
    ).astype(bool)
    return PreprocessedBatch(
        np.ascontiguousarray(rgb_output.transpose(2, 0, 1), dtype=np.float32),
        np.ascontiguousarray(rel_output.transpose(2, 0, 1), dtype=np.float32),
        np.ascontiguousarray(label_output),
        np.ascontiguousarray(valid_output),
        transform,
    )


def _quantiles_by_channel(values):
    if values.size == 0:
        return [{"p05": None, "p50": None, "p95": None} for _ in range(3)]
    return [
        {
            "p05": float(np.quantile(values[:, channel], 0.05)),
            "p50": float(np.quantile(values[:, channel], 0.50)),
            "p95": float(np.quantile(values[:, channel], 0.95)),
        }
        for channel in range(3)
    ]


def analyze_invalid_interpolation(
    rel_plus_uint8,
    valid_mask,
    transform,
    *,
    norm_mean=CMX_NORM_MEAN,
    norm_std=CMX_NORM_STD
):
    """Measure invalid=255 bilinear contamination without changing production input."""
    rel_array = np.asarray(rel_plus_uint8)
    valid_array = np.asarray(valid_mask)
    dummy_rgb = np.zeros_like(rel_array)
    dummy_label = np.zeros(valid_array.shape, dtype=np.uint8)
    _validate_input_contract(dummy_rgb, rel_array, dummy_label, valid_array)
    _validate_source_shapes(dummy_rgb, rel_array, dummy_label, valid_array, transform)
    expected_scaled = (
        max(1, int(transform.source_height * transform.scale)),
        max(1, int(transform.source_width * transform.scale)),
    )
    if expected_scaled != (transform.scaled_height, transform.scaled_width):
        raise ValueError("SpatialTransform scaled shape is inconsistent")

    formal = _resize(rel_array, transform, cv2.INTER_LINEAR).astype(np.float32)
    nearest_valid = _resize(
        valid_array.astype(np.uint8), transform, cv2.INTER_NEAREST
    ).astype(bool)
    valid_weight = _resize(
        valid_array.astype(np.float32), transform, cv2.INTER_LINEAR
    )
    weighted = _resize(
        rel_array.astype(np.float32) * valid_array[..., None],
        transform,
        cv2.INTER_LINEAR,
    )
    reference = np.full_like(weighted, 255.0)
    supported = valid_weight > 1e-6
    reference[supported] = weighted[supported] / valid_weight[supported, None]
    mixed_support = supported & (valid_weight < 1.0 - 1e-6)

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_valid = cv2.dilate(nearest_valid.astype(np.uint8), kernel).astype(bool)
    dilated_invalid = cv2.dilate((~nearest_valid).astype(np.uint8), kernel).astype(bool)
    boundary = dilated_valid & dilated_invalid
    deviation = np.abs(formal - reference)
    affected = mixed_support & np.any(deviation > 1e-6, axis=2)

    formal = _crop_and_center_pad(formal, transform, 0.0)
    reference = _crop_and_center_pad(reference, transform, 0.0)
    nearest_valid = _crop_and_center_pad(
        nearest_valid.astype(np.uint8), transform, 0
    ).astype(bool)
    boundary = _crop_and_center_pad(boundary.astype(np.uint8), transform, 0).astype(bool)
    affected = _crop_and_center_pad(affected.astype(np.uint8), transform, 0).astype(bool)
    deviation = np.abs(formal - reference)
    mean, std = _validated_norm(norm_mean, norm_std)
    formal_normalized = (formal / np.float32(255.0) - mean) / std
    affected_values = formal_normalized[affected]

    affected_deviation = deviation[affected]
    pixel_count = max(1, int(affected.size))
    return {
        "source_invalid_ratio": float(np.mean(~valid_array)),
        "transformed_nearest_invalid_ratio": float(np.mean(~nearest_valid)),
        "invalid_boundary_near_ratio": float(np.count_nonzero(boundary) / pixel_count),
        "bilinear_invalid_affected_ratio": float(np.count_nonzero(affected) / pixel_count),
        "affected_mean_channel_deviation": (
            float(np.mean(affected_deviation)) if affected_deviation.size else 0.0
        ),
        "affected_max_channel_deviation": (
            float(np.max(affected_deviation)) if affected_deviation.size else 0.0
        ),
        "affected_normalized_value_quantiles": _quantiles_by_channel(
            affected_values.reshape(-1, 3)
        ),
    }


def load_and_preprocess_for_cmx(
    rgb, rel_plus_path, label, rel_plus_valid_mask, transform, **kwargs
):
    """Load stored [EGVIA, LOA, ReD] bytes without a colour conversion."""
    rel_plus = load_rel_plus_png(rel_plus_path)
    return apply_cmx_compatible_preprocess(
        rgb, rel_plus, label, rel_plus_valid_mask, transform, **kwargs
    )
