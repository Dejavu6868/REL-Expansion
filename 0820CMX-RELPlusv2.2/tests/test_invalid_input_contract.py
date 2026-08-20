import numpy as np
import pytest

from rel_plus.integration.cmx_preprocess import (
    SpatialTransform,
    analyze_invalid_interpolation,
    apply_cmx_compatible_preprocess,
)


def identity_transform(shape):
    return SpatialTransform(
        source_height=shape[0],
        source_width=shape[1],
        scale=1.0,
        scaled_height=shape[0],
        scaled_width=shape[1],
        crop_top=0,
        crop_left=0,
        output_height=shape[0],
        output_width=shape[1],
    )


def test_invalid_mask_is_diagnostic_and_never_zeroes_formal_input():
    shape = (6, 8)
    rgb = np.zeros(shape + (3,), dtype=np.uint8)
    rel = np.full(shape + (3,), [11, 22, 33], dtype=np.uint8)
    valid = np.ones(shape, dtype=bool)
    valid[:, :3] = False
    rel[~valid] = 255
    label = np.zeros(shape, dtype=np.uint8)
    output = apply_cmx_compatible_preprocess(
        rgb, rel, label, valid, identity_transform(shape)
    )
    expected_invalid = (
        np.ones(3, dtype=np.float32)
        - np.array([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    np.testing.assert_allclose(output.modal_x[:, 0, 0], expected_invalid)
    assert not output.modal_x_valid_mask[0, 0]
    assert output.modal_x.shape[0] == 3
    assert output.modal_x_valid_mask.ndim == 2


def test_invalid_boundary_contamination_is_quantified_without_changing_output():
    shape = (8, 8)
    rel = np.full(shape + (3,), 32, dtype=np.uint8)
    valid = np.ones(shape, dtype=bool)
    valid[:, :4] = False
    rel[~valid] = 255
    transform = SpatialTransform(8, 8, 1.5, 12, 12, 0, 0, 12, 12)
    metrics = analyze_invalid_interpolation(rel, valid, transform)
    assert metrics["source_invalid_ratio"] == 0.5
    assert metrics["transformed_nearest_invalid_ratio"] == 0.5
    assert metrics["invalid_boundary_near_ratio"] > 0.0
    assert metrics["bilinear_invalid_affected_ratio"] > 0.0
    assert metrics["affected_mean_channel_deviation"] > 0.0
    assert metrics["affected_max_channel_deviation"] > 0.0
    assert len(metrics["affected_normalized_value_quantiles"]) == 3


@pytest.mark.parametrize(
    "rgb,match",
    [
        (np.zeros((4, 4, 3), dtype=np.float32), "rgb.*uint8"),
        (np.zeros((4, 4), dtype=np.uint8), "rgb.*HxWx3"),
    ],
)
def test_input_dtype_contract_rejects_undeclared_rgb(rgb, match):
    rel = np.zeros((4, 4, 3), dtype=np.uint8)
    label = np.zeros((4, 4), dtype=np.uint8)
    valid = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match=match):
        apply_cmx_compatible_preprocess(
            rgb, rel, label, valid, identity_transform((4, 4))
        )


def test_photometric_callback_shape_dtype_and_range_are_checked():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rel = np.zeros_like(rgb)
    label = np.zeros((4, 4), dtype=np.uint8)
    valid = np.ones((4, 4), dtype=bool)
    transform = identity_transform((4, 4))
    with pytest.raises(ValueError, match="dtype"):
        apply_cmx_compatible_preprocess(
            rgb, rel, label, valid, transform,
            photometric_augmentation=lambda value: value.astype(np.float32) / 255.0,
        )
    with pytest.raises(ValueError, match="shape"):
        apply_cmx_compatible_preprocess(
            rgb, rel, label, valid, transform,
            photometric_augmentation=lambda value: value[:2],
        )


def test_wrong_source_or_scaled_shape_fails_closed(monkeypatch):
    shape = (4, 4)
    rgb = np.zeros(shape + (3,), dtype=np.uint8)
    rel = np.zeros_like(rgb)
    label = np.zeros(shape, dtype=np.uint8)
    valid = np.ones(shape, dtype=bool)
    wrong_source = SpatialTransform(5, 4, 1.0, 5, 4, 0, 0, 4, 4)
    with pytest.raises(ValueError, match="source shape"):
        apply_cmx_compatible_preprocess(rgb, rel, label, valid, wrong_source)

    bad_scaled = SpatialTransform(4, 4, 1.0, 5, 4, 0, 0, 4, 4)
    with pytest.raises(ValueError, match="scaled shape"):
        apply_cmx_compatible_preprocess(rgb, rel, label, valid, bad_scaled)
