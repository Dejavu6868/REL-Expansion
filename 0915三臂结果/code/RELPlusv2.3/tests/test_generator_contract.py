import inspect

import numpy as np

from rel_plus.camera import CameraGeometry
from rel_plus.constants import (
    REL_PLUS_V2_ALPHA,
    REL_PLUS_V2_CANONICAL_SHAPE,
    REL_PLUS_V2_LAMBDA,
    REL_PLUS_V2_NORMAL_RADIUS,
)
from rel_plus.generator import generate_rel_plus_v2, generate_rel_plus_v2_1


def test_public_generator_exposes_only_frozen_options_and_unit_named_debug():
    signature = inspect.signature(generate_rel_plus_v2)
    assert list(signature.parameters) == ["raw_depth", "camera", "return_debug"]
    assert REL_PLUS_V2_ALPHA == 45.0
    assert REL_PLUS_V2_LAMBDA == 0.5
    assert REL_PLUS_V2_NORMAL_RADIUS == 2
    assert REL_PLUS_V2_CANONICAL_SHAPE == (480, 480)

    raw = np.full((12, 12), 1024, dtype=np.uint16)
    raw[0, 0] = 0
    camera = CameraGeometry.from_json_k(
        np.array([[30.0, 0.0, 6.0], [0.0, 30.0, 6.0], [0.0, 0.0, 1.0]]),
        raw.shape,
        np.eye(3),
        np.zeros(3),
    )
    rel_plus, debug = generate_rel_plus_v2(raw, camera, return_debug=True)
    assert rel_plus.shape == (12, 12, 3) and rel_plus.dtype == np.uint8
    np.testing.assert_array_equal(rel_plus[0, 0], [255, 255, 255])
    required = {
        "points_camera_m", "points_aligned_m", "points_for_encoding_cm",
        "red_raw_cm", "height_raw_cm", "horizontal_radius_cm",
        "normal_finite_mask", "normal_nonzero_mask", "normal_support_count",
        "normal_quality_mask", "normal_invalid_ratio", "zero_normal_ratio",
        "low_support_ratio", "encoding_valid_mask",
    }
    assert required.issubset(debug)
    forbidden = {"points_camera", "points_aligned", "red_raw", "height_raw", "horizontal_radius"}
    assert not forbidden.intersection(debug)
    np.testing.assert_allclose(debug["points_for_encoding_cm"], debug["points_aligned_m"] * 100.0)


def test_v2_1_generator_is_byte_identical_to_v2():
    raw = np.full((12, 12), 1024, dtype=np.uint16)
    raw[0, 0] = 0
    camera = CameraGeometry.from_json_k(
        np.array([[30.0, 0.0, 6.0], [0.0, 30.0, 6.0], [0.0, 0.0, 1.0]]),
        raw.shape, np.eye(3), np.zeros(3),
    )
    np.testing.assert_array_equal(
        generate_rel_plus_v2_1(raw, camera),
        generate_rel_plus_v2(raw, camera),
    )


def test_generator_sends_aligned_metres_times_100_to_encoder(monkeypatch):
    import rel_plus.generator as generator_module

    captured = {}

    def recording_encoder(points, normals, valid, **kwargs):
        captured["points"] = np.asarray(points).copy()
        return np.zeros(valid.shape + (3,), dtype=np.uint8), {}

    monkeypatch.setattr(generator_module, "encode_rel_channels", recording_encoder)
    raw = np.full((12, 12), 1024, dtype=np.uint16)
    camera = CameraGeometry.from_json_k(
        np.array([[30.0, 0.0, 6.0], [0.0, 30.0, 6.0], [0.0, 0.0, 1.0]]),
        raw.shape, np.eye(3), np.zeros(3),
    )
    _, debug = generate_rel_plus_v2_1(raw, camera, return_debug=True)
    np.testing.assert_allclose(captured["points"], debug["points_aligned_m"] * 100.0)
