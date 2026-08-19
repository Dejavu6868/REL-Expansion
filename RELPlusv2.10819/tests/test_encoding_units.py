import numpy as np
from rel_plus.encoding import encode_rel_channels, perspective_tangent_field


def encode_centimetres(points, normals, valid):
    tangent, _ = perspective_tangent_field(points)
    return encode_rel_channels(points, normals, valid, tangent=tangent)[0]


def test_dense_nondegenerate_bytes_match_frozen_v2_golden():
    rows, columns = np.indices((3, 4), dtype=np.float64)
    points_m = np.stack([0.2 + columns, 0.3 + rows, 1.0 + rows * 0.1], axis=-1)
    normals = np.dstack([np.zeros_like(rows), np.zeros_like(rows), np.ones_like(rows)])
    valid = np.ones(rows.shape, dtype=bool)
    expected = np.array(
        [
            [[255, 90, 0], [255, 90, 62], [255, 90, 132], [255, 90, 203]],
            [[255, 90, 68], [255, 90, 100], [255, 90, 156], [255, 90, 220]],
            [[255, 90, 138], [255, 90, 159], [255, 90, 201], [255, 90, 255]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(
        encode_centimetres(points_m * 100.0, normals, valid), expected
    )


def test_single_valid_pixel_matches_frozen_v2_golden():
    points_m = np.zeros((3, 4, 3), dtype=np.float64)
    normals = np.zeros_like(points_m)
    normals[..., 2] = 1.0
    valid = np.zeros((3, 4), dtype=bool)
    valid[1, 2] = True
    points_m[1, 2] = [0.01, 0.02, 0.03]
    expected = np.full((3, 4, 3), 255, dtype=np.uint8)
    expected[1, 2] = [255, 90, 254]
    np.testing.assert_array_equal(
        encode_centimetres(points_m * 100.0, normals, valid), expected
    )


def test_p1_equals_p99_constant_height_matches_frozen_v2_golden():
    rows, columns = np.indices((3, 4), dtype=np.float64)
    points_m = np.zeros((3, 4, 3), dtype=np.float64)
    points_m[..., 0] = columns * 0.01
    points_m[..., 1] = rows * 0.01
    points_m[..., 2] = 0.02
    normals = np.zeros_like(points_m)
    normals[..., 2] = 1.0
    expected = np.array(
        [
            [[255, 90, 0], [255, 90, 70], [255, 90, 141], [255, 90, 212]],
            [[255, 90, 70], [255, 90, 100], [255, 90, 158], [255, 90, 223]],
            [[255, 90, 141], [255, 90, 158], [255, 90, 200], [255, 90, 255]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(
        encode_centimetres(
            points_m * 100.0, normals, np.ones((3, 4), dtype=bool)
        ),
        expected,
    )


def test_red_min_equals_max_constant_radius_matches_frozen_v2_golden():
    points_m = np.zeros((3, 4, 3), dtype=np.float64)
    points_m[..., 0] = 0.02
    points_m[..., 2] = np.arange(12).reshape(3, 4) * 0.01
    normals = np.zeros_like(points_m)
    normals[..., 2] = 1.0
    expected = np.full((3, 4, 3), [255, 90, 2], dtype=np.uint8)
    np.testing.assert_array_equal(
        encode_centimetres(
            points_m * 100.0, normals, np.ones((3, 4), dtype=bool)
        ),
        expected,
    )


def test_nan_and_zero_normal_source_semantics_are_channel_local():
    points_cm = np.array([[[1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]])
    normals = np.array([[[np.nan, np.nan, np.nan], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    actual = encode_centimetres(points_cm, normals, np.ones((1, 3), bool))
    assert actual[0, 0, 0] == 255 and actual[0, 0, 1] == 90 and actual[0, 0, 2] != 255
    assert actual[0, 1, 0] == 127 and actual[0, 1, 1] == 90 and actual[0, 1, 2] != 255
