import numpy as np

from rel_plus.camera import (
    CameraGeometry,
    backproject_z_depth,
    json_k_to_rel_helper_k,
    resize_camera_geometry,
)
from rel_plus.depth import decode_stanford_s2d_depth, resize_raw_depth_nearest


def test_depth_decode_uses_raw_over_512_and_two_invalid_sentinels():
    raw = np.array([[0, 512, 1024, 65535]], dtype=np.uint16)
    depth, valid = decode_stanford_s2d_depth(raw)

    np.testing.assert_array_equal(valid, [[False, True, True, False]])
    np.testing.assert_allclose(depth, [[0.0, 1.0, 2.0, 0.0]], rtol=0.0, atol=0.0)
    assert depth.dtype == np.float32


def test_half_pixel_backprojection_round_trips_to_pixel_centres():
    k = np.array([[100.0, 0.0, 2.5], [0.0, 120.0, 1.5], [0.0, 0.0, 1.0]])
    depth = np.full((3, 4), 2.0, dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    points = backproject_z_depth(depth, valid, k)

    projected_u = k[0, 0] * points[..., 0] / points[..., 2] + k[0, 2]
    projected_v = k[1, 1] * points[..., 1] / points[..., 2] + k[1, 2]
    rows, columns = np.indices(depth.shape)
    np.testing.assert_allclose(projected_u, columns + 0.5, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(projected_v, rows + 0.5, rtol=0.0, atol=1e-6)


def test_json_k_to_one_based_source_helper_is_a_single_plus_half_adapter():
    k = np.array([[100.0, 0.0, 2.5], [0.0, 120.0, 1.5], [0.0, 0.0, 1.0]])
    helper = json_k_to_rel_helper_k(k)
    np.testing.assert_allclose(helper[:2, 2], [3.0, 2.0])
    np.testing.assert_allclose(helper[:2, :2], k[:2, :2])


def test_zero_based_center_constructor_converts_once_to_json_half_pixel_k():
    camera = CameraGeometry.from_zero_based_center_k(
        np.array([[100.0, 0.0, 2.0], [0.0, 120.0, 1.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
    )
    np.testing.assert_allclose(camera.K_json[:2, 2], [2.5, 1.5])


def test_canonical_resize_is_nearest_and_scales_json_k_rows():
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4)
    resized = resize_raw_depth_nearest(raw, (2, 2))
    np.testing.assert_array_equal(resized, [[0, 2], [8, 10]])

    camera = CameraGeometry.from_json_k(
        np.array([[8.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
    )
    canonical = resize_camera_geometry(camera, (4, 4), (2, 2))
    np.testing.assert_allclose(
        canonical.K_json,
        [[4.0, 0.0, 1.0], [0.0, 5.0, 1.0], [0.0, 0.0, 1.0]],
    )
