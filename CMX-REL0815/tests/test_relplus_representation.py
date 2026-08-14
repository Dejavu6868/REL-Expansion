import unittest

import numpy as np

from relplus.geometry import CameraMetadata
from relplus.representation import (
    compute_relplus,
    decode_stanford_depth,
    encode_relplus_channels,
    estimate_rel_normals,
)


class RelplusRepresentationTest(unittest.TestCase):
    def test_depth_decode_uses_documented_scale_and_invalid_sentinel(self):
        raw = np.array([[0, 511, 65535]], dtype=np.uint16)
        depth, valid = decode_stanford_depth(raw)
        np.testing.assert_allclose(depth[0, :2], [1.0 / 512.0, 1.0])
        self.assertEqual(depth[0, 2], 0.0)
        np.testing.assert_array_equal(valid, [[True, True, False]])

    def test_channel_order_formula_range_dtype_and_invalid_value(self):
        points = np.array([[[1.0, 0.0, 0.0], [2.0, 0.0, 1.0], [3.0, 0.0, 2.0]]])
        normals = np.broadcast_to(np.array([0.0, 0.0, -1.0]), points.shape).copy()
        valid = np.array([[True, True, True]])
        rel, aux = encode_relplus_channels(points, normals, valid)
        self.assertEqual(rel.dtype, np.uint8)
        np.testing.assert_array_equal(rel[0, :, 0], [0, 128, 255])  # ReD
        np.testing.assert_array_equal(rel[0, :, 1], [0, 64, 128])  # EGVIA
        np.testing.assert_array_equal(rel[0, :, 2], [128, 128, 128])  # LOA
        self.assertTrue(np.isfinite(aux["red"]).all())
        self.assertGreaterEqual(int(rel.min()), 0)
        self.assertLessEqual(int(rel.max()), 255)

        valid[0, 1] = False
        rel, _ = encode_relplus_channels(points, normals, valid)
        np.testing.assert_array_equal(rel[0, 1], [255, 255, 255])

    def test_egvia_gate_uses_angle_without_height_for_vertical_surface(self):
        points = np.array([[[1.0, 0.0, 0.0], [2.0, 0.0, 2.0]]])
        normals = np.broadcast_to(np.array([1.0, 0.0, 0.0]), points.shape).copy()
        rel, aux = encode_relplus_channels(points, normals, np.ones((1, 2), bool))
        np.testing.assert_array_equal(rel[0, :, 1], [128, 128])
        np.testing.assert_array_equal(rel[0, :, 2], [128, 128])
        np.testing.assert_allclose(aux["angle_degrees"], 90.0)

    def test_rel_plane_fit_normal_estimator_is_finite_with_hole(self):
        depth = np.full((15, 15), 2.0, dtype=np.float32)
        depth[7, 7] = 0.0
        valid = depth > 0
        k = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]])
        from relplus.geometry import backproject_z_depth

        points = backproject_z_depth(depth, k)
        normals, normal_valid = estimate_rel_normals(points, valid, radius=3)
        self.assertTrue(np.isfinite(normals).all())
        self.assertFalse(normal_valid[7, 7])
        np.testing.assert_allclose(normals[5, 5], [0.0, 0.0, 1.0], atol=1e-4)

    def test_synthetic_horizontal_and_vertical_planes_after_pose_rotation(self):
        k = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]])
        depth = np.full((15, 15), 2.0, dtype=np.float32)
        valid = np.ones_like(depth, dtype=bool)

        floor_camera = CameraMetadata(
            k=k,
            r_world_to_camera=np.diag([1.0, -1.0, -1.0]),
            t_world_to_camera=np.array([0.0, 0.0, 2.0]),
            camera_center_world=np.array([0.0, 0.0, 2.0]),
            center_residual=0.0,
        )
        floor_rel, floor_aux = compute_relplus(depth, valid, floor_camera)
        self.assertLessEqual(int(floor_rel[7, 8, 1]), 1)
        np.testing.assert_allclose(floor_aux["normals_world"][7, 8], [0.0, 0.0, -1.0], atol=1e-4)

        wall_r_wc = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        wall_camera = CameraMetadata(
            k=k,
            r_world_to_camera=wall_r_wc,
            t_world_to_camera=np.zeros(3),
            camera_center_world=np.zeros(3),
            center_residual=0.0,
        )
        wall_rel, wall_aux = compute_relplus(depth, valid, wall_camera)
        self.assertTrue(125 <= int(wall_rel[7, 7, 1]) <= 129)
        np.testing.assert_allclose(wall_aux["normals_world"][7, 7], [0.0, 1.0, 0.0], atol=1e-4)

    def test_rel_default_is_invariant_to_world_origin_translation(self):
        k = np.array([[12.0, 0.0, 8.0], [0.0, 11.0, 8.0], [0.0, 0.0, 1.0]])
        rows, columns = np.indices((15, 15), dtype=np.float64)
        depth = 1.5 + 0.02 * rows + 0.01 * columns
        valid = np.ones_like(depth, dtype=bool)
        r_wc = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        cameras = []
        for center in (np.zeros(3), np.array([10.0, -5.0, 3.0])):
            cameras.append(
                CameraMetadata(
                    k=k,
                    r_world_to_camera=r_wc,
                    t_world_to_camera=-r_wc @ center,
                    camera_center_world=center,
                    center_residual=0.0,
                )
            )

        first, first_aux = compute_relplus(depth, valid, cameras[0])
        translated, translated_aux = compute_relplus(depth, valid, cameras[1])

        np.testing.assert_array_equal(first, translated)
        np.testing.assert_array_equal(first_aux["valid"], translated_aux["valid"])
        np.testing.assert_allclose(
            first_aux["points_rel"], translated_aux["points_rel"], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            first_aux["points_rel"],
            first_aux["points_camera"] @ r_wc,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            translated_aux["points_rel"],
            translated_aux["points_world"] - cameras[1].camera_center_world,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            translated_aux["points_world"] - first_aux["points_world"],
            np.broadcast_to(cameras[1].camera_center_world, first_aux["points_world"].shape),
            rtol=0.0,
            atol=1e-12,
        )

    def test_invalid_nan_inf_inputs_do_not_escape(self):
        points = np.array([[[np.nan, 0.0, 1.0], [np.inf, 0.0, 1.0]]])
        normals = np.zeros_like(points)
        valid = np.array([[False, False]])
        rel, aux = encode_relplus_channels(points, normals, valid)
        np.testing.assert_array_equal(rel, np.full((1, 2, 3), 255, dtype=np.uint8))
        self.assertFalse(np.isnan(rel.astype(np.float32)).any())
        self.assertEqual(aux["valid_count"], 0)


if __name__ == "__main__":
    unittest.main()
