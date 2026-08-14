import json
import os
import tempfile
import unittest

import numpy as np

from relplus.geometry import (
    backproject_z_depth,
    camera_to_world,
    crop_intrinsics,
    horizontal_flip_intrinsics,
    load_camera_metadata,
    pad_intrinsics,
    resize_intrinsics,
    rotate_camera_vectors_to_world,
    world_to_camera,
)


class CameraGeometryTest(unittest.TestCase):
    def setUp(self):
        self.k = np.array(
            [[2.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def test_backproject_camera_z_depth_uses_one_based_pixels(self):
        depth = np.full((3, 3), 2.0, dtype=np.float32)
        points = backproject_z_depth(depth, self.k)
        np.testing.assert_allclose(points[1, 1], [0.0, 0.0, 2.0])
        np.testing.assert_allclose(points[0, 0], [-1.0, -0.5, 2.0])

    def test_resize_crop_pad_and_flip_intrinsics(self):
        resized = resize_intrinsics(self.k, (3, 3), (6, 12))
        np.testing.assert_allclose(
            resized, [[8.0, 0.0, 8.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]
        )
        cropped = crop_intrinsics(resized, left=3, top=1)
        np.testing.assert_allclose(cropped[:2, 2], [5.0, 3.0])
        padded = pad_intrinsics(cropped, left=2, top=4)
        np.testing.assert_allclose(padded[:2, 2], [7.0, 7.0])
        flipped = horizontal_flip_intrinsics(padded, width=12)
        self.assertEqual(flipped[0, 0], -8.0)
        self.assertEqual(flipped[0, 2], 6.0)  # u' = width + 1 - u

    def test_world_camera_roundtrip_and_vector_rotation(self):
        r_wc = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        center = np.array([10.0, 20.0, 3.0])
        p_cam = np.array([[[1.0, 2.0, 4.0]]])
        p_world = camera_to_world(p_cam, r_wc, center)
        np.testing.assert_allclose(world_to_camera(p_world, r_wc, center), p_cam)
        vectors = rotate_camera_vectors_to_world(np.array([[[0.0, 0.0, 1.0]]]), r_wc)
        np.testing.assert_allclose(vectors, [[[0.0, 0.0, 1.0]]])

    def test_pose_json_is_validated_as_world_to_camera(self):
        r_wc = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        center = np.array([10.0, 20.0, 3.0])
        t_wc = -r_wc @ center
        payload = {
            "camera_k_matrix": self.k.tolist(),
            "camera_rt_matrix": np.column_stack([r_wc, t_wc]).tolist(),
            "camera_location": center.tolist(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pose.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            camera = load_camera_metadata(path)
        np.testing.assert_allclose(camera.r_world_to_camera, r_wc)
        np.testing.assert_allclose(camera.camera_center_world, center)
        self.assertLess(camera.center_residual, 1e-8)

    def test_inconsistent_pose_is_rejected_instead_of_guessing(self):
        payload = {
            "camera_k_matrix": self.k.tolist(),
            "camera_rt_matrix": np.column_stack([np.eye(3), np.zeros(3)]).tolist(),
            "camera_location": [1.0, 0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pose.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "world-to-camera"):
                load_camera_metadata(path)


if __name__ == "__main__":
    unittest.main()
