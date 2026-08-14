import inspect
import unittest

import numpy as np

from relplus.pipeline import (
    estimate_gravity_down_camera,
    generate_relplus_from_depth_local,
)


class Stage2AGravityTest(unittest.TestCase):
    def test_estgravity_recovers_structured_direction(self):
        gravity = np.array([0.2, 0.9, 0.38], dtype=np.float64)
        gravity /= np.linalg.norm(gravity)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        wall_a = axis - gravity * np.dot(axis, gravity)
        wall_a /= np.linalg.norm(wall_a)
        wall_b = np.cross(gravity, wall_a)
        normals = np.concatenate([
            np.tile(gravity, (400, 1)), np.tile(-gravity, (400, 1)),
            np.tile(wall_a, (400, 1)), np.tile(-wall_a, (400, 1)),
            np.tile(wall_b, (400, 1)), np.tile(-wall_b, (400, 1)),
        ], axis=0).reshape(40, 60, 3)
        valid = np.ones((40, 60), dtype=bool)
        estimated = estimate_gravity_down_camera(
            normals, valid, initial_gravity=gravity + np.array([0.03, -0.01, 0.02])
        )
        self.assertTrue(np.isfinite(estimated).all())
        self.assertAlmostEqual(float(np.linalg.norm(estimated)), 1.0, places=12)
        angle = np.degrees(np.arccos(np.clip(np.dot(estimated, gravity), -1.0, 1.0)))
        self.assertLess(angle, 0.1)

    def test_local_generator_has_no_pose_rotation_parameter(self):
        parameters = inspect.signature(generate_relplus_from_depth_local).parameters
        self.assertNotIn("r_world_to_camera", parameters)
        self.assertNotIn("pose", parameters)

    def test_local_generator_returns_finite_unit_gravity(self):
        yy, xx = np.indices((32, 32), dtype=np.float64)
        depth = 2.0 + 0.001 * xx + 0.002 * yy
        valid = np.ones(depth.shape, dtype=bool)
        k = np.array([[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]])
        rel, rel_valid, auxiliary = generate_relplus_from_depth_local(depth, valid, k, normal_radius=2)
        self.assertEqual(rel.shape, (32, 32, 3))
        self.assertTrue(rel_valid.any())
        self.assertTrue(np.isfinite(auxiliary["gravity_down_camera"]).all())
        self.assertAlmostEqual(float(np.linalg.norm(auxiliary["gravity_down_camera"])), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
