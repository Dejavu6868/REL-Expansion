import unittest

import numpy as np

from relplus.pipeline import (
    SpatialTransformParameters,
    transform_depth_geometry,
    update_intrinsics,
)
from dataloader.dataloader import build_relplus_parameters


class RelPlusPipelineTest(unittest.TestCase):
    def test_training_parameters_center_pad_when_scaled_image_is_smaller_than_crop(self):
        params = build_relplus_parameters((480, 480), 0.5, (480, 480), (0, 0))
        self.assertEqual((params.resize_height, params.resize_width), (240, 240))
        self.assertEqual((params.crop_height, params.crop_width), (240, 240))
        self.assertEqual(
            (params.pad_top, params.pad_bottom, params.pad_left, params.pad_right),
            (120, 120, 120, 120),
        )

    def test_resize_crop_pad_updates_intrinsics_under_half_pixel_coordinates(self):
        k = np.array([[100.0, 0.0, 5.0], [0.0, 120.0, 4.0], [0.0, 0.0, 1.0]])
        params = SpatialTransformParameters(
            resize_height=12,
            resize_width=16,
            crop_y=2,
            crop_x=3,
            crop_height=8,
            crop_width=9,
            pad_top=1,
            pad_bottom=1,
            pad_left=2,
            pad_right=1,
            flip=False,
        )
        actual = update_intrinsics(k, (6, 8), params)
        expected = np.array([[200.0, 0.0, 9.0], [0.0, 240.0, 7.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_depth_transform_uses_shared_crop_and_marks_padding_invalid(self):
        depth = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        valid = np.ones((4, 4), dtype=bool)
        k = np.eye(3, dtype=np.float64)
        params = SpatialTransformParameters(
            resize_height=4,
            resize_width=4,
            crop_y=1,
            crop_x=1,
            crop_height=2,
            crop_width=2,
            pad_top=1,
            pad_bottom=1,
            pad_left=1,
            pad_right=1,
            flip=False,
        )
        transformed_depth, transformed_valid, _ = transform_depth_geometry(
            depth, valid, k, params
        )
        self.assertEqual(transformed_depth.shape, (4, 4))
        np.testing.assert_array_equal(transformed_depth[1:3, 1:3], depth[1:3, 1:3])
        self.assertEqual(int(transformed_valid.sum()), 4)
        self.assertTrue(np.all(transformed_depth[~transformed_valid] == 0.0))

    def test_horizontal_flip_is_rejected_by_geometry_pipeline_policy(self):
        params = SpatialTransformParameters(
            resize_height=4,
            resize_width=4,
            crop_y=0,
            crop_x=0,
            crop_height=4,
            crop_width=4,
            pad_top=0,
            pad_bottom=0,
            pad_left=0,
            pad_right=0,
            flip=True,
        )
        with self.assertRaisesRegex(ValueError, "disabled"):
            transform_depth_geometry(
                np.ones((4, 4)), np.ones((4, 4), dtype=bool), np.eye(3), params
            )


if __name__ == "__main__":
    unittest.main()
