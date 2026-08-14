import os
import tempfile
import unittest

import cv2
import numpy as np

from relplus.io import (
    apply_synchronized_spatial_transform,
    read_relplus_png,
    read_split,
    resolve_sample_paths,
    validate_disjoint_splits,
    write_relplus_png,
)


class DatasetAlignmentTest(unittest.TestCase):
    def test_split_read_and_leakage_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = os.path.join(tmp, "train.txt")
            test = os.path.join(tmp, "test.txt")
            with open(train, "w", encoding="utf-8") as handle:
                handle.write("area_1/sample_a\narea_2/sample_b\n")
            with open(test, "w", encoding="utf-8") as handle:
                handle.write("area_5/sample_c\n")
            self.assertEqual(len(read_split(train)), 2)
            validate_disjoint_splits(read_split(train), read_split(test))
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_disjoint_splits(read_split(train), ["area_1/sample_a"])

    def test_expected_sample_paths_share_identifier(self):
        paths = resolve_sample_paths("/dataset", "area_1/camera_frame")
        self.assertEqual(paths["rgb"], "/dataset/RGB/area_1/camera_frame.png")
        self.assertEqual(paths["depth"], "/dataset/Depth16/area_1/camera_frame.png")
        self.assertEqual(paths["label"], "/dataset/Label/area_1/camera_frame.png")
        self.assertEqual(paths["pose"], "/dataset/Pose/area_1/camera_frame.json")

    def test_png_roundtrip_preserves_semantic_channel_order(self):
        rel = np.zeros((4, 5, 3), dtype=np.uint8)
        rel[..., 0] = 11  # ReD
        rel[..., 1] = 22  # EGVIA
        rel[..., 2] = 33  # LOA
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample.png")
            write_relplus_png(path, rel)
            loaded = read_relplus_png(path)
        np.testing.assert_array_equal(loaded, rel)

    def test_resize_flip_crop_keep_rgb_rel_and_label_spatially_aligned(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rel = np.zeros_like(rgb)
        label = np.zeros((4, 4), dtype=np.uint8)
        rgb[1:3, 1:3, 0] = 200
        rel[1:3, 1:3, 0] = 200
        label[1:3, 1:3] = 7
        rgb2, rel2, label2 = apply_synchronized_spatial_transform(
            rgb, rel, label, resize_shape=(8, 8), flip=True, crop=(1, 1, 6, 6)
        )
        self.assertEqual(rgb2.shape, (6, 6, 3))
        self.assertEqual(rel2.shape, (6, 6, 3))
        self.assertEqual(label2.shape, (6, 6))
        self.assertTrue(np.array_equal(rgb2[..., 0], rel2[..., 0]))
        continuous_mask = rgb2[..., 0] > 100
        label_mask = label2 == 7
        self.assertGreater(np.logical_and(continuous_mask, label_mask).sum(), 0)
        self.assertTrue(set(np.unique(label2)).issubset({0, 7}))


if __name__ == "__main__":
    unittest.main()
