import os
import tempfile
import unittest

import cv2
import numpy as np

from dataloader.RGBXDataset import RGBXDataset


class _OnlinePreprocessSpy:
    def __init__(self):
        self.called = False

    def __call__(self, rgb, label, raw_depth, pose_path):
        self.called = True
        self.raw_depth = raw_depth.copy()
        self.pose_path = pose_path
        return rgb.transpose(2, 0, 1), label, np.full((3, 4, 4), 7.0)


class RelPlusOnlineDatasetTest(unittest.TestCase):
    def test_online_mode_loads_depth_and_pose_instead_of_an_x_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {}
            for name in ("RGB", "Label", "Depth16", "Pose"):
                roots[name] = os.path.join(directory, name)
                os.makedirs(roots[name])
            sample = "fixture"
            cv2.imwrite(os.path.join(roots["RGB"], sample + ".png"), np.zeros((4, 4, 3), np.uint8))
            cv2.imwrite(os.path.join(roots["Label"], sample + ".png"), np.ones((4, 4), np.uint8))
            depth = np.arange(16, dtype=np.uint16).reshape(4, 4) + 1
            cv2.imwrite(os.path.join(roots["Depth16"], sample + ".png"), depth)
            pose_path = os.path.join(roots["Pose"], sample + ".json")
            with open(pose_path, "w", encoding="utf-8") as handle:
                handle.write("{}")
            split = os.path.join(directory, "train.txt")
            with open(split, "w", encoding="utf-8") as handle:
                handle.write(sample + "\n")
            setting = {
                "rgb_root": roots["RGB"], "rgb_format": ".png",
                "gt_root": roots["Label"], "gt_format": ".png", "transform_gt": True,
                "x_root": os.path.join(directory, "missing_cache"), "x_format": ".png",
                "x_single_channel": False, "x_online_relplus": True,
                "depth_root": roots["Depth16"], "depth_format": ".png",
                "pose_root": roots["Pose"], "pose_format": ".json",
                "train_source": split, "eval_source": split, "class_names": ["fixture"],
            }
            preprocess = _OnlinePreprocessSpy()
            dataset = RGBXDataset(setting, "train", preprocess)
            result = dataset[0]
        self.assertTrue(preprocess.called)
        np.testing.assert_array_equal(preprocess.raw_depth, depth)
        self.assertEqual(preprocess.pose_path, pose_path)
        self.assertEqual(tuple(result["modal_x"].shape), (3, 4, 4))
        self.assertTrue(np.all(result["modal_x"].numpy() == 7.0))


if __name__ == "__main__":
    unittest.main()
