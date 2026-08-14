import importlib
import random
import sys
import types
import unittest

import numpy as np


class Stage2AControlsTest(unittest.TestCase):
    def _load_dataloader(self, horizontal_flip):
        fake = types.ModuleType("config")
        fake.config = types.SimpleNamespace(
            train_horizontal_flip=horizontal_flip,
            train_scale_array=None,
            image_height=3,
            image_width=4,
        )
        sys.modules["config"] = fake
        sys.modules.pop("dataloader.dataloader", None)
        return importlib.import_module("dataloader.dataloader")

    def test_baseline_trainpre_respects_disabled_flip(self):
        module = self._load_dataloader(False)
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        gt = np.arange(12, dtype=np.uint8).reshape(3, 4)
        modal = rgb.copy()
        original_random = random.random
        random.random = lambda: 0.99
        try:
            out_rgb, out_gt, out_modal = module.TrainPre(
                np.zeros(3), np.ones(3)
            )(rgb, gt, modal)
        finally:
            random.random = original_random
        np.testing.assert_array_equal(out_rgb.transpose(1, 2, 0), rgb.astype(np.float64) / 255.0)
        np.testing.assert_array_equal(out_gt, gt)
        np.testing.assert_array_equal(out_modal.transpose(1, 2, 0), modal.astype(np.float64) / 255.0)


if __name__ == "__main__":
    unittest.main()
