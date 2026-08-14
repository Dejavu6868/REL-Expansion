import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stage2a.runtime import append_epoch_metrics, load_common_initial_model, seed_everything


class Stage2ARuntimeTest(unittest.TestCase):
    def test_seed_everything_repeats_python_numpy_and_torch(self):
        seed_everything(12345, deterministic=True)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        seed_everything(12345, deterministic=True)
        second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(first, second)

    def test_common_initial_model_loads_strictly(self):
        with tempfile.TemporaryDirectory() as directory:
            source = torch.nn.Linear(3, 2)
            path = Path(directory) / "initial.pth"
            torch.save({"model": source.state_dict()}, path)
            target = torch.nn.Linear(3, 2)
            report = load_common_initial_model(target, str(path))
            self.assertEqual(report["missing_keys"], [])
            self.assertEqual(report["unexpected_keys"], [])
            for left, right in zip(source.parameters(), target.parameters()):
                torch.testing.assert_allclose(left, right)

    def test_epoch_metrics_has_one_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            append_epoch_metrics(path, {"epoch": 1, "train_loss": 2.0, "learning_rate": 1e-4})
            append_epoch_metrics(path, {"epoch": 2, "train_loss": 1.0, "learning_rate": 5e-5})
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], "epoch,train_loss,learning_rate")


if __name__ == "__main__":
    unittest.main()
