import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from relplus.geometry import load_camera_metadata
from relplus.io import write_relplus_png
from relplus.representation import compute_relplus, decode_stanford_depth
from scripts.validate_relplus_semantics import validate


class RelplusSemanticsValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.dataset = root / "dataset"
        self.run = root / "run"
        (self.run / "data_reports").mkdir(parents=True)
        sample_ids = ("area_1/a", "area_5/b")
        selected = []
        for index, sample_id in enumerate(sample_ids):
            depth_path = self.dataset / "Depth16" / (sample_id + ".png")
            pose_path = self.dataset / "Pose" / (sample_id + ".json")
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            pose_path.parent.mkdir(parents=True, exist_ok=True)
            raw = np.full((15, 15), 1023 + index * 64, dtype=np.uint16)
            self.assertTrue(cv2.imwrite(str(depth_path), raw))
            center = np.array([4.0 + index, -2.0, 1.0])
            k = np.array(
                [[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]
            )
            pose_path.write_text(
                json.dumps(
                    {
                        "camera_k_matrix": k.tolist(),
                        "camera_rt_matrix": np.column_stack(
                            [np.eye(3), -center]
                        ).tolist(),
                        "camera_location": center.tolist(),
                    }
                ),
                encoding="utf-8",
            )
            camera = load_camera_metadata(str(pose_path))
            depth, valid = decode_stanford_depth(raw)
            rel_native, auxiliary = compute_relplus(depth, valid, camera)
            rel = cv2.resize(rel_native, (480, 480), interpolation=cv2.INTER_LINEAR)
            valid_output = cv2.resize(
                auxiliary["valid"].astype(np.uint8),
                (480, 480),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            rel[~valid_output] = 255
            write_relplus_png(
                str(self.run / "relplus_cache" / (sample_id + ".png")), rel
            )
            selected.append({"sample_id": sample_id})
        (self.run / "data_reports" / "data_audit.json").write_text(
            json.dumps({"dataset_root": str(self.dataset), "selected_samples": selected}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_recomputed_rel_default_samples_must_match_cache_exactly(self):
        report = validate(self.run)
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["matched_samples"], 2)

        cached = self.run / "relplus_cache" / "area_1" / "a.png"
        image = cv2.imread(str(cached), cv2.IMREAD_COLOR)
        image[240, 240, 0] ^= 1
        write_relplus_png(str(cached), image)
        report = validate(self.run)
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(report["matched_samples"], 1)


if __name__ == "__main__":
    unittest.main()
