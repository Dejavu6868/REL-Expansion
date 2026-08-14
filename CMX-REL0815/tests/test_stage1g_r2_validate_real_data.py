import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.stage1g_r2_validate_real_data import (
    AREAS,
    parse_rgb_name,
    roundtrip_png,
    select_four_per_area,
)


class Stage1GR2ValidationTest(unittest.TestCase):
    def test_parse_rgb_name_preserves_room_with_underscores(self):
        parsed = parse_rgb_name(
            "camera_0123456789abcdef0123456789abcdef_conference_room_3_"
            "frame_17_domain_rgb.png"
        )
        self.assertEqual(parsed[0], "0123456789abcdef0123456789abcdef")
        self.assertEqual(parsed[1], "conference_room_3")
        self.assertEqual(parsed[2], 17)

    def test_selection_is_deterministic_room_diverse_and_camera_unique(self):
        candidates = []
        for area_index, area in enumerate(AREAS):
            for camera_index in range(7):
                candidates.append(
                    {
                        "area": area,
                        "room": f"room_{camera_index % 5}",
                        "camera_id": f"{area_index:02d}{camera_index:030d}",
                        "frame_id": str(camera_index),
                    }
                )
        first = select_four_per_area(candidates, 20260805)
        second = select_four_per_area(list(reversed(candidates)), 20260805)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        for area in AREAS:
            rows = [row for row in first if row["area"] == area]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({row["camera_id"] for row in rows}), 4)
            self.assertEqual(len({row["room"] for row in rows}), 4)

    def test_png_roundtrip_preserves_hwc_order_mask_and_invalid_triplet(self):
        rel = np.array(
            [[[1, 2, 3], [255, 255, 255]], [[4, 5, 6], [7, 8, 9]]],
            dtype=np.uint8,
        )
        valid = np.array([[True, False], [True, True]])
        with tempfile.TemporaryDirectory() as directory:
            result = roundtrip_png(rel, valid, Path(directory), "fixture")
        self.assertEqual(result["rel_mismatch_count"], 0)
        self.assertEqual(result["mask_mismatch_count"], 0)
        self.assertTrue(result["channel_order_preserved"])
        self.assertTrue(result["invalid_triplet_preserved"])


if __name__ == "__main__":
    unittest.main()
