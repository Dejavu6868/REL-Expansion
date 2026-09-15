from pathlib import Path

import numpy as np

from rel_plus.source_helpers import get_r_matrix, rotate_pc


FIXTURE = Path(__file__).parent / "fixtures/source_golden.npz"


def test_source_rotation_offline_golden():
    with np.load(FIXTURE) as data:
        actual_rotation = get_r_matrix(data["rotation_initial"], data["rotation_final"])
        np.testing.assert_allclose(actual_rotation, data["rotation_expected"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            rotate_pc(data["rotation_points"], data["rotation_expected"]),
            data["rotated_points_expected"], rtol=0.0, atol=0.0
        )
