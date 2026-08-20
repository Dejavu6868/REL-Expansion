from pathlib import Path

import numpy as np

from rel_plus.camera import json_k_to_rel_helper_k
from rel_plus.source_helpers import compute_normals_square_support


FIXTURE = Path(__file__).parent / "fixtures/source_golden.npz"


def test_source_normal_offline_golden():
    with np.load(FIXTURE) as data:
        helper_k = json_k_to_rel_helper_k(data["normal_k_json"])
        with np.errstate(divide="ignore", invalid="ignore"):
            normals, offset = compute_normals_square_support(
                data["normal_depth_m"], data["normal_missing"], 2, helper_k,
                np.ones(data["normal_depth_m"].shape, dtype=np.float32)
            )
        np.testing.assert_allclose(normals, data["normal_expected"], rtol=0.0, atol=0.0, equal_nan=True)
        np.testing.assert_allclose(offset, data["normal_offset_expected"], rtol=0.0, atol=0.0, equal_nan=True)
