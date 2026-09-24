from pathlib import Path

import numpy as np

from rel_plus.encoding import encode_rel_channels, erp_tangent_field


FIXTURE = Path(__file__).parent / "fixtures/source_golden.npz"


def test_source_encoder_offline_golden():
    with np.load(FIXTURE) as data:
        height, width = data["encoding_valid"].shape
        tangent, _ = erp_tangent_field(height, width)
        actual, _ = encode_rel_channels(
            data["encoding_points_cm"], data["encoding_normals"],
            data["encoding_valid"], tangent=tangent
        )
        np.testing.assert_array_equal(actual, data["encoding_expected"])
