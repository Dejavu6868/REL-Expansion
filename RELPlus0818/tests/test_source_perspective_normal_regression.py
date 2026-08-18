import importlib.util
import sys
from pathlib import Path

import numpy as np

from rel_plus.camera import json_k_to_rel_helper_k
from rel_plus.source_helpers import compute_normals_square_support


REFERENCE_ROOT = Path("/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference")
COMPATIBLE_HHA = Path("/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py")


def _load_reference_rgbd_util():
    sys.path.insert(0, str(REFERENCE_ROOT))
    hha_spec = importlib.util.spec_from_file_location(
        "utils.hha_util", str(COMPATIBLE_HHA)
    )
    hha_module = importlib.util.module_from_spec(hha_spec)
    sys.modules[hha_spec.name] = hha_module
    hha_spec.loader.exec_module(hha_module)
    path = REFERENCE_ROOT / "utils/rgbd_util.py"
    spec = importlib.util.spec_from_file_location("rel_public_rgbd_util", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perspective_radius_two_normals_match_live_public_helper():
    height, width = 9, 11
    rows, columns = np.indices((height, width), dtype=np.float64)
    depth_m = (1.5 + 0.01 * columns + 0.02 * rows).astype(np.float32)
    missing = np.zeros((height, width), dtype=bool)
    missing[1, 2] = True
    k_json = np.array(
        [[80.0, 0.0, 5.5], [0.0, 82.0, 4.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    helper_k = json_k_to_rel_helper_k(k_json)
    superpixels = np.ones((height, width), dtype=np.float32)
    reference = _load_reference_rgbd_util()

    with np.errstate(divide="ignore", invalid="ignore"):
        expected, expected_offset = reference.computeNormalsSquareSupport(
            depth_m, missing, 2, 1, helper_k, superpixels.copy()
        )
        actual, actual_offset = compute_normals_square_support(
            depth_m, missing, 2, helper_k, superpixels.copy()
        )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(
        actual_offset, expected_offset, rtol=0.0, atol=0.0, equal_nan=True
    )
