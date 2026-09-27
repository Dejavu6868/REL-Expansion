import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from rel_plus.camera import json_k_to_rel_helper_k
from rel_plus.encoding import encode_rel_channels, erp_tangent_field
from rel_plus.source_helpers import compute_normals_square_support, get_r_matrix, rotate_pc


FIXTURE = Path(__file__).parent / "fixtures/source_golden.npz"


def load_live_sources():
    authoritative_root = os.environ.get("REL_AUTHORITATIVE_ROOT")
    perspective_root = os.environ.get("REL_PERSPECTIVE_REFERENCE_ROOT")
    if not authoritative_root or not perspective_root:
        pytest.skip("REL source roots were not supplied")
    authoritative = Path(authoritative_root)
    perspective = Path(perspective_root)
    if not authoritative.exists() or not perspective.exists():
        pytest.skip("REL source roots do not exist")
    if (authoritative / "third_party/rel_original/rel.py").is_file():
        sys.path.insert(0, str(authoritative))
    elif (authoritative / "rel.py").is_file():
        sys.path.insert(0, str(authoritative.parent.parent))
    else:
        pytest.skip("REL authoritative root has no rel.py")
    rel_module = importlib.import_module("third_party.rel_original.rel")

    hha_file = Path(
        os.environ.get("REL_HHA_REFERENCE_FILE", "/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py")
    )
    if not hha_file.is_file():
        pytest.skip("compatible hha_util.py is unavailable")
    hha_spec = importlib.util.spec_from_file_location("utils.hha_util", str(hha_file))
    hha_module = importlib.util.module_from_spec(hha_spec)
    sys.modules[hha_spec.name] = hha_module
    hha_spec.loader.exec_module(hha_module)
    rgbd_spec = importlib.util.spec_from_file_location(
        "live_rgbd_util", str(perspective / "utils/rgbd_util.py")
    )
    rgbd_module = importlib.util.module_from_spec(rgbd_spec)
    rgbd_spec.loader.exec_module(rgbd_module)
    return rel_module, rgbd_module, hha_module


@pytest.mark.live_source
def test_all_audited_helpers_match_live_sources():
    rel_module, rgbd_module, hha_module = load_live_sources()
    with np.load(FIXTURE) as data:
        source_rotation = hha_module.getRMatrix(data["rotation_initial"], data["rotation_final"])
        np.testing.assert_allclose(get_r_matrix(data["rotation_initial"], data["rotation_final"]), source_rotation, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(rotate_pc(data["rotation_points"], source_rotation), hha_module.rotatePC(data["rotation_points"], source_rotation), rtol=0.0, atol=0.0)

        helper_k = json_k_to_rel_helper_k(data["normal_k_json"])
        superpixels = np.ones(data["normal_depth_m"].shape, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_normal, expected_offset = rgbd_module.computeNormalsSquareSupport(
                data["normal_depth_m"], data["normal_missing"], 2, 1, helper_k, superpixels.copy()
            )
            actual_normal, actual_offset = compute_normals_square_support(
                data["normal_depth_m"], data["normal_missing"], 2, helper_k, superpixels.copy()
            )
        np.testing.assert_allclose(actual_normal, expected_normal, rtol=0.0, atol=0.0, equal_nan=True)
        np.testing.assert_allclose(actual_offset, expected_offset, rtol=0.0, atol=0.0, equal_nan=True)

        depth_marker = data["encoding_valid"].astype(np.float32)
        rel_module.processDepthImage_ERP = lambda _depth, _missing: (
            data["encoding_points_cm"].copy(), data["encoding_normals"].copy(), [0.0, 0.0]
        )
        expected_rel = rel_module.getREL(depth_marker, alpha=45.0, lam=0.5)
        tangent, _ = erp_tangent_field(*depth_marker.shape)
        actual_rel, _ = encode_rel_channels(
            data["encoding_points_cm"], data["encoding_normals"],
            data["encoding_valid"], tangent=tangent
        )
        np.testing.assert_array_equal(actual_rel, expected_rel)
