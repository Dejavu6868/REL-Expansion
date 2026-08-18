import importlib
import sys

import numpy as np

from rel_plus.encoding import encode_rel_channels, erp_tangent_field


AUTHORITATIVE_REPO = "/home/zhuzhaoziao/RELPlus/CMX-REL"


def _load_authoritative_rel():
    if AUTHORITATIVE_REPO not in sys.path:
        sys.path.insert(0, AUTHORITATIVE_REPO)
    return importlib.import_module("third_party.rel_original.rel")


def test_new_source_encoder_is_pixel_exact_against_live_authoritative_getrel(monkeypatch):
    height, width = 5, 8
    rows, columns = np.indices((height, width), dtype=np.float32)
    points = np.stack(
        [0.2 + columns, 0.3 + rows, 1.0 + 0.1 * columns + 0.2 * rows],
        axis=-1,
    )
    normals = np.stack(
        [0.2 + 0.01 * columns, -0.3 + 0.02 * rows, np.ones_like(rows)],
        axis=-1,
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    depth_marker = np.ones((height, width), dtype=np.float32)
    depth_marker[0, 0] = 0.0
    valid = depth_marker != 0.0

    authoritative = _load_authoritative_rel()
    monkeypatch.setattr(
        authoritative,
        "processDepthImage_ERP",
        lambda _depth_cm, _missing: (points.copy(), normals.copy(), [0.0, 0.0]),
    )
    expected = authoritative.getREL(depth_marker, alpha=45.0, lam=0.5)

    tangent, _ = erp_tangent_field(height, width)
    actual, _ = encode_rel_channels(
        points,
        normals,
        valid,
        tangent=tangent,
        alpha=45.0,
        lam=0.5,
    )

    np.testing.assert_array_equal(actual[..., 0], expected[..., 0])
    np.testing.assert_array_equal(actual[..., 1], expected[..., 1])
    np.testing.assert_array_equal(actual[..., 2], expected[..., 2])
    np.testing.assert_array_equal(actual, expected)

