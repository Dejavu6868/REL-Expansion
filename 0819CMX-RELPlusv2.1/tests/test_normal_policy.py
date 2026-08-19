import numpy as np

from rel_plus.camera import CameraGeometry
from rel_plus.generator import generate_rel_plus_v2
from rel_plus.normal_diagnostics import build_normal_diagnostics


def test_normal_quality_mask_is_diagnostic_only(monkeypatch):
    raw = np.full((9, 9), 1024, dtype=np.uint16)
    camera = CameraGeometry.from_json_k(
        np.array([[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]]),
        raw.shape,
        np.eye(3),
        np.zeros(3),
    )
    normals = np.zeros((9, 9, 3), dtype=np.float64)
    normals[..., 2] = 1.0
    normals[3, 3] = np.nan
    normals[4, 4] = 0.0
    diagnostics = build_normal_diagnostics(normals, np.ones((9, 9), bool), radius=2)

    def fake_estimator(*_args, **_kwargs):
        return normals, diagnostics

    monkeypatch.setattr("rel_plus.generator.estimate_source_perspective_normals", fake_estimator)
    rel_plus, debug = generate_rel_plus_v2(raw, camera, return_debug=True)
    assert debug["encoding_valid_mask"][3, 3]
    assert not debug["normal_quality_mask"][3, 3]
    np.testing.assert_array_equal(rel_plus[3, 3, :2], [255, 90])
    assert rel_plus[3, 3, 2] != 255
    assert rel_plus[4, 4, 0] != 255
    assert rel_plus[4, 4, 1] == 90
    assert rel_plus[4, 4, 2] != 255


def test_low_support_is_reported_without_invalidating_depth():
    valid = np.zeros((7, 7), dtype=bool)
    valid[3, 3] = True
    normals = np.zeros((7, 7, 3), dtype=np.float64)
    normals[..., 2] = 1.0
    diagnostics = build_normal_diagnostics(normals, valid, radius=2)
    assert diagnostics.support_count[3, 3] == 1
    assert not diagnostics.quality_mask[3, 3]
