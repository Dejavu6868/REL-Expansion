import numpy as np

from rel_plus.camera import CameraGeometry
from rel_plus.validation.canonical_geometry import validate_canonical_geometry


def test_analytic_native_to_canonical_nearest_geometry_passes():
    shape = (108, 108)
    rows, columns = np.indices(shape, dtype=np.float64)
    raw = np.rint((2.0 + 0.002 * columns + 0.001 * rows) * 512.0).astype(np.uint16)
    k = np.array([[120.0, 0.0, 54.0], [0.0, 118.0, 54.0], [0.0, 0.0, 1.0]])
    z = raw.astype(np.float64) / 512.0
    xyz = np.stack(
        [
            (columns + 0.5 - k[0, 2]) * z / k[0, 0],
            (rows + 0.5 - k[1, 2]) * z / k[1, 1],
            z,
        ],
        axis=-1,
    )
    camera = CameraGeometry.from_json_k(k, shape, np.eye(3), np.zeros(3))
    result = validate_canonical_geometry(raw, xyz, camera, (48, 48))
    assert result["status"] == "PASS"
    assert result["native"]["reprojection_pixels"]["p95"] < 1e-10
    assert 0.0 < result["canonical"]["reprojection_pixels"]["p95"] < 1.0
