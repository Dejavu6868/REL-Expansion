import numpy as np
import pytest

from rel_plus.geometry import (
    GravityAlignmentSingularity,
    align_points_and_normals_to_gravity,
)


def vector_from_angle_to_down(degrees):
    angle = np.radians(degrees)
    return np.array([np.sin(angle), 0.0, -np.cos(angle)])


@pytest.mark.parametrize("degrees", [0.0, 25.0, 179.9])
def test_nonsingular_gravity_matches_source_alignment(degrees):
    gravity = vector_from_angle_to_down(degrees)
    field = gravity.reshape(1, 1, 3)
    aligned, _, matrix = align_points_and_normals_to_gravity(field, field, gravity, sample_id="s")
    np.testing.assert_allclose(aligned[0, 0], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-10)


@pytest.mark.parametrize("degrees", [179.999999, 180.0])
def test_antiparallel_gravity_raises_structured_error(degrees):
    gravity = vector_from_angle_to_down(degrees)
    with pytest.raises(GravityAlignmentSingularity, match=r"sample_id=sample-42.*gravity=.*angle_deg="):
        align_points_and_normals_to_gravity(
            np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), gravity, sample_id="sample-42"
        )
