import numpy as np

from rel_plus.integration.cmx_preprocess import (
    SpatialTransform,
    analyze_invalid_interpolation,
)


def test_invalid_interpolation_reports_joint_label_statistics_without_changing_input():
    rel_plus = np.full((4, 4, 3), 30, dtype=np.uint8)
    valid = np.ones((4, 4), dtype=bool)
    valid[:, 0] = False
    rel_plus[~valid] = 255
    label = np.tile(np.array([255, 0, 1, 1], dtype=np.uint8), (4, 1))
    original = rel_plus.copy()
    transform = SpatialTransform(4, 4, 1.5, 6, 6, 0, 0, 6, 6)

    report = analyze_invalid_interpolation(
        rel_plus,
        valid,
        transform,
        label=label,
        ignore_index=255,
        num_classes=2,
    )
    np.testing.assert_array_equal(rel_plus, original)
    assert report["affected_pixel_count"] > 0
    assert 0.0 <= report["affected_label_ignore_ratio"] <= 1.0
    assert 0.0 <= report["affected_label_valid_semantic_ratio"] <= 1.0
    assert set(report["affected_label_class_counts"]) == {"0", "1"}
    assert sum(report["affected_label_class_counts"].values()) <= report["affected_pixel_count"]
