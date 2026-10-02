import numpy as np

from rel_plus.validation.v1_v2_diff import summarize_difference


def test_difference_summary_counts_pixels_channels_and_intent():
    before = np.zeros((2, 3, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0, 0] = 3
    after[1, 2, :2] = 4
    row = summarize_difference(
        "normal_nan", before, after, "INTENTIONAL_NORMAL_MASK_FIX"
    )
    assert row["changed_pixel_count"] == 2
    assert row["changed_channel_count"] == 3
    assert row["max_difference"] == 4
    assert row["intentional_changed_pixel_count"] == 2
    assert row["unexpected_difference_count"] == 0


def test_unchanged_classification_exposes_any_difference_as_unexpected():
    before = np.zeros((1, 1, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0, 2] = 1
    row = summarize_difference(
        "dense", before, after, "UNCHANGED_DENSE_VALID"
    )
    assert row["unexpected_difference_count"] == 1
