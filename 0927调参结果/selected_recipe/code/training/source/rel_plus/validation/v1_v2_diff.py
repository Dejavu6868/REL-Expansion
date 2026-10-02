"""Structured accounting for expected and unexpected v1-to-v2 byte changes."""

import numpy as np


def summarize_difference(case_name, v1_array, v2_array, classification):
    before = np.asarray(v1_array, dtype=np.uint8)
    after = np.asarray(v2_array, dtype=np.uint8)
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("v1_array and v2_array must be matching HxWx3 arrays")
    changed_channels = before != after
    changed_pixels = np.any(changed_channels, axis=2)
    absolute = np.abs(before.astype(np.int16) - after.astype(np.int16))
    intentional = classification.startswith("INTENTIONAL_")
    unchanged_expected = classification == "UNCHANGED_DENSE_VALID"
    unexpected_count = int(np.count_nonzero(changed_pixels)) if unchanged_expected else 0
    if not intentional and not unchanged_expected:
        unexpected_count = int(np.count_nonzero(changed_pixels))
    return {
        "case": case_name,
        "classification": classification,
        "changed_pixel_count": int(np.count_nonzero(changed_pixels)),
        "changed_channel_count": int(np.count_nonzero(changed_channels)),
        "max_difference": int(absolute.max()) if absolute.size else 0,
        "intentional_changed_pixel_count": (
            int(np.count_nonzero(changed_pixels)) if intentional else 0
        ),
        "unexpected_difference_count": unexpected_count,
    }
