"""Shared augmentation profile for future RGBD/HHA/REL+ comparisons."""

import random

from rel_plus.integration.cmx_preprocess import sample_spatial_transform


S2D_RELPLUS_COMPARISON_NO_FLIP = "S2D_RELPLUS_COMPARISON_NO_FLIP"


def author_epoch_seed(base_seed, epoch, rank):
    return int(base_seed) + int(epoch) + int(rank) * 1000


def sample_comparison_transform(input_shape, scales, output_shape, rng=None):
    return sample_spatial_transform(
        input_shape,
        scales,
        output_shape,
        random if rng is None else rng,
    )


def _trace_row(sample_id, epoch, rank, transform):
    cropped_height = min(
        transform.output_height,
        transform.scaled_height - transform.crop_top,
    )
    cropped_width = min(
        transform.output_width,
        transform.scaled_width - transform.crop_left,
    )
    pad_height = max(0, transform.output_height - cropped_height)
    pad_width = max(0, transform.output_width - cropped_width)
    return {
        "sample_id": sample_id,
        "epoch": int(epoch),
        "rank": int(rank),
        "scale": float(transform.scale),
        "crop_top": int(transform.crop_top),
        "crop_left": int(transform.crop_left),
        "pad_top": pad_height // 2,
        "pad_bottom": pad_height - pad_height // 2,
        "pad_left": pad_width // 2,
        "pad_right": pad_width - pad_width // 2,
    }


def trace_comparison_profile(
    sample_ids,
    *,
    input_shape,
    output_shape,
    scales,
    base_seed,
    epoch,
    rank,
    x_modes=("rgbd", "hha", "rel_plus_v2_1"),
):
    """Return arm-identical traces without loading data or running a model."""
    seed = author_epoch_seed(base_seed, epoch, rank)
    rows = []
    rng = random.Random(seed)
    for sample_id in sample_ids:
        transform = sample_comparison_transform(
            input_shape, scales, output_shape, rng
        )
        rows.append(_trace_row(sample_id, epoch, rank, transform))
    return {mode: [dict(row) for row in rows] for mode in x_modes}
