"""REL+ EGVIA variants: frozen v2.1 and REL+ 3.0, which mixes height into every surface.

Frozen v2.1 mixes normalised height into EGVIA only where the surface is
more than alpha = 45 degrees from horizontal, so floor and table tops get the
same EGVIA (and LOA ~90): only ReD tells them apart. 3.0 (``v3_0``) mixes
height into every surface. Both frozen encoders
(``rel_plus.encoding.encode_rel_channels`` and the ERP ``getREL``) clip the
angle to [0, 255] and then test ``angle <= t or angle >= 255 - t`` with
``t = alpha * 255 / 180``, so a negative alpha marks no pixel horizontal.
Nothing else changes: LOA, ReD, the valid mask and the invalid value 255.

REL+ 3.1 (``v3_1``, the ``banded_metric`` rule of check_egvia_rules.py) replaces the
EGVIA formula instead of alpha:
EGVIA = 85 [S((theta - 45) / 15) + S((theta - 135) / 15) + clip((z - z_lo) / (z_hi - z_lo), 0, 1)],
S(x) = 1/2 + 3/4 x - 1/4 x^3 on clip(x, -1, 1), with theta the unblended surface
angle in degrees and z the camera-relative height in cm, [z_lo, z_hi] =
METRIC_RANGE_CM (floor p5 to ceiling p95 of the training split,
check_egvia_rules.py). It is applied by swapping the generator's
``encode_rel_channels`` for a wrapper that rewrites channel 0, and
``frozen["getREL"]`` for ``get_rel_banded_metric``, which repeats the ERP
``getREL`` statements for LOA and ReD.

REL+ 3.2 (``v3_2``) is 3.1 with metric ReD: the horizontal distance from the
camera in cm, ReD = 255 clip(d / METRIC_RED_MAX_CM, 0, 1), instead of the
per-image min/max that invalid pixels (stored at d = 0) pull to 0. Invalid
pixels stay 255. METRIC_RED_MAX_CM is the p99 horizontal distance of the
training split (Phase 0, check_input_defects.py). EGVIA and LOA are 3.1's.
"""

import functools
import json
from pathlib import Path

import numpy as np


VARIANT_ALPHA = {"v2_1": 45.0, "v3_0": -1.0, "v3_1": None, "v3_2": None}  # None: not an alpha variant
MARKER_NAME = "relplus_variant.json"  # written into the cache root by relplus_variant_cache.py
METRIC_RANGE_CM = (-150.0, 210.0)
BAND_HALF_WIDTH_DEG = 15.0
METRIC_RED_MAX_CM = 1200.0  # v3_2 only


def describe(variant):
    """The EGVIA rule of a variant, for reports."""
    if VARIANT_ALPHA[variant] is not None:
        return "alpha {}".format(VARIANT_ALPHA[variant])
    rule = "banded_metric (metric range {} cm, band half-width {} deg)".format(
        list(METRIC_RANGE_CM), BAND_HALF_WIDTH_DEG
    )
    if variant == "v3_2":
        rule += ", metric ReD (0 to {} cm)".format(METRIC_RED_MAX_CM)
    return rule


def smooth_step(x):
    x = np.clip(x, -1.0, 1.0)
    return 0.5 + 0.75 * x - 0.25 * x ** 3


def banded_metric_egvia(angle, height_cm):
    """Float EGVIA from the unblended angle code (0..255) and the camera-relative height in cm."""
    degrees = angle * 180.0 / 255.0
    low, high = METRIC_RANGE_CM
    return 85.0 * (
        smooth_step((degrees - 45.0) / BAND_HALF_WIDTH_DEG)
        + smooth_step((degrees - 135.0) / BAND_HALF_WIDTH_DEG)
        + np.clip((height_cm - low) / (high - low), 0.0, 1.0)
    )


def egvia_byte(egvia, valid):
    """The frozen encoders' cast: NaN and invalid pixels to 255, clip, truncate."""
    egvia = np.nan_to_num(egvia, nan=255.0)
    egvia[~valid] = 255.0
    return np.clip(egvia, 0, 255).astype(np.uint8)


def metric_red_byte(distance_cm, valid):
    """v3_2 ReD from the horizontal distance in cm; invalid pixels 255."""
    return egvia_byte(255.0 * distance_cm / METRIC_RED_MAX_CM, valid)


def encode_banded_metric(points_for_encoding_cm, normals_aligned, valid_mask, metric_red=False, **kwargs):
    """Frozen ``encode_rel_channels`` with channel 0 rewritten by ``banded_metric_egvia``,
    and channel 2 by ``metric_red_byte`` when ``metric_red`` is set."""
    from rel_plus.encoding import encode_rel_channels

    encoded, debug = encode_rel_channels(points_for_encoding_cm, normals_aligned, valid_mask, **kwargs)
    encoded = encoded.copy()
    encoded[:, :, 0] = egvia_byte(
        banded_metric_egvia(debug["egvia_angle_before_blend"], debug["height_raw_cm"]),
        np.asarray(valid_mask, dtype=bool),
    )
    debug["egvia_encoded"] = encoded[:, :, 0]
    if metric_red:
        encoded[:, :, 2] = metric_red_byte(debug["red_raw_cm"], np.asarray(valid_mask, dtype=bool))
        debug["red_encoded"] = encoded[:, :, 2]
    return encoded, debug


def get_rel_banded_metric(D, metric_red=False):
    """ERP ``getREL`` with banded_metric EGVIA; LOA and ReD repeat its statements,
    except that ``metric_red`` replaces ReD with ``metric_red_byte``."""
    from third_party.rel_original.rgbd_util import processDepthImage_ERP

    missingMask = D == 0
    pcRot, N, rot = processDepthImage_ERP(D * 100, missingMask)
    h, w = D.shape
    u = np.arange(w)
    phi = (u / w) * 2 * np.pi - np.pi
    cos_theta = np.cos(phi)[np.newaxis, :]
    sin_theta = np.sin(phi)[np.newaxis, :]
    hcos = N[:, :, 0] * cos_theta - N[:, :, 1] * sin_theta
    hcos = np.nan_to_num(hcos, nan=0)
    hcos = np.clip(hcos, -1.0, 1.0)
    HA = (np.arccos(hcos) * 180 / np.pi).astype(np.uint8)

    RD = np.hypot(pcRot[:, :, 0], pcRot[:, :, 1])
    if metric_red:
        RD = metric_red_byte(RD, ~missingMask)
    else:
        RD_min = RD.min()
        RD_max = RD.max()
        if RD_max > RD_min:
            RD = (RD - RD_min) * 255.0 / (RD_max - RD_min)
        RD = np.clip(RD, 0, 255).astype(np.uint8)

    N_z = np.clip(-N[:, :, 2], -1.0, 1.0)
    angle = np.clip((np.arccos(N_z, dtype=np.float32) / np.pi) * 255.0, 0, 255).astype(np.float32)
    egvia = egvia_byte(banded_metric_egvia(angle, pcRot[:, :, 2]), ~missingMask)

    I = np.stack([egvia, HA, RD], axis=2).astype(np.float32)
    I = np.nan_to_num(I, nan=255.0)
    I[missingMask, :] = 255.0
    return np.clip(I, 0, 255).astype(np.uint8)


def apply_variant(variant, frozen=None):
    """Point the frozen generator (and processes forked later) and ``frozen["getREL"]`` at a variant."""
    import rel_plus.encoding as encoding
    import rel_plus.generator as generator
    from third_party.rel_original.rel import getREL

    alpha = VARIANT_ALPHA[variant]
    if alpha is not None:
        encode, get_rel = encoding.encode_rel_channels, functools.partial(getREL, alpha=alpha)
    elif variant == "v3_2":
        encode = functools.partial(encode_banded_metric, metric_red=True)
        get_rel = functools.partial(get_rel_banded_metric, metric_red=True)
    else:
        encode, get_rel = encode_banded_metric, get_rel_banded_metric
    # Both are read by generate_rel_plus_v2 at call time.
    generator.REL_PLUS_V2_ALPHA = VARIANT_ALPHA["v2_1"] if alpha is None else alpha
    generator.encode_rel_channels = encode
    if frozen is not None and "getREL" in frozen:
        frozen["getREL"] = get_rel
    return alpha


def read_marker(cache_root):
    """The variant a cache was generated with; a cache without a marker is frozen v2.1."""
    path = Path(cache_root) / MARKER_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def select_for_eval(variant, config, frozen):
    """Apply an eval's variant after checking it against the checkpoint's training cache."""
    if config.x_mode != "rel_plus_v2_1":
        if variant != "v2_1":
            raise ValueError("--relplus-variant applies to the REL+ arm only")
        return None
    cache_root = Path(config.x_root_folder).parent
    marker = read_marker(cache_root)
    cached = "v2_1" if marker is None else marker["variant"]
    if cached != variant:
        raise ValueError(
            "--relplus-variant {} but the checkpoint was trained on the {} cache {}".format(
                variant, cached, cache_root
            )
        )
    apply_variant(variant, frozen)
    return variant
