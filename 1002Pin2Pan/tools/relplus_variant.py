"""REL+ EGVIA variants: frozen v2.1 and height mixed into every surface.

Frozen v2.1 mixes normalised height into EGVIA only where the surface is
more than alpha = 45 degrees from horizontal, so floor and table tops get the
same EGVIA (and LOA ~90): only ReD tells them apart. ``height_everywhere``
mixes height into every surface. Both frozen encoders
(``rel_plus.encoding.encode_rel_channels`` and the ERP ``getREL``) clip the
angle to [0, 255] and then test ``angle <= t or angle >= 255 - t`` with
``t = alpha * 255 / 180``, so a negative alpha marks no pixel horizontal.
Nothing else changes: LOA, ReD, the valid mask and the invalid value 255.
"""

import functools
import json
from pathlib import Path


VARIANT_ALPHA = {"v2_1": 45.0, "height_everywhere": -1.0}
MARKER_NAME = "relplus_variant.json"  # written into the cache root by relplus_variant_cache.py


def apply_variant(variant, frozen=None):
    """Point the frozen generator (and processes forked later) and ``frozen["getREL"]`` at a variant."""
    import rel_plus.generator as generator

    alpha = VARIANT_ALPHA[variant]
    generator.REL_PLUS_V2_ALPHA = alpha  # read by generate_rel_plus_v2 at call time
    if frozen is not None and "getREL" in frozen:
        original = getattr(frozen["getREL"], "func", frozen["getREL"])
        frozen["getREL"] = functools.partial(original, alpha=alpha)
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
