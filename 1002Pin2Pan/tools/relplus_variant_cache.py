#!/usr/bin/env python3
"""Generate or audit a REL+ variant cache with the frozen cache tools.

``generate`` runs the frozen ``tools/generate_full_relplus_cache.py`` and
``audit`` runs the frozen ``tools/audit_full_relplus_cache.py``, unchanged,
after pointing the frozen generator at a variant (see ``relplus_variant.py``).
Every other argument goes to the frozen tool as is, so the cache layout and
the reports are the ones the training launcher already reads.

``generate`` writes ``relplus_variant.json`` into ``--output-root`` before any
image, and refuses an output root that holds another variant, or REL+ images
without a marker (the frozen ``--resume`` only checks that a PNG decodes).
``--egvia-param`` (3.1 only) builds a cache with other 3.1 parameters, for the
eval-only sensitivity sweep; the marker records them.
``audit`` takes the variant from the marker in ``--cache-root`` and regenerates
its samples byte for byte with it.
"""

import argparse
import importlib
import json
import multiprocessing
import sys
from pathlib import Path

import relplus_variant as rv
from cross_projection import DEFAULT_SOURCE_ROOT


FROZEN_TOOLS = {
    "generate": ("tools.generate_full_relplus_cache", "--output-root"),
    "audit": ("tools.audit_full_relplus_cache", "--cache-root"),
}


def prepare_output_root(output_root, variant, source_root, egvia_params=None):
    """Write the marker before any image, so a resumed run cannot mix variants."""
    output_root = Path(output_root)
    marker = rv.read_marker(output_root)
    if marker is None and (output_root / "RELPlus").exists():
        raise FileExistsError(
            "{} holds REL+ images without {}".format(output_root, rv.MARKER_NAME)
        )
    if marker is not None and (
        marker["variant"] != variant or marker.get("egvia_params") != egvia_params
    ):
        raise FileExistsError(
            "{} holds the {} variant with EGVIA parameters {}".format(
                output_root, marker["variant"], marker.get("egvia_params")
            )
        )
    output_root.mkdir(parents=True, exist_ok=True)
    marker = {
        "variant": variant,
        "egvia_alpha": rv.VARIANT_ALPHA[variant],
        "egvia_rule": rv.describe(variant),
        "egvia_params": egvia_params,
        "frozen_egvia_alpha": rv.VARIANT_ALPHA["v2_1"],
        "source_root": str(Path(source_root).resolve()),
    }
    (output_root / rv.MARKER_NAME).write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("command", choices=sorted(FROZEN_TOOLS))
    parser.add_argument(
        "--variant", choices=sorted(rv.VARIANT_ALPHA), default="v3_0",
        help="generate only; audit uses the cache's marker",
    )
    parser.add_argument(
        "--egvia-param", action="append", metavar="NAME=VALUE",
        help="generate with --variant v3_1 only: override a 3.1 parameter ({})".format(
            ", ".join(sorted(rv.DEFAULT_EGVIA_PARAMS))
        ),
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args, forwarded = parser.parse_known_args(argv)
    module_name, root_flag = FROZEN_TOOLS[args.command]
    locate = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    locate.add_argument(root_flag, dest="root", type=Path, required=True)
    root = locate.parse_known_args(forwarded)[0].root

    egvia_params = None
    if args.command == "generate":
        variant = args.variant
        if args.egvia_param:
            if variant != "v3_1":
                parser.error("--egvia-param needs --variant v3_1")
            egvia_params = rv.set_egvia_params(rv.parse_egvia_params(args.egvia_param))
        prepare_output_root(root, variant, args.source_root, egvia_params)
    else:
        marker = rv.read_marker(root)
        if marker is None:
            raise FileNotFoundError(
                "{} has no {}; audit frozen v2.1 caches with the frozen tool".format(
                    root, rv.MARKER_NAME
                )
            )
        variant = marker["variant"]
        if args.egvia_param:
            parser.error("audit takes the EGVIA parameters from the marker")
        egvia_params = marker.get("egvia_params")

    sys.dont_write_bytecode = True  # keep the frozen archive's file set unchanged
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    tool = importlib.import_module(module_name)
    rv.apply_variant(variant)
    if variant == "v3_1":
        rv.set_egvia_params(egvia_params)
    multiprocessing.set_start_method("fork", force=True)  # cache workers inherit the variant
    print(json.dumps({"relplus_variant": variant, "egvia_rule": rv.describe(variant)}), flush=True)
    sys.argv = [tool.__file__] + forwarded
    return tool.main()


if __name__ == "__main__":
    raise SystemExit(main())
