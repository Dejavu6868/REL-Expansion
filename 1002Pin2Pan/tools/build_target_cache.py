#!/usr/bin/env python3
"""Cache the target-domain panoramas the adaptation trains on.

Stanford2D3D panoramas of the training areas (1, 2, 3, 4 and 6 by default)
with exactly the inputs ``eval_pano_transfer.py`` feeds the network: RGB as
the frozen loader reads it and the arm's ERP X input (REL+ ``getREL`` in the
checkpoint's variant, or ERP HHA with the verified cache recipe), at the
evaluation size. Area 5 is the test set and is refused.

Writes RGB/, X/ and Label/ PNGs under ``--output`` (``<area>/<stem>.png``) and,
once every panorama is written, ``manifest.json``. Label/ is the ground truth:
training never reads it, and ``gen_pseudo_labels.py`` uses it only to report
how good the pseudo-labels are.
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import cv2

import eval_pano_transfer as ev
import relplus_variant as rv


TRAIN_AREAS = ("area_1", "area_2", "area_3", "area_4", "area_6")
MANIFEST_NAME = "manifest.json"
_JOB = {}  # set before the worker pool forks


def load_manifest(cache_root, config):
    """The cache's manifest, checked against the checkpoint config's arm and REL+ variant."""
    path = Path(cache_root) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError("{} is missing: the target cache is incomplete".format(path))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["x_mode"] != config.x_mode:
        raise ValueError(
            "the target cache holds {} inputs but the checkpoint config is {}".format(
                manifest["x_mode"], config.x_mode
            )
        )
    if config.x_mode == "rel_plus_v2_1":
        marker = rv.read_marker(Path(config.x_root_folder).parent)
        trained = "v2_1" if marker is None else marker["variant"]
        if manifest["relplus_variant"] != trained:
            raise ValueError(
                "the target cache holds REL+ {} inputs but the checkpoint was trained on {}".format(
                    manifest["relplus_variant"], trained
                )
            )
    return manifest


def write_sample(sample):
    rgb, modal_x, label = ev.load_sample(
        sample, _JOB["size"], _JOB["lookup"], _JOB["frozen"], _JOB["x_mode"], _JOB["hha_recipe"]
    )
    for folder, image in (("RGB", rgb), ("X", modal_x), ("Label", label)):
        path = _JOB["output"] / folder / (sample["sample_id"] + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError("could not write " + str(path))
    return sample["sample_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stanford-root", required=True, type=Path)
    parser.add_argument("--semantic-labels", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="resolved arm config JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=ev.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--areas", nargs="+", default=list(TRAIN_AREAS))
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    parser.add_argument("--hha-cache-report", type=Path, default=None)
    parser.add_argument("--accept-hha-near-match", action="store_true")
    parser.add_argument(
        "--relplus-variant", choices=sorted(rv.VARIANT_ALPHA), default="v2_1",
        help="REL+ EGVIA encoding; must match the checkpoint's training cache",
    )
    args = parser.parse_args(argv)
    test_areas = [area for area in args.areas if area.startswith("area_5")]
    if test_areas:
        parser.error("area 5 is the test set and cannot be a target training area: " + " ".join(test_areas))
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))

    frozen = ev.import_frozen(args.source_root)
    config = ev.load_config(args.config)
    relplus_variant = rv.select_for_eval(args.relplus_variant, config, frozen)
    hha_recipe = None
    if config.x_mode == "hha_frozen_cache":
        hha_recipe = ev.hha_cache_recipe(
            args.hha_cache_report, config.x_root_folder, args.accept_hha_near_match
        )
    samples = ev.list_panoramas(args.stanford_root, args.areas)[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    _JOB.update(
        size=(args.height, args.width),
        lookup=ev.label_lookup(args.semantic_labels),
        frozen=frozen,
        x_mode=config.x_mode,
        hha_recipe=hha_recipe,
        output=args.output,
    )

    started = time.time()
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for index, _ in enumerate(pool.imap_unordered(write_sample, samples), start=1):
            if index % 50 == 0 or index == len(samples):
                print("{}/{} panoramas".format(index, len(samples)), flush=True)

    manifest = {
        "status": "SMOKE" if args.limit is not None else "COMPLETED",
        "config": str(args.config),
        "x_mode": config.x_mode,
        "relplus_variant": relplus_variant,
        "panorama_x_input": (
            "ERP getREL (alpha {}) on nearest-resized depth".format(rv.VARIANT_ALPHA[relplus_variant])
            if hha_recipe is None
            else "ERP HHA (hha.erp_hha), {}, channel order {}".format(
                hha_recipe["variant"], hha_recipe["channel_order"]
            )
        ),
        "hha_cache_report": None if hha_recipe is None else str(args.hha_cache_report),
        "hha_recipe": hha_recipe,
        "stanford_root": str(args.stanford_root),
        "areas": args.areas,
        "size": [args.height, args.width],
        "seconds": time.time() - started,
        "samples": [sample["sample_id"] for sample in samples],
    }
    (args.output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("cached {} panoramas in {}".format(len(samples), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
