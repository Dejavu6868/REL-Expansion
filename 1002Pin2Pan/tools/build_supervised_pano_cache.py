#!/usr/bin/env python3
"""Write the REL+ v3.1new panorama X cache for supervised training with Max's Pano_MA_Seg code.

For every panorama in ``--data-root``'s fold split files (``area_X/<stem>`` lines, train and test),
it reads the raw Stanford2D3D depth ``<stanford-root>/<area>/pano/depth/<stem>_depth.png`` at its native
2048x4096 and writes ``getREL`` with the v3.1new EGVIA rule (``relplus_variant.get_rel_banded_metric``),
the same X that ``eval_pano_transfer.py --relplus-variant v3_1new`` feeds the network, to
``<output>/X/<area>/<stem>_hha.png``: the folder layout and file names of Max's ``rle`` folder.

The marker ``<output>/relplus_variant.json`` is written first and says v3_1new; it is the parent of the
X folder, as for every other REL+ cache, so the training and evaluation scripts can refuse anything else.
``manifest.json`` is written last, once every panorama is written.
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import cv2

import eval_pano_transfer as ev
import relplus_variant as rv
from cross_projection import decode_erp_depth

VARIANT = "v3_1new"
SIZE = (2048, 4096)
X_SUFFIX = "_hha.png"  # Max's rle folder names its REL images so
MANIFEST_NAME = "manifest.json"
_JOB = {}  # set before the worker pool forks


def split_names(data_root, fold):
    names = []
    for split in ("train", "test"):
        path = Path(data_root) / "fold{}_{}.txt".format(fold, split)
        names += [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(names)) != len(names):
        raise ValueError("the fold {} train and test splits share panoramas".format(fold))
    return names


def write_sample(name):
    area, stem = name.split("/")
    depth_path = _JOB["stanford_root"] / area / "pano" / "depth" / (stem + "_depth.png")
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != "uint16" or raw.shape != SIZE:
        raise ValueError("{} is not a {}x{} uint16 depth map".format(depth_path, *SIZE))
    modal_x = _JOB["getREL"](decode_erp_depth(raw))
    path = _JOB["x_root"] / area / (stem + X_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), modal_x):
        raise OSError("could not write " + str(path))
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stanford-root", required=True, type=Path, help="raw Stanford2D3D (area_*/pano/depth)")
    parser.add_argument("--data-root", required=True, type=Path, help="Max's datasets/Stanford2D3DPano (split files)")
    parser.add_argument("--output", required=True, type=Path, help="new, empty folder")
    parser.add_argument("--fold", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--source-root", type=Path, default=ev.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))

    frozen = ev.import_frozen(args.source_root)
    rv.apply_variant(VARIANT, frozen)
    names = split_names(args.data_root, args.fold)[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / rv.MARKER_NAME).write_text(json.dumps({
        "variant": VARIANT,
        "egvia_rule": rv.describe(VARIANT),
        "writer": "build_supervised_pano_cache.py",
    }, indent=2) + "\n", encoding="utf-8")
    _JOB.update(stanford_root=args.stanford_root, x_root=args.output / "X", getREL=frozen["getREL"])

    started = time.time()
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for index, _ in enumerate(pool.imap_unordered(write_sample, names), start=1):
            if index % 50 == 0 or index == len(names):
                print("{}/{} panoramas {:.0f}s".format(index, len(names), time.time() - started), flush=True)

    manifest = {
        "status": "SMOKE" if args.limit is not None else "COMPLETED",
        "relplus_variant": VARIANT,
        "panorama_x_input": "ERP getREL ({}) on the native {}x{} depth".format(rv.describe(VARIANT), *SIZE),
        "stanford_root": str(args.stanford_root),
        "data_root": str(args.data_root),
        "fold": args.fold,
        "panoramas": len(names),
        "seconds": round(time.time() - started),
    }
    (args.output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("cached {} panoramas in {}".format(len(names), args.output / "X"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
