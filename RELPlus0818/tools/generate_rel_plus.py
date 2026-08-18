#!/usr/bin/env python3
"""Generate one canonical Stanford2D3D S2D REL+ v1 PNG."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rel_plus.generator import generate_rel_plus
from rel_plus.stanford_s2d import load_canonical_frame
from rel_plus.storage import save_rel_plus_png
from visualize_rel_plus import save_review_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", required=True, help="native Depth16 PNG")
    parser.add_argument("--camera-json", required=True, help="Stanford pose JSON")
    parser.add_argument("--output", required=True, help="output REL+ PNG")
    parser.add_argument("--debug-dir", help="optional debug array and montage directory")
    args = parser.parse_args()

    raw_depth, camera, source_shape = load_canonical_frame(
        args.depth, args.camera_json, canonical_shape=(480, 480)
    )
    rel_plus, debug = generate_rel_plus(raw_depth, camera, return_debug=True)
    save_rel_plus_png(args.output, rel_plus)

    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(debug_dir / "debug_arrays.npz", **debug)
        rgb_stub = np.zeros((480, 480, 3), dtype=np.uint8)
        save_review_bundle(
            debug_dir, rgb_stub, debug, rgb_label="RGB not provided by this CLI"
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "source_shape": list(source_shape),
                "output_shape": list(rel_plus.shape),
                "dtype": str(rel_plus.dtype),
                "channel_order": ["EGVIA", "LOA", "ReD"],
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
