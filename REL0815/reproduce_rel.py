#!/usr/bin/env python3
"""Generate original ERP-REL PNGs for an explicit, frozen sample manifest."""

import argparse
import csv
from pathlib import Path

import cv2

from rel_original import getImage, getREL


def main():
    parser = argparse.ArgumentParser(description="Reproduce original ERP-REL")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=45)
    parser.add_argument("--lam", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))

    for sample in samples:
        depth = getImage(sample["absolute_depth_path"], "Stanford2D3DPano")
        rel = getREL(depth, alpha=args.alpha, lam=args.lam)
        output_path = output_dir / f"{sample['sample_id']}.png"
        if not cv2.imwrite(str(output_path), rel):
            raise RuntimeError(f"Failed to save {output_path}")
        print(f"{sample['sample_id']} -> {output_path}")


if __name__ == "__main__":
    main()
