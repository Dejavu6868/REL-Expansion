#!/usr/bin/env python3
"""Write a training subset of the S2D pinhole views for ``train_view_subset.py``.

``--rule greedy`` drops ``--drop-fraction`` of the training views with the greedy coverage
rule of ``audit_view_coverage.py`` (each time the view whose loss pushes the fewest
directions of its camera location below ``--greedy-k`` views). ``--rule random`` drops the
same number uniformly at random with ``--seed``. CPU only.

Writes ``--output`` (kept sample IDs, one per line, in manifest order) and ``--output`` with
``.json`` added: the rule, counts, SHA-256 of the list and of the manifest, and the coverage the
subset loses, in the same terms as ``audit_view_coverage.py``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import audit_view_coverage as coverage
import cross_projection as cp


def select(rule, masks, groups, drop_count, greedy_k, seed):
    if rule == "greedy":
        return coverage.greedy_drop(masks, groups, drop_count, greedy_k)
    keep = np.ones(len(masks), dtype=bool)
    keep[np.random.default_rng(seed).choice(len(masks), drop_count, replace=False)] = False
    return keep


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="full_manifest.csv")
    parser.add_argument("--rule", required=True, choices=("greedy", "random"))
    parser.add_argument("--output", required=True, type=Path, help="subset list to write (.txt)")
    parser.add_argument("--source-root", type=Path, default=cp.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--drop-fraction", type=float, default=0.15)
    parser.add_argument("--directions", type=int, default=4096)
    parser.add_argument("--greedy-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None, help="required for --rule random")
    args = parser.parse_args()
    if args.rule == "random" and args.seed is None:
        parser.error("--rule random needs --seed")
    if args.output.exists():
        raise FileExistsError("{} exists; refusing to overwrite".format(args.output))

    cp.import_frozen_source(args.source_root)
    rows = coverage.read_split(args.manifest, "train")
    directions = coverage.sphere_directions(args.directions)
    bands = coverage.band_masks(directions)
    pitches, _, _, masks, groups, _ = coverage.load_views(rows, directions)
    drop_count = int(round(args.drop_fraction * len(rows)))
    keep = select(args.rule, masks, groups, drop_count, args.greedy_k, args.seed)

    before = coverage.multiplicity_after(masks, groups, np.ones(len(rows), dtype=bool))
    after = coverage.multiplicity_after(masks, groups, keep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(row["sample_id"] + "\n" for row, kept in zip(rows, keep) if kept), encoding="utf-8")
    report = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "rule": args.rule,
        "drop_fraction": args.drop_fraction,
        "seed": args.seed,
        "greedy_k": args.greedy_k if args.rule == "greedy" else None,
        "directions": args.directions,
        "train_views": len(rows),
        "kept": int(keep.sum()),
        "dropped": int((~keep).sum()),
        "subset": str(args.output.resolve()),
        "subset_sha256": sha256(args.output),
        "max_abs_pitch_kept_deg": float(np.abs(pitches[keep]).max()),
        "min_views_left_at_a_location": int(min(keep[rows_].sum() for rows_ in groups)),
        "coverage": coverage.coverage_summary(after, bands),
        "lost_share_by_level": coverage.lost_shares(before, after, bands),
    }
    report_path = args.output.with_name(args.output.name + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.rule, "kept", report["kept"], "of", len(rows), "holes",
          report["lost_share_by_level"]["1"]["whole_sphere"], "->", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
