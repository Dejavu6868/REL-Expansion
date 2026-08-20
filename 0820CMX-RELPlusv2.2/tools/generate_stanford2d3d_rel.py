#!/usr/bin/env python3
"""Generate original three-channel REL from complete S3D ERP depth images."""

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from third_party.rel_original import getImage, getREL


def find_depth_files(depth_root, areas):
    root = depth_root
    files = sorted(root.glob("area_*/*_depth.png"))
    if not files:
        files = sorted(root.glob("area_*/pano/depth/*_depth.png"))
    if not files:
        files = sorted(root.glob("*_depth.png"))

    if areas:
        wanted = set(areas)
        files = [path for path in files if any(part in wanted for part in path.parts)]
    return root, files


def output_path(depth_path, discovery_root, output_root, output_suffix):
    relative = depth_path.relative_to(discovery_root)
    parts = list(relative.parts)
    if "depth" in parts:
        parts[parts.index("depth")] = "rel"
    filename = parts[-1]
    if filename.endswith("_depth.png"):
        filename = filename[: -len("_depth.png")] + output_suffix
    parts[-1] = filename
    return output_root.joinpath(*parts)


def generate_one(task):
    depth_path, rel_path, alpha, lam, overwrite = task
    record = {
        "depth_path": str(depth_path),
        "rel_path": str(rel_path),
        "status": "",
        "error": "",
    }
    try:
        if rel_path.exists() and not overwrite:
            record["status"] = "SKIPPED"
            return record

        depth = getImage(str(depth_path), "Stanford2D3DPano")
        rel = getREL(depth, alpha=alpha, lam=lam)
        rel_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(rel_path), rel):
            raise RuntimeError("OpenCV failed to write REL")
        record.update(
            status="OK",
            shape="x".join(str(value) for value in rel.shape),
            dtype=str(rel.dtype),
            channel_0_min=int(rel[:, :, 0].min()),
            channel_0_max=int(rel[:, :, 0].max()),
            channel_1_min=int(rel[:, :, 1].min()),
            channel_1_max=int(rel[:, :, 1].max()),
            channel_2_min=int(rel[:, :, 2].min()),
            channel_2_max=int(rel[:, :, 2].max()),
        )
    except Exception as exc:
        record["status"] = "ERROR"
        record["error"] = "{}: {}".format(type(exc).__name__, exc)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=45)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--area", action="append")
    parser.add_argument("--output-suffix", default="_rel.png")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    discovery_root, files = find_depth_files(args.depth_root, args.area)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No Stanford2D3D ERP depth files found")

    tasks = [
        (
            path,
            output_path(path, discovery_root, args.output_root, args.output_suffix),
            args.alpha,
            args.lam,
            args.overwrite,
        )
        for path in files
    ]
    if args.workers == 1:
        records = [generate_one(task) for task in tasks]
    else:
        with mp.Pool(processes=args.workers) as pool:
            records = list(pool.imap(generate_one, tasks))

    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "depth_path", "rel_path", "status", "error", "shape", "dtype",
        "channel_0_min", "channel_0_max", "channel_1_min", "channel_1_max",
        "channel_2_min", "channel_2_max",
    ]
    manifest = args.output_root / "generation_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})

    ok = sum(record["status"] in ("OK", "SKIPPED") for record in records)
    errors = len(records) - ok
    print("REL generation: total={} usable={} errors={}".format(len(records), ok, errors))
    print("manifest={}".format(manifest))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
