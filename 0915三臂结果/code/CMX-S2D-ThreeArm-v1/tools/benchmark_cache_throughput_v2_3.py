#!/usr/bin/env python3
"""Measure formal REL+ cache throughput on 500--1000 manifest rows."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_full_relplus_cache import generate_cache, read_manifest


FULL_SAMPLE_COUNT = 70496


def _cpu_seconds(usage):
    return float(usage.ru_utime + usage.ru_stime)


def _peak_rss_mib(self_usage, child_usage):
    value = max(float(self_usage.ru_maxrss), float(child_usage.ru_maxrss))
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--authorize-throughput-benchmark", action="store_true")
    args = parser.parse_args()

    if not args.authorize_throughput_benchmark:
        raise RuntimeError("--authorize-throughput-benchmark is required")
    if not 500 <= args.limit <= 1000:
        raise ValueError("throughput benchmark limit must be between 500 and 1000")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    rows = read_manifest(args.manifest)
    if len(rows) < args.limit:
        raise ValueError("manifest contains fewer rows than the requested benchmark")
    selected = rows[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        raise RuntimeError("throughput output root must be empty")

    self_before = resource.getrusage(resource.RUSAGE_SELF)
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    generated, failures = generate_cache(
        selected, args.output_root, resume=False, workers=args.workers
    )
    elapsed = time.monotonic() - started
    self_after = resource.getrusage(resource.RUSAGE_SELF)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = (
        _cpu_seconds(self_after)
        - _cpu_seconds(self_before)
        + _cpu_seconds(child_after)
        - _cpu_seconds(child_before)
    )
    pngs = list(args.output_root.rglob("*.png"))
    encoded_bytes = sum(path.stat().st_size for path in pngs)
    completed = len(generated)
    average_png_bytes = encoded_bytes / len(pngs) if pngs else 0.0
    average_pair_bytes = encoded_bytes / completed if completed else 0.0
    status = (
        "PASS"
        if not failures and completed == len(selected) and len(pngs) == 2 * completed
        else "FAIL"
    )
    report = {
        "status": status,
        "scope": "infrastructure_throughput_estimate_not_scientific_experiment",
        "manifest_path": str(args.manifest.resolve()),
        "output_root": str(args.output_root.resolve()),
        "selected_count": len(selected),
        "generated_count": completed,
        "failure_count": len(failures),
        "workers": args.workers,
        "opencv_threads_per_worker": 1,
        "elapsed_seconds": elapsed,
        "images_per_second": completed / elapsed if elapsed else 0.0,
        "aggregate_cpu_seconds": cpu_seconds,
        "aggregate_cpu_percent_of_one_core": (
            cpu_seconds / elapsed * 100.0 if elapsed else 0.0
        ),
        "peak_rss_mib": _peak_rss_mib(self_after, child_after),
        "png_file_count": len(pngs),
        "encoded_bytes_written": encoded_bytes,
        "encoded_mib_per_second": (
            encoded_bytes / (1024.0 ** 2) / elapsed if elapsed else 0.0
        ),
        "average_png_file_bytes": average_png_bytes,
        "average_relplus_mask_pair_bytes": average_pair_bytes,
        "estimated_full_cache_bytes": average_pair_bytes * FULL_SAMPLE_COUNT,
        "estimated_full_png_inode_count": 2 * FULL_SAMPLE_COUNT,
        "failure_preview": failures[:100],
        "file_hash_written": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
