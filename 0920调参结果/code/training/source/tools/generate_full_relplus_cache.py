#!/usr/bin/env python3
"""Generate a resumable REL+ cache from a frozen manifest."""

import argparse
import csv
import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rel_plus.generator import generate_rel_plus_v2_1
from rel_plus.profiles import STANFORD_S2D_PROFILE
from rel_plus.stanford_s2d import load_canonical_frame


REPRESENTATION_PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
PILOT_LIMIT = 36


def _initialize_worker():
    """Keep one OpenCV thread inside each Python cache worker."""
    cv2.setNumThreads(1)


def read_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_generation_scope(
    rows, *, limit, dry_run, authorized, pilot_limit=PILOT_LIMIT
):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected = list(rows if limit is None else rows[:limit])
    if not dry_run and not authorized:
        if limit is None or limit > pilot_limit:
            raise RuntimeError(
                "full_cache_authorized must be True unless an explicit "
                "pilot-sized --limit is used"
            )
    return selected


def validate_cached_pair(rel_path, mask_path, expected_shape=(480, 480)):
    failures = []
    rel_path = Path(rel_path)
    mask_path = Path(mask_path)
    rel = cv2.imread(str(rel_path), cv2.IMREAD_UNCHANGED) if rel_path.is_file() else None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.is_file() else None
    if rel is None:
        failures.append("rel_plus_decode")
    else:
        if rel.shape != tuple(expected_shape) + (3,):
            failures.append("rel_plus_shape_or_channels")
        if rel.dtype != np.uint8:
            failures.append("rel_plus_dtype")
    if mask is None:
        failures.append("valid_mask_decode")
    else:
        if mask.shape != tuple(expected_shape):
            failures.append("valid_mask_shape")
        if mask.dtype != np.uint8:
            failures.append("valid_mask_dtype")
        elif not set(np.unique(mask).tolist()).issubset({0, 255}):
            failures.append("valid_mask_binary")
    return failures


def is_full_cache_generated(
    *,
    dry_run,
    failures,
    selected_rows,
    manifest_rows,
    generated_or_verified_rows,
    pilot_limit=PILOT_LIMIT
):
    """Return True only for a complete, successful non-pilot cache."""
    return bool(
        not dry_run
        and len(failures) == 0
        and len(selected_rows) == len(manifest_rows)
        and len(generated_or_verified_rows) == len(manifest_rows)
        and len(manifest_rows) > pilot_limit
    )


def _atomic_png(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp.png".format(path.stem, os.getpid()))
    try:
        if not cv2.imwrite(str(temporary), value):
            raise OSError("failed to encode {}".format(path))
        decoded = cv2.imread(str(temporary), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.shape != value.shape or decoded.dtype != value.dtype:
            raise OSError("temporary PNG verification failed for {}".format(path))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _output_paths(output_root, sample_id):
    relative = Path(str(sample_id) + ".png")
    root = Path(output_root)
    return root / "RELPlus" / relative, root / "ValidMask" / relative


def generate_one(row, output_root, *, resume):
    sample_id = row["sample_id"]
    if row.get("protocol_id") != REPRESENTATION_PROTOCOL_ID:
        raise ValueError("manifest representation protocol mismatch")
    rel_path, mask_path = _output_paths(output_root, sample_id)
    if resume and not validate_cached_pair(rel_path, mask_path):
        result = dict(row)
        result.update(
            {
                "rel_plus_path": str(rel_path),
                "valid_mask_path": str(mask_path),
                "cache_status": "RESUMED_VALID",
            }
        )
        return result
    raw_depth, camera, _ = load_canonical_frame(
        row["depth_path"],
        row["camera_metadata_path"],
        dataset_profile=STANFORD_S2D_PROFILE,
    )
    rel_plus, debug = generate_rel_plus_v2_1(
        raw_depth, camera, return_debug=True
    )
    valid = np.asarray(debug["depth_valid"], dtype=bool)
    if rel_plus.shape != raw_depth.shape + (3,) or rel_plus.dtype != np.uint8:
        raise ValueError("frozen generator returned an invalid REL+ array")
    _atomic_png(rel_path, rel_plus)
    _atomic_png(mask_path, valid.astype(np.uint8) * 255)
    failures = validate_cached_pair(rel_path, mask_path, raw_depth.shape)
    if failures:
        raise RuntimeError("written cache failed validation: {}".format(failures))
    result = dict(row)
    result.update(
        {
            "rel_plus_path": str(rel_path),
            "valid_mask_path": str(mask_path),
            "cache_status": "GENERATED",
        }
    )
    return result


def _generate_task(task):
    row, output_root, resume = task
    try:
        return generate_one(row, output_root, resume=resume), None
    except Exception as error:
        return None, {
            "sample_id": row.get("sample_id", ""),
            "status": "FAIL",
            "reason": type(error).__name__,
            "detail": str(error),
        }


def generate_cache(rows, output_root, *, resume=False, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    generated = []
    failures = []

    tasks = ((row, str(output_root), resume) for row in rows)
    if workers == 1:
        _initialize_worker()
        outcomes = map(_generate_task, tasks)
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_worker
        )
        outcomes = executor.map(_generate_task, tasks)
    try:
        for result, failure in outcomes:
            if failure is None:
                generated.append(result)
            else:
                failures.append(failure)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return generated, failures


def _write_split_lists(rows, output_root):
    root = Path(output_root)
    for split, filename in (("train", "train.txt"), ("test", "test.txt")):
        values = [row["sample_id"] for row in rows if row.get("split") == split]
        (root / filename).write_text(
            "".join(value + "\n" for value in values), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config-module",
        default=(
            "configs.stanford2d3d_s2d."
            "cmx_mit_b2_rel_plus_v2_3_formal"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--authorize-full-cache",
        action="store_true",
        help="explicit one-command authorization; repository config stays false",
    )
    args = parser.parse_args()
    config = importlib.import_module(args.config_module).config
    rows = read_manifest(args.manifest)
    selected = resolve_generation_scope(
        rows,
        limit=args.limit,
        dry_run=args.dry_run,
        authorized=bool(
            config.full_cache_authorized or args.authorize_full_cache
        ),
    )
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("manifest contains duplicate sample IDs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        generated = [
            dict(row, cache_status="DRY_RUN_NOT_WRITTEN") for row in selected
        ]
        failures = []
    else:
        generated, failures = generate_cache(
            selected,
            args.output_root,
            resume=args.resume,
            workers=args.workers,
        )
        if not failures and len(generated) == len(selected):
            _write_split_lists(selected, args.output_root)
    fields = list(dict.fromkeys(list(selected[0].keys()) + [
        "rel_plus_path", "valid_mask_path", "cache_status",
    ]))
    write_csv(
        args.output_root / "cache_manifest_resolved.csv", generated, fields
    )
    write_csv(
        args.output_root / "cache_generation_failures.csv",
        failures,
        ["sample_id", "status", "reason", "detail"],
    )
    full_cache_generated = is_full_cache_generated(
        dry_run=args.dry_run,
        failures=failures,
        selected_rows=selected,
        manifest_rows=rows,
        generated_or_verified_rows=generated,
    )
    cache_root = args.output_root.resolve()
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "integration_protocol_id": config.integration_protocol_id,
        "representation_protocol_id": config.representation_protocol_id,
        "manifest_count": len(rows),
        "selected_count": len(selected),
        "generated_or_resumed_count": len(generated),
        "failure_count": len(failures),
        "dry_run": args.dry_run,
        "resume": args.resume,
        "workers": args.workers,
        "full_cache_authorized": bool(
            config.full_cache_authorized or args.authorize_full_cache
        ),
        "full_cache_generated": full_cache_generated,
        "cache_root": str(cache_root),
        "rel_plus_root": str(cache_root / "RELPlus"),
        "valid_mask_root": str(cache_root / "ValidMask"),
        "manifest_path": str(args.manifest.resolve()),
        "resolved_manifest_path": str(
            (cache_root / "cache_manifest_resolved.csv").resolve()
        ),
        "train_source": str((cache_root / "train.txt").resolve()),
        "eval_source": str((cache_root / "test.txt").resolve()),
        "train_count": sum(row.get("split") == "train" for row in rows),
        "test_count": sum(row.get("split") == "test" for row in rows),
        "file_hash_written": False,
    }
    (args.output_root / "cache_generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
