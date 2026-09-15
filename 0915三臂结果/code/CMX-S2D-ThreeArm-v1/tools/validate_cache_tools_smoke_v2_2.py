#!/usr/bin/env python3
"""Inject cache faults and verify the V2.2 auditor detects each one."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_full_relplus_cache import audit_cache_rows
from tools.generate_full_relplus_cache import read_manifest, validate_cached_pair


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), value):
        raise OSError("failed to write fault fixture {}".format(path))


def _reasons(row, root):
    _, failures = audit_cache_rows([row], root)
    return sorted({failure["reason"] for failure in failures})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--audit-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    row = read_manifest(args.manifest)[0]
    sample_id = row["sample_id"]
    source_rel = args.cache_root / "RELPlus" / (sample_id + ".png")
    source_mask = args.cache_root / "ValidMask" / (sample_id + ".png")
    outcomes = {}
    with tempfile.TemporaryDirectory(prefix="cmx_relplus_v22_cache_fault_") as temporary:
        root = Path(temporary)
        rel = root / "RELPlus" / (sample_id + ".png")
        mask = root / "ValidMask" / (sample_id + ".png")

        def restore():
            rel.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(source_rel), str(rel))
            shutil.copyfile(str(source_mask), str(mask))

        restore()
        outcomes["baseline"] = _reasons(row, root)
        outcomes["resume_validation"] = validate_cached_pair(rel, mask)

        rel.unlink()
        outcomes["missing_file"] = _reasons(row, root)
        restore()
        rel.write_bytes(b"corrupt png")
        outcomes["corrupt_png"] = _reasons(row, root)
        restore()
        _write(rel, np.zeros((12, 12, 3), dtype=np.uint8))
        outcomes["wrong_shape"] = _reasons(row, root)
        restore()
        _write(rel, np.zeros((480, 480, 3), dtype=np.uint16))
        outcomes["wrong_dtype"] = _reasons(row, root)
        restore()
        _write(rel, np.zeros((480, 480, 4), dtype=np.uint8))
        outcomes["wrong_channels"] = _reasons(row, root)
        restore()
        _write(root / "RELPlus" / "extra.png", np.zeros((480, 480, 3), np.uint8))
        outcomes["extra_file"] = _reasons(row, root)

    expected = {
        "baseline": set(),
        "resume_validation": set(),
        "missing_file": {"rel_plus_missing"},
        "corrupt_png": {"rel_plus_decode"},
        "wrong_shape": {"rel_plus_shape_or_channels"},
        "wrong_dtype": {"rel_plus_dtype"},
        "wrong_channels": {"rel_plus_shape_or_channels"},
        "extra_file": {"extra_rel_plus"},
    }
    checks = {
        name: expected_reasons.issubset(set(outcomes[name]))
        for name, expected_reasons in expected.items()
    }
    audit = json.loads(args.audit_summary.read_text(encoding="utf-8"))
    checks["sample_regeneration"] = (
        audit.get("status") == "PASS"
        and audit.get("regeneration_count") == 36
        and audit.get("regeneration_failure_count") == 0
    )
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sample_id": sample_id,
        "checks": checks,
        "detected_reasons": outcomes,
        "regeneration_count": audit.get("regeneration_count"),
        "regeneration_failure_count": audit.get("regeneration_failure_count"),
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
