#!/usr/bin/env python3
"""Fail closed unless the geometry-aware online REL+ input is fully wired."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path, PurePosixPath

import cv2


EXPECTED_CHANNELS = ["ReD", "EGVIA", "LOA"]


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    os.environ["CMX_RUN_DIR"] = str(run_dir)
    sys.path.insert(0, str(repo))
    cfg = importlib.import_module("configs.cmx_relplus_2d").config
    errors = []

    checks = (
        (getattr(cfg, "x_online_relplus", False) is True, "x_online_relplus must be true"),
        (list(getattr(cfg, "relplus_channel_order", [])) == EXPECTED_CHANNELS, "channel order mismatch"),
        (getattr(cfg, "relplus_pixel_origin", None) == 0.5, "pixel origin must be 0.5"),
        ("uint16/512" in getattr(cfg, "relplus_depth_definition", ""), "depth scale must be uint16/512"),
        (getattr(cfg, "train_horizontal_flip", None) is False, "horizontal flip must be disabled"),
        ("no encoded-channel resize" in getattr(cfg, "relplus_cache_generation", ""),
         "encoded REL+ resize must be forbidden"),
    )
    for condition, message in checks:
        if not condition:
            errors.append(message)

    roots = {
        "rgb": Path(cfg.rgb_root_folder), "label": Path(cfg.gt_root_folder),
        "depth": Path(cfg.depth_root_folder), "pose": Path(cfg.pose_root_folder),
    }
    for name, root in roots.items():
        if not root.is_dir():
            errors.append("{} root missing: {}".format(name, root))

    missing = []
    counts = {}
    seen = set()
    for split_name, split_path in (("train", Path(cfg.train_source)), ("test", Path(cfg.eval_source))):
        identifiers = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        counts[split_name] = len(identifiers)
        for item in identifiers:
            pure = PurePosixPath(item)
            if pure.is_absolute() or ".." in pure.parts:
                errors.append("unsafe sample id: {}".format(item))
                continue
            if item in seen:
                errors.append("duplicate or overlapping sample id: {}".format(item))
            seen.add(item)
            expected = (roots["rgb"] / (item + cfg.rgb_format),
                        roots["label"] / (item + cfg.gt_format),
                        roots["depth"] / (item + cfg.depth_format),
                        roots["pose"] / (item + cfg.pose_format))
            for path in expected:
                if not path.is_file() and len(missing) < 200:
                    missing.append(str(path))
    if missing:
        errors.append("required online source files are missing")

    probe_id = next(iter(seen)) if seen else None
    probe_depth = None
    if probe_id:
        probe_depth = cv2.imread(str(roots["depth"] / (probe_id + cfg.depth_format)), cv2.IMREAD_UNCHANGED)
        if probe_depth is None or str(probe_depth.dtype) != "uint16" or probe_depth.ndim != 2:
            errors.append("probe depth is not 2D uint16")

    report = {
        "status": "PASS_ONLINE_RELPLUS_PREFLIGHT" if not errors else "FAIL_ONLINE_RELPLUS_PREFLIGHT",
        "config": "configs.cmx_relplus_2d", "dataset_class": "dataloader.RGBXDataset.RGBXDataset",
        "transform": "dataloader.dataloader.RelPlusTrainPre", "mode": "online_geometry_regeneration",
        "channel_order": EXPECTED_CHANNELS, "pixel_origin": getattr(cfg, "relplus_pixel_origin", None),
        "depth_definition": getattr(cfg, "relplus_depth_definition", None),
        "horizontal_flip": getattr(cfg, "train_horizontal_flip", None),
        "train_count": counts.get("train", 0), "test_count": counts.get("test", 0),
        "missing_file_count": len(missing), "missing_examples": missing,
        "probe_depth_shape": list(probe_depth.shape) if probe_depth is not None else None,
        "probe_depth_dtype": str(probe_depth.dtype) if probe_depth is not None else None,
        "errors": errors,
    }
    atomic_write(run_dir / "data_reports/online_relplus_validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
