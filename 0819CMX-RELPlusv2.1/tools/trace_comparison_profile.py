#!/usr/bin/env python3
"""Write the first 50 no-training transform traces for all three future arms."""

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    selected = importlib.import_module(
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_1_formal"
    )
    config = selected.config
    from dataloader.profiles import trace_comparison_profile

    with open(config.train_source, encoding="utf-8") as handle:
        sample_ids = [line.strip() for line in handle if line.strip()][:50]
    traces = trace_comparison_profile(
        sample_ids,
        input_shape=(480, 480),
        output_shape=(config.image_height, config.image_width),
        scales=config.train_scale_array,
        base_seed=config.seed,
        epoch=args.epoch,
        rank=args.rank,
    )
    identical = traces["rgbd"] == traces["hha"] == traces["rel_plus_v2_1"]
    report = {
        "status": "PASS" if identical and len(sample_ids) == 50 else "FAIL",
        "profile": config.augmentation_profile,
        "base_seed": config.seed,
        "epoch": args.epoch,
        "rank": args.rank,
        "sample_count": len(sample_ids),
        "arm_traces_identical": identical,
        "traces": traces,
        "training_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "traces"}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
