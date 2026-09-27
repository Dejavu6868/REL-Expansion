#!/usr/bin/env python3
"""Compare complete old/new sampler sequences element by element."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    expected_epochs = [1, 2, 3, 10, 11]
    mismatches = []
    compared = 0
    for epoch in expected_epochs:
        for rank in range(8):
            left = old["sequences"][str(epoch)][str(rank)]
            right = new["sequences"][str(epoch)][str(rank)]
            if len(left) != 6613 or len(right) != 6613:
                mismatches.append(
                    {"epoch": epoch, "rank": rank, "reason": "length"}
                )
                continue
            for position, (old_id, new_id) in enumerate(zip(left, right)):
                compared += 1
                if old_id != new_id and len(mismatches) < 20:
                    mismatches.append(
                        {
                            "epoch": epoch,
                            "rank": rank,
                            "position": position,
                            "old": old_id,
                            "new": new_id,
                        }
                    )
    passed = not mismatches and compared == 5 * 8 * 6613
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol": "OLD_VS_NEW_PURE_SAMPLER_EPOCH_SEQUENCE_V1",
        "old_code_root": old["code_root"],
        "new_code_root": new["code_root"],
        "epochs": expected_epochs,
        "world_size": 8,
        "sample_ids_per_rank_epoch": 6613,
        "element_comparison_count": compared,
        "all_ordered_sample_ids_exact": not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches,
        "gpu_used": False,
        "backpropagation_executed": False,
        "optimizer_step_executed": False,
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
