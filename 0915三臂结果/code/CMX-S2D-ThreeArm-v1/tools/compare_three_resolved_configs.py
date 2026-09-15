#!/usr/bin/env python3
"""Verify that the three S3D Fold 1 configs differ only in X-modality plumbing."""

import argparse
import json
from pathlib import Path


ALLOWED = {
    "abs_dir", "code_root", "experiment_name", "log_dir", "run_dir",
    "tb_dir", "log_dir_link", "checkpoint_dir", "prediction_dir",
    "visualization_dir", "log_file", "link_log_file", "val_log_file",
    "link_val_log_file", "root_dir", "x_name", "x_root_folder", "x_format",
    "x_is_single_channel", "x_mode", "x_channel_semantics", "data_setting",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgbd", required=True, type=Path)
    parser.add_argument("--hha", required=True, type=Path)
    parser.add_argument("--rel", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    configs = {
        "CMX-RGBD": json.loads(args.rgbd.read_text(encoding="utf-8")),
        "CMX-HHA": json.loads(args.hha.read_text(encoding="utf-8")),
        "CMX-REL": json.loads(args.rel.read_text(encoding="utf-8")),
    }
    keys = sorted(set().union(*(set(config) for config in configs.values())))
    differing = []
    unexpected = []
    for key in keys:
        values = {name: config.get(key, "<MISSING>") for name, config in configs.items()}
        encoded = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values.values()}
        if len(encoded) > 1:
            differing.append((key, values))
            if key not in ALLOWED:
                unexpected.append(key)

    lines = [
        "# Three resolved config comparison",
        "",
        "Status: **{}**".format("PASS" if not unexpected else "FAIL"),
        "",
        "Only X-modality plumbing and per-run output/code paths may differ.",
        "",
        "| Field | CMX-RGBD | CMX-HHA | CMX-REL | Allowed |",
        "|---|---|---|---|---|",
    ]
    for key, values in differing:
        rendered = [json.dumps(values[name], ensure_ascii=False) for name in configs]
        rendered = [value.replace("|", "\\|") for value in rendered]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                key, rendered[0], rendered[1], rendered[2], "yes" if key in ALLOWED else "no"
            )
        )
    if unexpected:
        lines.extend(["", "Unexpected differences: " + ", ".join(unexpected)])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("status={}".format("PASS" if not unexpected else "FAIL"))
    print("report={}".format(args.report))
    raise SystemExit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
