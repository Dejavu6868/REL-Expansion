#!/usr/bin/env python3
"""Export the runtime CMX configuration as JSON and plain text."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-module", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    module = importlib.import_module(args.config_module)
    resolved = {
        str(key): serializable(value)
        for key, value in dict(module.config).items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "resolved_config.json"
    text_path = args.output_dir / "resolved_config.txt"
    json_path.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(
        "".join("{} = {}\n".format(key, resolved[key]) for key in sorted(resolved)),
        encoding="utf-8",
    )
    print(json_path)
    print(text_path)


if __name__ == "__main__":
    main()
