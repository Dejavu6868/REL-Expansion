#!/usr/bin/env python3
"""Run an unchanged CMX entry point with an explicit config module."""

import argparse
import importlib
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-module", required=True)
    parser.add_argument("entry", choices=("train.py", "eval.py"))
    args, forwarded = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.modules["config"] = importlib.import_module(args.config_module)
    sys.argv = [args.entry] + forwarded
    runpy.run_path(str(repo_root / args.entry), run_name="__main__")


if __name__ == "__main__":
    main()
