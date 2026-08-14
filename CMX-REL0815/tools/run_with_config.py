#!/usr/bin/env python3
import argparse
import importlib
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Dotted config module")
    parser.add_argument("entrypoint", choices=("train.py", "eval.py"))
    args, passthrough = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    selected_config = importlib.import_module(args.config)
    if not hasattr(selected_config, "config"):
        raise AttributeError("{} does not expose 'config'".format(args.config))

    sys.modules["config"] = selected_config
    entrypoint = repo_root / args.entrypoint
    sys.argv = [str(entrypoint)] + passthrough
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
