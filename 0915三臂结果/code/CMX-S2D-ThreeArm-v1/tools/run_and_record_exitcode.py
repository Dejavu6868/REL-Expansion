#!/usr/bin/env python3
"""Run one command and atomically record its integer exit code."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exitcode", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    completed = subprocess.run(command)
    args.exitcode.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.exitcode.with_name(
        ".{}.{}.tmp".format(args.exitcode.name, os.getpid())
    )
    temporary.write_text("{}\n".format(completed.returncode), encoding="utf-8")
    os.replace(str(temporary), str(args.exitcode))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
