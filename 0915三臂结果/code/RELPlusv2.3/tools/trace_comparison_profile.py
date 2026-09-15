#!/usr/bin/env python3
"""Retired V2.1 constructed-trace entry."""

import sys


def main():
    sys.stderr.write(
        "Constructed trace copying is not V2.2 evidence. Use "
        "tools/trace_three_arm_dataloaders_v2_2.py with three real DataLoaders.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
