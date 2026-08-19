#!/usr/bin/env python3
"""Compare frozen and integrated REL+ v2.1 generator bytes on a fixed input."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r"""
import json
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from rel_plus.camera import CameraGeometry
from rel_plus.generator import generate_rel_plus_v2_1
raw = np.full((12, 12), 1024, dtype=np.uint16)
raw[0, 0] = 0
raw[3, 4] = 65535
camera = CameraGeometry.from_json_k(
    np.array([[30.0, 0.0, 6.0], [0.0, 30.0, 6.0], [0.0, 0.0, 1.0]]),
    raw.shape,
    np.eye(3),
    np.zeros(3),
)
value = generate_rel_plus_v2_1(raw, camera)
print(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype), "value": value.tolist()}))
"""


def _generate(root):
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(root)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)
    return np.asarray(payload.pop("value"), dtype=np.uint8), payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    before, before_meta = _generate(args.frozen_root)
    after, after_meta = _generate(ROOT)
    difference = np.abs(after.astype(np.int16) - before.astype(np.int16))
    changed_channels = int(np.count_nonzero(difference))
    changed_pixels = int(np.count_nonzero(np.any(difference != 0, axis=2)))
    maximum = int(difference.max()) if difference.size else 0
    status = (
        "PASS"
        if before_meta == after_meta
        and changed_channels == 0
        and changed_pixels == 0
        and maximum == 0
        else "FAIL"
    )
    report = {
        "status": status,
        "frozen_root": str(args.frozen_root),
        "integrated_root": str(ROOT),
        "shape": after_meta["shape"],
        "dtype": after_meta["dtype"],
        "changed_pixels": changed_pixels,
        "changed_channels": changed_channels,
        "max_difference": maximum,
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
