#!/usr/bin/env python3
"""Compare frozen and integrated REL+ v2.1 generator bytes on a fixed input."""

import argparse
import base64
import csv
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
REAL_SCRIPT = r"""
import base64
import json
import sys
sys.path.insert(0, sys.argv[1])
from rel_plus.generator import generate_rel_plus_v2_1
from rel_plus.profiles import STANFORD_S2D_PROFILE
from rel_plus.stanford_s2d import load_canonical_frame
rows = json.load(sys.stdin)
outputs = []
for row in rows:
    raw, camera, _ = load_canonical_frame(
        row["depth_path"], row["camera_metadata_path"],
        dataset_profile=STANFORD_S2D_PROFILE,
    )
    value = generate_rel_plus_v2_1(raw, camera)
    outputs.append({
        "sample_id": row["sample_id"],
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": base64.b64encode(value.tobytes()).decode("ascii"),
    })
print(json.dumps(outputs))
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


def _generate_real(root, rows):
    completed = subprocess.run(
        [sys.executable, "-c", REAL_SCRIPT, str(root)],
        input=json.dumps(rows),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    values = []
    for payload in json.loads(completed.stdout):
        shape = tuple(payload["shape"])
        value = np.frombuffer(base64.b64decode(payload["bytes"]), dtype=np.uint8)
        values.append((payload["sample_id"], value.reshape(shape), payload["dtype"]))
    return values


def _compare(before, after):
    difference = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return {
        "changed_channels": int(np.count_nonzero(difference)),
        "changed_pixels": int(np.count_nonzero(np.any(difference != 0, axis=2))),
        "max_difference": int(difference.max()) if difference.size else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--real-limit", type=int, default=12)
    args = parser.parse_args()
    before, before_meta = _generate(args.frozen_root)
    after, after_meta = _generate(ROOT)
    synthetic = _compare(before, after)
    real_rows = []
    real_results = []
    if args.manifest is not None:
        with args.manifest.open("r", encoding="utf-8", newline="") as handle:
            real_rows = list(csv.DictReader(handle))[: args.real_limit]
        if len(real_rows) < 12:
            raise ValueError("real byte regression requires at least 12 samples")
        frozen_values = _generate_real(args.frozen_root, real_rows)
        integrated_values = _generate_real(ROOT, real_rows)
        for frozen, integrated in zip(frozen_values, integrated_values):
            if frozen[0] != integrated[0] or frozen[2] != integrated[2]:
                raise RuntimeError("real regression metadata mismatch")
            result = _compare(frozen[1], integrated[1])
            result.update(
                {
                    "sample_id": frozen[0],
                    "shape": list(integrated[1].shape),
                    "dtype": integrated[2],
                }
            )
            real_results.append(result)
    changed_channels = synthetic["changed_channels"] + sum(
        row["changed_channels"] for row in real_results
    )
    changed_pixels = synthetic["changed_pixels"] + sum(
        row["changed_pixels"] for row in real_results
    )
    maximum = max(
        [synthetic["max_difference"]]
        + [row["max_difference"] for row in real_results]
    )
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
        "synthetic": synthetic,
        "real_sample_count": len(real_results),
        "real_samples": real_results,
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
