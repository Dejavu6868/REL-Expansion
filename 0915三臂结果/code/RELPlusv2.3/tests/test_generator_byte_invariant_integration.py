import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(
    os.environ.get("RELPLUS_V21_FROZEN_ROOT", "/home/zhuzhaoziao/RELPlus/RELPlusv2.1")
)


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
array = generate_rel_plus_v2_1(raw, camera)
print(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype), "values": array.tolist()}))
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
    return np.asarray(payload["values"], dtype=np.uint8), payload


@pytest.mark.live_source
def test_integration_keeps_frozen_generator_bytes_exactly_identical():
    if not FROZEN_ROOT.is_dir():
        pytest.skip("frozen REL+ v2.1 source is unavailable")
    before, before_meta = _generate(FROZEN_ROOT)
    after, after_meta = _generate(ROOT)
    difference = np.abs(after.astype(np.int16) - before.astype(np.int16))
    assert before_meta["shape"] == after_meta["shape"]
    assert before_meta["dtype"] == after_meta["dtype"] == "uint8"
    assert int(np.count_nonzero(np.any(difference != 0, axis=2))) == 0
    assert int(np.count_nonzero(difference)) == 0
    assert (int(difference.max()) if difference.size else 0) == 0
