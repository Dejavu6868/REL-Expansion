import csv
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def load_common_initial_model(model, path):
    if not path:
        raise ValueError("common initial model path is required")
    payload = torch.load(path, map_location=torch.device("cpu"))
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    incompatible = model.load_state_dict(state, strict=True)
    return {
        "path": os.path.abspath(path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "tensor_count": len(state),
    }


def append_epoch_metrics(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
