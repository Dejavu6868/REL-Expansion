#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def command_output(command):
    return subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Dotted config module")
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    module = importlib.import_module(args.config)
    config = module.config
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    metadata = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "working_directory": os.getcwd(),
        "hostname": platform.node(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short", "--branch"]),
        "config_module": args.config,
        "config_path": str(Path(module.__file__).resolve()),
        "resolved_config": json_value(dict(config)),
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_names_visible": gpu_names,
            "nvidia_smi": command_output(["nvidia-smi"]),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "OPENCV_FOR_THREADS_NUM",
                )
            },
        },
    }
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
