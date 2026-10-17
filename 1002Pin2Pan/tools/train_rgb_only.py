#!/usr/bin/env python3
"""Train the RGB-only baseline with the frozen CMX training entry, on the HHA arm's recipe and data.

Launch it like the frozen training, one process per GPU::

    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29513 \\
        1002Pin2Pan/tools/train_rgb_only.py --config <resolved HHA config> --output <new folder>

``--config`` is a resolved HHA arm config (for example a seed suite's ``configs/hha.json``).
Seed, data, recipe and data checks all come from it unchanged. The runner only sets the
architecture, the arm name and the output paths, then runs the frozen ``source/train.py``
with ``rgb_only.model_class()`` in place of the CMX network. The frozen loop reseeds from
``config.seed`` every epoch, so at the same seed this arm sees the same batches and
augmentations as the HHA arm (checked on CPU with two loader workers against the CMX
network, ``tests/test_rgb_only.py``; not traced on the server). The HHA input is still
loaded and is ignored by the network.

As the suite's ``run_arm.py`` does, the run uses one OpenCV and one torch thread per
process and refuses NaN losses and any world size other than 8. Other arguments go to the frozen Engine, e.g. ``--continue <checkpoint>`` to resume
into a new output folder. ``--smoke-iterations N`` trains one epoch of N iterations and saves
``epoch-1.pth``, a check of memory and the whole path before the long run.
"""

import argparse
import json
import os
import runpy
import sys
import types
from pathlib import Path

import numpy as np

import rgb_only as ro
from eval_pano_transfer import DEFAULT_SOURCE_ROOT

WORLD_SIZE = 8
LOG_FILES = {"log_file": "train.log", "link_log_file": "train_last.log",
             "val_log_file": "val.log", "link_val_log_file": "val_last.log"}


def rgb_only_config(hha_config, output, source_root, smoke_iterations=None):
    """The HHA config with only the architecture, arm name and output paths changed."""
    if hha_config.get("x_mode") != "hha_frozen_cache" or hha_config.get("architecture") != "Original CMX":
        raise ValueError("--config must be a resolved CMX HHA arm config")
    config = json.loads(json.dumps(hha_config))
    config.update(
        architecture=ro.ARCHITECTURE,
        arm_name="RGB-only",
        experiment_name=config["experiment_name"] + "_rgb_only",
        root_dir=str(source_root), abs_dir=str(source_root),
        output_dir=str(output), log_dir=str(output / "logs"), tb_dir=str(output / "tensorboard"),
        checkpoint_dir=str(output / "checkpoints"), log_dir_link=str(output / "latest_logs"),
        ddp_smoke_report=None,
    )
    config.update({field: str(output / "logs" / name) for field, name in LOG_FILES.items()})
    if smoke_iterations is not None:
        config.update(nepochs=1, niters_per_epoch=smoke_iterations)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="resolved HHA arm config JSON")
    parser.add_argument("--output", required=True, type=Path, help="new, empty output folder")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--smoke-iterations", type=int, default=None)
    args, forwarded = parser.parse_known_args(argv)
    if args.smoke_iterations is not None and args.smoke_iterations <= 0:
        parser.error("--smoke-iterations must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))

    source_root = args.source_root.resolve()
    resolved = rgb_only_config(
        json.loads(args.config.read_text(encoding="utf-8")), args.output.resolve(), source_root,
        args.smoke_iterations,
    )
    from easydict import EasyDict

    config = EasyDict(resolved)
    config.norm_mean = np.asarray(config.norm_mean, dtype=np.float64)
    config.norm_std = np.asarray(config.norm_std, dtype=np.float64)
    module = types.ModuleType("_rgb_only_config")
    module.config = module.cfg = config
    sys.modules[module.__name__] = module
    sys.modules["config"] = module  # what the frozen train.py and loader import
    sys.dont_write_bytecode = True  # keep the frozen archive's file set unchanged
    sys.path.insert(0, str(source_root))

    import cv2
    import torch
    import utils.training_runtime as runtime_module

    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    original_build = runtime_module.build_training_runtime
    original_sanitize = runtime_module.sanitize_author_loss_map

    def finite_loss_map(loss_map):
        result, count = original_sanitize(loss_map)
        if count:
            raise FloatingPointError("NaN replacement encountered; refusing to continue this run")
        return result, count

    def build(*positional, **keyword):
        runtime = original_build(*positional, model_class=ro.model_class(), **keyword)
        expected_loader = -(-int(config.num_train_imgs) // int(config.batch_size))
        if runtime.engine.world_size != WORLD_SIZE or len(runtime.train_loader) != expected_loader:
            raise RuntimeError("world size {} or loader length {} differs from the recipe ({}, {})".format(
                runtime.engine.world_size, len(runtime.train_loader), WORLD_SIZE, expected_loader))
        if runtime.engine.local_rank == 0:
            base = runtime.model.module if hasattr(runtime.model, "module") else runtime.model
            (args.output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
            (args.output / "rgb_only.json").write_text(json.dumps({
                "architecture": ro.ARCHITECTURE,
                "hha_config": str(args.config.resolve()),
                "seed": config.seed,
                "source_root": str(source_root),
                "parameters": sum(parameter.numel() for parameter in base.parameters()),
                "removed_modules": list(ro.REMOVED_MODULES),
                "smoke_iterations": args.smoke_iterations,
                "engine_arguments": forwarded,
            }, indent=2) + "\n", encoding="utf-8")
        print("Architecture: {} (the frozen train.log still prints Original CMX)".format(ro.ARCHITECTURE), flush=True)
        return runtime

    runtime_module.build_training_runtime = build
    runtime_module.sanitize_author_loss_map = finite_loss_map
    if "MASTER_PORT" in os.environ and not any(item.split("=")[0] in ("-p", "--port") for item in forwarded):
        forwarded += ["--port", os.environ["MASTER_PORT"]]  # the Engine otherwise resets it to 16005
    entry = source_root / "train.py"
    sys.argv = [str(entry)] + forwarded
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as result:
        if result.code not in (0, None):
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
