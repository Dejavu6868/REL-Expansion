#!/usr/bin/env python3
"""Train a single-channel REL+ arm with the frozen CMX training entry, on a vanilla REL+ arm's recipe and data.

Launch it like the frozen training, one process per GPU::

    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29515 \\
        1002Pin2Pan/tools/train_single_channel.py --config <resolved REL+ config> --channel LOA --output <new folder>

``--config`` is the resolved config of the vanilla REL+ run to pair with (for v3.1new, the one its seed-12345
checkpoint was trained with). Seed, data, recipe, network and data checks all come from it unchanged. The runner
only sets ``single_channel``, the arm name and the output paths, then runs the frozen ``source/train.py`` with the
loader replicating that channel of every REL+ PNG into all three X channels (``single_channel.py``). It refuses a
config whose cache marker (``relplus_variant.json``) is not ``--relplus-variant`` (default v3_1new), so a copied
config can't silently train v2.1. The frozen loop reseeds from ``config.seed`` every epoch, so at the same seed this
arm sees the same images, crops and scales as the vanilla arm; only the X values differ.

As ``train_gated_relplus.py`` does, the run uses one OpenCV and one torch thread per process and refuses NaN losses
and any world size other than 8. Other arguments go to the frozen Engine, e.g. ``--continue <checkpoint>`` to
resume into a new output folder. ``--smoke-iterations N`` trains one epoch of N iterations and saves ``epoch-1.pth``.
"""

import argparse
import json
import os
import runpy
import sys
import types
from pathlib import Path

import numpy as np

import relplus_variant as rv
import single_channel as sc
from eval_pano_transfer import DEFAULT_SOURCE_ROOT
from train_gated_relplus import cache_variant
from train_rgb_only import LOG_FILES, WORLD_SIZE


def single_channel_config(relplus_config, channel, output, source_root, smoke_iterations=None):
    """The REL+ config with only ``single_channel``, the arm name and the output paths changed."""
    if relplus_config.get("x_mode") != "rel_plus_v2_1" or relplus_config.get("architecture") != "Original CMX":
        raise ValueError("--config must be a resolved CMX REL+ arm config")
    if relplus_config.get(sc.CONFIG_KEY) is not None:
        raise ValueError("--config is already a single-channel config")
    config = json.loads(json.dumps(relplus_config))
    config.update({sc.CONFIG_KEY: channel})
    config.update(
        arm_name=config["arm_name"] + " " + channel + " only",
        experiment_name=config["experiment_name"] + "_single_" + channel,
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
    parser.add_argument("--config", required=True, type=Path, help="resolved REL+ arm config JSON")
    parser.add_argument("--channel", required=True, choices=sc.CHANNELS)
    parser.add_argument("--output", required=True, type=Path, help="new, empty output folder")
    parser.add_argument("--relplus-variant", default="v3_1new", choices=sorted(rv.VARIANT_ALPHA))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--smoke-iterations", type=int, default=None)
    args, forwarded = parser.parse_known_args(argv)
    if args.smoke_iterations is not None and args.smoke_iterations <= 0:
        parser.error("--smoke-iterations must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))

    source_root = args.source_root.resolve()
    relplus_config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = single_channel_config(
        relplus_config, args.channel, args.output.resolve(), source_root, args.smoke_iterations
    )
    variant = cache_variant(resolved)
    if variant != args.relplus_variant:
        raise ValueError("--relplus-variant {} but the config's cache {} is {}".format(
            args.relplus_variant, Path(resolved["x_root_folder"]).parent, variant))
    from easydict import EasyDict

    config = EasyDict(resolved)
    config.norm_mean = np.asarray(config.norm_mean, dtype=np.float64)
    config.norm_std = np.asarray(config.norm_std, dtype=np.float64)
    module = types.ModuleType("_single_channel_config")
    module.config = module.cfg = config
    sys.modules[module.__name__] = module
    sys.modules["config"] = module  # what the frozen train.py and loader import
    sys.dont_write_bytecode = True  # keep the frozen archive's file set unchanged
    sys.path.insert(0, str(source_root))

    import cv2
    import torch
    import utils.training_runtime as runtime_module
    from dataloader.RGBXDataset import RGBXDataset

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    sc.install_loader(config, RGBXDataset)  # loader workers are forked and inherit it

    original_build = runtime_module.build_training_runtime
    original_sanitize = runtime_module.sanitize_author_loss_map

    def finite_loss_map(loss_map):
        result, count = original_sanitize(loss_map)
        if count:
            raise FloatingPointError("NaN replacement encountered; refusing to continue this run")
        return result, count

    def build(*positional, **keyword):
        runtime = original_build(*positional, **keyword)
        expected_loader = -(-int(config.num_train_imgs) // int(config.batch_size))
        if runtime.engine.world_size != WORLD_SIZE or len(runtime.train_loader) != expected_loader:
            raise RuntimeError("world size {} or loader length {} differs from the recipe ({}, {})".format(
                runtime.engine.world_size, len(runtime.train_loader), WORLD_SIZE, expected_loader))
        if runtime.engine.local_rank == 0:
            (args.output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
            (args.output / "single_channel.json").write_text(json.dumps({
                "single_channel": args.channel,
                "x_input": "channel {} of the REL+ cache, copied into all three X channels".format(args.channel),
                "relplus_config": str(args.config.resolve()),
                "relplus_variant": variant,
                "seed": config.seed,
                "source_root": str(source_root),
                "smoke_iterations": args.smoke_iterations,
                "engine_arguments": forwarded,
            }, indent=2) + "\n", encoding="utf-8")
        print("X input: REL+ {} only, replicated to 3 channels".format(args.channel), flush=True)
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
