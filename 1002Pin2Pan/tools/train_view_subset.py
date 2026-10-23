#!/usr/bin/env python3
"""Train an existing CMX arm on a subset of its training images, with the frozen training entry.

Launch it like the frozen training, one process per GPU::

    python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=29514 \\
        1002Pin2Pan/tools/train_view_subset.py --config <resolved arm config> \\
        --subset <list from make_view_subset.py> --output <new folder>

``--config`` is a resolved CMX arm config (REL+ or HHA). Seed, recipe, caches and data checks
come from it unchanged; the frozen checks still run on the full training list it names. The
subset is applied when the training dataset reads that list, so only the listed images are
loaded, and every listed ID must be in it. The epoch keeps the frozen rule
``niters_per_epoch = images // batch_size + 1``, so 200 epochs over 85% of the images take
about 85% of the iterations, and the 10 warm-up epochs shrink with them.

As in ``train_rgb_only.py``, the run uses one OpenCV and one torch thread per process and refuses
NaN losses and any world size other than 8. Other arguments go to the frozen Engine, e.g.
``--continue <checkpoint>``. ``--smoke-iterations N`` trains one epoch of N iterations.
"""

import argparse
import hashlib
import json
import os
import runpy
import sys
import types
from pathlib import Path

import numpy as np

import train_rgb_only as tr
from eval_pano_transfer import DEFAULT_SOURCE_ROOT


def read_ids(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError("{} lists no samples".format(path))
    if len(set(ids)) != len(ids):
        raise ValueError("{} lists a sample twice".format(path))
    return ids


def subset_config(arm_config, subset, subset_count, output, source_root, smoke_iterations=None):
    """The arm config with only the experiment name, output paths and epoch length changed."""
    if arm_config.get("architecture") != "Original CMX":
        raise ValueError("--config must be a resolved CMX arm config")
    if subset_count >= int(arm_config["num_train_imgs"]):
        raise ValueError("the subset must be smaller than the {} training images".format(arm_config["num_train_imgs"]))
    config = json.loads(json.dumps(arm_config))
    batch_size = int(config["batch_size"])
    iterations = subset_count // batch_size + 1
    config.update(
        experiment_name=config["experiment_name"] + "_" + subset.stem,
        root_dir=str(source_root), abs_dir=str(source_root),
        output_dir=str(output), log_dir=str(output / "logs"), tb_dir=str(output / "tensorboard"),
        checkpoint_dir=str(output / "checkpoints"), log_dir_link=str(output / "latest_logs"),
        ddp_smoke_report=None,
        niters_per_epoch=iterations, logical_samples_per_epoch=batch_size * iterations,
        view_subset=str(subset), view_subset_count=subset_count,
        view_subset_sha256=hashlib.sha256(subset.read_bytes()).hexdigest(),
    )
    config.update({field: str(output / "logs" / name) for field, name in tr.LOG_FILES.items()})
    if smoke_iterations is not None:
        config.update(nepochs=1, niters_per_epoch=smoke_iterations)
    return config


def subset_dataset(base, ids):
    """``base`` whose training split keeps only ``ids``, in the order of the training list."""
    keep = set(ids)

    class SubsetDataset(base):
        def _get_file_names(self, split_name):
            names = super(SubsetDataset, self)._get_file_names(split_name)
            if split_name != "train":
                return names
            kept = [name for name in names if name in keep]
            if len(kept) != len(keep):
                raise RuntimeError("{} subset IDs are not in the training list".format(len(keep) - len(kept)))
            return kept

    return SubsetDataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="resolved CMX arm config JSON")
    parser.add_argument("--subset", required=True, type=Path, help="sample IDs to train on")
    parser.add_argument("--output", required=True, type=Path, help="new, empty output folder")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--smoke-iterations", type=int, default=None)
    args, forwarded = parser.parse_known_args(argv)
    if args.smoke_iterations is not None and args.smoke_iterations <= 0:
        parser.error("--smoke-iterations must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))

    source_root = args.source_root.resolve()
    subset = args.subset.resolve()
    ids = read_ids(subset)
    resolved = subset_config(
        json.loads(args.config.read_text(encoding="utf-8")), subset, len(ids), args.output.resolve(),
        source_root, args.smoke_iterations,
    )
    from easydict import EasyDict

    config = EasyDict(resolved)
    config.norm_mean = np.asarray(config.norm_mean, dtype=np.float64)
    config.norm_std = np.asarray(config.norm_std, dtype=np.float64)
    module = types.ModuleType("_view_subset_config")
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

    original_build = runtime_module.build_training_runtime
    original_sanitize = runtime_module.sanitize_author_loss_map

    def finite_loss_map(loss_map):
        result, count = original_sanitize(loss_map)
        if count:
            raise FloatingPointError("NaN replacement encountered; refusing to continue this run")
        return result, count

    def build(*positional, **keyword):
        runtime = original_build(*positional, dataset_class=subset_dataset(RGBXDataset, ids), **keyword)
        expected_loader = int(config.logical_samples_per_epoch) // int(config.batch_size)
        if (runtime.engine.world_size != tr.WORLD_SIZE or len(runtime.train_loader) != expected_loader
                or len(runtime.train_loader.dataset) != len(ids)):
            raise RuntimeError("world size {}, loader length {} or dataset size {} differs from ({}, {}, {})".format(
                runtime.engine.world_size, len(runtime.train_loader), len(runtime.train_loader.dataset),
                tr.WORLD_SIZE, expected_loader, len(ids)))
        if runtime.engine.local_rank == 0:
            (args.output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
            (args.output / "view_subset.json").write_text(json.dumps({
                "arm_config": str(args.config.resolve()),
                "subset": str(subset),
                "subset_sha256": resolved["view_subset_sha256"],
                "subset_count": len(ids),
                "full_train_count": int(config.num_train_imgs),
                "niters_per_epoch": int(config.niters_per_epoch),
                "seed": config.seed,
                "source_root": str(source_root),
                "smoke_iterations": args.smoke_iterations,
                "engine_arguments": forwarded,
            }, indent=2) + "\n", encoding="utf-8")
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
