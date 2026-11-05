#!/usr/bin/env python3
"""Supervised panorama training (Stanford2D3D fold 1) with Max's original Pano_MA_Seg code, X = REL+ v3.1new.

This is the recipe of the conference's RGB-REL-SMMF row (the gated checkpoint scored 67.367%), with
only the X folder changed: gated CMX MiT-B2 + MLP decoder, Focal loss, AdamW lr 6e-5, poly 0.9 with
10 warm-up epochs, global batch 8 of 1080x1080 random crops at scales 0.5-1.75 with mirroring,
200 epochs of 131 iterations, soft gate before epoch 100 and hard after (temperature 1.0 -> 0.01),
seed 12345, checkpoints every 5 epochs from epoch 100. Launch it like Max's run_train.sh::

    python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        1002Pin2Pan/tools/train_supervised_pano.py --smmf-root <Pano_MA_Seg> --data-root <Stanford2D3DPano> \\
        --x-root <cache>/X --output <new folder>

Max's current config.py selects Matterport, so this sets every field that the Matterport branch changes
(and the Stanford branch defines) back to the Stanford values, plus the paths; nothing else. Then it runs
his unmodified ``train.py`` from ``--output`` (the gate writes gate_*.txt to the working directory).
``--x-variant v3_1new`` (default) refuses an X folder whose parent has no v3_1new marker, so the run can't
silently train on the original REL; ``--x-variant rel`` trains the original REL folder (``rle``) as a
same-code control. ``--smoke-iterations N`` trains one epoch of N iterations and saves model-1.pth.
Other arguments (``-c <checkpoint>`` to continue) go to Max's Engine.
"""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import relplus_variant as rv

CLASS_NAMES = ["beam", "board", "bookcase", "ceiling", "chair", "clutter", "column", "door",
               "floor", "sofa", "table", "wall", "window"]
FOLD1_COUNTS = (1040, 373)
# The Stanford values of every field the Matterport branch of config.py overrides or the Stanford branch sets.
STANFORD_RECIPE = {
    "dataset_name": "Stanford2D3DPano",
    "dataset_fold": 1,
    "rgb_format": "_rgb.png",
    "gt_format": "_semantic.png",
    "gt_transform": True,
    "x_format": "_hha.png",
    "x_name": "rle",
    "x_is_single_channel": False,
    "x_selected_channel": [],
    "in_chans": 3,
    "in_chans_x": 3,
    "num_classes": 13,
    "class_names": CLASS_NAMES,
    "num_train_imgs": FOLD1_COUNTS[0],
    "num_eval_imgs": FOLD1_COUNTS[1],
    "image_height": 1080,
    "image_width": 1080,
    "batch_size": 8,
    "nepochs": 200,
    "epoch_hard": 100,
    "eval_start_epoch": 100,
    "eval_step": 5,
    "checkpoint_start_epoch": 100,
    "checkpoint_step": 5,
    "eval_crop_size": [1080, 1080],
    "eval_batch_size": 4,
    "eval_stride_rate": 2 / 3,
    "smmf_pad_h": 652,
    "window_positions": [0, 484, 968],
    "backbone": "gate_mit_b2",
    "using_gate": True,
    "pretrained_model": None,
}
# Training-relevant values of config.py that the recipe keeps; checked so a changed server copy is noticed.
EXPECTED_DEFAULTS = {
    "lr": 6e-5, "lr_power": 0.9, "weight_decay": 0.01, "optimizer": "AdamW", "criterion": "Focal",
    "warm_up_epoch": 10, "train_scale_array": [0.5, 0.75, 1, 1.25, 1.5, 1.75], "decoder": "MLPDecoder",
    "decoder_embed_dim": 512, "start_temp": 1.0, "end_temp": 0.01, "tempsched_start_epoch": 1,
    "use_random_crop": True, "use_random_reprojection": False, "use_guided_rotation": False,
    "use_spherical_weighted_loss": False, "use_dpe": False, "background": 255,
}
SOURCE_FILES = (
    "config.py", "train.py", "engine/engine.py", "models/builder.py", "models/encoders/dual_gate_segformer.py",
    "models/net_utils.py", "models/decoders/MLPDecoder.py", "dataloader/RGBXDataset.py",
    "dataloader/dataloader.py", "utils/transforms.py", "utils/loss_opr.py", "utils/lr_policy.py",
    "utils/init_func.py",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def check_x_root(x_root, x_variant):
    marker = rv.read_marker(Path(x_root).parent)
    found = None if marker is None else marker["variant"]
    if x_variant == "v3_1new" and found != "v3_1new":
        raise ValueError("{} has no v3_1new marker in its parent (found {}); is it the REL+ cache?".format(
            x_root, found))
    if x_variant == "rel" and found is not None:
        raise ValueError("--x-variant rel but {} is marked {}".format(x_root, found))


def check_files(setting):
    """Every split panorama has its RGB, label and X file; returns the split sizes."""
    counts = []
    for split in (setting["train_source"], setting["eval_source"]):
        names = [line.strip() for line in Path(split).read_text(encoding="utf-8").splitlines() if line.strip()]
        for name in names:
            for root, suffix in (("rgb_root", "rgb_format"), ("gt_root", "gt_format"), ("x_root", "x_format")):
                path = Path(setting[root]) / (name + setting[suffix])
                if not path.is_file():
                    raise FileNotFoundError(path)
        counts.append(len(names))
    if tuple(counts) != FOLD1_COUNTS:
        raise ValueError("fold 1 split sizes {} are not {}".format(counts, FOLD1_COUNTS))
    return counts


def configure(config, data_root, x_root, output, pretrained, seed, smoke_iterations=None):
    """Max's config with the Stanford fold-1 recipe and this run's paths."""
    for key, value in EXPECTED_DEFAULTS.items():
        if config[key] != value:
            raise ValueError("Pano_MA_Seg config.py has {} = {}, expected {}".format(key, config[key], value))
    for key, value in STANFORD_RECIPE.items():
        config[key] = value
    config.seed = seed
    config.dataset_path = str(data_root)
    config.rgb_root_folder = str(data_root / "image")
    config.gt_root_folder = str(data_root / "label")
    config.x_root_folder = str(x_root)
    config.train_source = str(data_root / "fold1_train.txt")
    config.eval_source = str(data_root / "fold1_test.txt")
    config.backbone_pretrained_model = str(pretrained)
    config.niters_per_epoch = config.num_train_imgs // config.batch_size + 1
    if smoke_iterations is not None:
        config.nepochs, config.niters_per_epoch = 1, smoke_iterations
    config.train_gate_cfg = {"soft_gate": True, "temp": config.start_temp}
    config.data_setting = {
        "rgb_root": config.rgb_root_folder, "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder, "gt_format": config.gt_format, "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder, "x_format": config.x_format, "x_single_channel": config.x_is_single_channel,
        "regions_root": "", "regions_format": "", "lw_root": "", "lw_format": "",
        "train_source": config.train_source, "eval_source": config.eval_source, "class_names": CLASS_NAMES,
    }
    log = output / "log"
    config.log_dir = config.log_dir_link = str(log)
    config.tb_dir = str(log / "tb")
    config.checkpoint_dir = str(log / "checkpoint")
    config.val_log_file = str(log / "val.log")
    config.link_val_log_file = str(log / "val_last.log")
    config.log_file = str(log / "train.log")
    config.link_log_file = str(log / "log_last.log")
    config.val_tb_dir = str(log / "val_tb")
    config.best_dir = str(log / "bestmodels")
    config.resume = ""
    config.enable_resume = False
    return config


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value.tolist() if hasattr(value, "tolist") else value


def import_config(smmf_root):
    sys.path.insert(0, str(smmf_root))
    cwd = os.getcwd()
    os.chdir(smmf_root)  # config.py reads the working directory
    try:
        from config import config
    finally:
        os.chdir(cwd)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smmf-root", required=True, type=Path, help="Max's original Pano_MA_Seg folder")
    parser.add_argument("--data-root", required=True, type=Path, help="datasets/Stanford2D3DPano")
    parser.add_argument("--x-root", required=True, type=Path, help="X folder, e.g. <REL+ cache>/X")
    parser.add_argument("--x-variant", default="v3_1new", choices=("v3_1new", "rel"))
    parser.add_argument("--output", required=True, type=Path, help="new, empty folder")
    parser.add_argument("--pretrained", type=Path, default=None,
                        help="MiT-B2 ImageNet weights (default <smmf-root>/pretrained/segformer/mit_b2.pth)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--smoke-iterations", type=int, default=None)
    args, forwarded = parser.parse_known_args(argv)
    if args.smoke_iterations is not None and args.smoke_iterations <= 0:
        parser.error("--smoke-iterations must be positive")
    smmf_root, data_root, x_root, output = (
        path.resolve() for path in (args.smmf_root, args.data_root, args.x_root, args.output))
    pretrained = (args.pretrained or smmf_root / "pretrained" / "segformer" / "mit_b2.pth").resolve()
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    rank = int(os.environ.get("RANK", "0"))
    continuing = any(item in ("-c", "--continue") or item.startswith("--continue=") for item in forwarded)
    record_path = output / ("supervised_pano_continue.json" if continuing else "supervised_pano.json")
    if rank == 0 and not continuing and record_path.exists():  # not "empty": the other ranks may create log/
        raise FileExistsError("{} already holds a run; refusing to overwrite".format(output))
    check_x_root(x_root, args.x_variant)

    config = configure(import_config(smmf_root), data_root, x_root, output, pretrained, args.seed,
                       args.smoke_iterations)
    counts = check_files(config.data_setting)
    (output / "log").mkdir(parents=True, exist_ok=True)
    if rank == 0:
        record = {
            "x_variant": args.x_variant,
            "x_root": str(x_root),
            "relplus_marker": rv.read_marker(x_root.parent),
            "split_sizes": counts,
            "smoke_iterations": args.smoke_iterations,
            "engine_arguments": forwarded,
            "smmf_root": str(smmf_root),
            "source_sha256": {name: file_sha256(smmf_root / name) for name in SOURCE_FILES},
            "resolved_config": plain(dict(config)),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(smmf_root / "config.py", output / "config.py")  # train.py copies ./config.py at the end
    print("Supervised panorama training: X {} from {}".format(args.x_variant, x_root), flush=True)

    os.chdir(output)
    entry = smmf_root / "train.py"
    sys.argv = [str(entry)] + forwarded
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as result:
        if result.code not in (0, None):
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
