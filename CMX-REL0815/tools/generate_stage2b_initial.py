#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    os.environ["STAGE2B_SEED"] = str(args.seed)
    selected = importlib.import_module("configs.stage2b_rawdepth")
    sys.modules["config"] = selected
    from models.builder import EncoderDecoder
    from stage2a.runtime import seed_everything

    cfg = selected.config
    seed_everything(cfg.seed, deterministic=True)
    model = EncoderDecoder(
        cfg=cfg,
        criterion=nn.CrossEntropyLoss(ignore_index=cfg.background),
        norm_layer=nn.BatchNorm2d,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("{}.{}.tmp".format(args.output.name, os.getpid()))
    torch.save({"model": model.state_dict(), "seed": cfg.seed, "config": "configs.stage2b_rawdepth"}, temporary)
    os.replace(str(temporary), str(args.output))
    report = {
        "status": "PASS_COMMON_INITIAL_MODEL",
        "seed": cfg.seed,
        "source_config": "configs.stage2b_rawdepth",
        "pretrained_model": cfg.pretrained_model,
        "architecture": {"backbone": cfg.backbone, "decoder": cfg.decoder, "num_classes": cfg.num_classes},
        "tensor_count": len(model.state_dict()),
        "four_arms_load_this_exact_file": True,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
