#!/usr/bin/env python3
"""Audit actual dual-encoder MiT-B2 loading without writing file hashes."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _source_state(path):
    payload = torch.load(str(path), map_location="cpu")
    if "model" in payload:
        payload = payload["model"]
    if "state_dict" in payload:
        payload = payload["state_dict"]
    return payload


def _mapped_source(source):
    mapped = {}
    for key, value in source.items():
        if "patch_embed" in key:
            mapped[key] = (key, key.replace("patch_embed", "extra_patch_embed"))
        elif "block" in key:
            mapped[key] = (key, key.replace("block", "extra_block"))
        elif "norm" in key:
            mapped[key] = (key, key.replace("norm", "extra_norm"))
    return mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    selected = importlib.import_module(
        "configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_1_pilot"
    )
    sys.modules["config"] = selected
    import models.builder as builder_module
    from utils.training_protocol import build_author_criterion, set_author_seed

    config = selected.config
    set_author_seed(config.seed, epoch=0, local_rank=0, distributed=False)
    criterion = build_author_criterion(config)
    initialization_calls = []
    original_init_weight = builder_module.init_weight

    def traced_init_weight(module, *call_args, **call_kwargs):
        before = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }
        result = original_init_weight(module, *call_args, **call_kwargs)
        after = module.state_dict()
        initialization_calls.append(
            {
                "state_dict_key_count": len(after),
                "shape_mismatch_count": sum(
                    tuple(before[key].shape) != tuple(after[key].shape)
                    for key in before
                ),
                "changed_tensor_count": sum(
                    not torch.equal(before[key], after[key]) for key in before
                ),
            }
        )
        return result

    builder_module.init_weight = traced_init_weight
    try:
        model = builder_module.EncoderDecoder(
            cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d
        )
    finally:
        builder_module.init_weight = original_init_weight
    target = model.backbone.state_dict()
    source = _source_state(Path(config.pretrained_model))
    mapping = _mapped_source(source)
    ignored_source_keys = sorted(set(source) - set(mapping))
    expected_ignored_source_keys = ["head.bias", "head.weight"]
    unexpected_source_keys = sorted(
        set(ignored_source_keys) - set(expected_ignored_source_keys)
    )

    missing = []
    shape_mismatch = []
    value_mismatch = []
    rgb_loaded = 0
    x_loaded = 0
    for source_key, (rgb_key, x_key) in mapping.items():
        source_tensor = source[source_key]
        for arm, target_key in (("rgb", rgb_key), ("x", x_key)):
            if target_key not in target:
                missing.append({"arm": arm, "key": target_key})
                continue
            if tuple(target[target_key].shape) != tuple(source_tensor.shape):
                shape_mismatch.append(
                    {
                        "arm": arm,
                        "key": target_key,
                        "source_shape": list(source_tensor.shape),
                        "target_shape": list(target[target_key].shape),
                    }
                )
                continue
            if not torch.equal(target[target_key].cpu(), source_tensor.cpu()):
                value_mismatch.append({"arm": arm, "key": target_key})
                continue
            if arm == "rgb":
                rgb_loaded += 1
            else:
                x_loaded += 1

    decoder = model.decode_head.state_dict()
    decoder_finite = all(
        bool(torch.isfinite(value).all()) for value in decoder.values()
    )
    decoder_nonzero = sum(
        int(torch.count_nonzero(value).item()) for value in decoder.values()
    )
    status = (
        "PASS"
        if mapping
        and rgb_loaded == len(mapping)
        and x_loaded == len(mapping)
        and not missing
        and not shape_mismatch
        and not value_mismatch
        and not unexpected_source_keys
        and ignored_source_keys == expected_ignored_source_keys
        and len(initialization_calls) == 1
        and initialization_calls[0]["shape_mismatch_count"] == 0
        and initialization_calls[0]["changed_tensor_count"] > 0
        and decoder_finite
        and decoder_nonzero > 0
        else "FAIL"
    )
    report = {
        "status": status,
        "pretrained_path": config.pretrained_model,
        "source_mapped_key_count": len(mapping),
        "rgb_encoder_exact_tensor_matches": rgb_loaded,
        "x_encoder_exact_tensor_matches": x_loaded,
        "missing_keys": missing,
        "unexpected_keys": unexpected_source_keys,
        "expected_ignored_source_keys": ignored_source_keys,
        "shape_mismatch": shape_mismatch,
        "value_mismatch": value_mismatch,
        "decoder": {
            "initialization": "Original CMX default init_weight/kaiming_normal",
            "state_dict_key_count": len(decoder),
            "all_tensors_finite": decoder_finite,
            "nonzero_value_count": decoder_nonzero,
            "init_weight_trace": initialization_calls,
        },
        "model_math": {
            "backbone": config.backbone,
            "decoder": config.decoder,
            "gate": config.using_gate,
            "smmf": config.using_smmf,
            "dymm": config.using_dymm,
            "sga": config.using_sga,
        },
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
