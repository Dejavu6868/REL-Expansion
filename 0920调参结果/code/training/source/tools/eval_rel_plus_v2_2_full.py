#!/usr/bin/env python3
"""Full-test CMX-REL+ V2.2 evaluator for one checkpoint."""

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _unbatch(batch):
    sample = {"fn": batch["fn"][0]}
    for key in ("data", "modal_x", "label", "modal_x_valid_mask"):
        value = batch[key][0]
        sample[key] = (
            value.detach().cpu().numpy() if torch.is_tensor(value) else value
        )
    return sample


def load_checkpoint_once(network, checkpoint):
    payload = torch.load(str(checkpoint), map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    incompatible = network.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    return payload.get("epoch") if isinstance(payload, dict) else None


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _distributed_context(device_arg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_arg)
    return rank, local_rank, world_size, device, initialized_here


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "configs.stanford2d3d_s2d."
            "cmx_mit_b2_rel_plus_v2_2_formal"
        ),
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    args = parser.parse_args()

    config = importlib.import_module(args.config).config
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.data_setting import build_data_setting
    from dataloader.dataloader import ValPre
    from dataloader.samplers import DistributedEvalSamplerNoPad
    from engine.relplus_evaluator import (
        aggregate_rank_evaluations,
        evaluate_prepared_sample,
        metrics_from_confusion,
        prepare_eval_sample,
        save_prediction_pair,
    )
    from models.builder import EncoderDecoder
    from utils.training_protocol import assert_runtime_dataset_contract

    assert_runtime_dataset_contract(config, require_cache_audit=True)
    rank, local_rank, world_size, device, initialized_here = _distributed_context(
        args.device
    )
    args.output.mkdir(parents=True, exist_ok=True)
    setting = build_data_setting(config, split="val")
    dataset = RGBXDataset(
        setting, "val", ValPre(x_mode=setting["x_mode"])
    )
    sampler = DistributedEvalSamplerNoPad(
        dataset, num_replicas=world_size, rank=rank
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        sampler=sampler,
    )
    network = EncoderDecoder(
        cfg=config, criterion=None, norm_layer=nn.BatchNorm2d
    )
    checkpoint_epoch = load_checkpoint_once(network, args.checkpoint)
    network.to(device)
    network.eval()

    confusion = np.zeros(
        (config.num_classes, config.num_classes), dtype=np.int64
    )
    owned = []
    prediction_count = 0
    prediction_root = args.output / "predictions"
    visualization_root = args.output / "visualizations"
    for batch in loader:
        prepared = prepare_eval_sample(_unbatch(batch), config)
        result = evaluate_prepared_sample(
            network,
            prepared,
            class_num=config.num_classes,
            ignore_index=config.background,
            device=device,
        )
        confusion += result["hist"]
        owned.append(prepared.sample_id)
        if args.save_predictions:
            save_prediction_pair(
                result["prediction"],
                prepared.sample_id,
                prediction_root,
                RGBXDataset.get_class_colors(),
            )
            prediction_count += 1
        if args.save_visualizations:
            paths = save_prediction_pair(
                result["prediction"],
                prepared.sample_id,
                visualization_root,
                RGBXDataset.get_class_colors(),
            )
            Path(paths["raw"]).unlink()

    rank_report = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "sample_count": len(owned),
        "owned_sample_ids": owned,
        "confusion_matrix": confusion.tolist(),
        "prediction_count": prediction_count,
    }
    (args.output / "rank_{:02d}_evaluation.json".format(rank)).write_text(
        json.dumps(rank_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    confusion_tensor = torch.as_tensor(confusion, dtype=torch.long, device=device)
    count_tensor = torch.tensor([len(owned)], dtype=torch.long, device=device)
    if world_size > 1:
        dist.all_reduce(confusion_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
    total_count = int(count_tensor.item())
    if total_count != config.num_eval_imgs:
        raise RuntimeError(
            "full evaluator sample count mismatch: expected {}, found {}".format(
                config.num_eval_imgs, total_count
            )
        )

    if rank == 0:
        full_confusion = confusion_tensor.cpu().numpy().astype(np.int64)
        metrics = metrics_from_confusion(full_confusion)
        metrics.update(
            {
                "status": "PASS",
                "scientific_metric_reported": True,
                "integration_protocol_id": config.integration_protocol_id,
                "representation_protocol_id": config.representation_protocol_id,
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint_epoch,
                "evaluation_sample_count": total_count,
                "class_count": config.num_classes,
                "ignore_index": config.background,
                "eval_scale_array": list(config.eval_scale_array),
                "eval_flip": config.eval_flip,
                "eval_crop_size": list(config.eval_crop_size),
                "eval_align_corners": config.eval_align_corners,
                "world_size": world_size,
                "file_hash_written": False,
            }
        )
        (args.output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        per_class = [
            {"class_id": index, "class_name": name, "IoU": metrics["per_class_iou"][index]}
            for index, name in enumerate(config.class_names)
        ]
        _write_csv(
            args.output / "per_class_iou.csv",
            per_class,
            ["class_id", "class_name", "IoU"],
        )
        confusion_rows = [
            {"true_class": index, **{
                "pred_{}".format(column): int(full_confusion[index, column])
                for column in range(config.num_classes)
            }}
            for index in range(config.num_classes)
        ]
        _write_csv(
            args.output / "confusion_matrix.csv",
            confusion_rows,
            ["true_class"] + ["pred_{}".format(index) for index in range(config.num_classes)],
        )
        manifest_rows = []
        rank_reports = []
        for owner_rank in range(world_size):
            report = json.loads(
                (args.output / "rank_{:02d}_evaluation.json".format(owner_rank)).read_text(
                    encoding="utf-8"
                )
            )
            rank_reports.append(report)
            manifest_rows.extend(
                {"sample_id": sample_id, "rank": owner_rank}
                for sample_id in report["owned_sample_ids"]
            )
        merged_again, count_again, owned_again = aggregate_rank_evaluations(
            rank_reports
        )
        if count_again != total_count or owned_again != [
            row["sample_id"] for row in manifest_rows
        ]:
            raise RuntimeError("evaluation ownership aggregation mismatch")
        if not np.array_equal(merged_again, full_confusion):
            raise RuntimeError("rank-file confusion differs from all-reduce")
        _write_csv(
            args.output / "evaluation_manifest.csv",
            manifest_rows,
            ["sample_id", "rank"],
        )
        if args.save_predictions and sum(
            json.loads(
                (args.output / "rank_{:02d}_evaluation.json".format(owner_rank)).read_text(
                    encoding="utf-8"
                )
            )["prediction_count"]
            for owner_rank in range(world_size)
        ) != total_count:
            raise RuntimeError("saved prediction count is incomplete")
        print(json.dumps(metrics, ensure_ascii=False))
    if world_size > 1:
        dist.barrier()
    if initialized_here:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
