#!/usr/bin/env python3
"""Adapt a frozen pinhole checkpoint to the panoramas (Trans4PASS warm-up and MPA stages).

Run with ``python -m torch.distributed.launch --nproc_per_node=8`` (as the
frozen launcher does) or ``torchrun``; one process per GPU.

``--stage warmup`` (Trans4PASS ``train_warm.py``): source focal loss plus an
output-space adversarial loss on unlabelled target crops.
``--stage mpa`` (Trans4PASS ``train_mpa.py``): source focal loss, focal loss
on the target pseudo-labels (``gen_pseudo_labels.py``), feature-to-prototype
losses on both domains and the adversarial loss. Weights are Trans4PASS's:
adversarial 0.001, prototype 0.001 per domain, pseudo-label 1.

Source batches come from the frozen training loader (same cache, augmentation
and focal loss); target batches are full-height crops of the cached panoramas
at a random yaw (``pin2pan_adapt.TargetPanoramas``). The network starts from
the checkpoint and is saved once, at the last iteration, as
``{"epoch": <source epoch>, "model": ...}`` so the panorama evals load it with
``--expected-epoch`` of the source checkpoint. No checkpoint is chosen on area 5.

Where this differs from Trans4PASS (it fine-tunes a trained network rather
than training from ImageNet weights): AdamW at 1/10 of the frozen learning
rate with poly decay instead of SGD; 2,000 warm-up and 10,000 MPA iterations;
the frozen focal loss instead of cross-entropy; the prototype memory update
fixed (``pin2pan_adapt.PrototypeMemory``); features are the decoder's fused
512-channel map (``decode_head.linear_fuse``) at 1/4 resolution.
"""

import argparse
import copy
import functools
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import build_target_cache as tc
import eval_pano_transfer as ev


STAGE_ITERATIONS = {"warmup": 2000, "mpa": 10000}
LAMBDA_ADV = 0.001
LAMBDA_PROTOTYPE = 0.001
LAMBDA_PSEUDO = 1.0
SOURCE_LABEL, TARGET_LABEL = 0.0, 1.0  # discriminator targets, as in Trans4PASS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_ITERATIONS))
    parser.add_argument("--config", required=True, type=Path, help="resolved arm config JSON")
    parser.add_argument("--checkpoint", required=True, type=Path, help="checkpoint to adapt")
    parser.add_argument("--expected-epoch", type=int, default=200)
    parser.add_argument("--target-cache", required=True, type=Path, help="build_target_cache.py output")
    parser.add_argument("--pseudo-labels", type=Path, default=None, help="gen_pseudo_labels.py output (mpa)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=ev.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--iterations", type=int, default=None, help="default: warmup 2000, mpa 10000")
    parser.add_argument("--lr", type=float, default=1.2e-5)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--source-batch", type=int, default=2, help="per GPU")
    parser.add_argument("--target-batch", type=int, default=1, help="per GPU")
    parser.add_argument("--target-crop-width", type=int, default=1024)
    parser.add_argument("--memory-source-images", type=int, default=2000, help="mpa memory init")
    parser.add_argument("--num-workers", type=int, default=4, help="per loader and GPU")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda", help="cuda (one GPU per process) or cpu")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.stage == "mpa") != (args.pseudo_labels is not None):
        parser.error("--pseudo-labels is required for mpa and not used by warmup")
    if args.iterations is None:
        args.iterations = STAGE_ITERATIONS[args.stage]
    if min(args.iterations, args.source_batch, args.target_batch, args.target_crop_width) <= 0:
        parser.error("iterations, batch sizes and crop width must be positive")
    return args


def target_samples(manifest, pseudo_labels):
    """Warm-up uses every cached panorama; MPA those that have pseudo-labels."""
    if pseudo_labels is None:
        return manifest["samples"]
    report = json.loads((pseudo_labels / "pseudo_labels.json").read_text(encoding="utf-8"))
    samples = (pseudo_labels / "samples.txt").read_text(encoding="utf-8").split()
    if report["x_mode"] != manifest["x_mode"] or not set(samples) <= set(manifest["samples"]):
        raise ValueError("{} was not made from this target cache".format(pseudo_labels))
    return samples


def endless(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def average_gradients(parameters, world_size):
    """The discriminator is used with and without gradients in one iteration, so it is not under DDP."""
    import torch
    import torch.distributed as dist

    grads = [parameter.grad for parameter in parameters]
    if world_size > 1:
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat /= world_size
        for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(synced)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("{} is not empty; refusing to overwrite".format(args.output))

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    import pin2pan_adapt as pa

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if args.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0")) if args.local_rank is None else args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    is_writer = rank == 0

    frozen = ev.import_frozen(args.source_root)
    from dataloader.RGBXDataset import RGBXDataset
    from dataloader.dataloader import get_train_loader
    from utils.lr_policy import PolyLR
    from utils.training_protocol import (
        build_author_criterion,
        build_author_optimizer,
        configure_author_cudnn,
        set_author_seed,
    )
    from utils.training_runtime import sanitize_author_loss_map
    from utils.transforms import normalize

    config = ev.load_config(args.config)
    manifest = tc.load_manifest(args.target_cache, config)
    samples = target_samples(manifest, args.pseudo_labels)
    config.lr = args.lr
    config.batch_size = args.source_batch * world_size
    config.num_workers = args.num_workers
    configure_author_cudnn()
    set_author_seed(config.seed, local_rank=rank, distributed=world_size > 1)

    source_loader, source_sampler = get_train_loader(
        SimpleNamespace(world_size=world_size, local_rank=rank), RGBXDataset, cfg=config
    )
    source_batches = endless(source_loader, source_sampler)
    target_set = pa.TargetPanoramas(
        args.target_cache,
        samples,
        args.target_crop_width,
        functools.partial(normalize, mean=config.norm_mean, std=config.norm_std),
        None if args.pseudo_labels is None else args.pseudo_labels / "PseudoLabel",
    )
    target_sampler = DistributedSampler(
        target_set, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed, drop_last=True
    )
    target_loader = DataLoader(
        target_set, batch_size=args.target_batch, sampler=target_sampler, num_workers=args.num_workers,
        drop_last=True, pin_memory=device.type == "cuda",
    )
    target_batches = endless(target_loader, target_sampler)

    norm_layer = nn.SyncBatchNorm if world_size > 1 and device.type == "cuda" else nn.BatchNorm2d
    network = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=norm_layer)
    source_epoch = frozen["load_checkpoint_once"](network, args.checkpoint, expected_epoch=args.expected_epoch)
    for module in network.decode_head.modules():
        if isinstance(module, norm_layer):  # as the frozen training's init_weight sets them
            module.eps = config.bn_eps
            module.momentum = config.bn_momentum
    network.to(device)
    features = {}
    network.decode_head.linear_fuse.register_forward_hook(
        lambda module, inputs, output: features.__setitem__("fused", output)
    )
    optimizer = build_author_optimizer(network, norm_layer, config)
    model = network
    if world_size > 1:
        model = DistributedDataParallel(
            network,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    discriminator = pa.FCDiscriminator(config.num_classes).to(device)
    if world_size > 1:
        # Rank-specific seeds require identical initial weights before averaging gradients.
        for parameter in discriminator.parameters():
            dist.broadcast(parameter.detach(), src=0)
    criterion = build_author_criterion(config)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.99))
    schedule = PolyLR(args.lr, config.lr_power, args.iterations)
    schedule_d = PolyLR(args.lr_d, 0.9, args.iterations)
    bce = nn.BCEWithLogitsLoss()

    def segmentation_loss(logits, label):
        loss_map, _ = sanitize_author_loss_map(criterion(logits, label))
        return loss_map.mean()

    def adversarial(logits, domain):
        output = discriminator(F.softmax(logits, dim=1))
        return bce(output, torch.full_like(output, domain))

    def inputs(batch):
        return batch["data"].to(device), batch["modal_x"].to(device), batch["label"].to(device)

    memory = None
    memory_counts = None
    if args.stage == "mpa":
        network.eval()

        def predict(batch):
            with torch.no_grad():
                rgb, modal_x, _ = inputs(batch)
                logits = network(rgb, modal_x)
            return features["fused"], logits

        source_init = math.ceil(args.memory_source_images / (world_size * args.source_batch))
        init_sampler = DistributedSampler(target_set, num_replicas=world_size, rank=rank, shuffle=False)
        init_memory, source_counts, target_counts = pa.initial_memory(
            (next(source_batches) for _ in range(source_init)),
            DataLoader(target_set, batch_size=args.target_batch, sampler=init_sampler, num_workers=args.num_workers),
            predict, config.num_classes, config.decoder_embed_dim, device,
        )
        memory = pa.PrototypeMemory(init_memory)
        memory_counts = {"source_batches": source_counts, "target_batches": target_counts}
    model.train()
    discriminator.train()

    if is_writer:
        args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    names = ["source_focal", "adversarial", "discriminator"]
    if memory is not None:
        names += ["pseudo_focal", "prototype_source", "prototype_target"]
    running = torch.zeros(len(names), device=device)
    for iteration in range(args.iterations):
        for group in optimizer.param_groups:
            group["lr"] = schedule.get_lr(iteration)
        for group in optimizer_d.param_groups:
            group["lr"] = schedule_d.get_lr(iteration)
        optimizer.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        discriminator.requires_grad_(False)
        values = {}

        rgb, modal_x, label = inputs(next(source_batches))
        source_logits = model(rgb, modal_x)
        values["source_focal"] = segmentation_loss(source_logits, label)
        loss = values["source_focal"]
        if memory is not None:
            values["prototype_source"], source_means, source_present = pa.feat_kl_loss(
                features["fused"], label, memory.memory
            )
            loss = loss + LAMBDA_PROTOTYPE * values["prototype_source"]
        loss.backward()

        rgb, modal_x, label = inputs(next(target_batches))
        target_logits = model(rgb, modal_x)
        values["adversarial"] = adversarial(target_logits, SOURCE_LABEL)
        loss = LAMBDA_ADV * values["adversarial"]
        if memory is not None:
            values["pseudo_focal"] = segmentation_loss(target_logits, label)
            values["prototype_target"], target_means, target_present = pa.feat_kl_loss(
                features["fused"], label, memory.memory
            )
            loss = loss + LAMBDA_PSEUDO * values["pseudo_focal"] + LAMBDA_PROTOTYPE * values["prototype_target"]
        loss.backward()

        discriminator.requires_grad_(True)
        values["discriminator"] = (
            adversarial(source_logits.detach(), SOURCE_LABEL) + adversarial(target_logits.detach(), TARGET_LABEL)
        ) / 2
        values["discriminator"].backward()

        current = torch.stack([values[name].detach() for name in names])
        if not bool(torch.isfinite(current).all()):
            raise FloatingPointError("adaptation loss is NaN or Inf at iteration {}".format(iteration))
        if not all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in network.parameters()
            if parameter.grad is not None
        ):
            raise FloatingPointError("adaptation gradient is NaN or Inf at iteration {}".format(iteration))
        average_gradients(list(discriminator.parameters()), world_size)
        optimizer.step()
        optimizer_d.step()
        if memory is not None:
            memory.collect(source_means, source_present)
            memory.collect(target_means, target_present)
            memory.maybe_update(iteration)

        running += current
        done = iteration + 1
        if done % args.log_every == 0 or done == args.iterations:
            pa.all_reduce_sum(running)
            steps = args.log_every if done % args.log_every == 0 else done % args.log_every
            if is_writer:
                record = {"iteration": done, "lr": schedule.get_lr(iteration), "seconds": time.time() - started}
                record.update(zip(names, (running / (steps * world_size)).tolist()))
                with (args.output / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            running.zero_()

    if is_writer:
        metadata = {
            "stage": args.stage,
            "iterations": args.iterations,
            "source_checkpoint": str(args.checkpoint),
            "source_epoch": source_epoch,
            "config": str(args.config),
            "x_mode": config.x_mode,
            "relplus_variant": manifest["relplus_variant"],
            "target_cache": str(args.target_cache),
            "target_cache_status": manifest["status"],
            "pseudo_labels": None if args.pseudo_labels is None else str(args.pseudo_labels),
            "target_panoramas": len(samples),
            "world_size": world_size,
            "source_batch_per_gpu": args.source_batch,
            "target_batch_per_gpu": args.target_batch,
            "target_crop": [manifest["size"][0], args.target_crop_width],
            "lr": args.lr,
            "lr_d": args.lr_d,
            "weights": {"adversarial": LAMBDA_ADV, "prototype": LAMBDA_PROTOTYPE, "pseudo": LAMBDA_PSEUDO},
            "memory_init_counts": memory_counts,
            "seconds": time.time() - started,
        }
        torch.save(
            {"epoch": source_epoch, "model": network.state_dict(), "pin2pan": metadata},
            args.output / "checkpoint.pth",
        )
        torch.save(
            {
                "discriminator": discriminator.state_dict(),
                "memory": None if memory is None else memory.memory.cpu(),
            },
            args.output / "adaptation_state.pth",
        )
        (args.output / "pin2pan.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print("saved {}".format(args.output / "checkpoint.pth"), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
