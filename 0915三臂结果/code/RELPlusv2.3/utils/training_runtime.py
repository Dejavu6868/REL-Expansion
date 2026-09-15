"""Shared construction path used by formal training and no-step startup smoke."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import get_train_loader
from engine.logger import get_logger
from models.builder import EncoderDecoder
from utils.lr_policy import WarmUpPolyLR
from utils.training_protocol import (
    build_author_criterion,
    build_author_optimizer,
    configure_author_cudnn,
    set_author_seed,
)


@dataclass
class TrainingRuntime:
    engine: object
    logger: object
    seed: int
    cudnn: dict
    train_loader: object
    train_sampler: object
    criterion: object
    model: object
    optimizer: object
    scheduler: object
    device: torch.device
    norm_layer: object


def accumulate_loss(total, loss):
    return float(total) + float(loss.detach().item())


def sanitize_author_loss_map(loss_map):
    """Match the REL author branch: replace NaNs before spatial mean."""
    nan_count = int(torch.count_nonzero(torch.isnan(loss_map)).item())
    if nan_count:
        loss_map = torch.nan_to_num(loss_map, nan=0.0)
    return loss_map, nan_count


def build_training_runtime(
    config,
    engine,
    *,
    dataset_class=RGBXDataset,
    model_class=EncoderDecoder,
    device=None,
    wrap_distributed=True,
    norm_layer_override=None
):
    logger = get_logger(config.log_dir, config.log_file)
    cudnn = configure_author_cudnn()
    seed = set_author_seed(
        config.seed,
        local_rank=engine.local_rank,
        distributed=engine.distributed,
    )
    train_loader, train_sampler = get_train_loader(
        engine, dataset_class, cfg=config
    )
    criterion = build_author_criterion(config)
    norm_layer = (
        norm_layer_override
        if norm_layer_override is not None
        else (nn.SyncBatchNorm if engine.distributed else nn.BatchNorm2d)
    )
    model = model_class(
        cfg=config, criterion=criterion, norm_layer=norm_layer
    )
    optimizer = build_author_optimizer(model, norm_layer, config)
    total_iterations = config.nepochs * config.niters_per_epoch
    scheduler = WarmUpPolyLR(
        config.lr,
        config.lr_power,
        total_iterations,
        config.niters_per_epoch * config.warm_up_epoch,
    )
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda", engine.local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    model.to(device)
    if engine.distributed and wrap_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[engine.local_rank] if device.type == "cuda" else None,
            output_device=engine.local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    engine.register_state(
        dataloader=train_loader, model=model, optimizer=optimizer
    )
    return TrainingRuntime(
        engine=engine,
        logger=logger,
        seed=seed,
        cudnn=cudnn,
        train_loader=train_loader,
        train_sampler=train_sampler,
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        norm_layer=norm_layer,
    )
