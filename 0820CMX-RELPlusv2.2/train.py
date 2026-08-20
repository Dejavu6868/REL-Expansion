"""Formal CMX training entry; V2.2 configuration remains fail-closed."""

import argparse
import os
import sys
import time

from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import config
from engine.engine import Engine
from utils.pyt_utils import all_reduce_tensor
from utils.training_protocol import (
    assert_runtime_dataset_contract,
    assert_training_ready,
    set_author_seed,
)
from utils.training_runtime import (
    accumulate_loss,
    build_training_runtime,
    sanitize_author_loss_map,
)


def main():
    parser = argparse.ArgumentParser()
    os.environ.setdefault("MASTER_PORT", "169710")

    # This guard must precede Engine, logger files, model, optimizer and TB.
    assert_training_ready(config)
    assert_runtime_dataset_contract(config, require_cache_audit=True)
    with Engine(custom_parser=parser) as engine:
        runtime = build_training_runtime(config, engine)
        logger = runtime.logger
        logger.info("Architecture: Original CMX")
        logger.info("Backbone: %s", config.backbone)
        logger.info("Gate: %s", config.using_gate)
        logger.info("SMMF: %s", config.using_smmf)
        logger.info("DyMM: %s", config.using_dymm)
        logger.info("Dataset: %s", config.dataset_name)
        logger.info(
            "Split: %s", getattr(config, "dataset_split", "UNSPECIFIED")
        )
        logger.info("Seed: %s", runtime.seed)

        tensorboard = None
        is_writer = (not engine.distributed) or engine.local_rank == 0
        if is_writer:
            tb_dir = config.tb_dir + "/{}".format(
                time.strftime("%b%d_%d-%H-%M", time.localtime())
            )
            tensorboard = SummaryWriter(log_dir=tb_dir)

        if engine.continue_state_object:
            engine.restore_checkpoint()
        runtime.optimizer.zero_grad()
        runtime.model.train()
        logger.info("begin training")
        author_nan_replacement_count = 0

        for epoch in range(engine.state.epoch, config.nepochs + 1):
            set_author_seed(
                config.seed,
                epoch=epoch,
                local_rank=engine.local_rank,
                distributed=engine.distributed,
            )
            runtime.train_sampler.set_epoch(epoch)
            progress = tqdm(
                range(config.niters_per_epoch),
                file=sys.stdout,
                bar_format="{desc}[{elapsed}<{remaining},{rate_fmt}]",
            )
            iterator = iter(runtime.train_loader)
            sum_loss = 0.0

            for index in progress:
                engine.update_iteration(epoch, index)
                minibatch = next(iterator)
                rgb = minibatch["data"].to(runtime.device, non_blocking=True)
                label = minibatch["label"].to(runtime.device, non_blocking=True)
                modal_x = minibatch["modal_x"].to(
                    runtime.device, non_blocking=True
                )
                loss_map = runtime.model(rgb, modal_x, label)
                loss_map, nan_count = sanitize_author_loss_map(loss_map)
                author_nan_replacement_count += nan_count
                loss = loss_map.mean()
                displayed_loss = (
                    all_reduce_tensor(loss, world_size=engine.world_size)
                    if engine.distributed
                    else loss
                )
                runtime.optimizer.zero_grad()
                loss.backward()
                runtime.optimizer.step()

                iteration = (epoch - 1) * config.niters_per_epoch + index
                learning_rate = runtime.scheduler.get_lr(iteration)
                for group in runtime.optimizer.param_groups:
                    group["lr"] = learning_rate
                sum_loss = accumulate_loss(sum_loss, displayed_loss)
                progress.set_description(
                    "Epoch {}/{} Iter {}/{}: lr={:.4e} loss={:.4f} "
                    "total_loss={:.4f}".format(
                        epoch,
                        config.nepochs,
                        index + 1,
                        config.niters_per_epoch,
                        learning_rate,
                        float(displayed_loss.detach().item()),
                        sum_loss / (index + 1),
                    ),
                    refresh=False,
                )

            if tensorboard is not None:
                tensorboard.add_scalar(
                    "train_loss", sum_loss / len(progress), epoch
                )
                tensorboard.add_scalar(
                    "author_nan_replacement_count",
                    author_nan_replacement_count,
                    epoch,
                )
            checkpoint_due = (
                epoch >= config.checkpoint_start_epoch
                and epoch % config.checkpoint_step == 0
            ) or epoch == config.nepochs
            if checkpoint_due and is_writer:
                engine.save_and_link_checkpoint(
                    config.checkpoint_dir,
                    config.log_dir,
                    config.log_dir_link,
                )
        if tensorboard is not None:
            tensorboard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
