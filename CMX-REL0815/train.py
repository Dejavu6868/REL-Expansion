import os.path as osp
import os
import csv
import json
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from stage2a.runtime import append_epoch_metrics, load_common_initial_model, seed_everything

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

if (
    getattr(config, 'x_mode', 'precomputed') == 'rel_source_aligned'
    and not getattr(config, 'training_authorized', False)
):
    raise RuntimeError(
        'rel_source_aligned is review-only: training_authorized=false; '
        'wait for explicit user approval'
    )


def write_runtime_topology(path, topology):
    temporary = '{}.{}.tmp'.format(path, os.getpid())
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(topology, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    seed = config.seed
    reference_group = None
    reference_shards = 1
    reference_group_ranks = None
    reference_world_size = 1
    topology = None
    topology_path = None
    if engine.distributed:
        expected_world_size = getattr(config, 'physical_world_size', engine.world_size)
        if engine.world_size != expected_world_size:
            raise ValueError(
                'physical world size mismatch: expected {}, got {}'.format(
                    expected_world_size, engine.world_size
                )
            )
        rank_seeds = getattr(config, 'physical_rank_seeds', None)
        if rank_seeds is None or len(rank_seeds) != engine.world_size:
            raise ValueError('physical_rank_seeds must define one seed per physical rank')
        seed = rank_seeds[engine.local_rank]

        reference_world_size = getattr(config, 'reference_world_size', engine.world_size)
        if engine.world_size % reference_world_size != 0:
            raise ValueError('physical world size must be divisible by reference world size')
        reference_shards = engine.world_size // reference_world_size
        if reference_shards > 1:
            calculated_pairs = [
                [
                    reference_rank + shard * reference_world_size
                    for shard in range(reference_shards)
                ]
                for reference_rank in range(reference_world_size)
            ]
            if calculated_pairs != getattr(config, 'reference_rank_pairs', None):
                raise ValueError('calculated reference groups do not match config')
            for ranks in calculated_pairs:
                group = dist.new_group(ranks=ranks)
                if engine.local_rank in ranks:
                    reference_group = group
                    reference_group_ranks = ranks
    config.runtime_rank_seed = seed
    seed_everything(seed, deterministic=getattr(config, 'deterministic_training', False))

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
    if engine.distributed:
        expected_batch_sizes = getattr(config, 'physical_rank_batch_sizes', None)
        if expected_batch_sizes is None or len(expected_batch_sizes) != engine.world_size:
            raise ValueError('physical_rank_batch_sizes must cover every physical rank')
        if train_loader.batch_size != expected_batch_sizes[engine.local_rank]:
            raise ValueError(
                'physical-rank batch mismatch: expected {}, got {}'.format(
                    expected_batch_sizes[engine.local_rank], train_loader.batch_size
                )
            )
        if len(train_loader) != config.niters_per_epoch:
            raise ValueError(
                'loader step mismatch: expected {}, got {}'.format(
                    config.niters_per_epoch, len(train_loader)
                )
            )
        visible_gpu_ids = [
            int(value) for value in os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        ]
        if visible_gpu_ids != getattr(config, 'physical_gpu_ids', None):
            raise ValueError('CUDA_VISIBLE_DEVICES does not match physical_gpu_ids')
        inventory_path = os.environ.get('CMX_GPU_INVENTORY')
        if not inventory_path or not osp.isfile(inventory_path):
            raise ValueError('CMX_GPU_INVENTORY must identify the audited GPU inventory')
        inventory = {}
        with open(inventory_path, newline='', encoding='utf-8') as handle:
            for row in csv.reader(handle):
                gpu_id = int(row[0].strip())
                inventory[gpu_id] = {
                    'uuid': row[1].strip(),
                    'name': row[2].strip(),
                    'memory_total_mib': int(row[3].strip()),
                }
        physical_gpu_id = visible_gpu_ids[engine.local_rank]
        if physical_gpu_id not in inventory:
            raise ValueError('physical GPU is absent from the audited inventory')
        topology_status = osp.join(config.log_dir, 'status')
        os.makedirs(topology_status, exist_ok=True)
        topology_path = osp.join(
            topology_status, 'topology_rank_{}.json'.format(engine.local_rank)
        )
        topology = {
            'physical_rank': engine.local_rank,
            'physical_world_size': engine.world_size,
            'physical_gpu_id': physical_gpu_id,
            'physical_gpu_uuid': inventory[physical_gpu_id]['uuid'],
            'reference_rank': engine.local_rank % reference_world_size,
            'reference_world_size': reference_world_size,
            'reference_group_ranks': reference_group_ranks,
            'local_batch_size': train_loader.batch_size,
            'sampler_samples': len(train_sampler),
            'loader_steps': len(train_loader),
            'global_batch_size': config.batch_size,
            'rank_seed': seed,
            'sampler_class': type(train_sampler).__name__,
            'loss_weighting': '2 * local_cross_entropy_sum / paired_valid_pixels',
            'ignore_index': config.background,
            'pid': os.getpid(),
            'gpu_inventory': osp.abspath(inventory_path),
            'first_optimizer_step_completed': False,
            'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime()),
        }
        write_runtime_topology(topology_path, topology)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion_reduction = 'sum' if reference_group is not None else 'mean'
    criterion = nn.CrossEntropyLoss(
        reduction=criterion_reduction, ignore_index=config.background
    )

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    initialization_report = load_common_initial_model(
        model, getattr(config, 'common_initial_model', None)
    )
    if (not engine.distributed) or engine.local_rank == 0:
        initialization_report_path = osp.join(config.log_dir, 'common_initial_load.json')
        with open(initialization_report_path, 'w', encoding='utf-8') as handle:
            json.dump(initialization_report, handle, indent=2, sort_keys=True)
            handle.write('\n')
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logger.info('begin trainning:')
    
    for epoch in range(engine.state.epoch, config.nepochs+1):
        epoch_started = time.perf_counter()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        sum_data_time = 0.0
        sum_iteration_time = 0.0
        previous_iteration_end = time.perf_counter()

        for idx in pbar:
            iteration_started = time.perf_counter()
            sum_data_time += iteration_started - previous_iteration_end
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)
            backward_loss = loss

            # reduce the whole loss over multi-gpu
            if reference_group is not None:
                local_valid_pixels = (gts != config.background).sum().to(dtype=loss.dtype)
                reference_valid_pixels = local_valid_pixels.clone()
                dist.all_reduce(reference_valid_pixels, group=reference_group)
                if reference_valid_pixels.item() <= 0:
                    raise ValueError('reference-rank batch has no valid target pixels')
                backward_loss = loss * (reference_shards / reference_valid_pixels)
                reduce_loss = all_reduce_tensor(
                    backward_loss.detach(), world_size=engine.world_size
                )
            elif engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            backward_loss.backward()
            optimizer.step()
            if engine.distributed and not topology['first_optimizer_step_completed']:
                topology['first_optimizer_step_completed'] = True
                topology['first_optimizer_step_epoch'] = epoch
                topology['first_optimizer_step_iteration'] = idx
                topology['recorded_at'] = time.strftime(
                    '%Y-%m-%dT%H:%M:%S%z', time.localtime()
                )
                write_runtime_topology(topology_path, topology)

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

            del loss, backward_loss
            pbar.set_description(print_str, refresh=False)
            iteration_ended = time.perf_counter()
            sum_iteration_time += iteration_ended - iteration_started
            previous_iteration_end = iteration_ended
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            append_epoch_metrics(config.metrics_csv, {
                'epoch': epoch,
                'train_loss': sum_loss / len(pbar),
                'learning_rate': lr,
                'mean_data_time_seconds': sum_data_time / len(pbar),
                'mean_iteration_time_seconds': sum_iteration_time / len(pbar),
                'epoch_time_seconds': time.perf_counter() - epoch_started,
                'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            })

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
