#!/usr/bin/env python3
"""Run one new experiment arm through the unmodified frozen CMX entrypoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
import types

from suite_common import build_config, configure_imports, digest, dump_json, load_suite


def model_digest(model):
    value = hashlib.sha256()
    base = model.module if hasattr(model, 'module') else model
    for name, tensor in sorted(base.state_dict().items()):
        value.update(name.encode('utf-8'))
        value.update(str(tuple(tensor.shape)).encode('ascii'))
        value.update(str(tensor.dtype).encode('ascii'))
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('rgbd', 'hha', 'relplus'))
    parser.add_argument('--mode', required=True, choices=('smoke', 'formal'))
    parser.add_argument('--port', required=True)
    parser.add_argument('--local_rank', '--local-rank', type=int, required=True)
    args = parser.parse_args()
    suite = load_suite(args.suite)
    configure_imports(suite)
    root = Path(suite['remote_source_root'])
    out = Path(suite['output_root']) / ('smoke' if args.mode == 'smoke' else 'runs') / args.arm
    if not out.is_dir():
        raise RuntimeError('supervisor-created isolated output is required')
    rank = args.local_rank
    started = time.time()
    config = build_config(suite, args.arm, args.mode)
    selected = types.ModuleType('_suite_selected_config')
    selected.config = selected.cfg = config
    sys.modules[selected.__name__] = selected
    sys.modules['config'] = selected
    try:
        import cv2
        import torch
        import torch.distributed as dist
        import torch.nn as nn
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        torch.cuda.set_per_process_memory_fraction(0.75, rank)
        torch.cuda.reset_peak_memory_stats(rank)
        import utils.training_runtime as runtime_module
        original_build = runtime_module.build_training_runtime
        original_sanitize = runtime_module.sanitize_author_loss_map

        def finite_loss_map(loss_map):
            result, count = original_sanitize(loss_map)
            if count:
                raise FloatingPointError('NaN replacement encountered; refusing to continue this new run')
            return result, count

        def recorded_build(*positional, **keyword):
            runtime = original_build(*positional, **keyword)
            if runtime.engine.world_size != 8 or len(runtime.train_loader) != 945:
                raise RuntimeError('world size or batch56 loader length mismatch')
            if len(runtime.train_sampler) != 6615:
                raise RuntimeError('per-rank sample count mismatch')
            if any(group['lr'] != 0.00012 for group in runtime.optimizer.param_groups):
                raise RuntimeError('actual optimizer learning rate mismatch')
            base = runtime.model.module if hasattr(runtime.model, 'module') else runtime.model
            bns = [{'name': name, 'type': type(layer).__name__, 'eps': layer.eps,
                    'momentum': layer.momentum} for name, layer in base.named_modules()
                   if isinstance(layer, (nn.BatchNorm2d, nn.SyncBatchNorm))]
            dump_json(out / 'rank_init_{:02d}.json'.format(rank), {
                'status': 'INITIALIZED', 'suite_id': suite['suite_id'], 'arm': args.arm,
                'mode': args.mode, 'rank': rank, 'world_size': runtime.engine.world_size,
                'pid': os.getpid(), 'started_at': started, 'source_root': str(root / 'source'),
                'train_sha256': digest(root / 'source' / 'train.py'),
                'model_sha256': model_digest(runtime.model),
                'sampler_length': len(runtime.train_sampler), 'loader_length': len(runtime.train_loader),
                'local_batch': runtime.train_loader.batch_size, 'global_batch': config.batch_size,
                'initial_optimizer_lrs': [group['lr'] for group in runtime.optimizer.param_groups],
                'optimizer_betas': [group['betas'] for group in runtime.optimizer.param_groups],
                'optimizer_weight_decays': [group['weight_decay'] for group in runtime.optimizer.param_groups],
                'scheduler_total_iterations': runtime.scheduler.total_iters,
                'scheduler_warmup_steps': runtime.scheduler.warmup_steps,
                'batch_norm_layers': bns, 'cuda_allocator_fraction': 0.75,
                'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                'opencv_threads': cv2.getNumThreads(), 'torch_threads': torch.get_num_threads(),
                'config': config,
            })
            return runtime

        runtime_module.build_training_runtime = recorded_build
        runtime_module.sanitize_author_loss_map = finite_loss_map
        forwarded = ['--local_rank', str(rank), '--port', args.port]
        if args.mode == 'formal':
            entry = root / 'source' / 'train.py'
            sys.argv = [str(entry)] + forwarded
        else:
            entry = root / 'source' / 'tools' / 'run_ddp_optimizer_smoke_v2_3.py'
            sys.argv = [str(entry), '--config-module', selected.__name__,
                        '--output-dir', str(out), '--iterations', '20', '--resume-iterations', '2',
                        '--authorize-ddp-smoke'] + forwarded
            if args.arm == 'relplus':
                sys.argv += ['--cache-audit', config.cache_audit_report,
                             '--training-data-preflight', config.training_data_preflight_report,
                             '--accept-source-compatible-invalid']
            else:
                sys.argv += ['--input-audit', config.input_audit_report]
        try:
            runpy.run_path(str(entry), run_name='__main__')
        except SystemExit as result:
            if result.code not in (0, None):
                raise
        torch.cuda.synchronize(rank)
        dump_json(out / 'rank_done_{:02d}.json'.format(rank), {
            'status': 'COMPLETED', 'suite_id': suite['suite_id'], 'arm': args.arm,
            'mode': args.mode, 'rank': rank, 'pid': os.getpid(), 'started_at': started,
            'ended_at': time.time(), 'epoch': 200 if args.mode == 'formal' else None,
            'global_iteration': 189000 if args.mode == 'formal' else 22,
            'max_memory_allocated': torch.cuda.max_memory_allocated(rank),
            'max_memory_reserved': torch.cuda.max_memory_reserved(rank),
        })
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0
    except BaseException as error:
        dump_json(out / 'rank_error_{:02d}.json'.format(rank), {
            'status': 'FAIL', 'rank': rank, 'arm': args.arm, 'mode': args.mode,
            'pid': os.getpid(), 'time': time.time(), 'error': repr(error),
            'traceback': traceback.format_exc(),
        })
        raise


if __name__ == '__main__':
    raise SystemExit(main())
