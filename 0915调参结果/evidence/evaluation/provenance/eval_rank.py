"""Run the unchanged frozen evaluator with the actual B56 training config."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

TRAIN_BUNDLE = Path('/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-20260915')
sys.dont_write_bytecode = True
sys.path.insert(0, str(TRAIN_BUNDLE))
from suite_common import build_config, configure_imports, dump_json, load_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('rgbd', 'hha', 'relplus'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    report = args.output / ('rank_%02d_runtime.json' % rank)
    started = time.time()
    try:
        import cv2
        import torch
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.cuda.set_per_process_memory_fraction(0.75, int(os.environ['LOCAL_RANK']))
        suite = load_suite(TRAIN_BUNDLE / 'suite.json')
        configure_imports(suite)
        cfg = build_config(suite, args.arm)
        module = types.ModuleType('current_evaluation_config')
        module.config = cfg
        sys.modules[module.__name__] = module
        from tools import eval_rel_plus_v2_3_full as frozen
        from engine import relplus_evaluator as core
        from models import builder
        from dataloader import RGBXDataset
        source = (TRAIN_BUNDLE / 'source').resolve()
        loaded = {}
        if args.arm != 'relplus':
            path = Path(__file__).with_name('input_adapter.py')
            spec = importlib.util.spec_from_file_location('input_adapter', str(path))
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            core.prepare_eval_sample = adapter.prepare_three_arm_eval_sample
            frozen._unbatch = adapter.unbatch_three_arm
            if adapter.frozen_evaluator is not frozen or adapter.evaluator_core is not core:
                raise RuntimeError('adapter imported a different evaluator')
            loaded['adapter'] = str(path)
        for name, item in [('evaluator', frozen), ('core', core),
                           ('model', builder), ('dataset', RGBXDataset)]:
            actual = Path(item.__file__).resolve()
            if source not in actual.parents:
                raise RuntimeError('module escaped frozen snapshot: ' + str(actual))
            loaded[name] = str(actual)
        checkpoint = Path(suite['output_root']) / 'runs' / args.arm / 'checkpoints/epoch-200.pth'
        runtime = {'status': 'RUNNING', 'rank': rank, 'arm': args.arm,
                   'started_at': started, 'loaded_modules': loaded,
                   'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
                   'cudnn_benchmark': torch.backends.cudnn.benchmark,
                   'cudnn_deterministic': torch.backends.cudnn.deterministic,
                   'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                   'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'opencv_threads': cv2.getNumThreads(), 'torch_threads': torch.get_num_threads(),
                   'amp': False, 'checkpoint': str(checkpoint), 'processed_samples': 0}
        dump_json(report, runtime)
        original_load = frozen.load_checkpoint_once
        def checked_load(network, checkpoint, *, expected_epoch):
            epoch = original_load(network, checkpoint, expected_epoch=expected_epoch)
            state = network.state_dict()
            if not all(bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point()):
                raise FloatingPointError('nonfinite model checkpoint tensor')
            runtime['state_key_count'] = len(state)
            runtime['checkpoint_payload_epoch'] = epoch
            runtime['batch_norm'] = [dict(name=n, eps=m.eps, momentum=m.momentum,
                                          type=type(m).__name__)
                                     for n, m in network.named_modules()
                                     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            dump_json(report, runtime)
            return epoch
        frozen.load_checkpoint_once = checked_load
        original_evaluate = core.evaluate_prepared_sample
        def observed_evaluate(*positional, **keywords):
            result = original_evaluate(*positional, **keywords)
            runtime['processed_samples'] += 1
            if runtime['processed_samples'] % 100 == 0:
                runtime['updated_at'] = time.time()
                dump_json(report, runtime)
            return result
        core.evaluate_prepared_sample = observed_evaluate
        sys.argv = [str(frozen.__file__), '--config-module', module.__name__,
                    '--checkpoint', str(checkpoint), '--expected-epoch', '200',
                    '--output', str(args.output / 'evaluation')]
        code = frozen.main()
        runtime.update(status='COMPLETED', exitcode=code, finished_at=time.time())
        dump_json(report, runtime)
        return code
    except BaseException as error:
        dump_json(report, {'status': 'FAIL', 'arm': args.arm, 'rank': rank,
                           'started_at': started, 'finished_at': time.time(),
                           'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    sys.exit(main())
