import os
import os.path as osp
import time
import argparse
import random

import numpy as np
import torch
import torch.distributed as dist

from .logger import get_logger
from utils.pyt_utils import load_model, parse_devices, extant_file, link_file, ensure_dir

logger = get_logger()

class State(object):
    def __init__(self):
        self.epoch = 1
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.sampler = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ['epoch', 'iteration', 'dataloader', 'model',
                         'optimizer', 'sampler']
            setattr(self, k, v)
            

class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info(
            "PyTorch Version {}".format(torch.__version__))
        self.state = State()
        self.devices = None
        self.distributed = False
        self.local_rank = 0
        self.world_size = 1

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
        
        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            os.environ['MASTER_PORT'] = self.args.port
            dist.init_process_group(backend="nccl", world_size=self.world_size, init_method='env://')
            self.devices = [i for i in range(self.world_size)]
        else:
            self.devices = (
                parse_devices(self.args.devices)
                if self.args.devices
                else ([0] if torch.cuda.is_available() else [])
            )


    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('--local_rank', '--local-rank', default=0, type=int,
                       help='process rank on node')
        p.add_argument('-p', '--port', type=str,
                       default='16005',
                       dest="port",
                       help='port for init_process_group')

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path, rank_runtime_states=None):
        logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        state_dict = {}

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v
        state_dict['model'] = new_state_dict
        state_dict['optimizer'] = self.state.optimizer.state_dict()
        state_dict['epoch'] = self.state.epoch
        state_dict['iteration'] = self.state.iteration
        if rank_runtime_states is not None:
            state_dict['rank_runtime_states'] = rank_runtime_states

        t_iobegin = time.time()
        temporary = '{}.tmp-{}'.format(path, os.getpid())
        torch.save(state_dict, temporary)
        os.replace(temporary, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        logger.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare checkpoint: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin))
    
    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)


    def save_and_link_checkpoint(
        self,
        checkpoint_dir,
        log_dir,
        log_dir_link,
        rank_runtime_states=None,
    ):
        ensure_dir(checkpoint_dir)
        self.validate_log_link_paths(log_dir, log_dir_link)
        if log_dir_link and not osp.exists(log_dir_link):
            link_file(log_dir, log_dir_link)
        current_epoch_checkpoint = osp.join(checkpoint_dir, 'epoch-{}.pth'.format(
            self.state.epoch))
        if rank_runtime_states is None:
            self.save_checkpoint(current_epoch_checkpoint)
        else:
            self.save_checkpoint(current_epoch_checkpoint, rank_runtime_states)
        last_epoch_checkpoint = osp.join(checkpoint_dir, 'epoch-last.pth')
        link_file(current_epoch_checkpoint, last_epoch_checkpoint)

    def collect_rank_runtime_states(self):
        sampler = self.state.sampler
        rank_state = {
            'rank': int(self.local_rank),
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': (
                torch.cuda.get_rng_state(self.local_rank)
                if torch.cuda.is_available()
                else None
            ),
            'sampler': {
                'seed': int(getattr(sampler, 'seed', 0)),
                'epoch': int(getattr(sampler, 'epoch', 0)),
            },
        }
        if not self.distributed:
            return [rank_state]
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, rank_state)
        return gathered

    def save_recovery_checkpoint(
        self,
        checkpoint_dir,
        *,
        step,
        rank_runtime_states,
    ):
        ensure_dir(checkpoint_dir)
        step = int(step)
        if step <= 0 or self.state.epoch % step:
            raise ValueError('recovery checkpoint epoch must be divisible by step')
        slot = 'odd' if (self.state.epoch // step) % 2 else 'even'
        path = osp.join(checkpoint_dir, 'recovery_epoch_{}.pth'.format(slot))
        self.save_checkpoint(path, rank_runtime_states)
        return path

    @staticmethod
    def validate_log_link_paths(log_dir, log_dir_link):
        if log_dir_link and osp.abspath(log_dir) == osp.abspath(log_dir_link):
            raise ValueError("refusing a self-referential log directory link")


    def restore_checkpoint(self):
        t_start = time.time()
        if self.distributed:
            # load the model on cpu first to avoid GPU RAM surge
            # when loading a model checkpoint
            # tmp = torch.load(self.continue_state_object,
            #                  map_location=lambda storage, loc: storage.cuda(
            #                      self.local_rank))
            tmp = torch.load(self.continue_state_object, map_location=torch.device('cpu'))
        else:
            tmp = torch.load(self.continue_state_object)
        t_ioend = time.time()
        self.state.model = load_model(self.state.model, tmp['model'], is_restore=True)
        self.state.optimizer.load_state_dict(tmp['optimizer'])
        self.state.epoch = tmp['epoch'] + 1
        self.state.iteration = tmp['iteration']
        rank_states = tmp.get('rank_runtime_states')
        if rank_states:
            matches = [
                value
                for value in rank_states
                if int(value.get('rank', -1)) == int(self.local_rank)
            ]
            if len(matches) != 1:
                raise RuntimeError('checkpoint rank runtime state is incomplete')
            runtime_state = matches[0]
            random.setstate(runtime_state['python_rng_state'])
            np.random.set_state(runtime_state['numpy_rng_state'])
            torch.set_rng_state(runtime_state['torch_rng_state'])
            if torch.cuda.is_available() and runtime_state['cuda_rng_state'] is not None:
                torch.cuda.set_rng_state(
                    runtime_state['cuda_rng_state'], self.local_rank
                )
            if self.state.sampler is not None:
                sampler_state = runtime_state['sampler']
                if int(getattr(self.state.sampler, 'seed', -1)) != int(
                    sampler_state['seed']
                ):
                    raise RuntimeError('checkpoint sampler seed mismatch')
                self.state.sampler.set_epoch(int(sampler_state['epoch']))
        del tmp
        t_end = time.time()
        logger.info(
            "Load checkpoint from file {}, "
            "Time usage:\n\tIO: {}, restore checkpoint: {}".format(
                self.continue_state_object, t_ioend - t_start, t_end - t_ioend))


    def __enter__(self):
        return self


    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False
