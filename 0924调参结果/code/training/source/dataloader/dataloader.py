import cv2
import torch
import numpy as np
from torch.utils import data
import random

from rel_plus.integration.cmx_preprocess import (
    apply_cmx_compatible_preprocess,
    sample_spatial_transform,
)
from rel_plus.policy import validate_rel_plus_augmentation_policy
from config import config
from dataloader.data_setting import build_data_setting
from dataloader.samplers import (
    DistributedEvalSamplerNoPad,
    FixedLengthDistributedSampler,
)
from dataloader.profiles import (
    S2D_RELPLUS_COMPARISON_NO_FLIP,
    sample_comparison_transform,
)
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x

def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std, cfg=None, rng=None):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.cfg = config if cfg is None else cfg
        self.rng = random if rng is None else rng
        self.x_mode = getattr(self.cfg, 'x_mode', 'standard')
        self.augmentation_profile = getattr(self.cfg, 'augmentation_profile', None)
        self.train_horizontal_flip = getattr(
            self.cfg, 'train_horizontal_flip', self.x_mode != 'rel_plus_v2_1'
        )
        if self.augmentation_profile == S2D_RELPLUS_COMPARISON_NO_FLIP:
            if self.train_horizontal_flip:
                raise ValueError(
                    '{} is a no-flip profile'.format(
                        S2D_RELPLUS_COMPARISON_NO_FLIP
                    )
                )
        if self.x_mode == 'rel_plus_v2_1':
            validate_rel_plus_augmentation_policy(
                horizontal_flip=self.train_horizontal_flip,
                vertical_flip=getattr(self.cfg, 'train_vertical_flip', False),
                arbitrary_rotation=getattr(
                    self.cfg, 'train_arbitrary_rotation', False
                ),
                perspective_warp=getattr(
                    self.cfg, 'train_perspective_warp', False
                ),
            )
        self.last_transform = None

    def __call__(self, rgb, gt, modal_x, modal_x_valid_mask=None):
        if self.augmentation_profile == S2D_RELPLUS_COMPARISON_NO_FLIP:
            if self.x_mode == 'rel_plus_v2_1' and modal_x_valid_mask is None:
                raise ValueError('REL+ v2.1 requires its diagnostic valid mask')
            if modal_x_valid_mask is not None:
                valid_mask = np.asarray(modal_x_valid_mask, dtype=bool)
            else:
                valid_mask = np.ones(rgb.shape[:2], dtype=bool)
            if valid_mask.shape != rgb.shape[:2]:
                raise ValueError('modal_x valid mask shape mismatch')
            scales = self.cfg.train_scale_array or [1.0]
            transform = sample_comparison_transform(
                rgb.shape[:2],
                scales,
                (self.cfg.image_height, self.cfg.image_width),
                self.rng,
            )
            self.last_transform = transform
            batch = apply_cmx_compatible_preprocess(
                rgb,
                modal_x,
                gt,
                valid_mask,
                transform,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                horizontal_flip=False,
            )
            if self.x_mode == 'rel_plus_v2_1':
                return (
                    batch.rgb,
                    batch.label,
                    batch.modal_x,
                    batch.modal_x_valid_mask,
                )
            return batch.rgb, batch.label, batch.modal_x

        if self.x_mode == 'rel_plus_v2_1':
            if modal_x_valid_mask is None:
                raise ValueError('REL+ v2.1 requires its diagnostic valid mask')
            scales = (
                self.cfg.train_scale_array
                if self.cfg.train_scale_array is not None
                else [1.0]
            )
            transform = sample_spatial_transform(
                rgb.shape[:2],
                scales,
                (self.cfg.image_height, self.cfg.image_width),
                self.rng,
            )
            self.last_transform = transform
            batch = apply_cmx_compatible_preprocess(
                rgb,
                modal_x,
                gt,
                np.asarray(modal_x_valid_mask, dtype=bool),
                transform,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
                horizontal_flip=self.train_horizontal_flip,
            )
            return (
                batch.rgb,
                batch.label,
                batch.modal_x,
                batch.modal_x_valid_mask,
            )

        if self.train_horizontal_flip:
            rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.cfg.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(
                rgb, gt, modal_x, self.cfg.train_scale_array
            )
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (self.cfg.image_height, self.cfg.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        
        return p_rgb, p_gt, p_modal_x

class ValPre(object):
    def __init__(self, x_mode='standard'):
        self.x_mode = x_mode

    def __call__(self, rgb, gt, modal_x, modal_x_valid_mask=None):
        if self.x_mode == 'rel_plus_v2_1':
            if modal_x_valid_mask is None:
                raise ValueError('REL+ validation requires a valid mask')
            if modal_x_valid_mask.shape != modal_x.shape[:2]:
                raise ValueError('REL+ validation mask shape mismatch')
            return rgb, gt, modal_x, modal_x_valid_mask
        return rgb, gt, modal_x

def get_train_loader(engine, dataset, cfg=None):
    selected = config if cfg is None else cfg
    data_setting = build_data_setting(selected, split='train')
    train_preprocess = TrainPre(
        selected.norm_mean, selected.norm_std, cfg=selected
    )
    train_dataset = dataset(data_setting, "train", train_preprocess)
    logical_samples = int(
        getattr(
            selected,
            'logical_samples_per_epoch',
            selected.batch_size * selected.niters_per_epoch,
        )
    )
    train_sampler = FixedLengthDistributedSampler(
        train_dataset,
        logical_samples_per_epoch=logical_samples,
        num_replicas=engine.world_size,
        rank=engine.local_rank,
        shuffle=True,
        seed=selected.seed,
    )
    if selected.batch_size % engine.world_size:
        raise ValueError('global batch_size must be divisible by world_size')
    batch_size = selected.batch_size // engine.world_size

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=selected.num_workers,
                                   drop_last=True,
                                   shuffle=False,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler


def get_val_loader(engine, dataset, cfg=None):
    selected = config if cfg is None else cfg
    data_setting = build_data_setting(selected, split='val')
    val_preprocess = ValPre(x_mode=data_setting['x_mode'])
    val_dataset = dataset(data_setting, 'val', val_preprocess)
    val_sampler = DistributedEvalSamplerNoPad(
        val_dataset,
        num_replicas=engine.world_size,
        rank=engine.local_rank,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=getattr(selected, 'eval_batch_size', 1),
        num_workers=selected.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        sampler=val_sampler,
    )
    return val_loader, val_sampler
