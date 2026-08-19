import cv2
import torch
import numpy as np
from torch.utils import data
import random
import sys
from pathlib import Path

RELPLUS_ROOT = Path(__file__).resolve().parents[2]
if str(RELPLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(RELPLUS_ROOT))

from rel_plus.integration.cmx_preprocess import (
    apply_cmx_compatible_preprocess,
    sample_spatial_transform,
)
from rel_plus.policy import validate_rel_plus_augmentation_policy
from config import config
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
        self.rng = np.random if rng is None else rng
        self.x_mode = getattr(self.cfg, 'x_mode', 'standard')
        self.train_horizontal_flip = getattr(
            self.cfg, 'train_horizontal_flip', self.x_mode != 'rel_plus_v2_1'
        )
        if self.x_mode == 'rel_plus_v2_1':
            validate_rel_plus_augmentation_policy(
                horizontal_flip=self.train_horizontal_flip,
                vertical_flip=False,
                arbitrary_rotation=False,
                perspective_warp=False,
            )
        self.last_transform = None

    def __call__(self, rgb, gt, modal_x, modal_x_valid_mask=None):
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
    def __call__(self, rgb, gt, modal_x):
        return rgb, gt, modal_x

def get_train_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'x_mode': getattr(config, 'x_mode', 'standard'),
                    'x_valid_root': getattr(config, 'x_valid_root_folder', None),
                    'x_valid_format': getattr(config, 'x_valid_format', '.png'),
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, cfg=config)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
