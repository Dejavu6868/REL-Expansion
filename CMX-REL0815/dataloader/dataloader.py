import cv2
import torch
import numpy as np
from torch.utils import data
import random
import json
from config import config
from dataloader.distributed_batch import SplitLogicalDistributedSampler
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize
from relplus.geometry import load_camera_metadata
from relplus.pipeline import (
    SpatialTransformParameters,
    generate_relplus_from_depth,
    generate_relplus_from_depth_local,
    transform_depth_geometry,
)
from rel_source_aligned.adapters.stanford2d3d_perspective_adapter import (
    PerspectiveInputAdapter,
)
from rel_source_aligned.cmx.rel_dataset_adapter import apply_shared_spatial_transform

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
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, rgb, gt, modal_x):
        if getattr(config, 'train_horizontal_flip', True):
            rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, config.train_scale_array)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        
        return p_rgb, p_gt, p_modal_x


def build_relplus_parameters(rgb_shape, scale, crop_size, crop_pos):
    resize_height = int(rgb_shape[0] * scale)
    resize_width = int(rgb_shape[1] * scale)
    crop_y, crop_x = crop_pos
    target_height, target_width = crop_size
    crop_height = min(target_height, resize_height - crop_y)
    crop_width = min(target_width, resize_width - crop_x)
    pad_height = target_height - crop_height
    pad_width = target_width - crop_width
    return SpatialTransformParameters(
        resize_height=resize_height,
        resize_width=resize_width,
        crop_y=crop_y,
        crop_x=crop_x,
        crop_height=crop_height,
        crop_width=crop_width,
        pad_top=pad_height // 2,
        pad_bottom=pad_height // 2 + pad_height % 2,
        pad_left=pad_width // 2,
        pad_right=pad_width // 2 + pad_width % 2,
        flip=False,
    )


class RelPlusTrainPre(object):
    """Shared RGB/label transform followed by REL+ regeneration from transformed depth."""

    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.last_parameters = None
        self.last_gravity_source = None

    @staticmethod
    def _load_intrinsics_only(path):
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
        intrinsics = np.asarray(payload['camera_k_matrix'], dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError('camera_k_matrix must be finite 3x3: {}'.format(path))
        return intrinsics

    def apply_with_parameters(self, rgb, gt, raw_depth, pose_path, scale, crop_pos):
        if getattr(config, 'train_horizontal_flip', False):
            raise ValueError('horizontal flip must be disabled for all four experiment arms')
        crop_size = (config.image_height, config.image_width)
        parameters = build_relplus_parameters(rgb.shape[:2], scale, crop_size, crop_pos)
        resized_rgb = cv2.resize(
            rgb, (parameters.resize_width, parameters.resize_height), interpolation=cv2.INTER_LINEAR
        )
        resized_gt = cv2.resize(
            gt, (parameters.resize_width, parameters.resize_height), interpolation=cv2.INTER_NEAREST
        )
        normalized_rgb = normalize(resized_rgb, self.norm_mean, self.norm_std)
        p_rgb, _ = random_crop_pad_to_shape(normalized_rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(resized_gt, crop_pos, crop_size, 255)

        gravity_source = getattr(config, 'relplus_gravity_source', 'pose')
        if gravity_source == 'local':
            source_k = self._load_intrinsics_only(pose_path)
            rotation = None
        elif gravity_source == 'pose':
            camera = load_camera_metadata(pose_path)
            source_k = camera.k
            rotation = camera.r_world_to_camera
        else:
            raise ValueError('unknown REL+ gravity source: {}'.format(gravity_source))
        depth = raw_depth.astype(np.float64) / 512.0
        depth_valid = (raw_depth != 65535) & (raw_depth > 0) & np.isfinite(depth)
        transformed_depth, transformed_valid, transformed_k = transform_depth_geometry(
            depth, depth_valid, source_k, parameters
        )
        if gravity_source == 'local':
            relplus, rel_valid, _ = generate_relplus_from_depth_local(
                transformed_depth, transformed_valid, transformed_k,
                normal_radius=getattr(config, 'relplus_native_normal_radius', 3),
            )
        else:
            relplus, rel_valid, _ = generate_relplus_from_depth(
                transformed_depth, transformed_valid, transformed_k, rotation,
                normal_radius=getattr(config, 'relplus_native_normal_radius', 3),
            )
        normalized_relplus = normalize(relplus, self.norm_mean, self.norm_std)
        normalized_relplus[~rel_valid] = 0.0
        self.last_parameters = parameters
        self.last_gravity_source = gravity_source
        return (
            p_rgb.transpose(2, 0, 1),
            p_gt,
            normalized_relplus.transpose(2, 0, 1),
        )

    def __call__(self, rgb, gt, raw_depth, pose_path):
        scale = random.choice(config.train_scale_array) if config.train_scale_array is not None else 1.0
        scaled_shape = (int(rgb.shape[0] * scale), int(rgb.shape[1] * scale))
        crop_pos = generate_random_crop_pos(
            scaled_shape, (config.image_height, config.image_width)
        )
        return self.apply_with_parameters(rgb, gt, raw_depth, pose_path, scale, crop_pos)


class SourceAlignedRELTrainPre(object):
    """Generate source-aligned REL first, then transform RGB/REL/label together."""

    def __init__(
        self,
        norm_mean,
        norm_std,
        adapter=None,
        target_size=None,
        scale_array=None,
        horizontal_flip=None,
    ):
        authority_root = getattr(config, 'rel_authority_root', None)
        self.adapter = adapter or PerspectiveInputAdapter(authority_root)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.target_size = target_size or (config.image_height, config.image_width)
        self.scale_array = scale_array if scale_array is not None else config.train_scale_array
        self.horizontal_flip = (
            horizontal_flip
            if horizontal_flip is not None
            else getattr(config, 'train_horizontal_flip', False)
        )
        self.alpha = getattr(config, 'rel_source_alpha_degrees', 45.0)
        self.lam = getattr(config, 'rel_source_lambda', 0.5)
        self.last_stage_order = None
        self.last_generation_shape = None
        self.last_impl = None

    def apply_with_parameters(
        self, rgb, gt, raw_depth, pose_path, scale, crop_pos, mirror=False
    ):
        camera_matrix = RelPlusTrainPre._load_intrinsics_only(pose_path)
        encoded = self.adapter.encode(
            raw_depth, camera_matrix, alpha=self.alpha, lam=self.lam
        )
        canonical_rel = encoded.rel
        self.last_generation_shape = canonical_rel.shape[:2]

        rgb, gt, rel = apply_shared_spatial_transform(
            rgb,
            gt,
            canonical_rel,
            scale=scale,
            crop_size=self.target_size,
            crop_pos=crop_pos,
            mirror=mirror,
        )
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        rel = normalize(rel, self.norm_mean, self.norm_std)
        self.last_stage_order = (
            'source_aligned_rel_generation',
            'shared_spatial_transform',
            'normalization',
        )
        self.last_impl = 'official_source'
        return rgb.transpose(2, 0, 1), gt, rel.transpose(2, 0, 1)

    def __call__(self, rgb, gt, raw_depth, pose_path):
        scale = random.choice(self.scale_array) if self.scale_array is not None else 1.0
        scaled_shape = (int(rgb.shape[0] * scale), int(rgb.shape[1] * scale))
        crop_pos = generate_random_crop_pos(scaled_shape, self.target_size)
        mirror = self.horizontal_flip and random.random() >= 0.5
        return self.apply_with_parameters(
            rgb, gt, raw_depth, pose_path, scale, crop_pos, mirror=mirror
        )


class SourceAlignedRELValPre(object):
    def __init__(self, adapter=None):
        authority_root = getattr(config, 'rel_authority_root', None)
        self.adapter = adapter or PerspectiveInputAdapter(authority_root)
        self.alpha = getattr(config, 'rel_source_alpha_degrees', 45.0)
        self.lam = getattr(config, 'rel_source_lambda', 0.5)

    def __call__(self, rgb, gt, raw_depth, pose_path):
        camera_matrix = RelPlusTrainPre._load_intrinsics_only(pose_path)
        rel = self.adapter.encode(
            raw_depth, camera_matrix, alpha=self.alpha, lam=self.lam
        ).rel
        return rgb, gt, rel

class ValPre(object):
    def __call__(self, rgb, gt, modal_x, pose_path=None):
        if pose_path is None:
            return rgb, gt, modal_x
        gravity_source = getattr(config, 'relplus_gravity_source', 'pose')
        if gravity_source == 'local':
            source_k = RelPlusTrainPre._load_intrinsics_only(pose_path)
            rotation = None
        elif gravity_source == 'pose':
            camera = load_camera_metadata(pose_path)
            source_k = camera.k
            rotation = camera.r_world_to_camera
        else:
            raise ValueError('unknown REL+ gravity source: {}'.format(gravity_source))
        depth = modal_x.astype(np.float64) / 512.0
        valid = (modal_x != 65535) & (modal_x > 0) & np.isfinite(depth)
        parameters = build_relplus_parameters(
            rgb.shape[:2], 1.0, rgb.shape[:2], (0, 0)
        )
        transformed_depth, transformed_valid, transformed_k = transform_depth_geometry(
            depth, valid, source_k, parameters
        )
        if gravity_source == 'local':
            relplus, _, _ = generate_relplus_from_depth_local(
                transformed_depth, transformed_valid, transformed_k,
                normal_radius=getattr(config, 'relplus_native_normal_radius', 3),
            )
        else:
            relplus, _, _ = generate_relplus_from_depth(
                transformed_depth, transformed_valid, transformed_k, rotation,
                normal_radius=getattr(config, 'relplus_native_normal_radius', 3),
            )
        return rgb, gt, relplus

def get_train_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'x_mode': getattr(config, 'x_mode', 'precomputed'),
                    'rel_impl': getattr(config, 'rel_impl', None),
                    'x_online_relplus': getattr(config, 'x_online_relplus', False),
                    'depth_root': getattr(config, 'depth_root_folder', None),
                    'depth_format': getattr(config, 'depth_format', '.png'),
                    'pose_root': getattr(config, 'pose_root_folder', None),
                    'pose_format': getattr(config, 'pose_format', '.json'),
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    if getattr(config, 'x_mode', 'precomputed') == 'rel_source_aligned':
        train_preprocess = SourceAlignedRELTrainPre(config.norm_mean, config.norm_std)
    elif getattr(config, 'x_online_relplus', False):
        train_preprocess = RelPlusTrainPre(config.norm_mean, config.norm_std)
    else:
        train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        reference_world_size = getattr(config, "reference_world_size", engine.world_size)
        if engine.world_size == reference_world_size:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            batch_size = config.batch_size // engine.world_size
        else:
            train_sampler = SplitLogicalDistributedSampler(
                train_dataset,
                global_batch_size=config.batch_size,
                num_steps=config.niters_per_epoch,
                num_replicas=engine.world_size,
                rank=engine.local_rank,
                reference_replicas=reference_world_size,
            )
            batch_size = train_sampler.local_batch_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   worker_init_fn=_seed_worker,
                                   generator=_loader_generator())

    return train_loader, train_sampler


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_generator():
    generator = torch.Generator()
    generator.manual_seed(int(getattr(config, 'runtime_rank_seed', getattr(config, 'seed', 0))))
    return generator
