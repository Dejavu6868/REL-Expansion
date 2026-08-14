import os
from pickletools import uint8
import cv2
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._x_mode = setting.get('x_mode', 'precomputed')
        self._rel_impl = setting.get('rel_impl')
        self._x_online_relplus = setting.get('x_online_relplus', False)
        self._depth_path = setting.get('depth_root')
        self._depth_format = setting.get('depth_format', '.png')
        self._pose_path = setting.get('pose_root')
        self._pose_format = setting.get('pose_format', '.json')
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]
        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        gt_path = os.path.join(self._gt_path, item_name + self._gt_format)

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt) 

        if self._x_mode == 'rel_source_aligned':
            if self._rel_impl != 'official_source':
                raise ValueError("rel_source_aligned requires rel_impl=official_source")
            depth_path = os.path.join(self._depth_path, item_name + self._depth_format)
            pose_path = os.path.join(self._pose_path, item_name + self._pose_format)
            raw_depth = self._open_image(depth_path, cv2.IMREAD_UNCHANGED)
            if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
                raise ValueError("source-aligned REL requires uint16 Z-depth: {}".format(depth_path))
            if not os.path.isfile(pose_path):
                raise FileNotFoundError("camera metadata could not be read: {}".format(pose_path))
            if self.preprocess is None:
                raise ValueError("source-aligned REL requires its dedicated preprocess")
            rgb, gt, x = self.preprocess(rgb, gt, raw_depth, pose_path)
        elif self._x_online_relplus:
            depth_path = os.path.join(self._depth_path, item_name + self._depth_format)
            pose_path = os.path.join(self._pose_path, item_name + self._pose_format)
            raw_depth = self._open_image(depth_path, cv2.IMREAD_UNCHANGED)
            if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
                raise ValueError("online REL+ requires uint16 Z-depth: {}".format(depth_path))
            if not os.path.isfile(pose_path):
                raise FileNotFoundError("pose could not be read: {}".format(pose_path))
            if self.preprocess is None:
                raise ValueError("online REL+ requires a geometry-aware preprocess")
            rgb, gt, x = self.preprocess(rgb, gt, raw_depth, pose_path)
        else:
            x_path = os.path.join(self._x_path, item_name + self._x_format)
            if self._x_single_channel:
                x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
                x = cv2.merge([x, x, x])
            else:
                x = self._open_image(x_path, cv2.COLOR_BGR2RGB)
            if self.preprocess is not None:
                rgb, gt, x = self.preprocess(rgb, gt, x)

        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        raw = cv2.imread(filepath, mode)
        if raw is None:
            raise FileNotFoundError("image could not be read: {}".format(filepath))
        img = np.array(raw, dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
