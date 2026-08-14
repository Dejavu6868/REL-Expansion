"""Dataset and lossless REL+ PNG I/O helpers."""

import os

import cv2
import numpy as np


def read_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def validate_disjoint_splits(train_ids, test_ids):
    overlap = sorted(set(train_ids).intersection(test_ids))
    if overlap:
        raise ValueError("train/test overlap: {}".format(overlap[:10]))


def resolve_sample_paths(dataset_root, sample_id):
    return {
        "rgb": os.path.join(dataset_root, "RGB", sample_id + ".png"),
        "depth": os.path.join(dataset_root, "Depth16", sample_id + ".png"),
        "label": os.path.join(dataset_root, "Label", sample_id + ".png"),
        "pose": os.path.join(dataset_root, "Pose", sample_id + ".json"),
    }


def write_relplus_png(path, relplus):
    array = np.asarray(relplus)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("REL+ image must be HxWx3 uint8 in [ReD, EGVIA, LOA] order")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp.png"
    if not cv2.imwrite(temporary, array):
        raise OSError("failed to write {}".format(temporary))
    os.replace(temporary, path)


def read_relplus_png(path):
    array = cv2.imread(path, cv2.IMREAD_COLOR)
    if array is None:
        raise OSError("failed to read {}".format(path))
    return array


def apply_synchronized_spatial_transform(
    rgb, relplus, label, resize_shape=None, flip=False, crop=None
):
    """Mirror the CMX continuous/label interpolation and transform ordering."""

    rgb = np.asarray(rgb)
    relplus = np.asarray(relplus)
    label = np.asarray(label)
    if resize_shape is not None:
        height, width = resize_shape
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        relplus = cv2.resize(relplus, (width, height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
    if flip:
        rgb = cv2.flip(rgb, 1)
        relplus = cv2.flip(relplus, 1)
        label = cv2.flip(label, 1)
    if crop is not None:
        left, top, width, height = crop
        rgb = rgb[top : top + height, left : left + width]
        relplus = relplus[top : top + height, left : left + width]
        label = label[top : top + height, left : left + width]
    return rgb, relplus, label
