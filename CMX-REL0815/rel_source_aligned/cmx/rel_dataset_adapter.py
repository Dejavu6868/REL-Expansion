import cv2

from utils.transforms import random_crop_pad_to_shape


def apply_shared_spatial_transform(
    rgb, label, rel, scale, crop_size, crop_pos, mirror=False
):
    """Apply one sampled transform to RGB, pre-generated REL, and label."""
    if mirror:
        rgb = cv2.flip(rgb, 1)
        label = cv2.flip(label, 1)
        rel = cv2.flip(rel, 1)
    resize_height = int(rgb.shape[0] * scale)
    resize_width = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(
        label, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST
    )
    rel = cv2.resize(rel, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
    label, _ = random_crop_pad_to_shape(label, crop_pos, crop_size, 255)
    rel, _ = random_crop_pad_to_shape(rel, crop_pos, crop_size, 0)
    return rgb, label, rel

