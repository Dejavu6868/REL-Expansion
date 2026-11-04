"""Single-channel REL+ ablation: one channel of the REL+ encoding, replicated three times, as the X input.

REL-SF4PASS Table 5 trains Original CMX with one REL channel as X (RGB-ReD, RGB-EGVIA, RGB-LOA).
CMX feeds a one-channel X by copying it into all three input channels (``x_is_single_channel``:
``cv2.merge([x, x, x])``; the frozen ``rawdepth_uint8_repeat3`` mode does the same), so the network,
stem and normalisation stay those of the three-channel arm.

The config key ``single_channel`` (set by ``train_single_channel.py``) names the channel. The cache
is the full three-channel one and is not changed: the channel is copied after each REL+ PNG is read
(training and the pinhole eval, ``install_loader``) and after the ERP ``getREL`` (panorama eval,
``wrap_get_rel``). Invalid pixels are 255 in every channel, so they stay 255.
"""

import functools
import os

import numpy as np

CHANNELS = ("EGVIA", "LOA", "ReD")  # the frozen REL+ channel order, also the ERP getREL output order
CONFIG_KEY = "single_channel"


def channel_of(config):
    """The config's single channel, or None for the full three-channel arm."""
    name = config.get(CONFIG_KEY)
    if name is not None and name not in CHANNELS:
        raise ValueError("{} must be one of {}, not {!r}".format(CONFIG_KEY, CHANNELS, name))
    return name


def replicate(x, name):
    """Channel ``name`` of an H x W x 3 REL+ image, copied into all three channels."""
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("REL+ input must be H x W x 3, got {}".format(x.shape))
    index = CHANNELS.index(name)
    return np.repeat(x[:, :, index:index + 1], 3, axis=2)


def install_loader(config, dataset_class):
    """Make the frozen RGBXDataset replicate the channel of every PNG it reads under ``config.x_root_folder``."""
    name = channel_of(config)
    if name is None:
        return None
    if config.x_mode != "rel_plus_v2_1" or tuple(config.channel_order) != CHANNELS:
        raise ValueError("single-channel REL+ needs x_mode rel_plus_v2_1 with channel order {}".format(CHANNELS))
    root = os.path.join(os.path.abspath(config.x_root_folder), "")
    original = dataset_class._open_image
    if getattr(original, "single_channel", None) is not None:
        raise RuntimeError("the loader already replicates {}".format(original.single_channel))

    def open_image(filepath, *args, **kwargs):
        image = original(filepath, *args, **kwargs)
        if os.path.abspath(filepath).startswith(root):
            image = replicate(image, name)
        return image

    open_image.single_channel = name
    dataset_class._open_image = staticmethod(open_image)
    return name


def wrap_get_rel(get_rel, name):
    """The panorama ``getREL`` with its output replaced by channel ``name``, replicated."""

    @functools.wraps(get_rel)
    def single(*args, **kwargs):
        return replicate(get_rel(*args, **kwargs), name)

    return single
