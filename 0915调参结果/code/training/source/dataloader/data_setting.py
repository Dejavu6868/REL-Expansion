"""Single data-setting builder shared by CMX training and evaluation."""


RELPLUS_X_MODE = "rel_plus_v2_1"
RAWDEPTH_X_MODE = "rawdepth_uint8_repeat3"
HHA_X_MODE = "hha_frozen_cache"
RELPLUS_REPRESENTATION = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
RELPLUS_CHANNEL_ORDER = ("EGVIA", "LOA", "ReD")


def _required(config, name):
    if not hasattr(config, name):
        raise ValueError("missing required config field: {}".format(name))
    return getattr(config, name)


def build_data_setting(config, *, split):
    """Build the one RGB-X dataset contract used by both train and val."""
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")

    x_mode = getattr(config, "x_mode", None)
    declares_relplus = (
        getattr(config, "representation_protocol_id", None)
        == RELPLUS_REPRESENTATION
        or bool(getattr(config, "x_valid_root_folder", None))
    )
    if x_mode is None:
        if declares_relplus:
            raise ValueError("REL+ config requires explicit x_mode")
        x_mode = "standard"

    x_valid_root = getattr(config, "x_valid_root_folder", None)
    x_valid_format = getattr(config, "x_valid_format", None)
    if x_mode == RELPLUS_X_MODE:
        if not x_valid_root:
            raise ValueError("REL+ x_valid_root is required")
        if not x_valid_format:
            raise ValueError("REL+ x_valid_format is required")
        channel_order = tuple(
            getattr(config, "channel_order", RELPLUS_CHANNEL_ORDER)
        )
        if channel_order != RELPLUS_CHANNEL_ORDER:
            raise ValueError("REL+ channel_order must be {}".format(
                RELPLUS_CHANNEL_ORDER
            ))
    else:
        channel_order = getattr(config, "channel_order", None)

    if x_mode == RAWDEPTH_X_MODE and not _required(
        config, "x_is_single_channel"
    ):
        raise ValueError("RawDepth uint8 adapter requires x_is_single_channel=True")
    if x_mode == HHA_X_MODE and _required(config, "x_is_single_channel"):
        raise ValueError("HHA frozen cache requires x_is_single_channel=False")

    return {
        "rgb_root": _required(config, "rgb_root_folder"),
        "rgb_format": _required(config, "rgb_format"),
        "gt_root": _required(config, "gt_root_folder"),
        "gt_format": _required(config, "gt_format"),
        "transform_gt": _required(config, "gt_transform"),
        "x_root": _required(config, "x_root_folder"),
        "x_format": _required(config, "x_format"),
        "x_single_channel": _required(config, "x_is_single_channel"),
        "x_mode": x_mode,
        "x_valid_root": x_valid_root,
        "x_valid_format": x_valid_format,
        "channel_order": channel_order,
        "emit_transform_trace": bool(
            getattr(config, "emit_transform_trace", False)
        ),
        "train_horizontal_flip": bool(
            getattr(config, "train_horizontal_flip", False)
        ),
        "train_source": _required(config, "train_source"),
        "eval_source": _required(config, "eval_source"),
        "class_names": _required(config, "class_names"),
    }
