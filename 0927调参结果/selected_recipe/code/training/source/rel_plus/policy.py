"""Frozen augmentation policy for camera-dependent REL+ geometry."""


def validate_rel_plus_augmentation_policy(
    *,
    horizontal_flip=False,
    vertical_flip=False,
    arbitrary_rotation=False,
    perspective_warp=False
):
    policies = {
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
        "arbitrary_rotation": arbitrary_rotation,
        "perspective_warp": perspective_warp,
    }
    for name, enabled in policies.items():
        if enabled:
            raise ValueError("REL+ v2 rejects {}".format(name))
