"""Frozen augmentation policy for camera-dependent REL+ geometry."""


def validate_rel_plus_augmentation_policy(horizontal_flip=False):
    if horizontal_flip:
        raise ValueError(
            "REL+ v1 rejects horizontal flip because it changes camera geometry"
        )
