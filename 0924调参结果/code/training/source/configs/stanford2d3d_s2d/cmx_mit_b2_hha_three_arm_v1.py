"""Fail-closed CMX-HHA arm for the S2D three-arm comparison."""

from .comparison_three_arm_v1 import build_three_arm_config


config = build_three_arm_config("hha")
cfg = config
