"""Fail-closed RGBD comparison arm; X is canonical CMX RawDepth."""

from .comparison_v2_2 import build_comparison_config


config = build_comparison_config("rgbd")
cfg = config
