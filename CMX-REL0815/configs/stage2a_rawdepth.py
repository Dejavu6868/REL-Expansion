from configs.stage2a_common import apply_stage2a
from configs.stanford2d3d_b2_common import make_config


config = apply_stage2a(make_config("rawdepth"), "rawdepth", "CMX RawDepth three-channel replication", "not_applicable")
cfg = config
config.config_path = __file__
