from configs.stage2b_common import apply_stage2b
from configs.stanford2d3d_b2_common import make_config


config = apply_stage2b(make_config("hha"), "hha", "CMX HHA cache", "HHA baseline EstGravity")
cfg = config
config.config_path = __file__
