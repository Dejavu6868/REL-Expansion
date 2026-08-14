from configs.stage2b_common import apply_stage2b
from configs.cmx_relplus_2d import config


config.relplus_gravity_source = "pose"
apply_stage2b(config, "relplus_pose", "online [ReD, EGVIA, LOA]", "R_w2c @ [0,0,-1]")
cfg = config
config.config_path = __file__
