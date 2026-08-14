from configs.stage2a_common import apply_stage2a
from configs.cmx_relplus_2d import config


config.relplus_gravity_source = "pose"
apply_stage2a(config, "relplus_pose", "online [ReD, EGVIA, LOA]", "R_w2c @ [0,0,-1]")
cfg = config
config.config_path = __file__
