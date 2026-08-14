from configs.stage2b_common import apply_stage2b
from configs.cmx_relplus_2d import config


config.relplus_gravity_source = "local"
apply_stage2b(config, "relplus_local", "online [ReD, EGVIA, LOA]", "REL-default EstGravity")
cfg = config
config.config_path = __file__
