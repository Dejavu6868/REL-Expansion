from configs.stage2a_common import apply_stage2a
from configs.cmx_relplus_2d import config


config.relplus_gravity_source = "local"
apply_stage2a(config, "relplus_local", "online [ReD, EGVIA, LOA]", "REL-default EstGravity")
cfg = config
config.config_path = __file__
