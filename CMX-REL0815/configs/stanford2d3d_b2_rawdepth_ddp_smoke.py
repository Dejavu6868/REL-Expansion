from .stanford2d3d_b2_common import make_config


config = make_config("rawdepth")
cfg = config
config.config_path = __file__
config.batch_size = 12
config.nepochs = 1
config.niters_per_epoch = 2
config.num_workers = 1
config.warm_up_epoch = 0
config.checkpoint_start_epoch = 1
config.checkpoint_step = 1
