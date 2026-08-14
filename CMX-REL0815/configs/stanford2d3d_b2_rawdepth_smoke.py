from .stanford2d3d_b2_common import make_config


config = make_config("rawdepth")
cfg = config
config.config_path = __file__
config.effective_distributed_seeds = [config.seed]
config.batch_size = 3
config.nepochs = 1
config.niters_per_epoch = 2
config.num_workers = 2
config.warm_up_epoch = 0
config.checkpoint_start_epoch = 1
config.checkpoint_step = 1
