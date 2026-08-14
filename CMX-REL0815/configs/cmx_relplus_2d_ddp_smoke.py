import os
import os.path as osp

from configs.cmx_relplus_2d import config


config.nepochs = int(os.environ.get("CMX_SMOKE_EPOCHS", "1"))
config.niters_per_epoch = 1
config.num_workers = 0
config.checkpoint_start_epoch = 1
config.checkpoint_step = 1
config.warm_up_epoch = 1
config.x_root_folder = os.environ["CMX_RELPLUS_ROOT"]
config.train_source = os.environ["CMX_SMOKE_SOURCE"]
config.eval_source = os.environ["CMX_SMOKE_SOURCE"]
config.num_train_imgs = sum(1 for line in open(config.train_source) if line.strip())
config.log_dir = osp.abspath(os.environ["CMX_RUN_DIR"])
config.tb_dir = osp.join(config.log_dir, "tensorboard")
config.log_dir_link = config.log_dir
config.checkpoint_dir = osp.join(config.log_dir, "checkpoints")
