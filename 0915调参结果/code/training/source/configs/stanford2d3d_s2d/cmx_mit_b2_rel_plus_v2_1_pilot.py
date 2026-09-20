"""36-sample loader/backward/evaluator wiring configuration; never trains."""

import os.path as osp

from .common import PILOT_ROOT, base_config, count_list, finish_config


config = base_config(
    experiment_name="CMX_RELPlus_v2_1_S2D_pilot_wiring",
    x_cache_root=PILOT_ROOT,
    output_name="pilot",
)
config.train_source = osp.join(PILOT_ROOT, "train.txt")
config.eval_source = osp.join(PILOT_ROOT, "test.txt")
config.num_train_imgs = count_list(config.train_source, 36)
config.num_eval_imgs = count_list(config.eval_source, 36)
config.batch_size = 1
config.num_workers = 0
config.nepochs = 0
config.niters_per_epoch = 36
config.warm_up_epoch = 0
config.checkpoint_start_epoch = 10 ** 9
config.checkpoint_step = 10 ** 9
config.training_authorized = False
config.data_ready = True
finish_config(config)
cfg = config
