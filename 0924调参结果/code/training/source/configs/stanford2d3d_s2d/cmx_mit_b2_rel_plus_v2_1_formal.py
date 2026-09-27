"""Complete but fail-closed CMX-REL+ v2.1 S2D formal configuration."""

import os.path as osp

from .common import DATASET_ROOT, OUTPUT_ROOT, base_config, count_list, finish_config


FORMAL_CACHE_ROOT = osp.join(OUTPUT_ROOT, "formal_cache_not_generated")
config = base_config(
    experiment_name="CMX_RELPlus_v2_1_S2D_formal",
    x_cache_root=FORMAL_CACHE_ROOT,
    output_name="formal_not_authorized",
)
config.train_source = osp.join(DATASET_ROOT, "train.txt")
config.eval_source = osp.join(DATASET_ROOT, "test.txt")
config.num_train_imgs = count_list(config.train_source, 52903)
config.num_eval_imgs = count_list(config.eval_source, 17593)
config.niters_per_epoch = config.num_train_imgs // config.batch_size + 1
config.training_authorized = False
config.data_ready = False
finish_config(config)
cfg = config
