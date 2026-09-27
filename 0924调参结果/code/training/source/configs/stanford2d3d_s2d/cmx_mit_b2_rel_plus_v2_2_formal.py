"""Import-safe, fail-closed CMX-REL+ V2.2 formal S2D configuration."""

import os.path as osp

from .common import (
    DATASET_ROOT,
    OUTPUT_ROOT_V2_2,
    base_config,
    finish_config,
)


INTEGRATION_PROTOCOL_ID = "CMX_RELPLUS_V2_2"
FORMAL_CACHE_ROOT = osp.join(OUTPUT_ROOT_V2_2, "formal_cache_not_generated")
config = base_config(
    experiment_name="CMX_RELPlus_v2_2_S2D_formal",
    x_cache_root=FORMAL_CACHE_ROOT,
    output_name="formal_not_authorized",
    protocol_id=INTEGRATION_PROTOCOL_ID,
    output_root=OUTPUT_ROOT_V2_2,
)
config.integration_protocol_id = INTEGRATION_PROTOCOL_ID
config.representation_protocol_id = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
config.train_source = osp.join(DATASET_ROOT, "train.txt")
config.eval_source = osp.join(DATASET_ROOT, "test.txt")
config.full_manifest = (
    "/data/zhuzhaoziao/RELPlus/outputs/"
    "REL_plus_v2_1_implementation/full_manifest.csv"
)
config.num_train_imgs = 52903
config.num_eval_imgs = 17593
config.logical_samples_per_epoch = 52904
config.niters_per_epoch = 6613
config.sampler = "FixedLengthDistributedSampler"
config.sampler_padding_count = 1
config.optimizer_parameter_groups = "author_group_weight_decay_and_no_decay"
config.checkpoint_epochs = list(range(100, 201, 5))
config.evaluation_epochs = list(range(100, 201, 5))
config.primary_endpoint = "epoch_200"
config.secondary_endpoint = "test_selected_best"
config.cache_audit_report = osp.join(
    FORMAL_CACHE_ROOT, "audit", "cache_audit_summary.json"
)
config.training_authorized = False
config.full_cache_authorized = False
config.data_ready = False
finish_config(config)
cfg = config
