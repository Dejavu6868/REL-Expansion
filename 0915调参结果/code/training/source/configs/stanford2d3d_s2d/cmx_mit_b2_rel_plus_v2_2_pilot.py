"""Audited 36-sample V2.2 cache wiring configuration; never trains."""

import os.path as osp

from .common import OUTPUT_ROOT_V2_2, base_config, finish_config


PILOT_ROOT = osp.join(OUTPUT_ROOT_V2_2, "pilot_cache")
config = base_config(
    experiment_name="CMX_RELPlus_v2_2_S2D_pilot_wiring",
    x_cache_root=PILOT_ROOT,
    output_name="pilot_wiring",
    protocol_id="CMX_RELPLUS_V2_2",
    output_root=OUTPUT_ROOT_V2_2,
)
config.integration_protocol_id = "CMX_RELPLUS_V2_2"
config.train_source = osp.join(PILOT_ROOT, "train.txt")
config.eval_source = osp.join(PILOT_ROOT, "test.txt")
config.num_train_imgs = 30
config.num_eval_imgs = 6
config.logical_samples_per_epoch = 32
config.niters_per_epoch = 4
config.num_workers = 0
config.cache_audit_report = osp.join(
    PILOT_ROOT, "audit", "cache_audit_summary.json"
)
config.training_authorized = False
config.full_cache_authorized = False
config.data_ready = True
finish_config(config)
cfg = config
