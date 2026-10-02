"""Import-safe, fail-closed CMX-REL+ V2.3 formal S2D configuration."""

import os.path as osp

from .common import (
    DATASET_ROOT,
    OUTPUT_ROOT_V2_3,
    base_config,
    finish_config,
)


INTEGRATION_PROTOCOL_ID = "CMX_RELPLUS_V2_3"
REPRESENTATION_PROTOCOL_ID = "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"
FORMAL_CACHE_ROOT = osp.join(OUTPUT_ROOT_V2_3, "formal_cache")

config = base_config(
    experiment_name="CMX_RELPlus_v2_3_S2D_formal",
    x_cache_root=FORMAL_CACHE_ROOT,
    output_name="formal_training/CMX_RELPlus_v2_3_seed12345",
    protocol_id=INTEGRATION_PROTOCOL_ID,
    output_root=OUTPUT_ROOT_V2_3,
)
config.integration_protocol_id = INTEGRATION_PROTOCOL_ID
config.representation_protocol_id = REPRESENTATION_PROTOCOL_ID
config.formal_cache_root = FORMAL_CACHE_ROOT
config.train_source = osp.join(FORMAL_CACHE_ROOT, "train.txt")
config.eval_source = osp.join(FORMAL_CACHE_ROOT, "test.txt")
config.full_manifest = (
    "/data/zhuzhaoziao/RELPlus/outputs/"
    "REL_plus_v2_1_implementation/full_manifest.csv"
)
config.cache_generation_report = osp.join(
    FORMAL_CACHE_ROOT, "cache_generation_summary.json"
)
config.generation_resolved_manifest_path = osp.join(
    FORMAL_CACHE_ROOT, "cache_manifest_resolved.csv"
)
config.resolved_manifest_path = osp.join(
    FORMAL_CACHE_ROOT, "audit", "cache_manifest_resolved.csv"
)
config.class_mapping = osp.join(DATASET_ROOT, "class_mapping.json")
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
config.training_data_preflight_report = osp.join(
    FORMAL_CACHE_ROOT,
    "preflight",
    "cmx_training_data_preflight_summary.json",
)
config.ddp_smoke_report = osp.join(
    OUTPUT_ROOT_V2_3, "ddp_smoke", "ddp_optimizer_smoke_summary.json"
)

# Repository defaults remain fail-closed. Only the explicit V2.3 launcher may
# create a one-run resolved config that changes these runtime fields.
config.training_authorized = False
config.full_cache_authorized = False
config.source_compatible_invalid_accepted = False
config.data_ready = False
finish_config(config)
cfg = config
