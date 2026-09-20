"""Three-sample CMX-REL smoke configuration; no epoch is launched."""

import os

from .cmx_mit_b2_rel import make_config


DATASET_ROOT = os.environ.get(
    "CMX_REL_SMOKE_DATASET_ROOT",
    "/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/smoke_dataset",
)
RUN_ROOT = os.environ.get(
    "CMX_REL_SMOKE_RUN_DIR",
    "/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/smoke_runtime",
)

config = make_config(DATASET_ROOT, run_dir=RUN_ROOT, smoke=True)
cfg = config
config.config_path = __file__
