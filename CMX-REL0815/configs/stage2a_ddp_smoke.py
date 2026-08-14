import importlib
import os


arm = os.environ["STAGE2A_ARM"]
if arm not in {"rawdepth", "hha", "relplus_local", "relplus_pose"}:
    raise ValueError("invalid STAGE2A_ARM")
config = importlib.import_module("configs.stage2a_{}".format(arm)).config
config.nepochs = 1
config.niters_per_epoch = 1
config.num_workers = 0
config.checkpoint_start_epoch = 1
config.checkpoint_step = 1
config.metrics_csv = os.path.join(config.log_dir, "metrics.csv")
cfg = config
config.config_path = __file__
