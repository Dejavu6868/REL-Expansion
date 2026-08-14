import os
import os.path as osp

from configs.stanford2d3d_b2_common import make_config


config = make_config("hha")
config.stage = "REL_SOURCE_ALIGNMENT_REVIEW"
config.arm = "rel_source_aligned"
config.modality = "rel_source_aligned"
config.x_mode = "rel_source_aligned"
config.rel_impl = "official_source"
config.x_online_relplus = False
config.x_is_single_channel = False
config.depth_root_folder = osp.join(config.dataset_path, "Depth16")
config.depth_format = ".png"
config.pose_root_folder = osp.join(config.dataset_path, "Pose")
config.pose_format = ".json"
config.rel_authority_root = os.environ.get(
    "REL_AUTHORITY_ROOT",
    "/home/zhuzhaoziao/rel_exp/"
    "REL-SF4PASS_authority_16c1267608171d67b34ecc3d0190920a06f1017e",
)
config.rel_authority_commit = "16c1267608171d67b34ecc3d0190920a06f1017e"
config.rel_source_core_entry = "getREL.py::getREL"
config.rel_source_geometry_entries = (
    "utils/rgbd_util.py::processDepthImage",
    "utils/rgbd_util.py::processDepthImage_ERP",
)
config.rel_source_channel_order = [
    "EGVIA_source_code",
    "LOA_source_code",
    "ReD_source_code",
]
config.rel_source_alpha_degrees = 45.0
config.rel_source_lambda = 0.5
config.rel_source_generation_order = "REL before shared random augmentation"
config.rel_reference_dependency_status = "author repository omits utils/hha_util.py"
config.train_horizontal_flip = False
config.num_workers = 0
config.pretrained_model = None
config.training_authorized = False
if hasattr(config, "pretrained_sha256"):
    del config.pretrained_sha256
cfg = config
config.config_path = __file__

