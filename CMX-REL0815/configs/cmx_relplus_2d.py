import os
import os.path as osp

from configs.stanford2d3d_b2_common import make_config
from relplus.spec import RELPLUS_SPEC, RELPLUS_SPEC_SHA256


config = make_config("hha")
config.modality = "relplus"
config.x_root_folder = os.environ.get(
    "CMX_RELPLUS_ROOT", osp.join(os.environ["CMX_RUN_DIR"], "relplus_cache")
)
config.x_is_single_channel = False
config.x_online_relplus = True
config.depth_root_folder = osp.join(config.dataset_path, "Depth16")
config.depth_format = ".png"
config.pose_root_folder = osp.join(config.dataset_path, "Pose")
config.pose_format = ".json"
config.relplus_channel_order = ["ReD", "EGVIA", "LOA"]
config.relplus_alpha_degrees = 45.0
config.relplus_lambda = 0.5
config.relplus_depth_definition = "camera-z; metres=uint16/512; 65535 invalid"
config.relplus_pose_definition = (
    "world-to-camera [R|t]; p_rel=p_camera@R=p_world-C; t/C provenance-only"
)
config.model_name = RELPLUS_SPEC["model_name"]
config.config_name = RELPLUS_SPEC["config_name"]
config.representation_semantics = RELPLUS_SPEC["representation_semantics"]
config.relplus_representation_version = RELPLUS_SPEC["representation_version"]
config.relplus_point_frame = RELPLUS_SPEC["point_frame"]
config.relplus_translation_in_red_loa = RELPLUS_SPEC["translation_in_red_loa"]
config.relplus_representation_spec_sha256 = RELPLUS_SPEC_SHA256
config.relplus_pixel_origin = 0.5
config.relplus_native_normal_radius = 3
config.relplus_cache_generation = "online after shared depth/K resize-crop-pad; no encoded-channel resize"
config.physical_world_size = 8
config.physical_gpu_ids = list(range(8))
config.reference_world_size = 4
config.physical_rank_batch_sizes = [2, 2, 2, 2, 1, 1, 1, 1]
config.physical_rank_seeds = list(range(8))
config.reference_rank_base_seeds = [0, 1, 2, 3]
config.reference_rank_pairs = [[0, 4], [1, 5], [2, 6], [3, 7]]
config.distributed_batch_adapter = "split each reference batch 3 into physical batches 2+1"
config.distributed_loss_adapter = "2 * local cross-entropy sum / paired valid pixels"
config.reference_stochastic_trajectory_equivalent = False
