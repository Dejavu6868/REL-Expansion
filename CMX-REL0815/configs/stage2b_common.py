import os


ARM_NAMES = ("rawdepth", "hha", "relplus_local", "relplus_pose")
REGISTERED_SEEDS = (23456, 34567)


def apply_stage2b(config, arm, second_modality, gravity_source):
    if arm not in ARM_NAMES:
        raise ValueError("unknown Stage2B arm: {}".format(arm))
    seed = int(os.environ.get("STAGE2B_SEED", "0"))
    if seed not in REGISTERED_SEEDS:
        raise ValueError("STAGE2B_SEED must be a registered Stage2B seed: {}".format(REGISTERED_SEEDS))
    config.stage = "Stage2B"
    config.arm = arm
    config.second_modality_identity = second_modality
    config.gravity_source = gravity_source
    config.seed = seed
    config.physical_world_size = 8
    config.physical_gpu_ids = list(range(8))
    config.reference_world_size = 4
    config.physical_rank_batch_sizes = [2, 2, 2, 2, 1, 1, 1, 1]
    config.physical_rank_seeds = [seed + rank for rank in range(8)]
    config.reference_rank_base_seeds = [seed + rank for rank in range(4)]
    config.reference_rank_pairs = [[0, 4], [1, 5], [2, 6], [3, 7]]
    config.distributed_batch_adapter = "split each reference batch 3 into physical batches 2+1"
    config.distributed_loss_adapter = "2 * local cross-entropy sum / paired valid pixels"
    config.reference_stochastic_trajectory_equivalent = False
    config.num_workers = 8
    config.train_horizontal_flip = False
    config.deterministic_training = True
    config.amp_enabled = False
    config.gradient_clipping = None
    config.checkpoint_selection_rule = "final epoch 32; test evaluated once"
    config.common_initial_model = os.environ.get("STAGE2B_COMMON_INITIAL_MODEL")
    config.metrics_csv = os.path.join(config.log_dir, "metrics.csv")
    return config
