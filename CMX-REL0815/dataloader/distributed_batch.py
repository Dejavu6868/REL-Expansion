import torch
from torch.utils.data import Sampler


class SplitLogicalDistributedSampler(Sampler):
    """Split each reference-rank batch across multiple physical DDP ranks."""

    def __init__(
        self,
        dataset,
        global_batch_size,
        num_steps,
        num_replicas,
        rank,
        reference_replicas,
        shuffle=True,
        seed=0,
    ):
        if num_replicas % reference_replicas != 0:
            raise ValueError("physical replicas must be divisible by reference replicas")
        if global_batch_size % reference_replicas != 0:
            raise ValueError("global batch must be divisible by reference replicas")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank is outside the physical replica range")

        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.num_steps = num_steps
        self.num_replicas = num_replicas
        self.rank = rank
        self.reference_replicas = reference_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        expected_size = global_batch_size * num_steps
        if len(dataset) != expected_size:
            raise ValueError(
                "dataset length must equal global_batch_size * num_steps: "
                "{} != {}".format(len(dataset), expected_size)
            )

        self.reference_batch_size = global_batch_size // reference_replicas
        self.shards_per_reference_rank = num_replicas // reference_replicas
        if self.reference_batch_size < self.shards_per_reference_rank:
            raise ValueError("reference batch is too small for the physical rank split")

        base, remainder = divmod(
            self.reference_batch_size, self.shards_per_reference_rank
        )
        self.shard_batch_sizes = [
            base + (shard < remainder)
            for shard in range(self.shards_per_reference_rank)
        ]
        self.reference_rank = rank % reference_replicas
        self.shard_index = rank // reference_replicas
        self.local_batch_size = self.shard_batch_sizes[self.shard_index]
        self.num_samples = self.local_batch_size * num_steps

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        reference_indices = indices[self.reference_rank :: self.reference_replicas]
        expected_reference_samples = self.reference_batch_size * self.num_steps
        if len(reference_indices) != expected_reference_samples:
            raise RuntimeError("reference-rank sample count is inconsistent")

        start = sum(self.shard_batch_sizes[: self.shard_index])
        stop = start + self.local_batch_size
        selected = []
        for step in range(self.num_steps):
            offset = step * self.reference_batch_size
            selected.extend(reference_indices[offset + start : offset + stop])
        if len(selected) != self.num_samples:
            raise RuntimeError("physical-rank sample count is inconsistent")
        return iter(selected)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
