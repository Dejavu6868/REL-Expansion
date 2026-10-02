"""Auditable train and evaluation samplers for CMX-REL+ V2.2."""

import math

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def _distributed_value(explicit, getter, fallback):
    if explicit is not None:
        return int(explicit)
    if dist.is_available() and dist.is_initialized():
        return int(getter())
    return int(fallback)


class FixedLengthDistributedSampler(Sampler):
    """Shuffle once per epoch, pad once, then give each rank a disjoint stride."""

    def __init__(
        self,
        dataset,
        *,
        logical_samples_per_epoch,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0
    ):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.logical_samples_per_epoch = int(logical_samples_per_epoch)
        self.num_replicas = _distributed_value(
            num_replicas, dist.get_world_size, 1
        )
        self.rank = _distributed_value(rank, dist.get_rank, 0)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.dataset_size <= 0:
            raise ValueError("dataset must contain at least one real sample")
        if self.logical_samples_per_epoch < self.dataset_size:
            raise ValueError(
                "logical_samples_per_epoch cannot omit real dataset samples"
            )
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid distributed sampler rank/world size")
        if self.logical_samples_per_epoch % self.num_replicas:
            raise ValueError(
                "logical_samples_per_epoch must be divisible by num_replicas"
            )
        self.num_samples = self.logical_samples_per_epoch // self.num_replicas

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            base = torch.randperm(
                self.dataset_size, generator=generator
            ).tolist()
        else:
            base = list(range(self.dataset_size))
        repeats, remainder = divmod(
            self.logical_samples_per_epoch, self.dataset_size
        )
        indices = base * repeats + base[:remainder]
        owned = indices[self.rank : self.logical_samples_per_epoch : self.num_replicas]
        if len(owned) != self.num_samples:
            raise RuntimeError("fixed-length sampler partition is inconsistent")
        return iter(owned)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @property
    def padding_count(self):
        return self.logical_samples_per_epoch - self.dataset_size


class DistributedEvalSamplerNoPad(Sampler):
    """Partition evaluation as indices[rank::world_size] without duplicates."""

    def __init__(self, dataset, *, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = _distributed_value(
            num_replicas, dist.get_world_size, 1
        )
        self.rank = _distributed_value(rank, dist.get_rank, 0)
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid evaluation sampler rank/world size")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, int(math.ceil(remaining / self.num_replicas)))
