import unittest

import torch
from torch.utils.data import DistributedSampler

from dataloader.distributed_batch import SplitLogicalDistributedSampler


class _Dataset:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class SplitLogicalDistributedSamplerTest(unittest.TestCase):
    def _samplers(self, steps=5, shuffle=True):
        dataset = _Dataset(12 * steps)
        return [
            SplitLogicalDistributedSampler(
                dataset,
                global_batch_size=12,
                num_steps=steps,
                num_replicas=8,
                rank=rank,
                reference_replicas=4,
                shuffle=shuffle,
            )
            for rank in range(8)
        ]

    def test_rank_batch_sizes_and_exact_epoch_coverage(self):
        samplers = self._samplers(shuffle=False)
        self.assertEqual(
            [sampler.local_batch_size for sampler in samplers],
            [2, 2, 2, 2, 1, 1, 1, 1],
        )
        indices = [index for sampler in samplers for index in sampler]
        self.assertEqual(len(indices), 60)
        self.assertEqual(sorted(indices), list(range(60)))

    def test_reconstructs_four_rank_distributed_sampler_batches(self):
        steps = 7
        dataset = _Dataset(12 * steps)
        split_samplers = self._samplers(steps=steps, shuffle=True)
        for epoch in (1, 2, 17, 32):
            for sampler in split_samplers:
                sampler.set_epoch(epoch)
            physical_indices = [list(sampler) for sampler in split_samplers]
            for logical_rank in range(4):
                reference = DistributedSampler(
                    dataset,
                    num_replicas=4,
                    rank=logical_rank,
                    shuffle=True,
                    seed=0,
                    drop_last=False,
                )
                reference.set_epoch(epoch)
                reference_indices = list(reference)
                two = physical_indices[logical_rank]
                one = physical_indices[logical_rank + 4]
                reconstructed = []
                for step in range(steps):
                    reconstructed.extend(two[2 * step : 2 * step + 2])
                    reconstructed.append(one[step])
                self.assertEqual(reconstructed, reference_indices)

            generator = torch.Generator()
            generator.manual_seed(epoch)
            permutation = torch.randperm(len(dataset), generator=generator).tolist()
            for step in range(steps):
                physical_step = []
                for rank in range(4):
                    physical_step.extend(
                        physical_indices[rank][2 * step : 2 * step + 2]
                    )
                    physical_step.append(physical_indices[rank + 4][step])
                self.assertEqual(
                    sorted(physical_step),
                    sorted(permutation[12 * step : 12 * step + 12]),
                )

    def test_pairwise_loss_scaling_matches_four_rank_mean(self):
        local_sums = [11.0, 17.0, 23.0, 29.0, 5.0, 7.0, 13.0, 19.0]
        valid_pixels = [7.0, 11.0, 13.0, 17.0, 3.0, 5.0, 7.0, 11.0]
        physical_losses = []
        logical_losses = []
        for logical_rank in range(4):
            partner = logical_rank + 4
            pair_valid = valid_pixels[logical_rank] + valid_pixels[partner]
            physical_losses.extend(
                [
                    2.0 * local_sums[logical_rank] / pair_valid,
                    2.0 * local_sums[partner] / pair_valid,
                ]
            )
            logical_losses.append(
                (local_sums[logical_rank] + local_sums[partner]) / pair_valid
            )
        self.assertAlmostEqual(
            sum(physical_losses) / 8.0,
            sum(logical_losses) / 4.0,
        )

    def test_epoch_changes_shuffle_deterministically(self):
        samplers = self._samplers()
        first = list(samplers[0])
        samplers[0].set_epoch(1)
        second = list(samplers[0])
        self.assertNotEqual(first, second)
        samplers[0].set_epoch(0)
        self.assertEqual(first, list(samplers[0]))

    def test_rejects_incompatible_topology(self):
        dataset = _Dataset(24)
        with self.assertRaises(ValueError):
            SplitLogicalDistributedSampler(dataset, 12, 2, 7, 0, 4)
        with self.assertRaises(ValueError):
            SplitLogicalDistributedSampler(dataset, 10, 2, 8, 0, 4)


if __name__ == "__main__":
    unittest.main()
