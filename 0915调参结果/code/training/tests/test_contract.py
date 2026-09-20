import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite_common import budget, validate_paths, validate_shared


class ContractTests(unittest.TestCase):
    def test_full_epoch_covers_every_image_with_minimal_padding(self):
        values = budget(52903, 56)
        self.assertEqual(values, {'niters_per_epoch': 945,
            'logical_samples_per_epoch': 52920, 'sampler_padding_count': 17,
            'total_updates': 189000, 'warmup_updates': 9450})
        self.assertLess(values['sampler_padding_count'], 56)

    def test_original_output_cannot_be_reused(self):
        with self.assertRaises(ValueError):
            validate_paths({'remote_source_root': '/home/zhuzhaoziao/RELPlus/CMX-S2D-ThreeArm-v1',
                            'output_root': '/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_ThreeArm_v1'})

    def test_relative_output_cannot_be_used(self):
        with self.assertRaises(ValueError):
            validate_paths({'remote_source_root': '/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-20260915',
                            'output_root': 'runs'})

    def test_batch_change_without_recomputed_sampling_rejected(self):
        with self.assertRaises(ValueError):
            validate_shared({'batch_size': 56, 'lr': 0.00012,
                             'niters_per_epoch': 6613, 'logical_samples_per_epoch': 52904})

    def test_invalid_batch_rejected(self):
        with self.assertRaises(ValueError):
            budget(52903, 0)


if __name__ == '__main__':
    unittest.main()
