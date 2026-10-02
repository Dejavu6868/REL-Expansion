import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
from suite_common import baseline_config_comparison, load_suite


class GammaIsolationTests(unittest.TestCase):
    def config(self, arm):
        result = json.loads((BUNDLE / 'baseline' / (arm + '.json')).read_text())
        result['focal_gamma'] = 1.0
        for field in ('root_dir', 'abs_dir'):
            result[field] = '/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG1-20260920/source'
        return result

    def test_gamma_is_the_only_allowed_semantic_change_in_both_arms(self):
        for arm in ('hha', 'relplus'):
            cfg = self.config(arm)
            report = baseline_config_comparison(cfg, arm)
            self.assertEqual(report['semantic_changes'], ['focal_gamma'])
            for field, value in [('lr', 0.00006), ('num_workers', 4), ('bn_eps', 1e-5),
                                 ('train_horizontal_flip', True), ('focal_gamma', 2)]:
                altered = copy.deepcopy(cfg)
                altered[field] = value
                with self.assertRaises(ValueError):
                    baseline_config_comparison(altered, arm)

    def test_nested_input_change_is_rejected(self):
        cfg = self.config('relplus')
        cfg['data_setting']['channel_order'].reverse()
        with self.assertRaises(ValueError):
            baseline_config_comparison(cfg, 'relplus')

    def test_source_relocation_must_be_exact(self):
        cfg = self.config('hha')
        cfg['root_dir'] = '/home/unrelated/source'
        with self.assertRaisesRegex(ValueError, 'source identity'):
            baseline_config_comparison(cfg, 'hha')

    def test_extra_override_cannot_escape_frozen_control_check(self):
        suite = json.loads((BUNDLE / 'suite.json').read_text())
        suite['shared']['norm_mean'] = [0, 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'suite.json'
            path.write_text(json.dumps(suite))
            with self.assertRaisesRegex(ValueError, 'key set'):
                load_suite(path)


if __name__ == '__main__':
    unittest.main()
