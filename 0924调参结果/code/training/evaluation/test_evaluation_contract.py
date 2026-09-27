"""CPU-only checks for baseline attribution, fixed endpoint, and refusal gates."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

BUNDLE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(BUNDLE / 'evaluation'))
import compare_results as comparison
import eval_rank
import run_evaluation


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads((BUNDLE / 'suite.json').read_text())
        self.gamma2 = comparison.validate_baseline(BUNDLE, self.suite)
        self.baseline = comparison.validate_gamma1_baseline(BUNDLE, self.suite, self.gamma2)

    def test_audited_baseline_and_attribution(self):
        self.assertEqual(self.gamma2['metrics']['hha']['mIoU_percent'], 60.98827182233775)
        self.assertEqual(self.baseline['metrics']['hha']['mIoU_percent'], 60.182898850089714)
        self.assertEqual(self.baseline['metrics']['relplus']['mIoU_percent'], 61.40637239796889)
        for field, value in [('lr', 0.00006), ('focal_gamma', 1), ('batch_size', 48), ('seed', 1)]:
            changed = copy.deepcopy(self.suite)
            changed['shared'][field] = value
            with self.assertRaisesRegex(ValueError, 'only changed'):
                comparison.validate_baseline(BUNDLE, changed)
            with self.assertRaisesRegex(ValueError, 'only changed'):
                comparison.validate_gamma1_baseline(BUNDLE, changed, self.gamma2)

    def test_changed_baseline_bytes_refused(self):
        with tempfile.TemporaryDirectory() as root:
            baseline = Path(root) / 'baseline'
            baseline.mkdir()
            (baseline / 'comparison_verified.json').write_bytes(
                (BUNDLE / 'baseline/comparison_verified.json').read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                comparison.validate_baseline(root, self.suite)

    def test_changed_primary_baseline_or_config_bytes_refused(self):
        import shutil
        for relative in ('baseline_gamma1/comparison.json', 'baseline_gamma1/suite.json',
                         'baseline_gamma1/hha.json', 'baseline_gamma1/relplus.json',
                         'baseline/gamma2_hha.json', 'baseline/gamma2_relplus.json'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as root:
                for directory in ('baseline', 'baseline_gamma1'):
                    shutil.copytree(BUNDLE / directory, Path(root) / directory)
                path = Path(root) / relative
                path.write_bytes(path.read_bytes() + b' ')
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    comparison.validate_gamma1_baseline(root, self.suite, self.gamma2)

    def current_config(self, arm):
        old = json.loads((BUNDLE / 'baseline_gamma1' / (arm + '.json')).read_text())
        # Fixture rebasing mirrors the contract's exact permitted identities.
        prior = json.loads((BUNDLE / 'baseline_gamma1/suite.json').read_text())
        return comparison._expected_config(old, prior, self.suite)

    def comparison_fixture(self, improvements=(1, 2)):
        audits = {arm: {'status': 'PASS', 'metrics': {
                    key: self.baseline['metrics'][arm][key] + improvement
                    for key in comparison.METRICS},
                    'gt_histogram': self.baseline['gt_histogram'],
                    'test_source_sha256': self.baseline['test_source_sha256']}
                  for arm, improvement in zip(comparison.ARMS, improvements)}
        before = {'suite': self.suite, 'baseline_gamma1': self.baseline,
                  'baseline_gamma2': self.gamma2, 'arms': {}}
        for arm in comparison.ARMS:
            config = self.current_config(arm)
            before['arms'][arm] = {'config': config, 'config_comparison':
                comparison.validate_resolved_config(BUNDLE, self.suite, arm, config)}
        return audits, before

    def test_full_resolved_config_refuses_semantic_and_identity_drift(self):
        for arm in comparison.ARMS:
            config = self.current_config(arm)
            comparison.validate_resolved_config(BUNDLE, self.suite, arm, config)
            for field, value in [('lr', 0.00006), ('focal_gamma', 1), ('seed', 42),
                                 ('output_dir', '/other/run'), ('root_dir', '/other/source'),
                                 ('eval_align_corners', True)]:
                changed = dict(config, **{field: value})
                with self.assertRaisesRegex(ValueError, 'configuration change'):
                    comparison.validate_resolved_config(BUNDLE, self.suite, arm, changed)

    def test_synthetic_same_arm_and_gap_differences(self):
        # Synthetic arithmetic fixtures; these are not evaluated model results.
        audits, before = self.comparison_fixture()
        result = comparison.build_comparison(audits, before)
        for key in comparison.METRICS:
            self.assertAlmostEqual(result['gamma1_5_minus_gamma1_percentage_points']['hha'][key], 1)
            self.assertAlmostEqual(result['gamma1_5_minus_gamma1_percentage_points']['relplus'][key], 2)
            self.assertAlmostEqual(result['relplus_minus_hha_gap_change_percentage_points']['gamma1_5_minus_gamma1'][key], 1)
            for arm in comparison.ARMS:
                self.assertEqual(result['gamma1_5_minus_gamma2_percentage_points'][arm][key],
                                 audits[arm]['metrics'][key] - self.gamma2['metrics'][arm][key])
            self.assertEqual(result['relplus_minus_hha_gap_change_percentage_points']['gamma1_5_minus_gamma2'][key],
                             result['differences_percentage_points']['relplus-hha'][key] - self.gamma2['relplus_minus_hha_pp'][key])
        audits['hha']['gt_histogram'] = [0] * 13
        with self.assertRaisesRegex(ValueError, 'GT differs'):
            comparison.build_comparison(audits, before)

    def test_candidate_screening_uses_full_precision_and_strict_relplus_boundary(self):
        def adjacent_positive_float(value, direction):
            # Python 3.8 has no math.nextafter; fixture metrics are positive binary64 values.
            bits = struct.unpack('>Q', struct.pack('>d', value))[0]
            return struct.unpack('>d', struct.pack('>Q', bits + direction))[0]

        audits, before = self.comparison_fixture((0, 0))
        result = comparison.build_comparison(audits, before)
        self.assertEqual(result['candidate_screening']['status'], 'FAIL')
        audits['relplus']['metrics']['mIoU_percent'] = adjacent_positive_float(
            self.baseline['metrics']['relplus']['mIoU_percent'], 1)
        self.assertEqual(comparison.build_comparison(audits, before)['candidate_screening']['status'], 'PASS')
        audits['hha']['metrics']['mIoU_percent'] = adjacent_positive_float(
            self.baseline['metrics']['hha']['mIoU_percent'], -1)
        self.assertEqual(comparison.build_comparison(audits, before)['candidate_screening']['status'], 'FAIL')

    def test_seed_class_order_and_invalid_metrics_refused(self):
        for mutation in ('seed', 'class', 'nan', 'range'):
            audits, before = self.comparison_fixture()
            before = copy.deepcopy(before)
            if mutation == 'seed':
                before['baseline_gamma1']['seed'] = 42
            elif mutation == 'class':
                before['arms']['hha']['config']['class_names'].reverse()
            else:
                audits['hha']['metrics']['mIoU_percent'] = float('nan') if mutation == 'nan' else 101
            with self.assertRaises(ValueError):
                comparison.build_comparison(audits, before)

    def test_audit_computation_byte_identity(self):
        self.assertEqual(hashlib.sha256((BUNDLE / 'evaluation/audit_eval.py').read_bytes()).hexdigest(),
                         '2c59e75f999ecd9e350dd1a75d19448f316d7dcf5f263d6cb94a8d125dfd6c07')

    def worker_fixture(self):
        arm = 'hha'
        root = Path(self.suite['output_root'])
        output = root / 'evaluation_epoch200' / arm
        checkpoint = root / 'runs' / arm / 'checkpoints/epoch-200.pth'
        queue = {'status': 'PASS', 'phase': 'train', 'suite_sha256': 'fixture-hash',
                 'completed': [{'arm': a} for a in ['hha', 'relplus']]}
        gate = {'status': 'PASS', 'training': copy.deepcopy(queue), 'suite': self.suite,
                'bundle_manifest_sha256': 'fixture-hash',
                'arms': {arm: {'checkpoint': {'path': str(checkpoint), 'size': 1, 'mtime_ns': '2'}}}}
        records = {root / 'train_status.json': queue, output.parent / 'preflight.json': gate}
        return root, output, records

    def call_worker_gate(self, output, records, size=1):
        with mock.patch.object(Path, 'read_text', lambda path: json.dumps(records[path])), \
                mock.patch.object(Path, 'stat', return_value=types.SimpleNamespace(st_size=size, st_mtime_ns=2)), \
                mock.patch.object(eval_rank, 'digest', return_value='fixture-hash'):
            eval_rank.require_completed_training(self.suite, 'hha', output)

    def test_worker_accepts_matching_completed_gate(self):
        _, output, records = self.worker_fixture()
        self.call_worker_gate(output, records)

    def test_worker_refuses_training_not_completed(self):
        root, output, records = self.worker_fixture()
        records[root / 'train_status.json']['status'] = 'RUNNING'
        records[output.parent / 'preflight.json']['training']['status'] = 'RUNNING'
        with self.assertRaisesRegex(RuntimeError, 'successful formal-training'):
            self.call_worker_gate(output, records)

    def test_worker_refuses_changed_manifest_and_checkpoint(self):
        _, output, records = self.worker_fixture()
        with self.assertRaisesRegex(RuntimeError, 'checkpoint differs'):
            self.call_worker_gate(output, records, size=2)
        records[output.parent / 'preflight.json']['bundle_manifest_sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'successful formal-training'):
            self.call_worker_gate(output, records)

    def test_worker_refuses_output_redirect(self):
        _, output, records = self.worker_fixture()
        with self.assertRaisesRegex(RuntimeError, 'scope mismatch'):
            self.call_worker_gate(output / 'retry', records)

    def test_supervisor_refuses_existing_output_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            status = output / 'status.json'
            status.write_text('preserved evidence')
            with mock.patch.object(run_evaluation, 'OUTPUT', output), self.assertRaises(FileExistsError):
                run_evaluation.main()
            self.assertEqual(status.read_text(), 'preserved evidence')

    def test_new_smoke_report_requires_validated_runtime_hash(self):
        path = str(run_evaluation.TRAIN_ROOT / 'smoke/hha/ddp_optimizer_smoke_summary.json')
        prior = {'ddp_smoke_report': {'path': path, 'exists': False},
                 'train_source': {'path': '/frozen/train.txt', 'exists': True, 'sha256': 'fixed'}}
        evidence = copy.deepcopy(prior)
        evidence['ddp_smoke_report'].update(exists=True, sha256='validated-smoke')
        prerequisites = {'smoke_arms': {'hha': {'status': 'PASS', 'smoke_summary_sha256': 'validated-smoke'}}}
        run_evaluation.validate_data_evidence(evidence, prior, prerequisites, 'hha')
        evidence['ddp_smoke_report']['sha256'] = 'tampered'
        with self.assertRaisesRegex(RuntimeError, 'smoke evidence changed'):
            run_evaluation.validate_data_evidence(evidence, prior, prerequisites, 'hha')
        evidence['ddp_smoke_report']['sha256'] = 'validated-smoke'
        evidence['train_source']['exists'] = False
        with self.assertRaisesRegex(RuntimeError, 'data evidence changed'):
            run_evaluation.validate_data_evidence(evidence, prior, prerequisites, 'hha')


if __name__ == '__main__':
    unittest.main()
