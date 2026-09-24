"""Read-only gamma-2 evidence validation and fixed-endpoint gamma comparison."""
import hashlib
import json
import math
from pathlib import Path

ARMS = ('hha', 'relplus')
METRICS = ('mIoU_percent', 'pixel_accuracy_percent', 'mean_accuracy_percent')
BASELINE_COMPARISON_SHA256 = 'c5148dd245515fa1f97d15f0ad06c56c3d93c5f663af9edad0c34a3b3ba609af'
BASELINE_SUITE_SHA256 = 'a8f323e48d8b70fc2bb6f429a0450bbf47eaff1dfbb308edbfc2a458a2282090'
TEST_SHA256 = 'b9de196c6c1aa8f9ac37926910af0806ce59b91eb068998711ddbb78eb24423a'
BASELINE_MIOU = {'hha': 60.98827182233775, 'relplus': 61.016897597503004}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_pinned(path, expected):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == expected, 'gamma-2 baseline SHA-256 mismatch: ' + str(path))
    return json.loads(raw), {'path': str(path), 'size': len(raw), 'sha256': actual}


def validate_baseline(bundle, suite):
    directory = Path(bundle) / 'baseline'
    data, comparison_source = _read_pinned(directory / 'comparison_verified.json', BASELINE_COMPARISON_SHA256)
    prior_suite, suite_source = _read_pinned(directory / 'suite.json', BASELINE_SUITE_SHA256)
    expected_shared = dict(prior_suite['shared'], focal_gamma=1)
    _require(suite['shared'] == expected_shared, 'gamma must be the only changed shared training control')
    _require(suite['arms'] == list(ARMS), 'comparison requires exactly HHA and REL+')
    _require(data['status'] == 'PASS_LOCAL_INDEPENDENT_COMPARISON' and
             data['new_evaluation_status'] == 'PASS', 'gamma-2 baseline did not pass')
    _require(data['single_seed'] == suite['shared']['seed'] == 12345, 'baseline seed mismatch')
    _require(data['test_source_sha256'] == TEST_SHA256, 'baseline test list mismatch')
    _require(data['old_and_new_gt_histogram_equal'] is True, 'baseline GT audit did not pass')
    _require(data['decoder_BN_eps_eval'] == 1e-5 and
             data['decoder_BN_eps_training'] == 0.001, 'baseline decoder BN policy mismatch')
    histogram = data['gt_histogram']
    _require(isinstance(histogram, list) and len(histogram) == 13 and
             all(type(x) is int and x >= 0 for x in histogram) and
             sum(histogram) == 3973620198, 'baseline GT histogram is invalid')
    rows = data['rows']
    _require(len(rows) == 3 and {row['arm'] for row in rows} == {'RGBD', 'HHA', 'RELPlus'},
             'baseline rows must contain three unique audited arms')
    by_name = {row['arm']: row for row in rows}
    metrics = {}
    for arm, label in [('hha', 'HHA'), ('relplus', 'RELPlus')]:
        row = by_name[label]
        metrics[arm] = {key: row['new_mIoU_percent' if key == 'mIoU_percent' else key]
                        for key in METRICS}
        _require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 100
                     for x in metrics[arm].values()), 'invalid baseline metrics: ' + arm)
        _require(metrics[arm]['mIoU_percent'] == BASELINE_MIOU[arm], 'baseline mIoU differs: ' + arm)
        checkpoint = data['new_checkpoint_sha256'][arm]
        _require(isinstance(checkpoint, str) and len(checkpoint) == 64 and
                 all(c in '0123456789abcdef' for c in checkpoint), 'invalid baseline checkpoint SHA-256')
    gap = {key: metrics['relplus'][key] - metrics['hha'][key] for key in METRICS}
    _require(gap == data['new_pairwise_differences_pp']['relplus-hha'], 'baseline gap recomputation mismatch')
    return {'status': 'PASS', 'focal_gamma': 2, 'source': comparison_source,
            'suite_source': suite_source, 'training_bundle': prior_suite['remote_source_root'],
            'training_output_root': prior_suite['output_root'], 'metrics': metrics,
            'checkpoint_sha256': {arm: data['new_checkpoint_sha256'][arm] for arm in ARMS},
            'gt_histogram': histogram, 'test_source_sha256': TEST_SHA256,
            'relplus_minus_hha_pp': gap}


def build_comparison(audits, before):
    baseline = before['baseline_gamma2']
    _require(set(audits) == set(ARMS), 'comparison requires exactly two completed audits')
    _require(baseline['status'] == 'PASS', 'baseline validation did not pass')
    for arm in ARMS:
        _require(audits[arm]['status'] == 'PASS', 'new audit did not pass: ' + arm)
        _require(audits[arm]['gt_histogram'] == baseline['gt_histogram'], 'GT differs from gamma-2: ' + arm)
        _require(audits[arm]['test_source_sha256'] == baseline['test_source_sha256'],
                 'test list differs from gamma-2: ' + arm)
        _require(all(math.isfinite(audits[arm]['metrics'][key]) for key in METRICS),
                 'new nonfinite metric: ' + arm)
    new_gap = {key: audits['relplus']['metrics'][key] - audits['hha']['metrics'][key] for key in METRICS}
    same_arm = {arm: {key: audits[arm]['metrics'][key] - baseline['metrics'][arm][key]
                      for key in METRICS} for arm in ARMS}
    return {'status': 'PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON',
            'seed': 12345, 'training_batch_size': 56, 'training_lr': 0.00012,
            'training_focal_gamma': 1, 'baseline_focal_gamma': 2, 'checkpoint_epoch': 200,
            'evaluation_samples_per_arm': 17593,
            'class_names': before['arms']['hha']['config']['class_names'],
            'arms': audits, 'baseline_gamma2': baseline,
            'differences_percentage_points': {'relplus-hha': new_gap},
            'gamma1_minus_gamma2_percentage_points': same_arm,
            'relplus_minus_hha_gap_change_percentage_points': {
                key: new_gap[key] - baseline['relplus_minus_hha_pp'][key] for key in METRICS},
            'scientific_boundary': 'Single seed, fixed epoch 200; gamma is the sole changed shared control. '
                                   'Differences are descriptive and do not establish stable superiority.'}
