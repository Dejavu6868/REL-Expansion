"""Pinned gamma-1/gamma-2 evidence and fixed-epoch gamma-1.5 comparison."""
import copy
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
GAMMA1_COMPARISON_SHA256 = '5d535c50ef1c29fa55de00b3cb8069077f7dff0d117c9b6b8abf1324efd64125'
GAMMA1_SUITE_SHA256 = '319d3d63d207e1ca8c3594ed20eb119f954ecfcb24562262b313dd74eedb4731'
CONFIG_SHA256 = {
    1: {'hha': 'b2c73d1297ded7e3b3dc2aef3da4b26e5b1b50b62bc24f460006f225b90c1120',
        'relplus': 'a6ef3a63391b6d11a65726e2f753bfcc71898cfe956774fca73edab8d7728e01'},
    2: {'hha': '97bc3b5fd8e98a2663c0596685feda34a38804ac73a34b147376be5ea70f3b6d',
        'relplus': 'dece975cbb673f4a543ea14b73ac8d6ee3f4dbb40d920b116a773e949947eb34'},
}
IDENTITY_FIELDS = {
    'experiment_name', 'experiment_protocol_id', 'comparison_protocol_id',
    'output_dir', 'log_dir', 'tb_dir', 'checkpoint_dir', 'log_dir_link',
    'log_file', 'link_log_file', 'val_log_file', 'link_val_log_file', 'ddp_smoke_report',
    'root_dir', 'abs_dir',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_pinned(path, expected):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == expected, 'baseline SHA-256 mismatch: ' + str(path))
    return json.loads(raw), {'path': str(path), 'size': len(raw), 'sha256': actual}


def validate_baseline(bundle, suite):
    directory = Path(bundle) / 'baseline'
    data, comparison_source = _read_pinned(directory / 'comparison_verified.json', BASELINE_COMPARISON_SHA256)
    prior_suite, suite_source = _read_pinned(directory / 'suite.json', BASELINE_SUITE_SHA256)
    expected_shared = dict(prior_suite['shared'], focal_gamma=1.5)
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
            'seed': data['single_seed'],
            'suite_source': suite_source, 'training_bundle': prior_suite['remote_source_root'],
            'training_output_root': prior_suite['output_root'], 'metrics': metrics,
            'checkpoint_sha256': {arm: data['new_checkpoint_sha256'][arm] for arm in ARMS},
            'gt_histogram': histogram, 'test_source_sha256': TEST_SHA256,
            'relplus_minus_hha_pp': gap}


def _expected_config(old, old_suite, new_suite):
    """Rebase only exact run identities; retain every scientific/data field."""
    expected = copy.deepcopy(old)
    expected['focal_gamma'] = new_suite['shared']['focal_gamma']
    for field in IDENTITY_FIELDS:
        value = old[field]
        _require(isinstance(value, str), 'invalid configuration identity: ' + field)
        for key in ('remote_source_root', 'output_root', 'suite_id'):
            value = value.replace(old_suite[key], new_suite[key])
        expected[field] = value
    return expected


def _require_config(actual, expected, label):
    _require(set(actual) == set(expected), 'resolved configuration fields differ: ' + label)
    changed = sorted(key for key in actual if actual[key] != expected[key])
    _require(not changed, 'non-gamma or non-derived configuration change: %s: %s' % (label, changed))


def validate_gamma1_baseline(bundle, suite, gamma2):
    directory = Path(bundle) / 'baseline_gamma1'
    data, source = _read_pinned(directory / 'comparison.json', GAMMA1_COMPARISON_SHA256)
    prior, suite_source = _read_pinned(directory / 'suite.json', GAMMA1_SUITE_SHA256)
    gamma2_suite, _ = _read_pinned(Path(bundle) / 'baseline/suite.json', BASELINE_SUITE_SHA256)
    _require(prior['shared']['focal_gamma'] == 1 and
             suite['shared'] == dict(prior['shared'], focal_gamma=1.5),
             'gamma must be the only changed shared training control from gamma-1')
    _require(prior['shared'] == dict(gamma2_suite['shared'], focal_gamma=1),
             'gamma-1/gamma-2 frozen shared controls differ')
    _require(prior['arms'] == suite['arms'] == list(ARMS), 'gamma-1 arm scope mismatch')
    _require(data['status'] == 'PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON' and
             data['training_focal_gamma'] == 1 and data['baseline_focal_gamma'] == 2 and
             data['checkpoint_epoch'] == 200 and data['evaluation_samples_per_arm'] == 17593,
             'gamma-1 baseline endpoint did not pass')
    _require(data['seed'] == prior['shared']['seed'] == gamma2['seed'] == 12345,
             'gamma-1/gamma-2 seed mismatch')
    _require(data['training_batch_size'] == 56 and data['training_lr'] == 0.00012,
             'gamma-1 baseline training controls differ')
    for key in ('metrics', 'checkpoint_sha256', 'gt_histogram', 'test_source_sha256',
                'relplus_minus_hha_pp', 'training_bundle', 'training_output_root'):
        _require(data['baseline_gamma2'][key] == gamma2[key],
                 'gamma-1 historical gamma-2 attribution differs: ' + key)
    for key in ('source', 'suite_source'):
        _require(data['baseline_gamma2'][key]['sha256'] == gamma2[key]['sha256'],
                 'gamma-1 historical gamma-2 source fingerprint differs: ' + key)
    _require(set(data['arms']) == set(ARMS), 'gamma-1 requires exactly two audited arms')
    metrics, checkpoints, config_sources, config_links = {}, {}, {}, {}
    for arm in ARMS:
        config, config_sources[arm] = _read_pinned(directory / (arm + '.json'), CONFIG_SHA256[1][arm])
        old, old_source = _read_pinned(Path(bundle) / 'baseline' / ('gamma2_' + arm + '.json'),
                                     CONFIG_SHA256[2][arm])
        _require_config(config, _expected_config(old, gamma2_suite, prior), 'gamma2->gamma1/' + arm)
        _require(all(config.get(key) == value for key, value in prior['shared'].items()),
                 'gamma-1 resolved/shared configuration mismatch: ' + arm)
        _require(config['class_names'] == data['class_names'], 'gamma-1 class order mismatch')
        config_links[arm] = {'status': 'PASS_ONLY_FOCAL_GAMMA_AND_DERIVED_IDENTITIES_CHANGED',
                            'gamma2_config_source': old_source, 'gamma1_config_source': config_sources[arm]}
        audit = data['arms'][arm]
        row = audit['metrics']
        _require(audit['status'] == row['status'] == 'PASS' and
                 audit['gt_histogram'] == gamma2['gt_histogram'] and
                 audit['test_source_sha256'] == gamma2['test_source_sha256'],
                 'gamma-1 GT/test audit differs: ' + arm)
        _require(audit['rank_counts'] == [2200] + [2199] * 7, 'gamma-1 rank coverage mismatch')
        expected_metrics = {'checkpoint_epoch': 200, 'evaluation_sample_count': 17593,
                            'class_count': 13, 'ignore_index': 255, 'world_size': 8,
                            'eval_scale_array': [1], 'eval_flip': False,
                            'eval_crop_size': [480, 480], 'eval_align_corners': False,
                            'representation_protocol_id': config['representation_protocol_id'],
                            'integration_protocol_id': config['integration_protocol_id'],
                            'metric_unit': 'fraction_0_to_1', 'valid_pixel_count': 3973620198,
                            'scientific_metric_reported': True,
                            'checkpoint': str(Path(prior['output_root']) / 'runs' / arm / 'checkpoints/epoch-200.pth')}
        _require(all(row.get(key) == value for key, value in expected_metrics.items()),
                 'gamma-1 evaluation semantics differ: ' + arm)
        metrics[arm] = {key: row[key] for key in METRICS}
        _require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 100
                     for x in metrics[arm].values()), 'invalid gamma-1 metric: ' + arm)
        checkpoints[arm] = audit['checkpoint_sha256']
        _require(isinstance(checkpoints[arm], str) and len(checkpoints[arm]) == 64 and
                 all(c in '0123456789abcdef' for c in checkpoints[arm]), 'invalid gamma-1 checkpoint SHA-256')
    gap = {key: metrics['relplus'][key] - metrics['hha'][key] for key in METRICS}
    _require(data['differences_percentage_points']['relplus-hha'] == gap, 'gamma-1 gap mismatch')
    for key in METRICS:
        for arm in ARMS:
            _require(data['gamma1_minus_gamma2_percentage_points'][arm][key] ==
                     metrics[arm][key] - gamma2['metrics'][arm][key], 'gamma-1 historical same-arm delta mismatch')
        _require(data['relplus_minus_hha_gap_change_percentage_points'][key] ==
                 gap[key] - gamma2['relplus_minus_hha_pp'][key], 'gamma-1 historical gap change mismatch')
    return {'status': 'PASS', 'focal_gamma': 1, 'seed': data['seed'], 'source': source,
            'suite_source': suite_source, 'config_sources': config_sources,
            'frozen_config_comparisons': config_links, 'class_names': data['class_names'],
            'training_bundle': prior['remote_source_root'], 'training_output_root': prior['output_root'],
            'metrics': metrics, 'checkpoint_sha256': checkpoints,
            'gt_histogram': gamma2['gt_histogram'], 'test_source_sha256': TEST_SHA256,
            'relplus_minus_hha_pp': gap}


def validate_resolved_config(bundle, suite, arm, config):
    prior, _ = _read_pinned(Path(bundle) / 'baseline_gamma1/suite.json', GAMMA1_SUITE_SHA256)
    old, source = _read_pinned(Path(bundle) / 'baseline_gamma1' / (arm + '.json'), CONFIG_SHA256[1][arm])
    _require_config(config, _expected_config(old, prior, suite), 'gamma1->gamma1.5/' + arm)
    return {'status': 'PASS_ONLY_FOCAL_GAMMA_AND_DERIVED_IDENTITIES_CHANGED',
            'baseline_focal_gamma': 1, 'training_focal_gamma': 1.5,
            'resolved_field_count': len(config), 'baseline_config_source': source}


def build_comparison(audits, before):
    baselines = {name: before['baseline_' + name] for name in ('gamma1', 'gamma2')}
    _require(set(audits) == set(ARMS), 'comparison requires exactly two completed audits')
    suite = before['suite']
    _require(suite['shared']['focal_gamma'] == 1.5 and suite['shared']['seed'] == 12345,
             'new comparison gamma/seed mismatch')
    for label, baseline in baselines.items():
        _require(baseline['status'] == 'PASS' and baseline['seed'] == 12345,
                 'baseline validation/seed did not pass: ' + label)
        for arm in ARMS:
            _require(audits[arm]['gt_histogram'] == baseline['gt_histogram'], 'GT differs from ' + label + ': ' + arm)
            _require(audits[arm]['test_source_sha256'] == baseline['test_source_sha256'],
                     'test list differs from ' + label + ': ' + arm)
    for arm in ARMS:
        _require(audits[arm]['status'] == 'PASS', 'new audit did not pass: ' + arm)
        _require(before['arms'][arm]['config']['class_names'] == baselines['gamma1']['class_names'],
                 'new class order mismatch: ' + arm)
        _require(before['arms'][arm]['config_comparison']['status'] ==
                 'PASS_ONLY_FOCAL_GAMMA_AND_DERIVED_IDENTITIES_CHANGED', 'new config comparison missing')
        _require(all(type(audits[arm]['metrics'][key]) in (int, float) and
                     math.isfinite(audits[arm]['metrics'][key]) and 0 <= audits[arm]['metrics'][key] <= 100
                     for key in METRICS), 'new invalid metric: ' + arm)
    new_gap = {key: audits['relplus']['metrics'][key] - audits['hha']['metrics'][key] for key in METRICS}
    same_arm = {label: {arm: {key: audits[arm]['metrics'][key] - baseline['metrics'][arm][key]
                             for key in METRICS} for arm in ARMS} for label, baseline in baselines.items()}
    candidate_checks = {
        'relplus_mIoU_strictly_above_gamma1':
            audits['relplus']['metrics']['mIoU_percent'] > baselines['gamma1']['metrics']['relplus']['mIoU_percent'],
        'hha_mIoU_at_least_gamma1':
            audits['hha']['metrics']['mIoU_percent'] >= baselines['gamma1']['metrics']['hha']['mIoU_percent'],
    }
    return {'status': 'PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON',
            'seed': 12345, 'training_batch_size': 56, 'training_lr': 0.00012,
            'training_focal_gamma': 1.5, 'primary_baseline_focal_gamma': 1,
            'historical_baseline_focal_gamma': 2, 'checkpoint_epoch': 200,
            'evaluation_samples_per_arm': 17593,
            'class_names': before['arms']['hha']['config']['class_names'],
            'arms': audits, 'baseline_gamma1': baselines['gamma1'], 'baseline_gamma2': baselines['gamma2'],
            'config_comparisons': {arm: before['arms'][arm]['config_comparison'] for arm in ARMS},
            'differences_percentage_points': {'relplus-hha': new_gap},
            'gamma1_5_minus_gamma1_percentage_points': same_arm['gamma1'],
            'gamma1_5_minus_gamma2_percentage_points': same_arm['gamma2'],
            'relplus_minus_hha_percentage_points': dict(
                gamma1_5=new_gap, **{label: baseline['relplus_minus_hha_pp'] for label, baseline in baselines.items()}),
            'relplus_minus_hha_gap_change_percentage_points': {
                'gamma1_5_minus_' + label: {key: new_gap[key] - baseline['relplus_minus_hha_pp'][key]
                                         for key in METRICS} for label, baseline in baselines.items()},
            'candidate_screening': {
                'status': 'PASS' if all(candidate_checks.values()) else 'FAIL',
                'checks': candidate_checks, 'metric': 'mIoU_percent',
                'comparison_precision': 'full stored float precision; no rounding before comparison',
                'scope': 'Single-seed descriptive candidate screening; not a statistical superiority claim.',
            },
            'scientific_boundary': 'Single seed, fixed epoch 200; gamma is the sole changed shared control. '
                                   'Differences are descriptive and do not establish stable superiority.'}
