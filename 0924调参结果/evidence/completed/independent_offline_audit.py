"""Standard-library-only verification of retrieved gamma-1.5 evidence.

No imports from the experimental implementation, no network, no inference.
Only writes INDEPENDENT_OFFLINE_AUDIT.json alongside this script.
"""
import collections
import csv
import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parents[1]
REPO = BUNDLE.parents[1]
ARCHIVE = REPO / '0920调参结果'
ARMS = ('hha', 'relplus')
METRICS = ('mIoU_percent', 'pixel_accuracy_percent', 'mean_accuracy_percent')
IDENTITY = {'experiment_name', 'experiment_protocol_id', 'comparison_protocol_id',
            'output_dir', 'log_dir', 'tb_dir', 'checkpoint_dir', 'log_dir_link',
            'log_file', 'link_log_file', 'val_log_file', 'link_val_log_file',
            'ddp_smoke_report', 'root_dir', 'abs_dir'}
COUNTS = collections.Counter()
FAILURES = []
READ_HASHES = {}
REPORT = {'status': 'RUNNING', 'method': 'Independent standard-library integer matrices and exact Fraction arithmetic; no experimental audit modules imported',
          'scope': 'Retrieved evidence and local frozen code/configuration only; no network, training or inference',
          'checkpoint_bytes_available_locally': False,
          'checkpoint_boundary': 'Checkpoint paths, sizes and SHA256 records are cross-checked, not recomputed from local checkpoint bytes. Main agent must supply fresh remote checkpoint/source/data verification.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    READ_HASHES[str(path)] = sha(path)
    return json.loads(path.read_text())


def check(group, label, condition):
    COUNTS[group] += 1
    if not condition:
        FAILURES.append({'group': group, 'check': label})


def close(group, label, actual, expected, tolerance=1e-10):
    check(group, label, type(actual) in (int, float) and math.isfinite(actual)
          and abs(actual - expected) <= tolerance)


def rows(path):
    path = Path(path)
    READ_HASHES[str(path)] = sha(path)
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def matrix(path, label):
    fields, data = rows(path)
    check('matrix', label + ': columns', fields == ['true_class'] + ['pred_%d' % i for i in range(13)])
    check('matrix', label + ': ordered rows', [r['true_class'] for r in data] == list(map(str, range(13))))
    result = [[int(row['pred_%d' % i]) for i in range(13)] for row in data]
    check('matrix', label + ': dimensions and nonnegative integers', len(result) == 13 and all(len(r) == 13 and all(type(v) is int and v >= 0 for v in r) for r in result))
    return result


def calculate(cm):
    gt = [sum(row) for row in cm]
    predicted = [sum(cm[r][c] for r in range(13)) for c in range(13)]
    union = [gt[c] + predicted[c] - cm[c][c] for c in range(13)]
    iou = [Fraction(cm[c][c], union[c]) if union[c] else None for c in range(13)]
    accuracy = [Fraction(cm[c][c], gt[c]) if gt[c] else None for c in range(13)]
    mean = lambda seq: sum((v for v in seq if v is not None), Fraction()) / sum(v is not None for v in seq)
    fractions = {'mIoU': mean(iou), 'pixel_accuracy': Fraction(sum(cm[i][i] for i in range(13)), sum(gt)), 'mean_accuracy': mean(accuracy)}
    return {'metrics': {k + '_percent': float(v * 100) for k, v in fractions.items()},
            'fractions': {k: float(v) for k, v in fractions.items()},
            'per_class_iou': [float(v) if v is not None else None for v in iou],
            'per_class_iou_percent': [float(v * 100) if v is not None else None for v in iou],
            'per_class_accuracy_percent': [float(v * 100) if v is not None else None for v in accuracy],
            'gt_histogram': gt, 'valid_pixels': sum(gt)}


def compare_metrics(stored, computed, label):
    for k, v in computed['metrics'].items():
        close('metrics', label + ':' + k, stored[k], v)
    for k, v in computed['fractions'].items():
        close('metrics', label + ':' + k, stored[k], v, 1e-12)
    for key in ('per_class_iou', 'per_class_iou_percent'):
        check('metrics', label + ':' + key + ':count', len(stored[key]) == 13)
        for i, expected in enumerate(computed[key]):
            close('metrics', label + ':' + key + ':' + str(i), stored[key][i], expected)


def main():
    transfer = read(ROOT / 'TRANSFER_MANIFEST.json')
    for name, expected in transfer['files'].items():
        path = ROOT / name
        check('transfer', name, path.is_file() and path.stat().st_size == expected['size'] and sha(path) == expected['sha256'])
    for name, expected in transfer['inputs'].items():
        path = ROOT / 'inputs' / (name + '.txt')
        check('transfer', name, path.is_file() and path.stat().st_size == expected['size'] and sha(path) == expected['sha256'])
    check('transfer', '121 transferred files', len(transfer['files']) == 119 and len(transfer['inputs']) == 2)
    manifest = read(BUNDLE / 'bundle_manifest.json')
    for name, expected in manifest['files'].items():
        check('bundle', name, sha(BUNDLE / name) == expected)
    check('bundle', '206 files', len(manifest['files']) == 206)
    source = {n[len('source/'):]: h for n, h in manifest['files'].items() if n.startswith('source/')}
    frozen = read(BUNDLE / 'frozen_integrity.json')
    check('bundle', '174 frozen source hashes', len(source) == 174 and source == frozen['source_hashes'])
    for name, expected in source.items():
        check('frozen_source', name, sha(ARCHIVE / 'code/training/source' / name) == expected)
    suite = read(BUNDLE / 'suite.json')
    prior = read(BUNDLE / 'baseline_gamma1/suite.json')
    check('configuration', 'shared only gamma1->1.5', suite['shared'] == dict(prior['shared'], focal_gamma=1.5))
    check('configuration', 'exact two arms', suite['arms'] == list(ARMS))
    train_ids = (ROOT / 'inputs/train_source.txt').read_text().splitlines()
    test_ids = (ROOT / 'inputs/eval_source.txt').read_text().splitlines()
    check('dataset', 'train 52903 unique', len(train_ids) == len(set(train_ids)) == 52903)
    check('dataset', 'test 17593 unique', len(test_ids) == len(set(test_ids)) == 17593)
    check('dataset', 'train/test disjoint', not set(train_ids).intersection(test_ids))
    check('dataset', 'frozen train list', sha(ROOT / 'inputs/train_source.txt') == '96788184f2a1b318a05395a2c6b3867759526e0adb612a90eb6af59b1491b011')
    check('dataset', 'frozen test list', sha(ROOT / 'inputs/eval_source.txt') == 'b9de196c6c1aa8f9ac37926910af0806ce59b91eb068998711ddbb78eb24423a')
    pipeline = read(ROOT / 'pipeline_status.json')
    training = read(ROOT / 'train_status.json')
    prep = read(ROOT / 'preflight.json')
    evaluation = read(ROOT / 'evaluation_epoch200/status.json')
    before = read(ROOT / 'evaluation_epoch200/preflight.json')
    comparison = read(ROOT / 'evaluation_epoch200/dual_arm_comparison.json')
    for label, record in [('pipeline', pipeline), ('train', training), ('preflight', prep), ('evaluation', evaluation), ('evaluation_preflight', before)]:
        check('completion', label + ':PASS', record['status'] == 'PASS')
    check('completion', 'pipeline completed stages', pipeline['completed_stages'] == ['training', 'evaluation_epoch200'])
    check('completion', 'pipeline suite', pipeline['suite_id'] == suite['suite_id'])
    check('completion', 'pipeline identical bundle before/after', pipeline['bundle_integrity'] == pipeline['final_bundle_integrity'] == {'status': 'PASS', 'file_count': 206, 'manifest_sha256': sha(BUNDLE / 'bundle_manifest.json')})
    check('completion', 'training suite hash', training['suite_sha256'] == prep['suite_sha256'] == before['suite_sha256'] == sha(BUNDLE / 'suite.json'))
    check('completion', 'evaluation exact suite and bundle', before['suite'] == suite and before['source_hashes'] == manifest['files'] and before['bundle_manifest_sha256'] == sha(BUNDLE / 'bundle_manifest.json'))
    check('completion', 'evaluation exact training state', before['training'] == training)
    check('completion', 'training ordered two arms', training['arms'] == [x['arm'] for x in training['completed']] == list(ARMS))
    check('completion', 'smoke initialization', training['prerequisites']['status'] == 'PASS' and training['prerequisites']['rank_initialization_count'] == 16 and training['prerequisites']['model_sha256'] == suite['baseline_initial_model_sha256'])
    check('completion', 'preflight SHA', training['prerequisites']['preflight_sha256'] == sha(ROOT / 'preflight.json'))
    current, checkpoints, configs = {}, {}, {}
    for arm, completion in zip(ARMS, training['completed']):
        cfg = read(ROOT / 'configs' / (arm + '.json'))
        configs[arm] = cfg
        old = read(BUNDLE / 'baseline_gamma1' / (arm + '.json'))
        check('configuration', arm + ':archived baseline bytes', sha(BUNDLE / 'baseline_gamma1' / (arm + '.json')) == sha(BUNDLE / 'baseline' / (arm + '.json')) == sha(ARCHIVE / 'configs' / (arm + '.json')))
        expected = dict(old, focal_gamma=1.5)
        for field in IDENTITY:
            value = old[field]
            for key in ('remote_source_root', 'output_root', 'suite_id'):
                value = value.replace(prior[key], suite[key])
            expected[field] = value
        check('configuration', arm + ':full config only gamma and exact derived paths', cfg == expected)
        check('configuration', arm + ':shared values', all(cfg[k] == v for k, v in suite['shared'].items()))
        check('configuration', arm + ':saved config SHA', sha(ROOT / 'configs' / (arm + '.json')) == prep['arms'][arm]['resolved_config_sha256'])
        check('configuration', arm + ':eval reconstructed config', before['arms'][arm]['config'] == cfg)
        run = ROOT / 'runs' / arm
        check('training', arm + ':same completion records', read(run / 'arm_status.json') == completion)
        runtime = read(run / 'runtime_status.json')
        check('training', arm + ':200/189000/8rank/NaN0', runtime == {'status': 'FORMAL_TRAINING_COMPLETED', 'epoch': 200, 'iteration_in_epoch': 945, 'global_iteration': 189000, 'optimizer_step_executed': True, 'author_nan_replacement_count': 0, 'world_size': 8})
        check('training', arm + ':exit0', (run / 'exitcode').read_text().strip() == '0')
        check('training', arm + ':completion endpoint', completion['status'] == 'PASS' and completion['epoch'] == 200 and completion['global_iteration'] == 189000 and completion['rank_done_count'] == 8 and completion['model_sha256'] == suite['baseline_initial_model_sha256'])
        check('training', arm + ':no rank errors', not list(run.glob('rank_error_*.json')))
        for rank in range(8):
            label = arm + '/rank' + str(rank)
            init = read(run / ('rank_init_%02d.json' % rank))
            done = read(run / ('rank_done_%02d.json' % rank))
            check('training_rank', label + ':complete', done['status'] == 'COMPLETED' and done['rank'] == rank and done['epoch'] == 200 and done['global_iteration'] == 189000 and done['mode'] == 'formal' and done['suite_id'] == suite['suite_id'] and done['arm'] == arm)
            check('training_rank', label + ':initialization and config', init['model_sha256'] == suite['baseline_initial_model_sha256'] and init['rank'] == rank and init['world_size'] == 8 and init['config'] == cfg)
            check('training_rank', label + ':actual loss gamma1.5', init['actual_criterion'] == {'type': 'FocalLoss2d', 'focal_gamma': 1.5, 'reduction': 'none', 'ignore_index': 255})
            check('training_rank', label + ':source path/hash', init['source_root'] == suite['remote_source_root'] + '/source' and init['train_sha256'] == manifest['files']['source/train.py'])
            check('training_rank', label + ':optimization', init['local_batch'] == 7 and init['global_batch'] == 56 and init['loader_length'] == 945 and init['sampler_length'] == 6615 and init['initial_optimizer_lrs'] == [0.00012, 0.00012] and init['optimizer_betas'] == [[0.9, 0.999], [0.9, 0.999]] and init['optimizer_weight_decays'] == [0.01, 0.0] and init['scheduler_total_iterations'] == 189000 and init['scheduler_warmup_steps'] == 9450)
            check('training_rank', label + ':training decoder BN', [b['eps'] for b in init['batch_norm_layers'] if b['name'].startswith('decode_head.')] == [0.001])
        arm_eval = ROOT / 'evaluation_epoch200' / arm
        ev = arm_eval / 'evaluation'
        check('evaluation', arm + ':exit0', (arm_eval / 'exitcode').read_text().strip() == '0')
        check('evaluation', arm + ':PASS', read(arm_eval / 'status.json')['status'] == 'PASS')
        cm = matrix(ev / 'confusion_matrix.csv', arm)
        computed = calculate(cm)
        current[arm] = computed
        stored = read(ev / 'metrics.json')
        compare_metrics(stored, computed, arm)
        required = {'status': 'PASS', 'checkpoint_epoch': 200, 'evaluation_sample_count': 17593, 'class_count': 13, 'ignore_index': 255, 'world_size': 8, 'eval_scale_array': [1], 'eval_flip': False, 'eval_crop_size': [480, 480], 'eval_align_corners': False, 'metric_unit': 'fraction_0_to_1', 'valid_pixel_count': 3973620198, 'scientific_metric_reported': True, 'integration_protocol_id': cfg['integration_protocol_id'], 'representation_protocol_id': cfg['representation_protocol_id']}
        check('evaluation', arm + ':frozen semantics', all(stored.get(k) == v for k, v in required.items()))
        combined = [[0] * 13 for _ in range(13)]
        expected_manifest, all_ids = [], []
        for rank in range(8):
            label = arm + '/rank' + str(rank)
            rr = read(ev / ('rank_%02d_evaluation.json' % rank))
            er = read(arm_eval / ('rank_%02d_runtime.json' % rank))
            ids = test_ids[rank::8]
            check('evaluation_rank', label + ':exact ordered shard', rr['owned_sample_ids'] == ids and rr['sample_count'] == len(ids) and rr['rank'] == rr['local_rank'] == rank and rr['world_size'] == 8)
            check('evaluation_rank', label + ':runtime completion', er['status'] == 'COMPLETED' and er['processed_samples'] == len(ids) and er['rank'] == rank and er['arm'] == arm and er['exitcode'] == 0 and er['checkpoint_payload_epoch'] == 200 and er['amp'] is False)
            check('evaluation_rank', label + ':decoder BN', [b['eps'] for b in er['batch_norm'] if b['name'].startswith('decode_head.')] == [1e-5])
            check('evaluation_rank', label + ':frozen module roots', all(v.startswith(suite['remote_source_root'] + '/') for v in er['loaded_modules'].values()))
            cells = rr['confusion_matrix']
            valid = len(cells) == 13 and all(len(row) == 13 and all(type(v) is int and v >= 0 for v in row) for row in cells)
            check('evaluation_rank', label + ':integer confusion', valid)
            for i in range(13):
                for j in range(13):
                    combined[i][j] += cells[i][j]
            all_ids.extend(rr['owned_sample_ids'])
            expected_manifest.extend({'sample_id': sample_id, 'rank': str(rank)} for sample_id in ids)
        check('evaluation', arm + ':all rank matrices sum exactly', combined == cm)
        check('evaluation', arm + ':17593 unique sample union', len(all_ids) == len(set(all_ids)) == 17593 and set(all_ids) == set(test_ids))
        fields, listed = rows(ev / 'evaluation_manifest.csv')
        check('evaluation', arm + ':exact manifest order', fields == ['sample_id', 'rank'] and listed == expected_manifest)
        fields, classes = rows(ev / 'per_class_iou.csv')
        check('evaluation', arm + ':class CSV schema and ordering', fields == ['class_id', 'class_name', 'IoU_fraction', 'IoU_percent'] and [r['class_id'] for r in classes] == list(map(str, range(13))) and [r['class_name'] for r in classes] == cfg['class_names'])
        for i, row in enumerate(classes):
            close('metrics', arm + ':CSV class fraction ' + str(i), float(row['IoU_fraction']), computed['per_class_iou'][i])
            close('metrics', arm + ':CSV class percent ' + str(i), float(row['IoU_percent']), computed['per_class_iou_percent'][i])
        audit = read(arm_eval / 'integrity_audit.json')
        check('evaluation', arm + ':audit linkage', comparison['arms'][arm] == audit and audit['metrics'] == stored and audit['gt_histogram'] == computed['gt_histogram'] and audit['rank_counts'] == [2200] + [2199] * 7)
        check('evaluation', arm + ':valid pixels', computed['valid_pixels'] == 3973620198)
        cp = before['arms'][arm]['checkpoint']
        expected_path = suite['output_root'] + '/runs/' + arm + '/checkpoints/epoch-200.pth'
        check('checkpoint_records', arm + ':path/size/hash agree', cp['path'] == stored['checkpoint'] == completion['checkpoint'] == expected_path and cp['size'] == completion['checkpoint_size'] and cp['sha256'] == completion['checkpoint_sha256'] == audit['checkpoint_sha256'])
        checkpoints[arm] = cp
    check('evaluation', 'cross arm GT exact', current['hha']['gt_histogram'] == current['relplus']['gt_histogram'])
    baselines = {}
    archive_sums = {}
    for line in (ARCHIVE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        archive_sums[name.lstrip('*')] = digest
    for gamma in (1, 2):
        label = 'gamma' + str(gamma)
        baselines[label] = {}
        for arm in ARMS:
            folder = ARCHIVE / ('evidence/completed/evaluation_epoch200/' + arm + '/evaluation' if gamma == 1 else 'baseline_gamma2/' + arm)
            for filename in ('confusion_matrix.csv', 'metrics.json'):
                p = folder / filename
                check('baseline', label + '/' + arm + ':' + filename + ':archive hash', sha(p) == archive_sums[str(p.relative_to(ARCHIVE))])
            computed = calculate(matrix(folder / 'confusion_matrix.csv', label + '/' + arm))
            compare_metrics(read(folder / 'metrics.json'), computed, label + '/' + arm)
            for key, value in computed['metrics'].items():
                close('baseline', label + '/' + arm + ':pinned comparison:' + key, comparison['baseline_' + label]['metrics'][arm][key], value)
            check('baseline', label + '/' + arm + ':same GT', computed['gt_histogram'] == current[arm]['gt_histogram'])
            baselines[label][arm] = computed
    new_gap = {k: current['relplus']['metrics'][k] - current['hha']['metrics'][k] for k in METRICS}
    changes, gaps = {}, {'gamma1_5': new_gap}
    for label, arms in baselines.items():
        changes[label] = {}
        gaps[label] = {k: arms['relplus']['metrics'][k] - arms['hha']['metrics'][k] for k in METRICS}
        for arm in ARMS:
            changes[label][arm] = {k: current[arm]['metrics'][k] - arms[arm]['metrics'][k] for k in METRICS}
            for k, value in changes[label][arm].items():
                close('comparison', label + '/' + arm + ':' + k, comparison['gamma1_5_minus_' + label + '_percentage_points'][arm][k], value)
        for k in METRICS:
            close('comparison', label + ':gap:' + k, comparison['relplus_minus_hha_percentage_points'][label][k], gaps[label][k])
            close('comparison', label + ':gap change:' + k, comparison['relplus_minus_hha_gap_change_percentage_points']['gamma1_5_minus_' + label][k], new_gap[k] - gaps[label][k])
    for k, value in new_gap.items():
        close('comparison', 'new gap:' + k, comparison['differences_percentage_points']['relplus-hha'][k], value)
        close('comparison', 'new gap alias:' + k, comparison['relplus_minus_hha_percentage_points']['gamma1_5'][k], value)
    candidate = {'relplus_mIoU_strictly_above_gamma1': current['relplus']['metrics']['mIoU_percent'] > baselines['gamma1']['relplus']['metrics']['mIoU_percent'], 'hha_mIoU_at_least_gamma1': current['hha']['metrics']['mIoU_percent'] >= baselines['gamma1']['hha']['metrics']['mIoU_percent']}
    check('comparison', 'candidate independent flags', comparison['candidate_screening']['checks'] == candidate)
    candidate_status = 'PASS' if all(candidate.values()) else 'FAIL'
    check('comparison', 'candidate independent status', comparison['candidate_screening']['status'] == candidate_status)
    check('comparison', 'fixed endpoint attribution', comparison['status'] == 'PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON' and comparison['training_focal_gamma'] == 1.5 and comparison['primary_baseline_focal_gamma'] == 1 and comparison['historical_baseline_focal_gamma'] == 2 and comparison['checkpoint_epoch'] == 200 and comparison['seed'] == 12345 and comparison['class_names'] == configs['hha']['class_names'] == configs['relplus']['class_names'])
    REPORT.update(current=current, baseline_metrics={label: {arm: c['metrics'] for arm, c in arms.items()} for label, arms in baselines.items()},
                  changes_percentage_points=changes, relplus_minus_hha_percentage_points=gaps,
                  candidate_screening={'status': candidate_status, 'checks': candidate}, checkpoint_records=checkpoints,
                  actual_training_focal_gamma_all_16_ranks=1.5, shared_initialization_sha256=suite['baseline_initial_model_sha256'],
                  training_completed_ranks=16, evaluation_completed_ranks=16, unique_test_samples_per_arm=17593,
                  transferred_files_verified=121, bundle_files_verified=206, frozen_source_files_verified=174,
                  full_config_fields={'hha': len(configs['hha']), 'relplus': len(configs['relplus'])},
                  pipeline_finished_beijing=datetime.datetime.fromtimestamp(pipeline['finished_at'], datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  training_wall_seconds=training['finished_at']-training['started_at'],
                  evaluation_wall_seconds=evaluation['finished_at']-evaluation['started_at'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        FAILURES.append({'group': 'audit_execution', 'check': repr(error), 'traceback': traceback.format_exc()})
    REPORT.update(status='PASS_INDEPENDENT_OFFLINE_AUDIT' if not FAILURES else 'FAIL_INDEPENDENT_OFFLINE_AUDIT',
                  checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  check_count=sum(COUNTS.values()), check_counts_by_group=dict(COUNTS), failures=FAILURES,
                  files_read_sha256=READ_HASHES, audit_script_sha256=sha(__file__))
    (ROOT / 'INDEPENDENT_OFFLINE_AUDIT.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: REPORT.get(k) for k in ('status', 'check_count', 'check_counts_by_group', 'failures', 'changes_percentage_points', 'candidate_screening', 'pipeline_finished_beijing')}, ensure_ascii=False))
    raise SystemExit(0 if not FAILURES else 1)
