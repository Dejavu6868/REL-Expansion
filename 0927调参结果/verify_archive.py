#!/usr/bin/env python3
"""Standard-library read-only verification of the 0927 selected-recipe archive.

No model code, training, inference, network, or unarchived weights are accessed.
The original selected archive's verifier is executed as a subprocess. Its helper
functions are also used to independently recompute comparison confusion metrics.
"""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
ARMS = ('hha', 'relplus')
SOURCE_MANIFEST = '26be6da003118b90cb7309e7104894dd830e3725dd9a6fc400b800d44253c937'
EXPECTED = {'optimizer': 'AdamW', 'lr': 0.00012, 'focal_gamma': 1.0, 'weight_decay': 0.01,
            'batch_size': 56, 'warm_up_epoch': 10, 'lr_power': 0.9, 'nepochs': 200, 'seed': 12345}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10), label)


def files(root):
    result = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Symlink: ' + str(path))
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
    return result


def relative(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and p.as_posix() == name, 'Unsafe path')
    return name


def verify_inventory():
    entries = {}
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        require(match is not None, 'Malformed SHA256SUMS')
        digest, name = match.groups()
        relative(name)
        require(name not in entries and name != 'SHA256SUMS', 'Duplicate inventory member')
        entries[name] = digest
    require(files(ROOT) == set(entries) | {'SHA256SUMS'}, 'Archive file set differs')
    for name, digest in entries.items():
        require(sha(ROOT / name) == digest, 'SHA mismatch: ' + name)
    copied = read(ROOT / 'provenance/copied_files.json')
    for name, info in copied.items():
        relative(name)
        require(entries.get(name) == info['sha256'], 'Copy provenance digest mismatch: ' + name)
        require((ROOT / name).stat().st_size == info['size'], 'Copy size mismatch: ' + name)
    return entries


def main():
    inventory = verify_inventory()
    selected = ROOT / 'selected_recipe'
    require(len(files(selected)) == 459, 'Selected archive count differs')
    require(sha(selected / 'SHA256SUMS') == SOURCE_MANIFEST, 'Selected published inventory differs')
    result = subprocess.run([sys.executable, '-B', str(selected / 'verify_archive.py')],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    require(result.returncode == 0, 'Selected archive failed verification: ' + result.stdout + result.stderr)
    selected_check = json.loads(result.stdout)
    require(selected_check['status'] == 'PASS_OFFLINE_ARCHIVE_VERIFICATION', 'Selected verification status differs')
    spec = importlib.util.spec_from_file_location('selected_archive_metric_helpers', selected / 'verify_archive.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = read(ROOT / 'FROZEN_PARAMETERS.json')
    require(summary['status'] == 'FROZEN_SELECTED_EXISTING_RECIPE', 'Summary status differs')
    require(summary['selected_parameters'] == EXPECTED, 'Wrong selected recipe')
    require(summary['selected_recipe_id'] == 'gamma1_lr12', 'Wrong selection')
    require(summary['source_archive']['commit'] == 'c8ff74502cf6717fc554adc47db4d14e9ddd42a9', 'Wrong source commit')
    require(summary['source_archive']['manifest_sha256'] == SOURCE_MANIFEST, 'Wrong source manifest')
    require(summary['source_archive']['file_count'] == 459 and summary['source_archive']['matches_git_commit'] is True, 'Wrong source identity')
    require(summary['adamw_betas'] == [0.9, 0.999] and summary['optimizer_group_weight_decays'] == [0.01, 0.0], 'Optimizer differs')
    require(summary['world_size'] == 8 and summary['updates_per_arm'] == 189000, 'Budget differs')
    require(summary['new_training_started'] is False and summary['retrained_for_this_archive'] is False, 'Archive is not a retraining')
    require(summary['full_shared_training_config'] == read(selected / 'code/training/suite.json')['shared'], 'Shared config differs')
    require((ROOT / 'provenance/checkpoints.json').read_bytes() == (selected / 'provenance/checkpoints.json').read_bytes(), 'Checkpoint records differ')
    for arm in ARMS:
        cfg = read(ROOT / 'configs' / (arm + '.json'))
        require(all(cfg.get(k) == v for k, v in EXPECTED.items()), 'Resolved selected config differs: ' + arm)
        require((ROOT / 'configs' / (arm + '.json')).read_bytes() == (selected / 'configs' / (arm + '.json')).read_bytes(), 'Selected config bytes differ')

    values, ground_truth = {}, []
    dirs = {'gamma1_lr12': selected / 'evidence/completed/evaluation_epoch200',
            'gamma2_lr12': selected / 'baseline_gamma2',
            'gamma1_lr10': ROOT / 'comparisons/gamma1_lr10',
            'gamma15_lr12': ROOT / 'comparisons/gamma15_lr12'}
    for recipe, directory in dirs.items():
        values[recipe] = {}
        for arm in ARMS:
            arm_dir = directory / arm / 'evaluation' if recipe == 'gamma1_lr12' else directory / arm
            checked = module.check_metrics(arm_dir, arm)
            values[recipe][arm] = checked['metrics']['mIoU_percent']
            ground_truth.append(checked['gt'])
    require(all(x == ground_truth[0] for x in ground_truth), 'Ground truth histograms differ')

    for recipe, gamma, lr in [('gamma1_lr10', 1, 0.00010), ('gamma15_lr12', 1.5, 0.00012)]:
        folder = ROOT / 'comparisons' / recipe
        comp = read(folder / 'comparison.json')
        require(comp['seed'] == 12345 and comp['checkpoint_epoch'] == 200 and comp['evaluation_samples_per_arm'] == 17593, 'Comparison endpoint differs')
        require(comp['training_lr'] == lr and comp['training_focal_gamma'] == gamma, 'Comparison recipe differs')
        require(read(folder / 'pipeline_status.json')['status'] == 'PASS' and read(folder / 'train_status.json')['status'] == 'PASS', 'Comparison not completed')
        live = read(folder / 'LIVE_FINAL_RECHECK.json')
        expected_live = ('PASS_LIVE_FINAL_RECHECK', 292) if recipe == 'gamma1_lr10' else ('PASS', 309)
        require((live['status'], live['checks_count']) == expected_live and len(live['checks']) == expected_live[1], 'Prior live audit did not pass')
        require(read(folder / 'INDEPENDENT_OFFLINE_AUDIT.json')['status'] == 'PASS_INDEPENDENT_OFFLINE_AUDIT', 'Prior offline audit did not pass')
        for arm in ARMS:
            config = read(folder / 'configs' / (arm + '.json'))
            require(config['lr'] == lr and config['focal_gamma'] == gamma and config['weight_decay'] == 0.01 and config['seed'] == 12345, 'Comparison resolved config differs')
            close(comp['arms'][arm]['metrics']['mIoU_percent'], values[recipe][arm], 'Comparison metrics differ')
        candidate = values[recipe]['relplus'] > values['gamma1_lr12']['relplus'] and values[recipe]['hha'] >= values['gamma1_lr12']['hha']
        require(comp['candidate_screening']['status'] == ('PASS' if candidate else 'FAIL'), 'Candidate screen differs')

    full = read(ROOT / 'results/completed_recipes.json')
    rows = full['rows']
    require(len(rows) == 4 and {r['recipe'] for r in rows} == set(values), 'Wrong recipe coverage')
    params = {'gamma1_lr12': (1, 0.00012), 'gamma1_lr10': (1, 0.00010),
              'gamma15_lr12': (1.5, 0.00012), 'gamma2_lr12': (2, 0.00012)}
    for row in rows:
        recipe = row['recipe']
        require((row['focal_gamma'], row['lr']) == params[recipe] and row['weight_decay'] == 0.01, 'Score table recipe differs')
        for arm in ARMS:
            close(row[arm + '_mIoU_percent'], values[recipe][arm], 'Score table metric differs')
        close(row['relplus_minus_hha_pp'], values[recipe]['relplus'] - values[recipe]['hha'], 'Score table gap differs')
    with (ROOT / 'results/completed_recipes.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 4, 'CSV row count differs')
    for observed, expected in zip(csv_rows, rows):
        require(set(observed) == set(expected), 'CSV fields differ')
        for key, value in expected.items():
            close(observed[key], value, 'CSV value differs') if isinstance(value, (float, int)) else require(observed[key] == value, 'CSV text differs')
    require(max(values, key=lambda r: values[r]['relplus']) == full['selected_recipe'] == 'gamma1_lr12', 'Highest observed recipe selection differs')
    for arm in ARMS:
        close(summary['selected_metrics_percent'][arm]['mIoU_percent'], values['gamma1_lr12'][arm], 'Selected summary metric differs')
    print(json.dumps({'status': 'PASS_OFFLINE_ARCHIVE_VERIFICATION', 'archive_files': len(inventory) + 1,
                      'selected_archive_byte_identical_files': 459, 'selected_full_archive_verification': selected_check['status'],
                      'source_snapshot_files': 174, 'training_bundle_files': 200, 'recipes_compared': 4,
                      'confusion_matrices_recomputed': 8, 'selected_recipe': 'gamma1_lr12',
                      'selected_parameters': EXPECTED, 'mIoU_percent': values,
                      'data_or_weight_bytes_rehashed': False, 'new_training_or_inference': False}, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'FAIL_OFFLINE_ARCHIVE_VERIFICATION', 'error': str(error)}, ensure_ascii=False))
        sys.exit(1)
