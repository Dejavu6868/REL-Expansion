"""Independent B56/LR1.2e-4 experiment contract; no frozen source mutation."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ARMS = ('hha', 'relplus')
EXPECTED_SHARED = {
    'batch_size': 56, 'lr': 0.00012, 'nepochs': 200,
    'num_train_imgs': 52903, 'num_eval_imgs': 17593,
    'niters_per_epoch': 945, 'logical_samples_per_epoch': 52920,
    'sampler_padding_count': 17, 'warm_up_epoch': 10, 'lr_power': 0.9,
    'optimizer': 'AdamW', 'weight_decay': 0.01, 'criterion': 'Focal',
    'focal_gamma': 1.0, 'loss_reduction': 'none_then_mean',
    'seed': 12345, 'num_workers': 16, 'image_height': 480, 'image_width': 480,
    'num_classes': 13, 'background': 255, 'backbone': 'mit_b2',
    'decoder': 'MLPDecoder', 'decoder_embed_dim': 512,
    'amp': False, 'sync_bn': True, 'bn_eps': 0.001, 'bn_momentum': 0.1,
    'train_horizontal_flip': False, 'train_vertical_flip': False,
    'train_arbitrary_rotation': False, 'train_perspective_warp': False,
    'train_scale_array': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    'scheduler': 'WarmUpPolyLR', 'checkpoint_start_epoch': 100,
    'checkpoint_step': 5, 'recovery_checkpoint_step': 5,
    'recovery_checkpoint_before_epoch': 100,
    'primary_endpoint': 'epoch_200', 'eval_scale_array': [1], 'eval_flip': False,
}


def budget(n, batch, epochs=200, warmup=10):
    if min(n, batch, epochs) <= 0 or not 0 <= warmup <= epochs:
        raise ValueError('invalid training budget')
    steps = (n + batch - 1) // batch
    return {'niters_per_epoch': steps, 'logical_samples_per_epoch': steps * batch,
            'sampler_padding_count': steps * batch - n,
            'total_updates': steps * epochs, 'warmup_updates': steps * warmup}


def validate_paths(suite):
    source = Path(suite['remote_source_root'])
    output = Path(suite['output_root'])
    if not source.is_absolute() or not output.is_absolute():
        raise ValueError('all runtime roots must be absolute')
    if source.parent != Path('/home/zhuzhaoziao/RELPlus') or source.name != 'CMX-S2D-B56-LR12-FG1-20260920':
        raise ValueError('independent source directory required')
    if (output.parent.parent != Path('/data/zhuzhaoziao/RELPlus/outputs') or
            output.parent.name != 'CMX_S2D_B56_LR12_FG1_20260920' or
            not output.name.startswith('attempt')):
        raise ValueError('independent experiment/attempt output required')


def validate_shared(values):
    for key, expected in EXPECTED_SHARED.items():
        if key not in values or values[key] != expected:
            raise ValueError('shared control mismatch: {}: {!r} != {!r}'.format(key, values.get(key), expected))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError('not JSON serializable: {}'.format(type(value).__name__))


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(path))


def load_suite(path):
    suite = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_paths(suite)
    if set(suite['shared']) != set(EXPECTED_SHARED):
        raise ValueError('shared key set must exactly match frozen controls')
    validate_shared(suite['shared'])
    if suite.get('arms') != list(ARMS) or suite.get('authorization', {}).get('approved') is not True:
        raise ValueError('exact two-arm scope and explicit training authorization required')
    return suite


def configure_imports(suite):
    source = Path(suite['remote_source_root']) / 'source'
    if not (source / 'train.py').is_file():
        raise FileNotFoundError(source / 'train.py')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))


BASELINE_CONFIG_HASHES = {
    'hha': '97bc3b5fd8e98a2663c0596685feda34a38804ac73a34b147376be5ea70f3b6d',
    'relplus': 'dece975cbb673f4a543ea14b73ac8d6ee3f4dbb40d920b116a773e949947eb34',
}
IDENTITY_FIELDS = {
    'experiment_name', 'experiment_protocol_id', 'comparison_protocol_id',
    'output_dir', 'log_dir', 'tb_dir', 'checkpoint_dir', 'log_dir_link',
    'log_file', 'link_log_file', 'val_log_file', 'link_val_log_file', 'ddp_smoke_report',
    'root_dir', 'abs_dir',
}


def baseline_config_comparison(config, arm):
    path = Path(__file__).resolve().parent / 'baseline' / (arm + '.json')
    if digest(path) != BASELINE_CONFIG_HASHES[arm]:
        raise ValueError('frozen baseline config fingerprint changed: ' + arm)
    old = json.loads(path.read_text())
    actual = json.loads(json.dumps(dict(config), default=json_default))
    if set(old) != set(actual):
        raise ValueError('full resolved config key set differs from baseline')
    expected_source = '/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-FG1-20260920/source'
    if any(actual[k] != expected_source for k in ('root_dir', 'abs_dir')):
        raise ValueError('source identity must point to the exact new frozen snapshot')
    differences = {k: {'before': old[k], 'after': actual[k]}
                   for k in old if old[k] != actual[k]}
    semantic_changes = set(differences) - IDENTITY_FIELDS
    if semantic_changes != {'focal_gamma'} or old['focal_gamma'] != 2 or actual['focal_gamma'] != 1:
        raise ValueError('non-gamma training/data change: {}'.format(semantic_changes))
    return {'status': 'PASS_ONLY_FOCAL_GAMMA_CHANGED', 'arm': arm,
            'baseline_sha256': BASELINE_CONFIG_HASHES[arm],
            'resolved_field_count': len(actual), 'differences': differences,
            'semantic_changes': sorted(semantic_changes)}


def build_config(suite, arm, mode='formal'):
    if arm not in ARMS or mode not in ('formal', 'smoke'):
        raise ValueError('unsupported arm or mode')
    module_name = ('configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_3_formal'
                   if arm == 'relplus' else
                   'configs.stanford2d3d_s2d.cmx_mit_b2_{}_three_arm_v1'.format(arm))
    config = copy.deepcopy(importlib.import_module(module_name).config)
    for key, value in suite['shared'].items():
        setattr(config, key, copy.deepcopy(value))
    config.experiment_name = '{}_{}'.format(suite['suite_id'], arm)
    config.experiment_protocol_id = suite['suite_id']
    config.comparison_protocol_id = suite['suite_id']
    config.comparison_arm = arm
    config.arm_name = {'rgbd': 'CMX-RGBD', 'hha': 'CMX-HHA', 'relplus': 'CMX-REL+'}[arm]
    config.training_authorized = suite['authorization']['approved'] is True
    config.source_compatible_invalid_accepted = arm == 'relplus'
    config.full_cache_authorized = False
    run = Path(suite['output_root']) / ('runs' if mode == 'formal' else 'smoke') / arm
    config.output_dir = str(run)
    config.log_dir = str(run / 'logs')
    config.tb_dir = str(run / 'tensorboard')
    config.checkpoint_dir = str(run / 'checkpoints')
    config.log_dir_link = str(run / 'latest_logs')
    for key, name in (('log_file', 'train.log'), ('link_log_file', 'train_last.log'),
                      ('val_log_file', 'val.log'), ('link_val_log_file', 'val_last.log')):
        setattr(config, key, str(run / 'logs' / name))
    config.ddp_smoke_report = str(Path(suite['output_root']) / 'smoke' / arm / 'ddp_optimizer_smoke_summary.json')
    config.checkpoint_epochs = list(range(100, 201, 5))
    config.secondary_endpoint = 'not_selected'
    from dataloader.data_setting import build_data_setting
    config.data_setting = build_data_setting(config, split='train')
    validate_config(config, suite, arm)
    baseline_config_comparison(config, arm)
    return config


def validate_config(config, suite, arm):
    validate_shared(config)
    if config.training_authorized is not True or config.comparison_arm != arm:
        raise ValueError('config authorization or arm mismatch')
    expected_integration = 'CMX_RELPLUS_V2_3' if arm == 'relplus' else 'CMX_S2D_THREE_ARM_V1'
    if config.integration_protocol_id != expected_integration:
        raise ValueError('original strict data integration identity must be preserved')
    if config.source_compatible_invalid_accepted != (arm == 'relplus'):
        raise ValueError('invalid-storage acceptance mismatch')
    root = Path(suite['output_root']).resolve()
    for field in ('output_dir', 'log_dir', 'tb_dir', 'checkpoint_dir', 'log_dir_link',
                  'log_file', 'link_log_file', 'val_log_file', 'link_val_log_file'):
        if root not in Path(config[field]).resolve().parents:
            raise ValueError('output escapes new experiment: {}'.format(field))
