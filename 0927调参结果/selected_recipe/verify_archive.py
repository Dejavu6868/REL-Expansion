#!/usr/bin/env python3
"""Read-only, offline verification of the frozen 0920 gamma=1 archive.

Usage: python3 -B verify_archive.py [ARCHIVE_DIRECTORY]
Only Python 3.8+ standard library is used. No model code is imported or executed;
no network, inference, training, or file writes occur. Absent weights/raw data
cannot be rehashed here: only their archived identity records are compared.
SHA256SUMS checks integrity, not independent experimental authenticity.
"""
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parent
ARMS = ("hha", "relplus")
LABELS = {"hha": "HHA", "relplus": "RELPlus"}
N, C, WORLD, PIXELS = 17593, 13, 8, 3973620198
TEST_SHA = "b9de196c6c1aa8f9ac37926910af0806ce59b91eb068998711ddbb78eb24423a"
TRAIN_SHA = "96788184f2a1b318a05395a2c6b3867759526e0adb612a90eb6af59b1491b011"
BUNDLE_SHA = "42df431764b0c0aad2f2a71214bda28dfc6a34d9d06eea2ab9bebad6b342892c"
SUITE_SHA = "319d3d63d207e1ca8c3594ed20eb119f954ecfcb24562262b313dd74eedb4731"
INITIAL_SHA = "87254603a9898e551964e4a92c561a6d74273d678ee4ed631c3fc5fc223c25fa"
CLASSES = ["beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
           "door", "floor", "sofa", "table", "wall", "window"]
METRICS = ("mIoU", "pixel_accuracy", "mean_accuracy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object, parse_constant=invalid_constant)


def integer(value, label):
    if isinstance(value, str):
        require(re.fullmatch(r"[0-9]+", value) is not None, label + ": not integer text")
        value = int(value)
    require(type(value) is int and 0 <= value <= 2 ** 63 - 1,
            label + ": not a nonnegative int64")
    return value


def close(value, expected, label, tolerance=1e-10):
    require(not isinstance(value, bool), label + ": Boolean is not a metric")
    actual = float(value)
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
            label + ": value differs from independent recomputation")


def relative_path(name):
    require(isinstance(name, str) and name != "" and "\\" not in name,
            "invalid inventory filename")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and
            path.as_posix() == name and "." not in path.parts,
            "unsafe inventory filename: " + name)
    return path


def file_set(root):
    result = set()
    require(root.is_dir(), "missing directory: " + str(root))
    for p in root.rglob("*"):
        require(not p.is_symlink(), "archive symlinks are not allowed: " + str(p))
        if p.is_file():
            result.add(p.relative_to(root).as_posix())
        else:
            require(p.is_dir(), "unexpected filesystem entry: " + str(p))
    return result


def verify_inventory(root, name, count=None):
    inventory = {}
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed SHA inventory line: " + name)
        digest, relative = match.groups()
        relative_path(relative)
        require(relative not in inventory and relative != name,
                "duplicate or self-referential SHA entry: " + relative)
        inventory[relative] = digest
    if count is not None:
        require(len(inventory) == count, name + ": wrong inventory count")
    require(file_set(root) == set(inventory) | {name},
            name + ": actual file set does not equal the inventory")
    for relative, expected in inventory.items():
        require(sha(root / relative) == expected, "SHA-256 mismatch: " + relative)
    return inventory


def read_csv(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, "CSV header mismatch: " + str(path))
        rows = list(reader)
    require(all(set(row) == set(fields) and all(v is not None for v in row.values())
                for row in rows), "malformed CSV row: " + str(path))
    return rows


def matrix(value, label, strings=False):
    require(isinstance(value, list) and len(value) == C, label + ": expected 13 rows")
    result = []
    for row in value:
        require(isinstance(row, list) and len(row) == C, label + ": expected 13 columns")
        if not strings:
            require(all(type(v) is int for v in row), label + ": matrix must contain integers")
        result.append([integer(v, label) for v in row])
    require(sum(map(sum, result)) <= PIXELS, label + ": excess pixel count")
    return result


def check_metrics(directory, arm):
    observed = read_json(directory / "metrics.json")
    expected_protocol = ("CMX_RELPLUS_V2_3" if arm == "relplus" else "CMX_S2D_THREE_ARM_V1")
    representation = {"rgbd": "RGBD_UINT8_REPEAT3_SOURCECOMPAT",
                      "hha": "HHA_FROZEN_CACHE_EXECUTABLE_ORDER",
                      "relplus": "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"}[arm]
    fixed = {"status": "PASS", "checkpoint_epoch": 200, "evaluation_sample_count": N,
             "class_count": C, "ignore_index": 255, "world_size": WORLD,
             "eval_scale_array": [1], "eval_flip": False, "eval_crop_size": [480, 480],
             "eval_align_corners": False, "metric_unit": "fraction_0_to_1",
             "scientific_metric_reported": True, "integration_protocol_id": expected_protocol,
             "representation_protocol_id": representation, "valid_pixel_count": PIXELS}
    for key, value in fixed.items():
        require(key in observed and type(observed[key]) is type(value) and observed[key] == value,
                str(directory) + ": invalid metric protocol field " + key)
    rows = read_csv(directory / "confusion_matrix.csv", ["true_class"] +
                    ["pred_{}".format(i) for i in range(C)])
    require([integer(r["true_class"], "true_class") for r in rows] == list(range(C)),
            "confusion CSV class order differs")
    hist = matrix([[r["pred_{}".format(i)] for i in range(C)] for r in rows],
                  str(directory), strings=True)
    require(sum(map(sum, hist)) == PIXELS, "total valid pixels differ")
    gt = [sum(row) for row in hist]
    pred = [sum(hist[r][c] for r in range(C)) for c in range(C)]
    require(all(v > 0 for v in gt), "each frozen class must have ground-truth pixels")
    iou = [hist[i][i] / (gt[i] + pred[i] - hist[i][i]) for i in range(C)]
    accuracy = [hist[i][i] / gt[i] for i in range(C)]
    computed = {"mIoU": sum(iou) / C, "pixel_accuracy": sum(hist[i][i] for i in range(C)) / PIXELS,
                "mean_accuracy": sum(accuracy) / C}
    for key, value in computed.items():
        close(observed[key], value, key, 1e-12)
        close(observed[key + "_percent"], value * 100, key + "_percent")
    for key, scale in (("per_class_iou", 1), ("per_class_iou_percent", 100)):
        require(isinstance(observed[key], list) and len(observed[key]) == C, key + ": invalid shape")
        for i in range(C):
            close(observed[key][i], iou[i] * scale, key)
    per_class = read_csv(directory / "per_class_iou.csv",
                         ["class_id", "class_name", "IoU_fraction", "IoU_percent"])
    require([integer(r["class_id"], "class_id") for r in per_class] == list(range(C)) and
            [r["class_name"] for r in per_class] == CLASSES, "per-class CSV order differs")
    for i, row in enumerate(per_class):
        close(row["IoU_fraction"], iou[i], "class IoU", 1e-12)
        close(row["IoU_percent"], iou[i] * 100, "class IoU percent")
    return {"metrics": observed, "matrix": hist, "gt": gt, "iou": iou}


def fixed_fields(observed, expected, label):
    for key, value in expected.items():
        require(key in observed and type(observed[key]) is type(value) and observed[key] == value,
                label + ': invalid field ' + key)


def verify_transfer(completed):
    manifest = read_json(completed / 'TRANSFER_MANIFEST.json')
    require(len(manifest['files']) == 114, 'completed transfer must contain 114 files')
    for name, row in manifest['files'].items():
        relative_path(name)
        path = completed / name
        require(path.stat().st_size == row['size'] and sha(path) == row['sha256'],
                'completed transfer mismatch: ' + name)
    extras = {'TRANSFER_MANIFEST.json', 'TRANSFER_VERIFICATION.json', 'LIVE_FINAL_RECHECK.json',
              'INDEPENDENT_OFFLINE_AUDIT.json', 'local_status_before_completion.json'}
    require(file_set(completed) == set(manifest['files']) | extras,
            'completed evidence exact file set differs')
    receipt = read_json(completed / 'TRANSFER_VERIFICATION.json')
    fixed_fields(receipt, {'status': 'PASS', 'file_count': 114,
                          'total_bytes': sum(r['size'] for r in manifest['files'].values()),
                          'manifest_sha256': sha(completed / 'TRANSFER_MANIFEST.json')}, 'transfer receipt')
    prep = ROOT / 'evidence/preparation'
    prep_manifest = read_json(prep / 'TRANSFER_MANIFEST.json')
    require(len(prep_manifest['files']) == 80, 'preparation transfer must contain 80 files')
    require(file_set(prep) == set(prep_manifest['files']) | {'TRANSFER_MANIFEST.json'},
            'preparation exact file set differs')
    for name, digest in prep_manifest['files'].items():
        relative_path(name)
        require(sha(prep / name) == digest, 'preparation transfer mismatch: ' + name)
    return manifest


def verify_code(preflight):
    training = ROOT / 'code/training'
    bundle = read_json(training / 'bundle_manifest.json')
    expected = bundle['files']
    require(len(expected) == 200 and len([p for p in expected if p.startswith('source/')]) == 174,
            'frozen bundle must contain 200 files and 174 source files')
    require(file_set(training) == set(expected) | {'bundle_manifest.json'},
            'training bundle exact file set differs')
    require(expected == preflight['source_hashes'], 'evaluation used a different bundle')
    require(sha(training / 'bundle_manifest.json') == preflight['bundle_manifest_sha256'] == BUNDLE_SHA,
            'bundle manifest fingerprint differs')
    require(sha(training / 'suite.json') == preflight['suite_sha256'] == SUITE_SHA,
            'suite fingerprint differs')
    for name, digest in expected.items():
        relative_path(name)
        require(sha(training / name) == digest, 'bundle SHA-256 mismatch: ' + name)
    for name, digest in preflight['implementation_hashes'].items():
        relative_path(name)
        require(expected['evaluation/' + name] == digest, 'evaluation implementation differs: ' + name)
    suite = read_json(training / 'suite.json')
    require(suite == preflight['suite'] and suite['arms'] == list(ARMS), 'suite/arm set differs')
    prior_suite = read_json(training / 'baseline/suite.json')
    require(set(suite['shared']) == set(prior_suite['shared']), 'shared control field set differs')
    delta = {k for k in suite['shared'] if suite['shared'][k] != prior_suite['shared'][k]}
    require(delta == {'focal_gamma'} and suite['shared']['focal_gamma'] == 1 and
            prior_suite['shared']['focal_gamma'] == 2, 'sole shared changed factor is not gamma2 to gamma1')
    require(suite['baseline_initial_model_sha256'] == INITIAL_SHA, 'initial model identity differs')
    require(suite['budget']['optimizer_updates_per_arm'] == 189000 and
            suite['budget']['warmup_updates'] == 9450, 'update budget differs')
    return suite


def verify_config(arm, preflight, suite):
    cfg = read_json(ROOT / 'configs' / (arm + '.json'))
    raw_cfg = ROOT / 'evidence/completed/configs' / (arm + '.json')
    require(cfg == preflight['arms'][arm]['config'] == read_json(raw_cfg) and
            sha(ROOT / 'configs' / (arm + '.json')) == sha(raw_cfg), arm + ': config copies differ')
    old = read_json(ROOT / 'code/training/baseline' / (arm + '.json'))
    require(set(cfg) == set(old), arm + ': complete config key set changed')
    out = suite['output_root'] + '/runs/' + arm
    identity = {
        'root_dir': suite['remote_source_root'] + '/source',
        'abs_dir': suite['remote_source_root'] + '/source',
        'experiment_name': suite['suite_id'] + '_' + arm,
        'experiment_protocol_id': suite['suite_id'], 'comparison_protocol_id': suite['suite_id'],
        'output_dir': out, 'log_dir': out + '/logs', 'tb_dir': out + '/tensorboard',
        'log_dir_link': out + '/latest_logs', 'checkpoint_dir': out + '/checkpoints',
        'log_file': out + '/logs/train.log', 'link_log_file': out + '/logs/train_last.log',
        'val_log_file': out + '/logs/val.log', 'link_val_log_file': out + '/logs/val_last.log',
        'ddp_smoke_report': suite['output_root'] + '/smoke/' + arm + '/ddp_optimizer_smoke_summary.json'}
    expected = dict(old, focal_gamma=1.0, **identity)
    require(cfg == expected, arm + ': config differs outside gamma and approved exact identities')
    require(cfg['class_names'] == CLASSES and cfg['comparison_arm'] == arm,
            arm + ': class order/config arm differs')
    for key, value in suite['shared'].items():
        require(cfg[key] == value, arm + ': shared config differs: ' + key)
    return cfg


def verify_training(preflight, suite, completed):
    pipeline = read_json(completed / 'pipeline_status.json')
    fixed_fields(pipeline, {'status': 'PASS', 'phase': 'execute', 'suite_id': suite['suite_id'],
                           'completed_stages': ['training', 'evaluation_epoch200'], 'active_stage': None},
                 'pipeline')
    for name in ('bundle_integrity', 'final_bundle_integrity'):
        fixed_fields(pipeline[name], {'status': 'PASS', 'file_count': 200, 'manifest_sha256': BUNDLE_SHA}, name)
    queue = read_json(completed / 'train_status.json')
    require(queue == preflight['training'], 'training completion differs from evaluation prerequisite')
    fixed_fields(queue, {'status': 'PASS', 'phase': 'train', 'suite_id': suite['suite_id'],
                        'arms': list(ARMS), 'active_arm': None, 'suite_sha256': SUITE_SHA}, 'training queue')
    require([r['arm'] for r in queue['completed']] == list(ARMS), 'training arm completion set differs')
    prerequisites = queue['prerequisites']
    require(prerequisites == preflight['training_prerequisites'], 'initialization prerequisites differ')
    fixed_fields(prerequisites, {'status': 'PASS', 'model_sha256': INITIAL_SHA,
                                'rank_initialization_count': 16}, 'training prerequisites')
    require(prerequisites['preflight_sha256'] == sha(ROOT / 'evidence/preparation/preflight.json'),
            'training prerequisite preflight fingerprint differs')
    preparation = read_json(ROOT / 'evidence/preparation/preflight.json')
    require(preparation['status'] == 'PASS', 'preparation preflight failed')
    for arm, summary in zip(ARMS, queue['completed']):
        cfg = verify_config(arm, preflight, suite)
        prep_arm = preparation['arms'][arm]
        require(prep_arm['status'] == 'PASS' and prep_arm['train_count'] == 52903 and
                prep_arm['test_count'] == N and prep_arm['train_test_disjoint'] is True and
                prep_arm['resolved_config_sha256'] == sha(completed / 'configs' / (arm + '.json')),
                arm + ': preparation split/config identity differs')
        directory = completed / 'runs' / arm
        require(read_json(directory / 'arm_status.json') == summary, arm + ': arm/queue summary differs')
        require((directory / 'exitcode').read_text().strip() == '0', arm + ': training exit nonzero')
        require(read_json(directory / 'cleanup.json')['remaining_members'] == [], arm + ': cleanup incomplete')
        fixed_fields(summary, {'status': 'PASS', 'epoch': 200, 'global_iteration': 189000,
                               'rank_done_count': WORLD, 'model_sha256': INITIAL_SHA}, arm + ' training endpoint')
        fixed_fields(read_json(directory / 'runtime_status.json'),
                     {'status': 'FORMAL_TRAINING_COMPLETED', 'epoch': 200, 'global_iteration': 189000,
                      'iteration_in_epoch': 945, 'author_nan_replacement_count': 0, 'world_size': WORLD,
                      'optimizer_step_executed': True}, arm + ' runtime')
        fixed_fields(read_json(directory / 'initialization_check.json'),
                     {'status': 'PASS', 'model_sha256': INITIAL_SHA, 'rank_initialization_count': WORLD},
                     arm + ' initialization')
        require(prerequisites['smoke_arms'][arm]['smoke_summary_sha256'] ==
                sha(ROOT / 'evidence/preparation/smoke' / arm / 'ddp_optimizer_smoke_summary.json'),
                arm + ': smoke prerequisite fingerprint differs')
        for rank in range(WORLD):
            done = read_json(directory / ('rank_done_%02d.json' % rank))
            fixed_fields(done, {'status': 'COMPLETED', 'suite_id': suite['suite_id'], 'arm': arm,
                                'mode': 'formal', 'rank': rank, 'epoch': 200, 'global_iteration': 189000},
                         arm + ' training rank completion')
            init = read_json(directory / ('rank_init_%02d.json' % rank))
            fixed_fields(init, {'status': 'INITIALIZED', 'suite_id': suite['suite_id'], 'arm': arm,
                                'mode': 'formal', 'rank': rank, 'world_size': WORLD,
                                'model_sha256': INITIAL_SHA, 'source_root': suite['remote_source_root'] + '/source',
                                'sampler_length': 6615, 'loader_length': 945, 'local_batch': 7, 'global_batch': 56,
                                'scheduler_warmup_steps': 9450,
                                'actual_criterion': {'type': 'FocalLoss2d', 'focal_gamma': 1.0,
                                                     'reduction': 'none', 'ignore_index': 255}},
                         arm + ' actual rank initialization')
            require(init['config'] == cfg and init['train_sha256'] ==
                    sha(ROOT / 'code/training/source/train.py'), arm + ': initialized code/config differs')
            require(init['initial_optimizer_lrs'] == [0.00012, 0.00012] and
                    init['optimizer_betas'] == [[0.9, 0.999], [0.9, 0.999]] and
                    init['optimizer_weight_decays'] == [0.01, 0.0] and
                    init['scheduler_total_iterations'] == 189000, arm + ': actual optimizer/scheduler differs')
            decoder = [r for r in init['batch_norm_layers'] if r['name'] == 'decode_head.linear_fuse.1']
            require(len(decoder) == 1 and decoder[0]['type'] == 'SyncBatchNorm' and
                    decoder[0]['eps'] == 0.001, arm + ': training BN differs')
        cp = preflight['arms'][arm]['checkpoint']
        require(cp['path'] == summary['checkpoint'] and cp['size'] == summary['checkpoint_size'] and
                cp['sha256'] == summary['checkpoint_sha256'], arm + ': checkpoint record mismatch')
        require(re.fullmatch(r'[0-9a-f]{64}', cp['sha256']) is not None and cp['size'] > 0,
                arm + ': invalid checkpoint identity')
        integer(cp['mtime_ns'], 'checkpoint mtime')
    return queue


def verify_new_arm(arm, preflight, suite):
    out = ROOT / "evidence/completed/evaluation_epoch200" / arm
    evidence = preflight["arms"][arm]
    cfg = evidence["config"]
    cp = evidence["checkpoint"]
    require(cfg["class_names"] == CLASSES and cfg["comparison_arm"] == arm,
            arm + ": class order or config arm differs")
    for key, value in suite["shared"].items():
        require(cfg[key] == value, arm + ": training/evaluation config differs: " + key)
    require(evidence["ordered_test_count"] == N and evidence["test_list_sha256"] == TEST_SHA,
            arm + ": preflight test identity differs")
    require((out / "exitcode").read_text().strip() == "0", arm + ": evaluation exit nonzero")
    require(read_json(out / "status.json")["status"] == "PASS", arm + ": evaluation did not pass")
    require(read_json(out / "cleanup.json")["remaining_members"] == [], arm + ": cleanup incomplete")
    result = check_metrics(out / "evaluation", arm)
    require(result["metrics"]["checkpoint"] == cp["path"], arm + ": evaluated checkpoint path differs")
    test_path = ROOT / "evidence/completed/inputs" / (arm + "_test.txt")
    require(sha(test_path) == TEST_SHA, arm + ": archived ordered test fingerprint differs")
    test_ids = test_path.read_text(encoding="utf-8").splitlines()
    require(len(test_ids) == N and len(set(test_ids)) == N and all(test_ids),
            arm + ": archived ordered test IDs are not unique nonempty 17593")
    shards, matrices = [], []
    expected_modules = {"evaluator": "tools/eval_rel_plus_v2_3_full.py",
                        "core": "engine/relplus_evaluator.py", "model": "models/builder.py",
                        "dataset": "dataloader/RGBXDataset.py"}
    for rank in range(WORLD):
        runtime = read_json(out / "rank_{:02d}_runtime.json".format(rank))
        count = len(range(rank, N, WORLD))
        fixed = {"status": "COMPLETED", "rank": rank, "arm": arm, "exitcode": 0,
                 "processed_samples": count, "state_key_count": 837,
                 "checkpoint_payload_epoch": 200, "checkpoint": cp["path"], "amp": False}
        for key, value in fixed.items():
            require(type(runtime[key]) is type(value) and runtime[key] == value,
                    arm + ": rank runtime differs: " + key)
        for key, relative in expected_modules.items():
            require(runtime["loaded_modules"][key] == suite["remote_source_root"] + "/source/" + relative,
                    arm + ": module escaped frozen source: " + key)
        decoder_bn = [r for r in runtime["batch_norm"] if r["name"] == "decode_head.linear_fuse.1"]
        require(len(decoder_bn) == 1 and decoder_bn[0]["eps"] == 1e-5 and
                decoder_bn[0]["type"] == "BatchNorm2d", arm + ": evaluation decoder BN differs")
        report = read_json(out / "evaluation/rank_{:02d}_evaluation.json".format(rank))
        require(report["rank"] == report["local_rank"] == rank and report["world_size"] == WORLD and
                report["sample_count"] == count, arm + ": rank report count/identity differs")
        ids = report["owned_sample_ids"]
        require(len(ids) == count and all(isinstance(v, str) and v and not any(c in v for c in "\r\n") for v in ids),
                arm + ": invalid owned sample IDs")
        require(ids == test_ids[rank::WORLD], arm + ": rank ordered IDs differ from test_ids[rank::8]")
        shards.append(ids)
        matrices.append(matrix(report["confusion_matrix"], arm + " rank matrix"))
    ids = [shards[i % WORLD][i // WORLD] for i in range(N)]
    require(len(set(ids)) == N, arm + ": duplicated test IDs")
    test_bytes = ("\n".join(ids) + "\n").encode("utf-8")
    require(hashlib.sha256(test_bytes).hexdigest() == TEST_SHA,
            arm + ": reconstructed ordered test.txt hash differs")
    manifest = read_csv(out / "evaluation/evaluation_manifest.csv", ["sample_id", "rank"])
    expected_manifest = [{"sample_id": value, "rank": str(rank)}
                         for rank, shard in enumerate(shards) for value in shard]
    require(manifest == expected_manifest, arm + ": manifest does not match exact rank ownership")
    summed = [[sum(m[r][c] for m in matrices) for c in range(C)] for r in range(C)]
    require(summed == result["matrix"], arm + ": rank matrices do not equal final confusion")
    audit = read_json(out / "integrity_audit.json")
    require(audit["status"] == "PASS" and audit["metrics"] == result["metrics"] and
            audit["gt_histogram"] == result["gt"] and audit["test_source_sha256"] == TEST_SHA and
            audit["checkpoint_sha256"] == cp["sha256"] and
            audit["rank_counts"] == [len(s) for s in shards], arm + ": archived audit differs")
    fixed_fields(audit["checks"], {"exitcode_zero": True, "fixed_epoch_200": True,
                 "ordered_rank_shards_match": True, "unique_test_samples": N,
                 "manifest_matches_rank_shards": True, "rank_confusions_match_csv": True,
                 "nonnegative_integer_confusions": True, "metrics_independently_recomputed": True,
                 "valid_pixel_count": PIXELS, "class_order_matches": True,
                 "checkpoint_path_size_mtime_sha256_unchanged": True}, arm + " evaluation audit")
    require(read_json(out / "status.json")["metrics"] == result["metrics"], arm + ": status metrics differ")
    result["audit"] = audit
    result["ids"] = ids
    return result


def verify_comparison(new, old, preflight, completed, status):
    comparison = read_json(ROOT / 'results/dual_arm_comparison.json')
    require(comparison == read_json(completed / 'evaluation_epoch200/dual_arm_comparison.json'),
            'comparison copies differ')
    fixed_fields(comparison, {'status': 'PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON', 'seed': 12345,
                              'training_batch_size': 56, 'training_lr': 0.00012,
                              'training_focal_gamma': 1, 'baseline_focal_gamma': 2,
                              'checkpoint_epoch': 200, 'evaluation_samples_per_arm': N,
                              'class_names': CLASSES}, 'comparison metadata')
    baseline = comparison['baseline_gamma2']
    require(baseline == preflight['baseline_gamma2'] and baseline['status'] == 'PASS' and
            baseline['focal_gamma'] == 2 and baseline['test_source_sha256'] == TEST_SHA and
            baseline['gt_histogram'] == new['hha']['gt'], 'baseline reference differs')
    baseline_summary_path = ROOT / 'baseline_gamma2/comparison_verified.json'
    require(sha(baseline_summary_path) == sha(ROOT / 'code/training/baseline/comparison_verified.json') ==
            baseline['source']['sha256'], 'baseline summary fingerprint differs')
    require(sha(ROOT / 'code/training/baseline/suite.json') == baseline['suite_source']['sha256'],
            'baseline suite fingerprint differs')
    prior = read_json(baseline_summary_path)
    require(prior['status'] == 'PASS_LOCAL_INDEPENDENT_COMPARISON' and
            prior['new_evaluation_status'] == 'PASS' and prior['single_seed'] == 12345 and
            prior['test_source_sha256'] == TEST_SHA and prior['gt_histogram'] == new['hha']['gt'],
            'baseline summary scope differs')
    rows = read_csv(ROOT / 'results/gamma1_vs_gamma2_metrics.csv',
                    ['arm', 'metric', 'gamma2_percent', 'gamma1_percent', 'gamma1_minus_gamma2_pp'])
    require([(r['arm'], r['metric']) for r in rows] ==
            [(a, k + '_percent') for a in ARMS for k in METRICS], 'summary CSV metric/arm order differs')
    for arm in ARMS:
        current, original = new[arm]['metrics'], old[arm]['metrics']
        require(comparison['arms'][arm] == new[arm]['audit'], arm + ': comparison embedded audit differs')
        require(status['metrics'][arm] == current, arm + ': evaluation queue metrics differ')
        require(baseline['checkpoint_sha256'][arm] == prior['new_checkpoint_sha256'][arm],
                arm + ': baseline checkpoint identity records differ')
        row = [r for r in prior['rows'] if r['arm'] == LABELS[arm]]
        require(len(row) == 1, arm + ': baseline summary arm missing/duplicated')
        for key in METRICS:
            field = key + '_percent'
            before, after = original[field], current[field]
            close(baseline['metrics'][arm][field], before, arm + ' baseline ' + field)
            close(row[0]['new_mIoU_percent' if key == 'mIoU' else field], before, 'baseline summary ' + field)
            close(comparison['gamma1_minus_gamma2_percentage_points'][arm][field], after - before,
                  arm + ' gamma change ' + field)
            csv_row = next(r for r in rows if r['arm'] == arm and r['metric'] == field)
            for name, value in [('gamma2_percent', before), ('gamma1_percent', after),
                                ('gamma1_minus_gamma2_pp', after - before)]:
                close(csv_row[name], value, arm + ' metrics CSV ' + name)
    for key in METRICS:
        field = key + '_percent'
        current_gap = new['relplus']['metrics'][field] - new['hha']['metrics'][field]
        prior_gap = old['relplus']['metrics'][field] - old['hha']['metrics'][field]
        close(comparison['differences_percentage_points']['relplus-hha'][field], current_gap,
              'gamma1 pairwise gap ' + field)
        close(baseline['relplus_minus_hha_pp'][field], prior_gap, 'gamma2 pairwise gap ' + field)
        close(prior['new_pairwise_differences_pp']['relplus-hha'][field], prior_gap,
              'baseline summary pairwise gap ' + field)
        close(comparison['relplus_minus_hha_gap_change_percentage_points'][field], current_gap - prior_gap,
              'change in pairwise gap ' + field)
    rows = read_csv(ROOT / 'results/gamma1_per_class.csv',
                    ['class_id', 'class_name', 'hha_gamma1_iou_percent',
                     'relplus_gamma1_iou_percent', 'relplus_minus_hha_pp'])
    require(len(rows) == C, 'per-class summary has wrong number of rows')
    for i, row in enumerate(rows):
        require(integer(row['class_id'], 'class ID') == i and row['class_name'] == CLASSES[i],
                'summary class order differs')
        for arm in ARMS:
            close(row[arm + '_gamma1_iou_percent'], new[arm]['iou'][i] * 100, 'summary class IoU')
        close(row['relplus_minus_hha_pp'], (new['relplus']['iou'][i] - new['hha']['iou'][i]) * 100,
              'summary class IoU difference')
    return comparison


def verify_audits(new, old, preflight, completed, comparison):
    live = read_json(completed / 'LIVE_FINAL_RECHECK.json')
    fixed_fields(live, {'status': 'PASS_READONLY_LIVE_RECHECK'}, 'archived live audit')
    fixed_fields(live['bundle'], {'status': 'PASS', 'file_count': 200, 'manifest_sha256': BUNDLE_SHA},
                 'live bundle audit')
    offline = read_json(completed / 'INDEPENDENT_OFFLINE_AUDIT.json')
    require(offline['status'] == 'PASS' and offline['failures'] == [] and
            offline['check_count'] == len(offline['checks']) == 189 and
            all(v is True for v in offline['checks'].values()), 'archived independent audit failed')
    for arm in ARMS:
        cp = preflight['arms'][arm]['checkpoint']
        require(live['arms'][arm] == new[arm]['audit'], arm + ': live audit differs from final evaluation')
        base_cp = live['baseline_checkpoints'][arm]
        require(base_cp['sha256'] == comparison['baseline_gamma2']['checkpoint_sha256'][arm] and
                base_cp['path'] == old[arm]['metrics']['checkpoint'] and base_cp['size'] > 0,
                arm + ': live baseline checkpoint records differ')
        data = preflight['arms'][arm]['data_evidence']
        require(data['eval_source']['sha256'] == TEST_SHA and data['train_source']['sha256'] == TRAIN_SHA,
                arm + ': frozen split fingerprint differs')
        for record in data.values():
            if record.get('exists'):
                actual = live['inputs'][record['path']]
                require(actual['sha256'] == record['sha256'] and actual['size'] == record['size_bytes'],
                        arm + ': live data fingerprint differs')
        record = offline['arms'][arm]
        fixed_fields(record, {'status': 'PASS', 'test_list_sha256': TEST_SHA,
                              'actual_focal_gamma_all_eight_ranks': 1.0,
                              'training_epoch': 200, 'training_global_iteration': 189000,
                              'runtime_author_nan_replacement_count': 0,
                              'checkpoint_identity_record': cp,
                              'rank_sample_counts': [len(range(r, N, WORLD)) for r in range(WORLD)]},
                     arm + ' independent audit')
        m = record['recomputed_metrics']
        om = offline['baseline_gamma2'][arm]['metrics_from_archived_confusion_matrix']
        require(m['gt_histogram'] == new[arm]['gt'] and m['valid_pixel_count'] == PIXELS,
                arm + ': independent audit pixel counts differ')
        for key in METRICS:
            for suffix, tol in (('', 1e-12), ('_percent', 1e-10)):
                close(m[key + suffix], new[arm]['metrics'][key + suffix], 'independent new metric', tol)
                close(om[key + suffix], old[arm]['metrics'][key + suffix], 'independent old metric', tol)
            close(record['gamma1_minus_gamma2_percentage_points'][key + '_percent'],
                  comparison['gamma1_minus_gamma2_percentage_points'][arm][key + '_percent'],
                  'independent gamma change')
        for i in range(C):
            close(m['per_class_iou'][i], new[arm]['iou'][i], 'independent class IoU', 1e-12)
            close(m['per_class_iou_percent'][i], new[arm]['iou'][i] * 100, 'independent class IoU percent')
    gap_names = {'gamma1_relplus_minus_hha': comparison['differences_percentage_points']['relplus-hha'],
                 'gamma2_relplus_minus_hha': comparison['baseline_gamma2']['relplus_minus_hha_pp'],
                 'change_in_relplus_minus_hha_gap': comparison['relplus_minus_hha_gap_change_percentage_points']}
    for name, expected in gap_names.items():
        for key, value in expected.items():
            close(offline['differences_percentage_points'][name][key], value, 'independent gap ' + key)


def verify_provenance(inventory):
    copied = read_json(ROOT / 'provenance/copied_files.json')
    require(isinstance(copied, dict) and len(copied) > 0, 'copied-file provenance is empty')
    for name, record in copied.items():
        relative_path(name)
        relative_path(record['source_workspace_relative'])
        require(name in inventory and inventory[name] == record['sha256'],
                'copied-file provenance hash differs: ' + name)
    generated_reference = 'baseline_gamma2/SOURCE_REFERENCE.json'
    fixed_fields(read_json(ROOT / generated_reference),
                 {'repository': 'https://github.com/Dejavu6868/REL-Expansion',
                  'commit': '791f11db670239a19e78f3956c6f3d4466f17dc3',
                  'path': '0915调参结果', 'focal_gamma': 2,
                  'policy': 'Exact byte copies; no new baseline training/evaluation.'},
                 'baseline source reference')
    for prefix in ('code/training/', 'configs/', 'evidence/completed/', 'evidence/preparation/',
                   'baseline_gamma2/'):
        require(all(name in copied for name in inventory
                    if name.startswith(prefix) and name != generated_reference),
                'copied-file provenance incomplete: ' + prefix)
    dependencies = read_json(ROOT / 'provenance/local_audit_dependency_map.json')
    audited_inputs = read_json(ROOT / 'evidence/completed/INDEPENDENT_OFFLINE_AUDIT.json')['input_file_sha256']
    require(dependencies['status'] == 'ALL_ORIGINAL_AUDIT_INPUTS_INCLUDED' and
            len(audited_inputs) == 100 and set(dependencies['files']) == set(audited_inputs),
            'independent audit dependency set is incomplete')
    for original, record in dependencies['files'].items():
        require(record['sha256'] == audited_inputs[original], 'audit dependency hash record differs')
        paths = record['archive_paths']
        require(isinstance(paths, list) and paths and len(paths) == len(set(paths)),
                'audit dependency needs nonempty unique archive paths')
        for name in paths:
            relative_path(name)
            require(name in inventory and inventory[name] == record['sha256'],
                    'audit dependency has missing or mismatched archived content: ' + name)
    require(not any(PurePosixPath(name).suffix.lower() in ('.pth', '.pt', '.ckpt', '.safetensors')
                    for name in inventory), 'unexpected model weights in evidence-only archive')


def main():
    global ROOT
    require(len(sys.argv) <= 2, 'usage: verify_archive.py [ARCHIVE_DIRECTORY]')
    if len(sys.argv) == 2:
        ROOT = Path(sys.argv[1]).resolve()
    inventory = verify_inventory(ROOT, 'SHA256SUMS')
    completed = ROOT / 'evidence/completed'
    verify_transfer(completed)
    preflight = read_json(completed / 'evaluation_epoch200/preflight.json')
    require(preflight['status'] == 'PASS' and set(preflight['arms']) == set(ARMS),
            'evaluation preflight incomplete')
    suite = verify_code(preflight)
    verify_training(preflight, suite, completed)
    status = read_json(completed / 'evaluation_epoch200/status.json')
    fixed_fields(status, {'status': 'PASS', 'completed': list(ARMS), 'active_arm': None}, 'evaluation queue')
    new = {arm: verify_new_arm(arm, preflight, suite) for arm in ARMS}
    old = {arm: check_metrics(ROOT / 'baseline_gamma2' / arm, arm) for arm in ARMS}
    require(all(row['gt'] == new['hha']['gt'] for row in list(new.values()) + list(old.values())),
            'new/baseline ground-truth histograms differ')
    require(new['hha']['ids'] == new['relplus']['ids'], 'test ID order differs across arms')
    comparison = verify_comparison(new, old, preflight, completed, status)
    verify_audits(new, old, preflight, completed, comparison)
    verify_provenance(inventory)
    print(json.dumps({'status': 'PASS_OFFLINE_ARCHIVE_VERIFICATION', 'archive_files_hashed': len(inventory),
                      'transferred_completed_files_hashed': 114, 'transferred_preparation_files_hashed': 80,
                      'training_bundle_files': 200, 'source_snapshot_files': 174,
                      'training_completed_ranks': 16, 'training_actual_focal_gamma_all_ranks': 1.0,
                      'evaluation_completed_ranks': 16, 'unique_test_samples_per_arm': N,
                      'ordered_test_sha256': TEST_SHA, 'valid_pixels_per_arm': PIXELS,
                      'sole_scientific_changed_factor': 'focal_gamma: 2 -> 1',
                      'metrics_independently_recomputed': True,
                      'mIoU_percent': {arm: new[arm]['metrics']['mIoU_percent'] for arm in ARMS},
                      'gamma1_minus_gamma2_percentage_points': comparison['gamma1_minus_gamma2_percentage_points'],
                      'unarchived_weights_or_datasets_rehashed': False, 'inference_rerun': False},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'status': 'FAIL_OFFLINE_ARCHIVE_VERIFICATION', 'error': str(error)},
                         ensure_ascii=False))
        sys.exit(1)
