#!/usr/bin/env python3
"""Read-only, offline verification of the frozen 0924 gamma=1.5 archive.

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
BUNDLE_SHA = "e488b4974ccf70095a3be6010a3c6ecc4f488e38b6d136475e690850c1a5605f"
SUITE_SHA = "ccd10368db7f700eb06c846e0d65a12367f5b0b5b6c71dbb54ae189639dd7e64"
INITIAL_SHA = "87254603a9898e551964e4a92c561a6d74273d678ee4ed631c3fc5fc223c25fa"
CLASSES = ["beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
           "door", "floor", "sofa", "table", "wall", "window"]
METRICS = ("mIoU", "pixel_accuracy", "mean_accuracy")


CHECK_COUNT = 0


def require(condition, message):
    global CHECK_COUNT
    CHECK_COUNT += 1
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


def verify_code(preflight):
    training = ROOT / 'code/training'
    bundle = read_json(training / 'bundle_manifest.json')
    expected = bundle['files']
    require(len(expected) == 206 and len([p for p in expected if p.startswith('source/')]) == 174,
            'frozen bundle must contain 206 files and 174 source files')
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
    prior_suite = read_json(training / 'baseline_gamma1/suite.json')
    require(set(suite['shared']) == set(prior_suite['shared']), 'shared control field set differs')
    delta = {k for k in suite['shared'] if suite['shared'][k] != prior_suite['shared'][k]}
    require(delta == {'focal_gamma'} and suite['shared']['focal_gamma'] == 1.5 and
            prior_suite['shared']['focal_gamma'] == 1, 'sole shared changed factor is not gamma1 to gamma1.5')
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
    expected = dict(old, focal_gamma=1.5, **identity)
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
        fixed_fields(pipeline[name], {'status': 'PASS', 'file_count': 206, 'manifest_sha256': BUNDLE_SHA}, name)
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
                                'actual_criterion': {'type': 'FocalLoss2d', 'focal_gamma': 1.5,
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
    test_path = ROOT / "evidence/completed/inputs/eval_source.txt"
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


def verify_transfer(completed):
    transfer = read_json(completed / 'TRANSFER_MANIFEST.json')
    require(len(transfer['files']) == 119 and set(transfer['inputs']) == {'train_source', 'eval_source'},
            'completed transfer must contain 119 outputs and 2 inputs')
    total_bytes = 0
    for name, record in transfer['files'].items():
        relative_path(name)
        path = completed / name
        require(path.stat().st_size == record['size'] and sha(path) == record['sha256'],
                'completed transfer mismatch: ' + name)
        total_bytes += record['size']
    for name, expected_sha, count in [('train_source', TRAIN_SHA, 52903), ('eval_source', TEST_SHA, N)]:
        path = completed / 'inputs' / (name + '.txt')
        record = transfer['inputs'][name]
        require(path.stat().st_size == record['size'] and sha(path) == record['sha256'] == expected_sha,
                'input transfer mismatch: ' + name)
        ids = path.read_text().splitlines()
        require(len(ids) == len(set(ids)) == count and all(ids), 'invalid ordered input list: ' + name)
        total_bytes += record['size']
    train_ids = set((completed / 'inputs/train_source.txt').read_text().splitlines())
    test_ids = set((completed / 'inputs/eval_source.txt').read_text().splitlines())
    require(not train_ids.intersection(test_ids), 'training and test sample IDs overlap')
    fixed_fields(read_json(completed / 'TRANSFER_VERIFICATION.json'),
                 {'status': 'PASS', 'output_files': 119, 'input_files': 2,
                  'all_sha256_verified': True, 'total_bytes': total_bytes}, 'completed transfer receipt')
    preparation = ROOT / 'evidence/preparation'
    prep_receipt = read_json(ROOT / 'evidence/startup/PREPARATION_TRANSFER_VERIFICATION.json')
    require(prep_receipt['status'] == 'PASS' and prep_receipt['file_count'] == len(prep_receipt['files']) == 79,
            'preparation transfer receipt must contain 79 files')
    require(file_set(preparation) == set(prep_receipt['files']), 'preparation file set differs from original transfer')
    for name, record in prep_receipt['files'].items():
        relative_path(name)
        path = preparation / name
        require(path.stat().st_size == record['size'] and sha(path) == record['sha256'],
                'preparation original transfer mismatch: ' + name)
    require(sha(preparation / 'preflight.json') == sha(completed / 'preflight.json'),
            'preparation and completed preflight copies differ')
    smoke = read_json(preparation / 'smoke_status.json')
    require(smoke['status'] == 'PASS', 'preparation smoke phase failed')
    finite_fields = ('loss_finite', 'logits_finite', 'gradients_finite', 'optimizer_step_executed',
                     'checkpoint_saved', 'checkpoint_resumed', 'parameters_match_after_restore',
                     'lr_continuous_across_restore', 'lr_updated_after_resume', 'pretrained_model_loaded')
    for arm in ARMS:
        directory = preparation / 'smoke' / arm
        summary = read_json(directory / 'ddp_optimizer_smoke_summary.json')
        require(summary['status'] == 'PASS' and summary['rank_count'] == summary['gpu_count'] == WORLD,
                arm + ': smoke summary rank count differs')
        for key in finite_fields:
            require(summary[key] is True, arm + ': smoke summary failed: ' + key)
        for rank in range(WORLD):
            row = read_json(directory / ('rank_%02d.json' % rank))
            require(row['status'] == 'PASS' and row['rank'] == rank and row['world_size'] == WORLD and
                    row['author_nan_replacement_count'] == 0, arm + ': smoke rank failed')
            for key in finite_fields:
                require(row[key] is True, arm + ': smoke rank failed: ' + key)
    return transfer


def verify_baseline_configuration_chain(suite):
    training = ROOT / 'code/training'
    prior_suites = {1: read_json(training / 'baseline_gamma1/suite.json'),
                   2: read_json(training / 'baseline/suite.json')}
    require(prior_suites[1]['shared'] == dict(prior_suites[2]['shared'], focal_gamma=1),
            'gamma1 and gamma2 baseline shared parameters differ beyond gamma')
    identities = ('root_dir', 'abs_dir', 'experiment_name', 'experiment_protocol_id',
                  'comparison_protocol_id', 'output_dir', 'log_dir', 'tb_dir', 'checkpoint_dir',
                  'log_dir_link', 'log_file', 'link_log_file', 'val_log_file', 'link_val_log_file',
                  'ddp_smoke_report')
    for arm in ARMS:
        gamma1_path = training / 'baseline_gamma1' / (arm + '.json')
        require(sha(gamma1_path) == sha(training / 'baseline' / (arm + '.json')),
                arm + ': duplicate gamma1 baseline configs differ')
        gamma1 = read_json(gamma1_path)
        gamma2 = read_json(training / 'baseline' / ('gamma2_' + arm + '.json'))
        expected = dict(gamma2, focal_gamma=1.0)
        for field in identities:
            value = gamma2[field]
            for key in ('remote_source_root', 'output_root', 'suite_id'):
                value = value.replace(prior_suites[2][key], prior_suites[1][key])
            expected[field] = value
        require(gamma1 == expected, arm + ': original gamma2->gamma1 full config changed beyond gamma/identity')
    require(suite['shared']['lr'] == 0.00012 and suite['shared']['focal_gamma'] == 1.5,
            'archive must contain completed gamma1.5/lr0.00012 experiment')


def verify_comparison(new, baselines, preflight, completed, status):
    path = ROOT / 'results/dual_arm_comparison.json'
    comparison = read_json(path)
    require(sha(path) == sha(completed / 'evaluation_epoch200/dual_arm_comparison.json'),
            'comparison copies are not exact bytes')
    fixed_fields(comparison, {'status': 'PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON', 'seed': 12345,
                             'training_batch_size': 56, 'training_lr': 0.00012,
                             'training_focal_gamma': 1.5, 'primary_baseline_focal_gamma': 1,
                             'historical_baseline_focal_gamma': 2, 'checkpoint_epoch': 200,
                             'evaluation_samples_per_arm': N, 'class_names': CLASSES}, 'comparison')
    for gamma in (1, 2):
        label = 'gamma' + str(gamma)
        baseline = comparison['baseline_' + label]
        require(baseline == preflight['baseline_' + label] and baseline['status'] == 'PASS' and
                baseline['focal_gamma'] == gamma and baseline['test_source_sha256'] == TEST_SHA and
                baseline['gt_histogram'] == new['hha']['gt'], label + ': embedded reference differs')
        summary_path = ROOT / ('baseline_' + label) / 'comparison_verified.json'
        original_path = ROOT / ('code/training/baseline_gamma1/comparison.json' if gamma == 1 else
                                'code/training/baseline/comparison_verified.json')
        suite_path = ROOT / ('code/training/baseline_gamma1/suite.json' if gamma == 1 else
                            'code/training/baseline/suite.json')
        require(sha(summary_path) == sha(original_path) == baseline['source']['sha256'],
                label + ': original baseline comparison hash differs')
        require(sha(suite_path) == baseline['suite_source']['sha256'], label + ': original baseline suite hash differs')
        summary = read_json(summary_path)
        if gamma == 1:
            fixed_fields(summary, {'status': 'PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON',
                                   'training_focal_gamma': 1, 'checkpoint_epoch': 200,
                                   'evaluation_samples_per_arm': N, 'seed': 12345}, 'gamma1 summary')
        else:
            fixed_fields(summary, {'status': 'PASS_LOCAL_INDEPENDENT_COMPARISON', 'new_evaluation_status': 'PASS',
                                   'single_seed': 12345, 'test_source_sha256': TEST_SHA}, 'gamma2 summary')
        for arm in ARMS:
            old = baselines[label][arm]['metrics']
            current = new[arm]['metrics']
            if gamma == 1:
                reference = summary['arms'][arm]['metrics']
                checkpoint_sha = summary['arms'][arm]['checkpoint_sha256']
            else:
                matches = [r for r in summary['rows'] if r['arm'] == LABELS[arm]]
                require(len(matches) == 1, arm + ': duplicate/missing gamma2 summary arm')
                reference = dict(matches[0], mIoU_percent=matches[0]['new_mIoU_percent'])
                checkpoint_sha = summary['new_checkpoint_sha256'][arm]
            require(baseline['checkpoint_sha256'][arm] == checkpoint_sha, label + '/' + arm + ': checkpoint identity differs')
            for key in METRICS:
                field = key + '_percent'
                close(baseline['metrics'][arm][field], old[field], label + '/' + arm + ': baseline metric')
                close(reference[field], old[field], label + '/' + arm + ': original summary metric')
                close(comparison['gamma1_5_minus_' + label + '_percentage_points'][arm][field],
                      current[field] - old[field], label + '/' + arm + ': same-arm delta')
        for key in METRICS:
            field = key + '_percent'
            gap = baselines[label]['relplus']['metrics'][field] - baselines[label]['hha']['metrics'][field]
            new_gap = new['relplus']['metrics'][field] - new['hha']['metrics'][field]
            close(baseline['relplus_minus_hha_pp'][field], gap, label + ': prior gap')
            close(comparison['relplus_minus_hha_percentage_points'][label][field], gap, label + ': gap alias')
            close(comparison['relplus_minus_hha_gap_change_percentage_points']['gamma1_5_minus_' + label][field],
                  new_gap - gap, label + ': gap change')
    for arm in ARMS:
        require(comparison['arms'][arm] == new[arm]['audit'] and status['metrics'][arm] == new[arm]['metrics'],
                arm + ': evaluation/comparison audit metrics differ')
    for key in METRICS:
        field = key + '_percent'
        gap = new['relplus']['metrics'][field] - new['hha']['metrics'][field]
        close(comparison['differences_percentage_points']['relplus-hha'][field], gap, 'new gap')
        close(comparison['relplus_minus_hha_percentage_points']['gamma1_5'][field], gap, 'new gap alias')
    candidate = {'relplus_mIoU_strictly_above_gamma1': new['relplus']['metrics']['mIoU_percent'] > baselines['gamma1']['relplus']['metrics']['mIoU_percent'],
                 'hha_mIoU_at_least_gamma1': new['hha']['metrics']['mIoU_percent'] >= baselines['gamma1']['hha']['metrics']['mIoU_percent']}
    require(candidate == {'relplus_mIoU_strictly_above_gamma1': False, 'hha_mIoU_at_least_gamma1': True} and
            comparison['candidate_screening']['checks'] == candidate and
            comparison['candidate_screening']['status'] == 'FAIL',
            'candidate screening FAIL must be distinct from successful execution')
    rows = read_csv(ROOT / 'results/gamma15_vs_baselines_metrics.csv',
                    ['arm', 'gamma1_mIoU_percent', 'gamma1_5_mIoU_percent', 'gamma2_mIoU_percent',
                     'gamma1_5_minus_gamma1_pp', 'gamma1_5_minus_gamma2_pp',
                     'gamma1_5_pixel_accuracy_percent', 'gamma1_5_mean_accuracy_percent'])
    require([r['arm'] for r in rows] == list(ARMS), 'metrics CSV arm ordering differs')
    for arm, row in zip(ARMS, rows):
        expected = {'gamma1_mIoU_percent': baselines['gamma1'][arm]['metrics']['mIoU_percent'],
                    'gamma1_5_mIoU_percent': new[arm]['metrics']['mIoU_percent'],
                    'gamma2_mIoU_percent': baselines['gamma2'][arm]['metrics']['mIoU_percent'],
                    'gamma1_5_minus_gamma1_pp': comparison['gamma1_5_minus_gamma1_percentage_points'][arm]['mIoU_percent'],
                    'gamma1_5_minus_gamma2_pp': comparison['gamma1_5_minus_gamma2_percentage_points'][arm]['mIoU_percent'],
                    'gamma1_5_pixel_accuracy_percent': new[arm]['metrics']['pixel_accuracy_percent'],
                    'gamma1_5_mean_accuracy_percent': new[arm]['metrics']['mean_accuracy_percent']}
        for key, value in expected.items():
            close(row[key], value, arm + ': metric summary CSV ' + key)
    rows = read_csv(ROOT / 'results/gamma15_per_class.csv',
                    ['arm', 'class', 'gamma1_IoU_percent', 'gamma1_5_IoU_percent', 'difference_pp'])
    require([(r['arm'], r['class']) for r in rows] == [(a, c) for a in ARMS for c in CLASSES],
            'per-class summary ordering differs')
    for row in rows:
        arm, index = row['arm'], CLASSES.index(row['class'])
        before, after = baselines['gamma1'][arm]['iou'][index] * 100, new[arm]['iou'][index] * 100
        for key, value in [('gamma1_IoU_percent', before), ('gamma1_5_IoU_percent', after), ('difference_pp', after - before)]:
            close(row[key], value, arm + ': class summary ' + key)
    return comparison


def verify_audits(new, baselines, preflight, completed, comparison):
    live = read_json(completed / 'LIVE_FINAL_RECHECK.json')
    require(live['status'] == 'PASS' and live['checks_count'] == len(live['checks']) == 309 and
            len(set(live['checks'])) == 309 and live['bundle_file_count'] == 206 and
            live['input_file_count'] == len(live['input_sha256']) == 18, 'archived live audit is incomplete')
    require(live['candidate_screening'] == comparison['candidate_screening'], 'live candidate screening differs')
    weights = {(r['gamma'], r['arm']): r for r in live['weights']}
    require(len(weights) == len(live['weights']) == 6, 'live audit must identify all 6 checkpoint records')
    offline = read_json(completed / 'INDEPENDENT_OFFLINE_AUDIT.json')
    require(offline['status'] == 'PASS_INDEPENDENT_OFFLINE_AUDIT' and not offline['failures'] and
            offline['check_count'] == sum(offline['check_counts_by_group'].values()) == 1073,
            'archived independent offline audit did not pass')
    require(offline['actual_training_focal_gamma_all_16_ranks'] == 1.5 and
            offline['shared_initialization_sha256'] == INITIAL_SHA and
            offline['candidate_screening']['status'] == 'FAIL', 'independent actual gamma/initialization/screening differs')
    require(sha(completed / 'independent_offline_audit.py') == offline['audit_script_sha256'],
            'original independent audit script hash differs')
    for arm in ARMS:
        cp = preflight['arms'][arm]['checkpoint']
        require(all(weights[(1.5, arm)][k] == v for k, v in cp.items()), arm + ': live new checkpoint identity differs')
        require(live['metrics'][arm] == new[arm]['metrics'], arm + ': live metrics differ')
        require(offline['checkpoint_records'][arm] == cp and offline['current'][arm]['gt_histogram'] == new[arm]['gt'],
                arm + ': independent checkpoint/GT evidence differs')
        for key in METRICS:
            field = key + '_percent'
            close(offline['current'][arm]['metrics'][field], new[arm]['metrics'][field], 'independent new metric')
        for i in range(C):
            close(offline['current'][arm]['per_class_iou'][i], new[arm]['iou'][i], 'independent class IoU', 1e-12)
        for gamma in (1, 2):
            label = 'gamma' + str(gamma)
            record = weights[(gamma, arm)]
            require(record['sha256'] == comparison['baseline_' + label]['checkpoint_sha256'][arm] and
                    record['path'] == baselines[label][arm]['metrics']['checkpoint'] and record['size'] > 0,
                    label + '/' + arm + ': live baseline weight identity differs')
            for key in METRICS:
                field = key + '_percent'
                close(offline['baseline_metrics'][label][arm][field], baselines[label][arm]['metrics'][field],
                      'independent baseline metric')
                close(offline['changes_percentage_points'][label][arm][field],
                      comparison['gamma1_5_minus_' + label + '_percentage_points'][arm][field], 'independent delta')
        for record in preflight['arms'][arm]['data_evidence'].values():
            if record.get('exists'):
                require(live['input_sha256'][record['path']] == record['sha256'], arm + ': live input SHA differs')


def verify_provenance(inventory, preflight, suite, comparison):
    copied = read_json(ROOT / 'provenance/copied_files.json')
    require(isinstance(copied, dict) and copied, 'copied-file provenance missing')
    for name, record in copied.items():
        relative_path(name)
        relative_path(record['source_workspace_relative'])
        require(name in inventory and inventory[name] == record['sha256'], 'copied-file SHA differs: ' + name)
    for prefix in ('code/training/', 'configs/', 'evidence/completed/', 'evidence/preparation/'):
        require(all(name in copied for name in inventory if name.startswith(prefix)), 'copy provenance incomplete: ' + prefix)
    dependencies = read_json(ROOT / 'provenance/local_audit_dependency_map.json')
    original = read_json(ROOT / 'evidence/completed/INDEPENDENT_OFFLINE_AUDIT.json')['files_read_sha256']
    require(dependencies['status'] == 'ALL_ORIGINAL_AUDIT_INPUTS_INCLUDED' and
            set(dependencies['files']) == set(original) and len(original) == 103,
            'original independent audit dependency set incomplete')
    for name, record in dependencies['files'].items():
        require(record['sha256'] == original[name], 'dependency fingerprint differs: ' + name)
        paths = record['archive_paths']
        require(isinstance(paths, list) and paths and len(paths) == len(set(paths)), 'dependency archive paths missing')
        for path in paths:
            relative_path(path)
            require(inventory.get(path) == record['sha256'], 'dependency not available inside archive: ' + path)
    source = read_json(ROOT / 'provenance/source_provenance.json')
    require(source['bundle_manifest_sha256'] == BUNDLE_SHA and source['suite_sha256'] == SUITE_SHA and
            source['bundle_files'] == 206 and source['source_python_files'] == 174 and
            source['remote_bundle'] == suite['remote_source_root'] and source['remote_output'] == suite['output_root'],
            'source provenance differs from executed bundle')
    weights = read_json(ROOT / 'provenance/checkpoints.json')
    require(weights['weights_included'] is False and weights['seed'] == 12345 and weights['endpoint'] == 'fixed_epoch_200',
            'weight provenance scope differs')
    for arm in ARMS:
        require(weights['arms'][arm] == preflight['arms'][arm]['checkpoint'], arm + ': weight provenance differs')
        for label in ('gamma1', 'gamma2'):
            require(weights['baseline_' + label][arm] == comparison['baseline_' + label]['checkpoint_sha256'][arm],
                    arm + ': baseline weight provenance differs')
    environment = read_json(ROOT / 'provenance/observed_environment.json')
    require(environment['preparation_environment'] == read_json(ROOT / 'evidence/preparation/preflight.json')['environment'],
            'environment provenance differs')
    frozen = read_json(ROOT / 'provenance/frozen_protocol.json')
    require(frozen['training'] == suite['shared'] and frozen['arms'] == list(ARMS), 'frozen protocol training controls differ')
    require(not any(PurePosixPath(p).suffix.lower() in ('.pth', '.pt', '.ckpt', '.safetensors') for p in inventory),
            'unexpected model weights in evidence archive')


def main():
    global ROOT
    require(len(sys.argv) <= 2, 'usage: verify_archive.py [ARCHIVE_DIRECTORY]')
    if len(sys.argv) == 2:
        ROOT = Path(sys.argv[1]).resolve()
    inventory = verify_inventory(ROOT, 'SHA256SUMS')
    completed = ROOT / 'evidence/completed'
    verify_transfer(completed)
    preflight = read_json(completed / 'evaluation_epoch200/preflight.json')
    require(preflight['status'] == 'PASS' and set(preflight['arms']) == set(ARMS), 'evaluation preflight incomplete')
    suite = verify_code(preflight)
    verify_baseline_configuration_chain(suite)
    verify_training(preflight, suite, completed)
    status = read_json(completed / 'evaluation_epoch200/status.json')
    fixed_fields(status, {'status': 'PASS', 'completed': list(ARMS), 'active_arm': None}, 'evaluation queue')
    new = {arm: verify_new_arm(arm, preflight, suite) for arm in ARMS}
    baselines = {label: {arm: check_metrics(ROOT / ('baseline_' + label) / arm, arm) for arm in ARMS}
                 for label in ('gamma1', 'gamma2')}
    all_results = list(new.values()) + [r for arms in baselines.values() for r in arms.values()]
    require(all(r['gt'] == new['hha']['gt'] for r in all_results), 'new/old ground-truth histograms differ')
    require(new['hha']['ids'] == new['relplus']['ids'], 'new arm test IDs differ')
    comparison = verify_comparison(new, baselines, preflight, completed, status)
    verify_audits(new, baselines, preflight, completed, comparison)
    verify_provenance(inventory, preflight, suite, comparison)
    print(json.dumps({'status': 'PASS_OFFLINE_ARCHIVE_VERIFICATION', 'checks_count': CHECK_COUNT,
                      'failures': [], 'archive_files_hashed': len(inventory),
                      'transferred_completed_files_hashed': 121, 'transferred_preparation_files_hashed': 79,
                      'training_bundle_files': 206, 'source_snapshot_files': 174,
                      'training_completed_ranks': 16, 'training_actual_focal_gamma_all_ranks': 1.5,
                      'evaluation_completed_ranks': 16, 'unique_test_samples_per_arm': N,
                      'ordered_test_sha256': TEST_SHA, 'valid_pixels_per_arm': PIXELS,
                      'sole_scientific_changed_factor': 'focal_gamma: 1 -> 1.5',
                      'metrics_independently_recomputed': True,
                      'mIoU_percent': {a: new[a]['metrics']['mIoU_percent'] for a in ARMS},
                      'gamma1_5_minus_gamma1_percentage_points': comparison['gamma1_5_minus_gamma1_percentage_points'],
                      'gamma1_5_minus_gamma2_percentage_points': comparison['gamma1_5_minus_gamma2_percentage_points'],
                      'execution_status': 'PASS', 'candidate_screening_status': 'FAIL',
                      'unarchived_weights_or_datasets_rehashed': False, 'inference_rerun': False},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'status': 'FAIL_OFFLINE_ARCHIVE_VERIFICATION', 'checks_count': CHECK_COUNT,
                          'failures': [{'error': str(error), 'type': type(error).__name__}]}, ensure_ascii=False))
        sys.exit(1)
