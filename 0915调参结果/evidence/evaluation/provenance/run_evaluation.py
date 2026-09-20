"""Three fixed-endpoint evaluations; new outputs and existing resource guards."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

TRAIN_BUNDLE = Path('/home/zhuzhaoziao/RELPlus/CMX-S2D-B56-LR12-20260915')
TRAIN_ROOT = Path('/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_B56_LR12_20260915/attempt1')
OUTPUT = TRAIN_ROOT / 'evaluation_epoch200_20260920_attempt1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(TRAIN_BUNDLE))
from suite_common import build_config, configure_imports, digest, dump_json, load_suite
import resource_guard as guard
import preflight as training_preflight


def identity(path):
    p = Path(path)
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=str(s.st_mtime_ns), sha256=digest(p))


def precheck():
    suite = load_suite(TRAIN_BUNDLE / 'suite.json')
    configure_imports(suite)
    old = json.loads((TRAIN_ROOT / 'preflight.json').read_text())
    manifest = json.loads((TRAIN_BUNDLE / 'bundle_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest(TRAIN_BUNDLE / relative) != expected:
            raise RuntimeError('training bundle changed: ' + relative)
    queue = json.loads((TRAIN_ROOT / 'train_status.json').read_text())
    if queue['status'] != 'PASS' or [x['arm'] for x in queue['completed']] != suite['arms']:
        raise RuntimeError('three-arm training did not complete')
    from utils.training_protocol import assert_runtime_dataset_contract
    before = {'status': 'PASS', 'checked_at': time.time(), 'training': queue,
              'suite': suite, 'arms': {}, 'source_hashes': manifest['files'],
              'implementation_hashes': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
              'evaluation_authorization': 'User: 进行评估, 2026-09-20',
              'bn_policy': 'Frozen criterion=None evaluation, decoder eps1e-5; training eps1e-3; unchanged from 0915.'}
    for arm, completed in zip(suite['arms'], queue['completed']):
        run = TRAIN_ROOT / 'runs' / arm
        runtime = json.loads((run / 'runtime_status.json').read_text())
        if (run / 'exitcode').read_text().strip() != '0' or runtime['epoch'] != 200 or runtime['global_iteration'] != 189000 or runtime['author_nan_replacement_count'] != 0:
            raise RuntimeError('invalid training completion: ' + arm)
        for rank in range(8):
            done = json.loads((run / ('rank_done_%02d.json' % rank)).read_text())
            if (done['status'], done['epoch'], done['global_iteration']) != ('COMPLETED', 200, 189000):
                raise RuntimeError('incomplete training rank: ' + arm)
        checkpoint = identity(completed['checkpoint'])
        if checkpoint['sha256'] != completed['checkpoint_sha256']:
            raise RuntimeError('training checkpoint changed: ' + arm)
        cfg = build_config(suite, arm)
        assert_runtime_dataset_contract(cfg, require_cache_audit=True)
        config = training_preflight.json_value(cfg)
        resolved_path = Path(old['arms'][arm]['resolved_config'])
        if digest(resolved_path) != old['arms'][arm]['resolved_config_sha256']:
            raise RuntimeError('saved training configuration changed: ' + arm)
        if json.loads(resolved_path.read_text()) != config:
            raise RuntimeError('evaluation did not reconstruct the actual training config: ' + arm)
        evidence = training_preflight.file_evidence(cfg, arm)
        # Match each data-evidence file against the actual pre-training preflight.
        prior = old['arms'][arm]['data_evidence_files']
        for field, item in evidence.items():
            if item.get('exists') and prior.get(field, {}).get('exists') and item['sha256'] != prior[field]['sha256']:
                raise RuntimeError('data evidence changed: ' + field)
        ids = training_preflight.ordered_ids(cfg.eval_source, 17593)
        if digest(cfg.eval_source) != 'b9de196c6c1aa8f9ac37926910af0806ce59b91eb068998711ddbb78eb24423a':
            raise RuntimeError('frozen test list changed')
        before['arms'][arm] = {'checkpoint': checkpoint, 'config': config, 'data_evidence': evidence,
                               'ordered_test_count': len(ids), 'test_list_sha256': digest(cfg.eval_source)}
    dump_json(OUTPUT / 'preflight.json', before)
    return suite, before


def evaluate_arm(suite, before, arm):
    from audit_eval import audit_arm
    out = OUTPUT / arm
    out.mkdir()
    proc = None
    identities = {}
    state = {'status': 'RUNNING', 'arm': arm, 'started_at': time.time()}
    dump_json(out / 'status.json', state)
    try:
        guard.preflight(out)
        cmd = [suite['python'], '-B', '-m', 'torch.distributed.launch', '--nproc_per_node=8',
               '--master_addr=127.0.0.1', '--master_port=%d' % guard.free_port(), '--use_env',
               str(Path(__file__).with_name('eval_rank.py')), '--arm', arm, '--output', str(out)]
        overrides = dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTHONDONTWRITEBYTECODE='1',
                         PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        dump_json(out / 'command.json', {'argv': cmd, 'cwd': str(TRAIN_BUNDLE / 'source'),
                                        'environment_overrides': overrides})
        with (out / 'launcher.log').open('w') as log, (out / 'resources.jsonl').open('w') as samples:
            proc = subprocess.Popen(cmd, cwd=str(TRAIN_BUNDLE / 'source'), env=dict(os.environ, **overrides),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            identities[proc.pid] = guard.process_start(proc.pid)
            if identities[proc.pid] is None or os.getpgid(proc.pid) != proc.pid:
                raise RuntimeError('launcher identity unavailable')
            dump_json(out / 'process.json', {'launcher_pid': proc.pid, 'pgid': proc.pid,
                       'starttime': identities[proc.pid], 'supervisor_pid': os.getpid()})
            deadline = time.monotonic() + 1800
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('evaluation exceeded 30-minute operational limit; incomplete')
                for p in out.glob('rank_*_runtime.json'):
                    if json.loads(p.read_text()).get('status') == 'FAIL':
                        raise RuntimeError('rank failure: ' + p.name)
                snapshot = guard.resources()
                samples.write(json.dumps(snapshot) + '\n'); samples.flush()
                guard.remember_group_members(proc.pid, identities)
                guard.check_running(snapshot, proc.pid, identities)
                time.sleep(5)
            code = proc.wait()
            (out / 'exitcode').write_text(str(code) + '\n')
            if code != 0:
                raise RuntimeError('nonzero evaluator exit: %s' % code)
        runtimes = [json.loads((out / ('rank_%02d_runtime.json' % r)).read_text()) for r in range(8)]
        for rank, row in enumerate(runtimes):
            if row['status'] != 'COMPLETED' or row['processed_samples'] != len(range(rank, 17593, 8)):
                raise RuntimeError('missing runtime completion')
        audit = audit_arm(out, before['arms'][arm]['config'], before['arms'][arm]['checkpoint'])
        dump_json(out / 'integrity_audit.json', audit)
        state.update(status='PASS', metrics=audit['metrics'])
        return audit
    except BaseException as error:
        state.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if proc is not None:
            receipt = guard.stop_own_group(proc, identities)
            dump_json(out / 'cleanup.json', receipt)
            (out / 'exitcode').write_text(str(proc.poll()) + '\n')
            if receipt.get('remaining_members'):
                state.update(status='FAIL', error='own process cleanup incomplete')
                dump_json(out / 'status.json', state)
                raise RuntimeError('own process cleanup incomplete')
        state['finished_at'] = time.time()
        dump_json(out / 'status.json', state)


def main():
    OUTPUT.mkdir()
    status = {'status': 'RUNNING', 'started_at': time.time(), 'completed': [],
              'scope': 'Fixed epoch200; full17593; 3 arms; no training or checkpoint sweep.'}
    dump_json(OUTPUT / 'status.json', status)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, guard.interrupted)
    try:
        suite, before = precheck()
        audits = {}
        for arm in suite['arms']:
            status['active_arm'] = arm; dump_json(OUTPUT / 'status.json', status)
            audits[arm] = evaluate_arm(suite, before, arm)
            status['completed'].append(arm)
            dump_json(OUTPUT / 'status.json', status)
        gt = [audits[a]['gt_histogram'] for a in suite['arms']]
        if any(row != gt[0] for row in gt):
            raise RuntimeError('ground-truth histogram differs across arms')
        for rel, expected in before['source_hashes'].items():
            if digest(TRAIN_BUNDLE / rel) != expected:
                raise RuntimeError('training source changed during evaluation')
        for name, expected in before['implementation_hashes'].items():
            if digest(Path(__file__).parent / name) != expected:
                raise RuntimeError('evaluation implementation changed')
        for arm in suite['arms']:
            for item in before['arms'][arm]['data_evidence'].values():
                if item.get('exists') and digest(item['path']) != item['sha256']:
                    raise RuntimeError('data evidence changed during evaluation')
        comparison = {'status': 'PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON',
                      'seed': 12345, 'training_batch_size': 56, 'training_lr': 0.00012,
                      'evaluation_samples_per_arm': 17593, 'class_names': before['arms']['rgbd']['config']['class_names'],
                      'arms': audits, 'differences_percentage_points': {}}
        for a, b in [('relplus', 'hha'), ('relplus', 'rgbd'), ('hha', 'rgbd')]:
            comparison['differences_percentage_points'][a+'-'+b] = {
                key: audits[a]['metrics'][key] - audits[b]['metrics'][key]
                for key in ('mIoU_percent', 'pixel_accuracy_percent', 'mean_accuracy_percent')}
        dump_json(OUTPUT / 'three_arm_comparison.json', comparison)
        status.update(status='PASS', active_arm=None, metrics={a:audits[a]['metrics'] for a in suite['arms']})
    except BaseException as error:
        status.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        status['finished_at'] = time.time()
        dump_json(OUTPUT / 'status.json', status)
        print(json.dumps(status), flush=True)
    return 0 if status['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
