#!/usr/bin/env python3
"""Durable two-arm training followed by the authorized fixed-endpoint evaluation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
from suite_common import digest, dump_json, load_suite
import resource_guard as guard
import run_suite

BUNDLE = Path(__file__).resolve().parent


def verify_bundle():
    manifest = json.loads((BUNDLE / 'bundle_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = BUNDLE / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError('bundle integrity mismatch: ' + relative)
    return {'status': 'PASS', 'file_count': len(manifest['files']),
            'manifest_sha256': digest(BUNDLE / 'bundle_manifest.json')}


def require_pass(result, stage):
    if result.get('status') != 'PASS':
        raise RuntimeError('{} failed: {}'.format(stage, result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('prepare', 'execute'), required=True)
    args = parser.parse_args()
    suite_path = BUNDLE / 'suite.json'
    suite = load_suite(suite_path)
    if Path(suite['remote_source_root']).resolve() != BUNDLE:
        raise RuntimeError('pipeline must run inside its own deployed bundle')
    if suite['scope'].get('training_then_evaluation') is not True:
        raise RuntimeError('train-and-evaluate scope required')
    output = Path(suite['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    run_suite._claim_phase(output, 'pipeline_' + args.phase, suite)
    status_path = output / ('preparation_status.json' if args.phase == 'prepare' else 'pipeline_status.json')
    state = {'status': 'RUNNING', 'phase': args.phase, 'suite_id': suite['suite_id'],
             'started_at': time.time(), 'supervisor_pid': os.getpid(),
             'supervisor_starttime': guard.process_start(os.getpid()),
             'completed_stages': [], 'active_stage': 'bundle_integrity'}
    dump_json(status_path, state)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, run_suite.interrupted)
    try:
        state['bundle_integrity'] = verify_bundle()
        if args.phase == 'prepare':
            import preflight
            state['active_stage'] = 'preflight'
            dump_json(status_path, state)
            if preflight.main(['--suite', str(suite_path)]) != 0:
                raise RuntimeError('preflight nonzero exit')
            state['completed_stages'].append('preflight')
            state['active_stage'] = 'smoke'
            dump_json(status_path, state)
            require_pass(run_suite.run_phase(suite, suite_path, 'smoke'), 'smoke')
            run_suite.validate_train_prerequisites(suite, suite_path)
            state['completed_stages'].append('smoke')
        else:
            prepared = json.loads((output / 'preparation_status.json').read_text())
            require_pass(prepared, 'preparation')
            if (prepared['suite_id'] != suite['suite_id'] or
                    prepared['final_bundle_integrity'] != state['bundle_integrity']):
                raise RuntimeError('bundle changed after successful preparation')
            run_suite.validate_train_prerequisites(suite, suite_path)
            state['active_stage'] = 'training'
            dump_json(status_path, state)
            require_pass(run_suite.run_phase(suite, suite_path, 'train'), 'training')
            state['completed_stages'].append('training')
            verify_bundle()
            state['active_stage'] = 'evaluation_epoch200'
            dump_json(status_path, state)
            evaluation = BUNDLE / 'evaluation'
            sys.path.insert(0, str(evaluation))
            spec = importlib.util.spec_from_file_location('dual_endpoint_evaluation', str(evaluation / 'run_evaluation.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if module.main() != 0:
                raise RuntimeError('fixed epoch200 evaluation failed')
            state['completed_stages'].append('evaluation_epoch200')
            state['evaluation_status'] = str(output / 'evaluation_epoch200' / 'status.json')
        state['final_bundle_integrity'] = verify_bundle()
        state.update(status='PASS', active_stage=None)
    except BaseException as error:
        state.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        state['finished_at'] = time.time()
        dump_json(status_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
