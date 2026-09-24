"""Launch exactly one durable pipeline phase and record its process identity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from suite_common import digest, dump_json, load_suite
import resource_guard as guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('prepare', 'execute'))
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    suite = load_suite(bundle / 'suite.json')
    if bundle != Path(suite['remote_source_root']):
        raise RuntimeError('incorrect deployed bundle')
    output = Path(suite['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / (args.phase + '_supervisor_launch.json')
    with receipt.open('x') as handle:
        json.dump({'status': 'CLAIMED', 'phase': args.phase, 'time': time.time()}, handle)
    command = [suite['python'], '-B', str(bundle / 'pipeline.py'), '--phase', args.phase]
    overrides = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                     OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (output / (args.phase + '_supervisor.log')).open('x') as log:
        process = subprocess.Popen(command, cwd=str(bundle), env=dict(os.environ, **overrides),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    identity = guard.process_start(process.pid)
    if identity is None or os.getpgid(process.pid) != process.pid:
        raise RuntimeError('supervisor process identity unavailable')
    value = {'status': 'LAUNCHED', 'pid': process.pid, 'starttime': identity,
             'pgid': process.pid, 'argv': command, 'cwd': str(bundle),
             'phase': args.phase, 'started_at': time.time(), 'detached_session': True,
             'suite_sha256': digest(bundle / 'suite.json'),
             'bundle_manifest_sha256': digest(bundle / 'bundle_manifest.json')}
    dump_json(receipt, value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
