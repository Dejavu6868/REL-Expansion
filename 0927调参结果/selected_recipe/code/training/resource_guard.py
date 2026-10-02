#!/usr/bin/env python3
"""Run bounded, isolated batch probes. Python 3.8 standard library only."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


GIB = 1024 ** 3
MIB = 1024 ** 2
WORLD_SIZE = 8
MEASURED_IMAGES_PER_RANK = 336
WARMUP_IMAGES_PER_RANK = 168
BASE = Path('/data/zhuzhaoziao/RELPlus/outputs/CMX_S2D_ThreeArm_v1')
DIAGNOSTICS = BASE / 'diagnostics'
SOURCE = Path('/home/zhuzhaoziao/RELPlus/CMX-S2D-ThreeArm-v1')
PYTHON = '/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python'
CHECKPOINT = BASE / 'cmx_hha_seed12345/checkpoints/epoch-200.pth'
RESOLVED = BASE / 'launch_evidence/hha_20260908_194223/frozen_formal_resolved_config.json'


class StopProbe(RuntimeError):
    def __init__(self, status, reason, details=None):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.details = details


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(path))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * MIB), b''):
            value.update(chunk)
    return value.hexdigest()


def audit(expected):
    stat = CHECKPOINT.stat()
    result = {
        'source_hashes': {
            str(path.relative_to(SOURCE)): digest(path)
            for path in sorted(SOURCE.rglob('*.py')) if '__pycache__' not in path.parts
        },
        'checkpoint': {'path': str(CHECKPOINT), 'size': stat.st_size,
                       'mtime_ns': str(stat.st_mtime_ns), 'sha256': digest(CHECKPOINT)},
        'resolved_sha256': digest(RESOLVED),
    }
    result['additional_readonly_inputs'] = [
        {'path': row['path'], 'sha256': digest(row['path'])}
        for row in expected.get('additional_readonly_inputs', [])
    ]
    return result


def check_integrity(actual, expected):
    differences = [key for key in ('source_hashes', 'checkpoint', 'resolved_sha256', 'additional_readonly_inputs')
                   if actual.get(key) != expected.get(key)]
    if differences:
        raise StopProbe('BLOCKED', 'FROZEN_INTEGRITY_MISMATCH', differences)


def query(arguments):
    result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=5)
    if result.returncode:
        raise StopProbe('BLOCKED', 'RESOURCE_QUERY_FAILED', {
            'command': arguments, 'exitcode': result.returncode, 'stderr': result.stderr})
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def number(value):
    parsed = float(value.strip())
    if not math.isfinite(parsed):
        raise ValueError('Non-finite resource value')
    return parsed


def resources():
    gpu_rows = query(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                      '--format=csv,noheader,nounits'])
    process_rows = query(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory',
                          '--format=csv,noheader,nounits'])
    gpus = []
    for row in gpu_rows:
        if len(row) != 7:
            raise StopProbe('BLOCKED', 'UNEXPECTED_GPU_QUERY_FORMAT', row)
        gpus.append({'index': int(row[0]), 'uuid': row[1].strip(),
                     'total_bytes': int(number(row[2]) * MIB),
                     'used_bytes': int(number(row[3]) * MIB),
                     'free_bytes': int(number(row[4]) * MIB),
                     'utilization_percent': number(row[5]), 'temperature_c': number(row[6])})
    processes = []
    for row in process_rows:
        if not row or not any(item.strip() for item in row):
            continue
        if len(row) != 4:
            raise StopProbe('BLOCKED', 'UNEXPECTED_PROCESS_QUERY_FORMAT', row)
        processes.append({'pid': int(row[0]), 'name': row[1].strip(), 'gpu_uuid': row[2].strip(),
                          'used_gpu_memory_mib_raw': row[3].strip()})
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key == 'MemAvailable':
            memory['available_bytes'] = int(value.split()[0]) * 1024
    shm = os.statvfs('/dev/shm')
    if 'available_bytes' not in memory:
        raise StopProbe('BLOCKED', 'MEMAVAILABLE_UNAVAILABLE')
    return {'time': time.time(), 'gpus': sorted(gpus, key=lambda gpu: gpu['index']),
            'compute_processes': processes, 'host': memory,
            'shm_available_bytes': shm.f_bavail * shm.f_frsize}


def require_idle(snapshot):
    if [gpu['index'] for gpu in snapshot['gpus']] != list(range(WORLD_SIZE)):
        raise StopProbe('BLOCKED', 'EXPECTED_EIGHT_GPUS')
    if snapshot['compute_processes']:
        raise StopProbe('BLOCKED', 'OTHER_GPU_COMPUTE_PROCESS', snapshot['compute_processes'])
    if snapshot['host']['available_bytes'] < 64 * GIB:
        raise StopProbe('BLOCKED', 'HOST_PREFLIGHT_HEADROOM')
    if snapshot['shm_available_bytes'] < 4 * GIB:
        raise StopProbe('BLOCKED', 'SHM_PREFLIGHT_HEADROOM')
    for gpu in snapshot['gpus']:
        if gpu['used_bytes'] >= 512 * MIB or gpu['utilization_percent'] > 5:
            raise StopProbe('BLOCKED', 'GPU_NOT_IDLE', gpu)
        if gpu['temperature_c'] >= 85:
            raise StopProbe('BLOCKED', 'GPU_PREFLIGHT_TEMPERATURE', gpu)


def preflight(output):
    """Allow utilization accounting to settle after our preceding process exited."""
    deadline = time.monotonic() + 15
    with (output / 'idle_samples.jsonl').open('w') as handle:
        while True:
            snapshot = resources()
            handle.write(json.dumps(snapshot) + '\n')
            handle.flush()
            try:
                require_idle(snapshot)
                save(output / 'gpu_before.json', snapshot)
                return snapshot
            except StopProbe as error:
                if error.reason != 'GPU_NOT_IDLE' or time.monotonic() >= deadline:
                    raise
            time.sleep(1)


def process_start(pid):
    try:
        # The comm field may contain spaces or parentheses; fields after the final ')' are stable.
        tail = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()
        if tail[0] in ('Z', 'X'):
            return None
        return tail[19]  # Field 22: process start time in clock ticks.
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def remember_group_members(pgid, identities):
    anchors = {pid: identities[pid] for pid in verified_members(pgid, identities)}
    if not anchors:
        return
    discovered = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) == pgid:
                started = process_start(pid)
                if started is not None:
                    discovered[pid] = started
        except (ProcessLookupError, PermissionError):
            continue
    if verified_members(pgid, anchors):
        for pid, started in discovered.items():
            identities.setdefault(pid, started)


def verified_members(pgid, identities):
    result = []
    for pid, started in identities.items():
        if started is None:
            continue
        try:
            if process_start(pid) == started and os.getpgid(pid) == pgid:
                result.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    return result


def stop_own_group(proc, identities):
    """Signal only the new session's verified group, never a name or foreign PID."""
    pgid = proc.pid
    if pgid <= 1 or pgid == os.getpgrp():
        raise StopProbe('BLOCKED', 'UNSAFE_PROCESS_GROUP_ID', pgid)
    members = verified_members(pgid, identities)
    receipt = {'pgid': pgid, 'verified_members': members, 'signals': []}
    if not members:
        receipt['remaining_members'] = []
        receipt['reason'] = 'NO_LIVE_PREVIOUSLY_OWNED_GROUP_ANCHOR; no group was claimed or signalled.'
        return receipt
    anchors = {pid: identities[pid] for pid in members}
    remember_group_members(pgid, identities)
    # Recheck the known identities after enumeration; a numeric PGID alone is not ownership.
    if not verified_members(pgid, anchors):
        receipt['remaining_members'] = []
        return receipt
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        receipt['remaining_members'] = []
        return receipt
    receipt['signals'].append('SIGTERM')
    deadline = time.monotonic() + 10
    soft_deadline = deadline - 2
    while time.monotonic() < soft_deadline:
        proc.poll()
        if not verified_members(pgid, identities):
            break
        time.sleep(0.2)
    members = verified_members(pgid, identities)
    if members:
        try:
            os.killpg(pgid, signal.SIGKILL)
            receipt['signals'].append('SIGKILL')
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=max(0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        receipt['launcher_wait_timeout'] = True
    while time.monotonic() < deadline and verified_members(pgid, identities):
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    receipt['remaining_members'] = verified_members(pgid, identities)
    return receipt


def check_running(snapshot, pgid, identities):
    if [gpu['index'] for gpu in snapshot['gpus']] != list(range(WORLD_SIZE)):
        raise StopProbe('BLOCKED', 'GPU_INVENTORY_CHANGED')
    for process in snapshot['compute_processes']:
        pid = process['pid']
        try:
            observed_pgid = os.getpgid(pid)
        except ProcessLookupError:
            # A query result may describe a process that exited before inspection.
            continue
        except PermissionError:
            raise StopProbe('BLOCKED', 'GPU_PROCESS_OWNERSHIP_UNKNOWN', process)
        if observed_pgid != pgid:
            raise StopProbe('BLOCKED', 'FOREIGN_GPU_PROCESS_APPEARED', process)
        if not verified_members(pgid, identities):
            raise StopProbe('BLOCKED', 'OWN_PROCESS_GROUP_ANCHOR_LOST', process)
        started = process_start(pid)
        if started is not None:
            identities.setdefault(pid, started)
    if snapshot['host']['available_bytes'] < 32 * GIB:
        raise StopProbe('BLOCKED', 'HOST_RUNTIME_HEADROOM')
    if snapshot['shm_available_bytes'] < 2 * GIB:
        raise StopProbe('BLOCKED', 'SHM_RUNTIME_HEADROOM')
    for gpu in snapshot['gpus']:
        if gpu['temperature_c'] >= 85:
            raise StopProbe('BLOCKED', 'GPU_TEMPERATURE_LIMIT', gpu)
        if gpu['free_bytes'] < max(3 * GIB, 0.20 * gpu['total_bytes']):
            raise StopProbe('BLOCKED', 'GPU_RUNTIME_HEADROOM', gpu)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(('127.0.0.1', 0))
        return handle.getsockname()[1]


def validate_rank_results(case_dir, local_batch, repeat, baseline=None):
    if (WARMUP_IMAGES_PER_RANK % local_batch or
            MEASURED_IMAGES_PER_RANK % local_batch):
        raise StopProbe("FAIL", "NON_DIVISIBLE_SAMPLE_WINDOWS")
    records = []
    for rank in range(WORLD_SIZE):
        path = case_dir / 'rank_{}.json'.format(rank)
        if not path.is_file():
            raise StopProbe('FAIL', 'MISSING_RANK_RESULT', {'rank': rank})
        row = json.loads(path.read_text())
        expected = {'status': 'PASS', 'rank': rank, 'local_batch': local_batch, 'repeat': repeat,
                    'warmup_steps': WARMUP_IMAGES_PER_RANK // local_batch,
                    'measured_steps': MEASURED_IMAGES_PER_RANK // local_batch, 'error_count': 0}
        for key, value in expected.items():
            if row.get(key) != value:
                raise StopProbe('FAIL', 'RANK_RESULT_CONTRACT', {'rank': rank, 'field': key})
        for key in ('slowest_rank_seconds', 'global_images_per_sec', 'max_memory_allocated',
                    'max_memory_reserved', 'total_memory_bytes'):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] <= 0:
                raise StopProbe('FAIL', 'INVALID_RANK_MEASUREMENT', {'rank': rank, 'field': key})
        if row['max_memory_allocated'] > row['max_memory_reserved']:
            raise StopProbe('FAIL', 'INVALID_MEMORY_PEAK_ORDER', {'rank': rank})
        if row['max_memory_reserved'] > 0.75 * row['total_memory_bytes']:
            raise StopProbe('FAIL', 'ALLOCATOR_HEADROOM_LIMIT', {'rank': rank})
        if len(row.get('measured_sample_ids', [])) != MEASURED_IMAGES_PER_RANK:
            raise StopProbe('FAIL', 'MEASURED_IMAGE_COUNT', {'rank': rank})
        if len(row.get('input_trace', [])) != WARMUP_IMAGES_PER_RANK + MEASURED_IMAGES_PER_RANK:
            raise StopProbe('FAIL', 'INPUT_TRACE_IMAGE_COUNT', {'rank': rank})
        optimizer_entries = row.get('optimizer_state_entries')
        if not isinstance(optimizer_entries, int) or optimizer_entries <= 0:
            raise StopProbe('FAIL', 'ADAMW_STATE_NOT_CONFIRMED', {'rank': rank})
        total_steps = expected['warmup_steps'] + expected['measured_steps']
        losses = row.get('losses', [])
        if (len(losses) != total_steps or
                not all(isinstance(loss, (int, float)) and math.isfinite(loss) for loss in losses)):
            raise StopProbe('FAIL', 'INCOMPLETE_OR_NONFINITE_LOSS_TRACE', {'rank': rank})
        memory_rows = row.get('memory_rows', [])
        if len(memory_rows) != total_steps or [item.get('step') for item in memory_rows] != list(range(1, total_steps + 1)):
            raise StopProbe('FAIL', 'INCOMPLETE_OPTIMIZER_STEP_TRACE', {'rank': rank})
        trace = row['input_trace']
        if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in trace):
            raise StopProbe('FAIL', 'INVALID_SAMPLE_TRACE_FORMAT', {'rank': rank})
        if row['measured_sample_ids'] != [item[0] for item in trace[WARMUP_IMAGES_PER_RANK:]]:
            raise StopProbe('FAIL', 'MEASURED_IDS_DO_NOT_MATCH_TRACE', {'rank': rank})
        model_hash = row.get('initial_model_sha256', '')
        if not isinstance(model_hash, str) or len(model_hash) != 64 or any(c not in '0123456789abcdef' for c in model_hash):
            raise StopProbe('FAIL', 'MODEL_HASH_INVALID', {'rank': rank})
        calculated = WORLD_SIZE * MEASURED_IMAGES_PER_RANK / row['slowest_rank_seconds']
        if not math.isclose(calculated, row['global_images_per_sec'], rel_tol=1e-6):
            raise StopProbe('FAIL', 'THROUGHPUT_ARITHMETIC', {'rank': rank})
        if baseline is not None:
            for key in ('initial_model_sha256', 'measured_sample_ids', 'input_trace'):
                if row[key] != baseline[rank][key]:
                    raise StopProbe('FAIL', 'CROSS_BATCH_INPUT_OR_INITIALIZATION_MISMATCH', {'rank': rank, 'field': key})
        records.append(row)
    if len({row['initial_model_sha256'] for row in records}) != 1:
        raise StopProbe('FAIL', 'RANK_INITIALIZATION_MISMATCH')
    measured_ids = [sample_id for row in records for sample_id in row['measured_sample_ids']]
    if len(set(measured_ids)) != len(measured_ids):
        raise StopProbe('FAIL', 'DUPLICATE_MEASURED_SAMPLE_IDS_WITHIN_OR_ACROSS_RANKS')
    slowest = max(row['slowest_rank_seconds'] for row in records)
    if not all(math.isclose(row['slowest_rank_seconds'], slowest, rel_tol=1e-6) for row in records):
        raise StopProbe('FAIL', 'RANK_TIMING_AGGREGATION_MISMATCH')
    return records


def run_case(case_dir, local_batch, repeat, probe, baseline=None):
    case_dir.mkdir()
    outcome = {'status': 'RUNNING', 'local_batch': local_batch, 'repeat': repeat,
               'started_at': time.time(), 'no_checkpoint_written': True}
    save(case_dir / 'case_summary.json', outcome)
    proc = None
    identities = {}
    failure = None
    try:
        preflight(case_dir)
        command = [PYTHON, '-B', '-m', 'torch.distributed.launch', '--nproc_per_node=8',
                   '--master_addr=127.0.0.1', '--master_port={}'.format(free_port()),
                   str(probe), '--source', str(SOURCE), '--output', str(case_dir),
                   '--resolved', str(RESOLVED), '--local-batch', str(local_batch), '--repeat', str(repeat)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
        save(case_dir / 'command.json', {'argv': command, 'cwd': str(SOURCE),
             'environment_overrides': {key: env[key] for key in ('PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')}})
        with (case_dir / 'probe.log').open('w') as log, (case_dir / 'resource_samples.jsonl').open('w') as samples:
            proc = subprocess.Popen(command, cwd=str(SOURCE), env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            identities[proc.pid] = process_start(proc.pid)
            if os.getpgid(proc.pid) != proc.pid:
                raise StopProbe('BLOCKED', 'LAUNCHER_PROCESS_GROUP_MISMATCH')
            save(case_dir / 'process.json', {'supervisor_pid': os.getpid(), 'launcher_pid': proc.pid,
                                           'pgid': proc.pid, 'started_at': time.time()})
            deadline = time.monotonic() + 600
            while proc.poll() is None:
                if list(case_dir.glob('error_rank_*.json')):
                    raise StopProbe('FAIL', 'RANK_ERROR_RECORDED')
                if time.monotonic() >= deadline:
                    raise StopProbe('FAIL', 'CASE_TIMEOUT_600_SECONDS')
                sample_started = time.monotonic()
                snapshot = resources()
                samples.write(json.dumps(snapshot) + '\n')
                samples.flush()
                remember_group_members(proc.pid, identities)
                check_running(snapshot, proc.pid, identities)
                time.sleep(max(0.0, 1.0 - (time.monotonic() - sample_started)))
            code = proc.wait()
            if code != 0:
                raise StopProbe('FAIL', 'NONZERO_LAUNCHER_EXIT', code)
        if list(case_dir.glob('error_rank_*.json')):
            raise StopProbe('FAIL', 'RANK_ERROR_RECORDED')
        records = validate_rank_results(case_dir, local_batch, repeat, baseline)
        outcome.update(status='PASS', slowest_rank_seconds=max(row['slowest_rank_seconds'] for row in records),
                       global_images_per_sec=WORLD_SIZE * MEASURED_IMAGES_PER_RANK / max(row['slowest_rank_seconds'] for row in records))
    except StopProbe as error:
        failure = error
    except BaseException as error:
        failure = StopProbe('BLOCKED' if isinstance(error, KeyboardInterrupt) else 'FAIL',
                            'SUPERVISOR_EXCEPTION', '{}: {}'.format(type(error).__name__, error))
    finally:
        if proc is not None:
            try:
                receipt = stop_own_group(proc, identities)
                save(case_dir / 'cleanup.json', receipt)
                if receipt.get('remaining_members'):
                    failure = StopProbe('BLOCKED', 'OWN_PROCESS_CLEANUP_INCOMPLETE', receipt)
            except BaseException as error:
                failure = StopProbe('BLOCKED', 'OWN_PROCESS_CLEANUP_FAILED', str(error))
            (case_dir / 'exitcode').write_text(str(proc.poll()) + '\n')
        else:
            (case_dir / 'exitcode').write_text('NOT_LAUNCHED\n')
        if failure is not None:
            outcome.update(status=failure.status, reason=failure.reason, details=failure.details)
        try:
            # Post-run observers report any new services; they never terminate them.
            save(case_dir / 'resources_after.json', resources())
        except BaseException as error:
            outcome['resources_after_error'] = '{}: {}'.format(type(error).__name__, error)
        outcome['finished_at'] = time.time()
        save(case_dir / 'case_summary.json', outcome)
    if failure is not None:
        raise failure
    return records, outcome


def summarize_level(local_batch, cases):
    elapsed = sum(case['slowest_rank_seconds'] for case in cases)
    return {'status': 'PASS', 'local_batch': local_batch, 'global_batch': WORLD_SIZE * local_batch,
            'repeat_images_per_sec': [case['global_images_per_sec'] for case in cases],
            'combined_global_images_per_sec': 2 * WORLD_SIZE * MEASURED_IMAGES_PER_RANK / elapsed,
            'combined_slowest_rank_seconds': elapsed, 'case_paths': [case['case_path'] for case in cases]}


def predicted_headroom(records_by_repeat, next_batch):
    previous = next_batch - 1
    rows = []
    for rank in range(WORLD_SIZE):
        total = min(records[rank]['total_memory_bytes'] for records in records_by_repeat)
        reserved = max(records[rank]['max_memory_reserved'] for records in records_by_repeat)
        predicted = reserved * next_batch / previous
        rows.append({'rank': rank, 'previous_reserved_bytes': reserved, 'predicted_reserved_bytes': predicted,
                     'limit_bytes': 0.75 * total, 'within_limit': predicted <= 0.75 * total})
    return {'eligible': all(row['within_limit'] for row in rows), 'per_rank': rows,
            'method': 'previous observed peak reserved * next_batch / previous_batch',
            'interpretation': 'Conservative estimate, not an observed OOM or a guarantee that the next batch fits.'}


def useful_gain(previous, current):
    ratio = current['combined_global_images_per_sec'] / previous['combined_global_images_per_sec']
    repeats = [new >= old for old, new in zip(previous['repeat_images_per_sec'], current['repeat_images_per_sec'])]
    return {'eligible': ratio >= 1.05 and all(repeats), 'combined_speed_ratio': ratio,
            'combined_gain_percent': 100 * (ratio - 1), 'repeat_not_slower': repeats,
            'required_combined_speed_ratio': 1.05}


def validated_output(value):
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError('--output must be absolute')
    output = raw.resolve()
    if output.parent != DIAGNOSTICS.resolve() or not output.name.startswith('batch_probe_') or output.name == 'batch_probe_':
        raise ValueError('--output must be a new direct diagnostics/batch_probe_* directory')
    if os.path.lexists(str(raw)) or output.exists():
        raise ValueError('--output already exists; never overwrite an earlier attempt')
    return output


def interrupted(signum, frame):
    raise KeyboardInterrupt('Supervisor received signal {}'.format(signum))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-local-batch', type=int, choices=(6, 7, 8), default=8)
    args = parser.parse_args(argv)
    output = validated_output(args.output)
    script_dir = Path(__file__).resolve().parent
    expected_path = script_dir / 'expected_integrity.json'
    probe = script_dir / 'probe_batch.py'
    expected = json.loads(expected_path.read_text())
    output.mkdir()
    summary = {'status': 'RUNNING', 'planned': list(range(6, args.max_local_batch + 1)),
               'completed': [], 'latest': None, 'started_at': time.time(), 'repeats_per_level': 2,
               'scope': 'Disposable engineering probe only; no formal training or checkpoint writes.',
               'diagnostic_code_sha256': {'run_stage.py': digest(__file__), 'probe_batch.py': digest(probe),
                                          'expected_integrity.json': digest(expected_path)}}
    save(output / 'summary.json', summary)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    previous_level = None
    previous_records = None
    baseline = {}
    initial_hash = None
    before = None
    try:
        before = audit(expected)
        save(output / 'integrity_before.json', before)
        check_integrity(before, expected)
        for local_batch in summary['planned']:
            level_dir = output / 'local_batch_{}'.format(local_batch)
            level_dir.mkdir()
            summary['latest'] = {'local_batch': local_batch, 'status': 'RUNNING'}
            save(output / 'summary.json', summary)
            if previous_records is not None:
                prediction = predicted_headroom(previous_records, local_batch)
                save(level_dir / 'headroom_prediction.json', prediction)
                if not prediction['eligible']:
                    raise StopProbe('STOP_PREDICTED_HEADROOM', 'CONSERVATIVE_PREDICTION_EXCEEDS_ALLOCATOR_BUDGET', prediction)
            cases = []
            level_records = []
            for repeat in (0, 1):
                case_dir = output / 'local_batch_{}_repeat_{}'.format(local_batch, repeat)
                records, case = run_case(case_dir, local_batch, repeat, probe, baseline.get(repeat))
                if initial_hash is None:
                    initial_hash = records[0]['initial_model_sha256']
                if records[0]['initial_model_sha256'] != initial_hash:
                    raise StopProbe('FAIL', 'REPEAT_INITIALIZATION_MISMATCH')
                if local_batch == 6:
                    baseline[repeat] = records
                case['case_path'] = str(case_dir.relative_to(output))
                cases.append(case)
                level_records.append(records)
            level = summarize_level(local_batch, cases)
            level['observed_peak_reserved_by_rank'] = [max(rows[rank]['max_memory_reserved'] for rows in level_records)
                                                       for rank in range(WORLD_SIZE)]
            if previous_level is not None:
                level['gain_gate'] = useful_gain(previous_level, level)
                if not level['gain_gate']['eligible']:
                    level['status'] = 'STOP_NO_USEFUL_GAIN'
            save(level_dir / 'level_summary.json', level)
            summary['completed'].append(level)
            summary['latest'] = level
            save(output / 'summary.json', summary)
            if level['status'] == 'STOP_NO_USEFUL_GAIN':
                raise StopProbe('STOP_NO_USEFUL_GAIN', 'GAIN_UNDER_5_PERCENT_OR_A_REPEAT_REGRESSED', level['gain_gate'])
            previous_level, previous_records = level, level_records
        summary['status'] = 'PASS'
        summary['recommended_local_batch_within_tested_range'] = previous_level['local_batch']
        summary['recommendation_scope'] = 'SHORT_PROBE_ONLY; does not authorize or validate formal training.'
    except StopProbe as error:
        summary.update(status=error.status, reason=error.reason, details=error.details)
        if previous_level is not None:
            summary['last_level_passing_all_gates'] = previous_level['local_batch']
    except BaseException as error:
        summary.update(status='BLOCKED' if isinstance(error, KeyboardInterrupt) else 'FAIL',
                       reason='SUPERVISOR_EXCEPTION', details='{}: {}'.format(type(error).__name__, error))
    finally:
        try:
            after = audit(expected)
            save(output / 'integrity_after.json', after)
            check_integrity(after, expected)
            summary['frozen_integrity_unchanged'] = before == after
            if before != after:
                summary.update(status='FAIL', reason='FROZEN_INTEGRITY_CHANGED_DURING_PROBE')
        except BaseException as error:
            summary['prior_status'] = summary['status']
            summary.update(status='BLOCKED', reason='FINAL_INTEGRITY_NOT_CONFIRMED', details=str(error))
        summary['finished_at'] = time.time()
        if isinstance(summary.get('latest'), dict) and summary['latest'].get('status') == 'RUNNING':
            summary['latest'].update(status=summary['status'], reason=summary.get('reason'))
            level_dir = output / 'local_batch_{}'.format(summary['latest']['local_batch'])
            save(level_dir / 'level_summary.json', summary['latest'])
        save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['status'] in ('PASS', 'STOP_NO_USEFUL_GAIN', 'STOP_PREDICTED_HEADROOM') else 1


if __name__ == '__main__':
    sys.exit(main())
