#!/usr/bin/env python3
"""Run the authorized three-arm suite sequentially with resource protection."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import resource_guard as guard


WORLD_SIZE = 8
SAMPLE_SECONDS = 5.0
SMOKE_TIMEOUT_SECONDS = 1200.0
STALL_TIMEOUT_SECONDS = 1800.0
ARMS = ["rgbd", "hha", "relplus"]
FINITE_FIELDS = ("loss_finite", "logits_finite", "gradients_finite")
SMOKE_REQUIRED_TRUE = FINITE_FIELDS + (
    "optimizer_step_executed", "checkpoint_saved", "checkpoint_resumed",
    "parameters_match_after_restore", "lr_continuous_across_restore",
    "lr_updated_after_resume", "pretrained_model_loaded",
)


class SuiteFailure(RuntimeError):
    def __init__(self, status, reason, details=None):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.details = details


def read_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SuiteFailure("BLOCKED", "EVIDENCE_UNREADABLE", {"path": str(path), "error": str(error)})
    if not isinstance(value, dict):
        raise SuiteFailure("BLOCKED", "EVIDENCE_NOT_OBJECT", str(path))
    return value


def require(row, field, expected, label):
    if row.get(field) != expected:
        raise SuiteFailure("FAIL", "{}_{}".format(label, field),
                           {"expected": expected, "found": row.get(field)})


def validate_suite(suite):
    if suite.get("authorization", {}).get("approved") is not True:
        raise SuiteFailure("BLOCKED", "AUTHORIZATION_REQUIRED")
    if not isinstance(suite.get("suite_id"), str) or not suite["suite_id"]:
        raise SuiteFailure("BLOCKED", "SUITE_ID_REQUIRED")
    if suite.get("arms") != ARMS:
        raise SuiteFailure("BLOCKED", "ARM_ORDER_MISMATCH")
    for field in ("remote_source_root", "output_root", "python"):
        value = suite.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise SuiteFailure("BLOCKED", "ABSOLUTE_PATH_REQUIRED", field)
    shared = suite.get("shared", {})
    if shared.get("nepochs") != 200 or shared.get("niters_per_epoch") != 945:
        raise SuiteFailure("BLOCKED", "TRAINING_ENDPOINT_MISMATCH")
    source, output = Path(suite["remote_source_root"]).resolve(), Path(suite["output_root"]).resolve()
    if output == source or output in source.parents:
        raise SuiteFailure("BLOCKED", "OUTPUT_SOURCE_PATH_COLLISION")
    return suite


def validate_initializations(directory):
    hashes = []
    for rank in range(WORLD_SIZE):
        row = read_json(Path(directory) / "rank_init_{:02d}.json".format(rank))
        require(row, "rank", rank, "INITIALIZATION")
        model_hash = row.get("model_sha256", "")
        if (not isinstance(model_hash, str) or len(model_hash) != 64 or
                any(character not in "0123456789abcdef" for character in model_hash)):
            raise SuiteFailure("FAIL", "INITIALIZATION_HASH_INVALID", {"rank": rank})
        hashes.append(model_hash)
    if len(set(hashes)) != 1:
        raise SuiteFailure("FAIL", "RANK_INITIALIZATION_MISMATCH")
    return hashes


def validate_smoke(directory):
    directory = Path(directory)
    summary = read_json(directory / "ddp_optimizer_smoke_summary.json")
    require(summary, "status", "PASS", "SMOKE_SUMMARY")
    for field in SMOKE_REQUIRED_TRUE:
        require(summary, field, True, "SMOKE_SUMMARY")
    require(summary, "rank_count", WORLD_SIZE, "SMOKE_SUMMARY")
    require(summary, "gpu_count", WORLD_SIZE, "SMOKE_SUMMARY")
    for rank in range(WORLD_SIZE):
        row = read_json(directory / "rank_{:02d}.json".format(rank))
        require(row, "status", "PASS", "SMOKE_RANK")
        require(row, "rank", rank, "SMOKE_RANK")
        require(row, "world_size", WORLD_SIZE, "SMOKE_RANK")
        for field in SMOKE_REQUIRED_TRUE:
            require(row, field, True, "SMOKE_RANK")
        if row.get("author_nan_replacement_count") != 0:
            raise SuiteFailure("FAIL", "SMOKE_RANK_NAN_REPLACEMENT", {"rank": rank})
    hashes = validate_initializations(directory)
    return {"status": "PASS", "model_sha256": hashes[0],
            "rank_initialization_count": len(hashes),
            "smoke_summary_sha256": guard.digest(directory / "ddp_optimizer_smoke_summary.json")}


def validate_train_prerequisites(suite, suite_path=None):
    output = Path(suite["output_root"])
    preflight = Path(suite.get("preflight_report", str(output / "preflight.json")))
    preflight_report = read_json(preflight)
    require(preflight_report, "status", "PASS", "PREFLIGHT")
    smoke_phase = read_json(output / "smoke_status.json")
    require(smoke_phase, "status", "PASS", "SMOKE_PHASE")
    require(smoke_phase, "suite_id", suite["suite_id"], "SMOKE_PHASE")
    if suite_path is not None:
        suite_hash = guard.digest(suite_path)
        require(preflight_report, "suite_sha256", suite_hash, "PREFLIGHT_SUITE_FINGERPRINT")
        require(smoke_phase, "suite_sha256", suite_hash, "SMOKE_SUITE_FINGERPRINT")
    reports = {arm: validate_smoke(output / "smoke" / arm) for arm in ARMS}
    hashes = {report["model_sha256"] for report in reports.values()}
    if len(hashes) != 1:
        raise SuiteFailure("FAIL", "CROSS_ARM_INITIALIZATION_MISMATCH", reports)
    return {"status": "PASS", "model_sha256": next(iter(hashes)),
            "rank_initialization_count": sum(row["rank_initialization_count"] for row in reports.values()),
            "preflight_sha256": guard.digest(preflight), "smoke_arms": reports}


def check_health(directory, phase, state, now, started):
    directory = Path(directory)
    errors = list(directory.glob("rank_error_*.json"))
    if errors:
        raise SuiteFailure("FAIL", "RANK_ERROR_RECORDED", [str(path) for path in errors])
    expected_model_hash = state.get("expected_model_sha256")
    if phase == "train" and expected_model_hash is not None and not state.get("initialization_verified"):
        if all((directory / "rank_init_{:02d}.json".format(rank)).is_file() for rank in range(WORLD_SIZE)):
            hashes = validate_initializations(directory)
            if hashes[0] != expected_model_hash:
                raise SuiteFailure("FAIL", "FORMAL_SMOKE_INITIALIZATION_MISMATCH",
                                   {"expected": expected_model_hash, "found": hashes[0]})
            guard.save(directory / "initialization_check.json", {
                "status": "PASS", "model_sha256": hashes[0],
                "rank_initialization_count": len(hashes), "checked_at": time.time(),
                "checked_after_launch_seconds": now - started,
            })
            state["initialization_verified"] = True
    status_path = directory / "runtime_status.json"
    if status_path.is_file():
        row = read_json(status_path)
        if row.get("status") in ("FAIL", "FAILED", "BLOCKED", "ERROR"):
            raise SuiteFailure("FAIL", "RUNTIME_REPORTED_FAILURE", row)
        if "author_nan_replacement_count" in row and row["author_nan_replacement_count"] != 0:
            raise SuiteFailure("FAIL", "RUNTIME_NAN_REPLACEMENT", row)
        for field in ("loss", "learning_rate"):
            if field in row and (not isinstance(row[field], (int, float)) or not math.isfinite(row[field])):
                raise SuiteFailure("FAIL", "RUNTIME_NONFINITE_{}".format(field), row)
        iteration = row.get("global_iteration")
        if iteration is not None:
            if not isinstance(iteration, int) or iteration < 0:
                raise SuiteFailure("FAIL", "INVALID_RUNTIME_ITERATION", row)
            previous = state["last_iteration"]
            if previous is not None and iteration < previous:
                raise SuiteFailure("FAIL", "RUNTIME_ITERATION_REGRESSED", row)
            if previous is None or iteration > previous:
                state.update(last_progress=now, last_iteration=iteration)
    if phase == "smoke" and now - started >= SMOKE_TIMEOUT_SECONDS:
        raise SuiteFailure("FAIL", "SMOKE_TIMEOUT_1200_SECONDS")
    if phase == "train" and now - state["last_progress"] >= STALL_TIMEOUT_SECONDS:
        raise SuiteFailure("FAIL", "FORMAL_PROGRESS_STALLED_1800_SECONDS", state.copy())


def validate_formal(directory, suite):
    directory = Path(directory)
    expected_epoch = suite["shared"]["nepochs"]
    expected_iteration = expected_epoch * suite["shared"]["niters_per_epoch"]
    runtime = read_json(directory / "runtime_status.json")
    for field, expected in (("status", "FORMAL_TRAINING_COMPLETED"),
                            ("epoch", expected_epoch), ("global_iteration", expected_iteration),
                            ("author_nan_replacement_count", 0)):
        require(runtime, field, expected, "FORMAL_ENDPOINT")
    for rank in range(WORLD_SIZE):
        done = read_json(directory / "rank_done_{:02d}.json".format(rank))
        require(done, "status", "COMPLETED", "FORMAL_RANK")
        require(done, "rank", rank, "FORMAL_RANK")
        for field, expected in (("epoch", expected_epoch), ("global_iteration", expected_iteration),
                                ("author_nan_replacement_count", 0)):
            if field in done:
                require(done, field, expected, "FORMAL_RANK")
    checkpoint = directory / "checkpoints" / "epoch-{}.pth".format(expected_epoch)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise SuiteFailure("FAIL", "FORMAL_CHECKPOINT_MISSING_OR_EMPTY", str(checkpoint))
    return {"status": "PASS", "epoch": expected_epoch, "global_iteration": expected_iteration,
            "checkpoint": str(checkpoint), "checkpoint_size": checkpoint.stat().st_size,
            "checkpoint_sha256": guard.digest(checkpoint), "rank_done_count": WORLD_SIZE}


def build_command(suite, suite_path, arm, phase, port):
    return [suite["python"], "-B", "-m", "torch.distributed.launch", "--nproc_per_node=8",
            "--master_addr=127.0.0.1", "--master_port={}".format(port),
            str(Path(suite["remote_source_root"]) / "run_arm.py"),
            "--suite", str(Path(suite_path).resolve()), "--arm", arm,
            "--mode", "smoke" if phase == "smoke" else "formal", "--port", str(port)]


def _failure(error):
    if isinstance(error, (SuiteFailure, guard.StopProbe)):
        return SuiteFailure(error.status, error.reason, error.details)
    return SuiteFailure("BLOCKED" if isinstance(error, KeyboardInterrupt) else "FAIL",
                        "SUPERVISOR_EXCEPTION", "{}: {}".format(type(error).__name__, error))


def run_arm(suite, suite_path, arm, phase):
    directory = Path(suite["output_root"]) / ("smoke" if phase == "smoke" else "runs") / arm
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError:
        raise SuiteFailure("BLOCKED", "ARM_OUTPUT_EXISTS", str(directory))
    outcome = {"status": "RUNNING", "arm": arm, "phase": phase,
               "suite_id": suite["suite_id"], "started_at": time.time()}
    guard.save(directory / "arm_status.json", outcome)
    proc, failure, identities = None, None, {}
    try:
        expected_model_hash = None
        if phase == "train":
            expected_model_hash = validate_smoke(
                Path(suite["output_root"]) / "smoke" / arm)["model_sha256"]
        guard.preflight(directory)
        port = guard.free_port()
        command = build_command(suite, suite_path, arm, phase, port)
        overrides = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
        environment = dict(os.environ, **overrides)
        root = Path(suite["remote_source_root"])
        guard.save(directory / "command.json", {
            "argv": command, "cwd": str(root), "environment_overrides": overrides,
            "suite_sha256": guard.digest(suite_path),
            "run_arm_sha256": guard.digest(root / "run_arm.py"),
            "run_suite_sha256": guard.digest(Path(__file__)),
            "resource_guard_sha256": guard.digest(Path(guard.__file__)),
        })
        with (directory / "launcher.log").open("w") as log, (directory / "resource_samples.jsonl").open("w") as samples:
            proc = subprocess.Popen(command, cwd=str(root), env=environment,
                                    stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            identities[proc.pid] = guard.process_start(proc.pid)
            if identities[proc.pid] is None or os.getpgid(proc.pid) != proc.pid:
                raise SuiteFailure("BLOCKED", "LAUNCHER_PROCESS_IDENTITY_NOT_CONFIRMED")
            guard.save(directory / "process.json", {
                "supervisor_pid": os.getpid(), "supervisor_starttime": guard.process_start(os.getpid()),
                "launcher_pid": proc.pid, "launcher_starttime": identities[proc.pid],
                "pgid": proc.pid, "started_at": time.time(),
            })
            started = time.monotonic()
            state = {"last_progress": started, "last_iteration": None,
                     "expected_model_sha256": expected_model_hash, "initialization_verified": False}
            while proc.poll() is None:
                tick = time.monotonic()
                check_health(directory, phase, state, tick, started)
                snapshot = guard.resources()
                samples.write(json.dumps(snapshot) + "\n")
                samples.flush()
                guard.remember_group_members(proc.pid, identities)
                guard.check_running(snapshot, proc.pid, identities)
                time.sleep(max(0.0, SAMPLE_SECONDS - (time.monotonic() - tick)))
            code = proc.wait()
            if code != 0:
                raise SuiteFailure("FAIL", "NONZERO_LAUNCHER_EXIT", code)
            check_health(directory, phase, state, time.monotonic(), started)
        if phase == "smoke":
            outcome.update(validate_smoke(directory))
        else:
            outcome.update(validate_formal(directory, suite))
            formal_hashes = validate_initializations(directory)
            smoke_hash = validate_smoke(Path(suite["output_root"]) / "smoke" / arm)["model_sha256"]
            if formal_hashes[0] != smoke_hash:
                raise SuiteFailure("FAIL", "FORMAL_SMOKE_INITIALIZATION_MISMATCH")
            outcome["model_sha256"] = formal_hashes[0]
    except BaseException as error:
        failure = _failure(error)
    finally:
        if proc is not None:
            try:
                receipt = guard.stop_own_group(proc, identities)
                guard.save(directory / "cleanup.json", receipt)
                if receipt.get("remaining_members"):
                    failure = SuiteFailure("BLOCKED", "OWN_PROCESS_CLEANUP_INCOMPLETE", receipt)
            except BaseException as error:
                failure = SuiteFailure("BLOCKED", "OWN_PROCESS_CLEANUP_FAILED", str(error))
            (directory / "exitcode").write_text(str(proc.poll()) + "\n", encoding="utf-8")
        else:
            (directory / "exitcode").write_text("NOT_LAUNCHED\n", encoding="utf-8")
        try:
            guard.save(directory / "resources_after.json", guard.resources())
        except BaseException as error:
            outcome["resources_after_error"] = str(error)
        if failure is not None:
            outcome.update(status=failure.status, reason=failure.reason, details=failure.details)
        outcome["finished_at"] = time.time()
        guard.save(directory / "arm_status.json", outcome)
    if failure is not None:
        raise failure
    return outcome


def _claim_phase(output, phase, suite):
    path = output / "{}.lock".format(phase)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise SuiteFailure("BLOCKED", "PHASE_ALREADY_CLAIMED", str(path))
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid(), "starttime": guard.process_start(os.getpid()),
                   "suite_id": suite["suite_id"], "phase": phase,
                   "created_at": time.time(), "restart_allowed": False}, handle, indent=2)
        handle.write("\n")
    # Kept as a permanent launch receipt: this runner has no resume/retry path.


def run_phase(suite, suite_path, phase):
    validate_suite(suite)
    if phase not in ("smoke", "train"):
        raise SuiteFailure("BLOCKED", "INVALID_PHASE")
    output = Path(suite["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    _claim_phase(output, phase, suite)
    status_path = output / "{}_status.json".format(phase)
    if status_path.exists():
        raise SuiteFailure("BLOCKED", "PHASE_ALREADY_HAS_STATUS", str(status_path))
    status = {"status": "RUNNING", "suite_id": suite["suite_id"], "phase": phase,
              "started_at": time.time(), "supervisor_pid": os.getpid(),
              "suite_sha256": guard.digest(suite_path), "arms": ARMS, "completed": []}
    guard.save(status_path, status)
    try:
        if phase == "train":
            status["prerequisites"] = validate_train_prerequisites(suite, suite_path)
            guard.save(status_path, status)
        for arm in ARMS:
            status["active_arm"] = arm
            guard.save(status_path, status)
            result = run_arm(suite, suite_path, arm, phase)
            status["completed"].append(result)
            guard.save(status_path, status)
        if phase == "smoke":
            if len({row["model_sha256"] for row in status["completed"]}) != 1:
                raise SuiteFailure("FAIL", "CROSS_ARM_INITIALIZATION_MISMATCH")
        status.update(status="PASS", active_arm=None)
    except BaseException as error:
        failure = _failure(error)
        status.update(status=failure.status, reason=failure.reason, details=failure.details)
    finally:
        status["finished_at"] = time.time()
        guard.save(status_path, status)
    return status


def interrupted(signum, frame):
    raise SuiteFailure("BLOCKED", "SUPERVISOR_INTERRUPTED", signum)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("smoke", "train"))
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        suite_path = args.suite.resolve()
        suite = read_json(suite_path)
        validate_suite(suite)
        if Path(suite["remote_source_root"]).resolve() != Path(__file__).resolve().parent:
            raise SuiteFailure("BLOCKED", "BUNDLE_ROOT_MISMATCH")
        result = run_phase(suite, suite_path, args.phase)
    except BaseException as error:
        failure = _failure(error)
        result = {"status": failure.status, "reason": failure.reason, "details": failure.details}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
