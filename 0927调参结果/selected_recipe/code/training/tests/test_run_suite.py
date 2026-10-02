import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
import run_suite as runner
from suite_common import EXPECTED_SHARED


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.suite = {
            "suite_id": "isolated-test", "remote_source_root": str(BUNDLE),
            "output_root": str(self.root / "output"), "python": sys.executable,
            "arms": ["hha", "relplus"],
            "shared": dict(EXPECTED_SHARED),
            "authorization": {"approved": True},
        }
        self.suite_path = self.root / "suite.json"
        write(self.suite_path, self.suite)

    def smoke_reports(self, arm, model_hash="a" * 64):
        directory = Path(self.suite["output_root"]) / "smoke" / arm
        ranks = []
        for rank in range(8):
            report = {
                "status": "PASS", "rank": rank, "world_size": 8,
                "loss_finite": True, "logits_finite": True,
                "gradients_finite": True, "author_nan_replacement_count": 0,
                "optimizer_step_executed": True, "checkpoint_saved": True,
                "checkpoint_resumed": True, "parameters_match_after_restore": True,
                "lr_continuous_across_restore": True, "lr_updated_after_resume": True,
                "pretrained_model_loaded": True,
            }
            ranks.append(report)
            write(directory / "rank_{:02d}.json".format(rank), report)
            write(directory / "rank_init_{:02d}.json".format(rank),
                  {"rank": rank, "model_sha256": model_hash})
        summary = dict(ranks[0], rank_count=8, gpu_count=8, ranks=ranks)
        summary.pop("rank")
        write(directory / "ddp_optimizer_smoke_summary.json", summary)
        return directory

    def prerequisites(self):
        output = Path(self.suite["output_root"])
        write(output / "preflight.json", {
            "status": "PASS", "suite_sha256": runner.guard.digest(self.suite_path)})
        write(output / "smoke_status.json", {
            "status": "PASS", "suite_id": self.suite["suite_id"],
            "suite_sha256": runner.guard.digest(self.suite_path)})
        for arm in self.suite["arms"]:
            self.smoke_reports(arm)

    def test_unapproved_suite_rejected(self):
        self.suite["authorization"]["approved"] = False
        with self.assertRaisesRegex(runner.SuiteFailure, "AUTHORIZATION"):
            runner.validate_suite(self.suite)

    def test_missing_preflight_prevents_train(self):
        with self.assertRaises(runner.SuiteFailure):
            runner.validate_train_prerequisites(self.suite)

    def test_matching_16_initializations_permit_train(self):
        self.prerequisites()
        receipt = runner.validate_train_prerequisites(self.suite)
        self.assertEqual(receipt["model_sha256"], "a" * 64)
        self.assertEqual(receipt["rank_initialization_count"], 16)

    def test_one_arm_initialization_mismatch_prevents_train(self):
        self.prerequisites()
        self.smoke_reports("relplus", "b" * 64)
        with self.assertRaisesRegex(runner.SuiteFailure, "INITIALIZATION"):
            runner.validate_train_prerequisites(self.suite)

    def test_parameters_changed_after_smoke_prevent_training(self):
        self.prerequisites()
        self.suite["shared"]["lr"] = 0.0009
        write(self.suite_path, self.suite)
        write(Path(self.suite["output_root"]) / "preflight.json", {
            "status": "PASS", "suite_sha256": runner.guard.digest(self.suite_path)})
        with self.assertRaisesRegex(runner.SuiteFailure, "SMOKE_SUITE_FINGERPRINT"):
            runner.validate_train_prerequisites(self.suite, self.suite_path)

    def test_stale_preflight_suite_fingerprint_prevents_training(self):
        self.prerequisites()
        write(Path(self.suite["output_root"]) / "preflight.json", {
            "status": "PASS", "suite_sha256": "0" * 64})
        with self.assertRaisesRegex(runner.SuiteFailure, "PREFLIGHT_SUITE_FINGERPRINT"):
            runner.validate_train_prerequisites(self.suite, self.suite_path)

    def test_nan_replaced_rank_rejected_even_when_summary_passes(self):
        self.prerequisites()
        path = Path(self.suite["output_root"]) / "smoke/hha/rank_04.json"
        row = json.loads(path.read_text())
        row["author_nan_replacement_count"] = 1
        write(path, row)
        with self.assertRaisesRegex(runner.SuiteFailure, "NAN"):
            runner.validate_train_prerequisites(self.suite)

    def test_missing_finite_evidence_rejected(self):
        directory = self.smoke_reports("hha")
        path = directory / "rank_02.json"
        row = json.loads(path.read_text())
        del row["gradients_finite"]
        write(path, row)
        with self.assertRaisesRegex(runner.SuiteFailure, "gradients_finite"):
            runner.validate_smoke(directory)

    def test_legacy_launcher_uses_explicit_local_rank_contract(self):
        command = runner.build_command(self.suite, self.suite_path, "relplus", "train", 23456)
        self.assertIn("torch.distributed.launch", command)
        self.assertNotIn("--use_env", command)
        self.assertEqual(command[command.index("--mode") + 1], "formal")
        self.assertIn("--master_port=23456", command)
        self.assertEqual(command[command.index("--port") + 1], "23456")

    def test_existing_arm_directory_is_not_overwritten(self):
        directory = Path(self.suite["output_root"]) / "runs/hha"
        directory.mkdir(parents=True)
        sentinel = directory / "checkpoint.bin"
        sentinel.write_bytes(b"keep")
        with patch.object(runner.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(runner.SuiteFailure, "OUTPUT_EXISTS"):
                runner.run_arm(self.suite, self.suite_path, "hha", "train")
            popen.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_formal_completion_requires_all_ranks_and_checkpoint(self):
        directory = self.root / "formal"
        write(directory / "runtime_status.json", {
            "status": "FORMAL_TRAINING_COMPLETED", "epoch": 200,
            "global_iteration": 189000, "author_nan_replacement_count": 0,
        })
        for rank in range(8):
            write(directory / "rank_done_{:02d}.json".format(rank),
                  {"status": "COMPLETED", "rank": rank})
        with self.assertRaisesRegex(runner.SuiteFailure, "CHECKPOINT"):
            runner.validate_formal(directory, self.suite)
        checkpoint = directory / "checkpoints/epoch-200.pth"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint-content")
        receipt = runner.validate_formal(directory, self.suite)
        self.assertEqual(receipt["checkpoint_sha256"], runner.guard.digest(checkpoint))
        (directory / "rank_done_07.json").unlink()
        with self.assertRaises(runner.SuiteFailure):
            runner.validate_formal(directory, self.suite)

    def test_incorrect_final_iteration_rejected(self):
        directory = self.root / "wrong-endpoint"
        write(directory / "runtime_status.json", {
            "status": "FORMAL_TRAINING_COMPLETED", "epoch": 200,
            "global_iteration": 188999, "author_nan_replacement_count": 0,
        })
        with self.assertRaisesRegex(runner.SuiteFailure, "global_iteration"):
            runner.validate_formal(directory, self.suite)

    def test_progress_watchdog_is_based_on_progress_not_file_rewrites(self):
        directory = self.root / "stuck"
        status = {"status": "RUNNING", "global_iteration": 20,
                  "loss": 1.0, "learning_rate": 0.0001,
                  "author_nan_replacement_count": 0}
        write(directory / "runtime_status.json", status)
        state = {"last_progress": 0.0, "last_iteration": None}
        runner.check_health(directory, "train", state, 1.0, 0.0)
        self.assertEqual(state["last_progress"], 1.0)
        write(directory / "runtime_status.json", status)
        with self.assertRaisesRegex(runner.SuiteFailure, "STALLED"):
            runner.check_health(directory, "train", state, 1802.0, 0.0)

    def test_nan_runtime_and_rank_errors_stop_work(self):
        directory = self.root / "unhealthy"
        write(directory / "runtime_status.json", {
            "status": "RUNNING", "global_iteration": 1,
            "loss": float("nan"), "author_nan_replacement_count": 0})
        state = {"last_progress": 0.0, "last_iteration": None}
        with self.assertRaisesRegex(runner.SuiteFailure, "NONFINITE"):
            runner.check_health(directory, "train", state, 1.0, 0.0)
        write(directory / "rank_error_00.json", {"error": "worker failed"})
        with self.assertRaisesRegex(runner.SuiteFailure, "RANK_ERROR"):
            runner.check_health(directory, "train", state, 1.0, 0.0)

    def test_formal_initialization_mismatch_stops_as_soon_as_all_ranks_ready(self):
        directory = self.root / "init-mismatch"
        for rank in range(8):
            write(directory / "rank_init_{:02d}.json".format(rank),
                  {"rank": rank, "model_sha256": "b" * 64})
        state = {"last_progress": 0.0, "last_iteration": None,
                 "expected_model_sha256": "a" * 64}
        with self.assertRaisesRegex(runner.SuiteFailure, "FORMAL_SMOKE_INITIALIZATION_MISMATCH"):
            runner.check_health(directory, "train", state, 1.0, 0.0)

    def test_formal_initialization_is_checked_once_after_eighth_rank_is_ready(self):
        directory = self.root / "init-once"
        for rank in range(7):
            write(directory / "rank_init_{:02d}.json".format(rank),
                  {"rank": rank, "model_sha256": "a" * 64})
        state = {"last_progress": 0.0, "last_iteration": None,
                 "expected_model_sha256": "a" * 64}
        with patch.object(runner, "validate_initializations", wraps=runner.validate_initializations) as validate:
            runner.check_health(directory, "train", state, 1.0, 0.0)
            validate.assert_not_called()
            write(directory / "rank_init_07.json", {"rank": 7, "model_sha256": "a" * 64})
            runner.check_health(directory, "train", state, 2.0, 0.0)
            runner.check_health(directory, "train", state, 3.0, 0.0)
            validate.assert_called_once_with(directory)
        self.assertTrue(state["initialization_verified"])
        receipt = json.loads((directory / "initialization_check.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["model_sha256"], "a" * 64)

    def test_failed_arm_prevents_later_arms_and_phase_restart(self):
        calls = []
        def fake_arm(suite, suite_path, arm, phase):
            calls.append(arm)
            if arm == "relplus":
                raise runner.SuiteFailure("FAIL", "TEST_ARM_FAILURE")
            return {"status": "PASS", "arm": arm}
        with patch.object(runner, "run_arm", side_effect=fake_arm):
            result = runner.run_phase(self.suite, self.suite_path, "smoke")
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(calls, ["hha", "relplus"])
        with patch.object(runner, "run_arm") as launch:
            with self.assertRaisesRegex(runner.SuiteFailure, "PHASE_ALREADY"):
                runner.run_phase(self.suite, self.suite_path, "smoke")
            launch.assert_not_called()

    def test_foreign_process_failure_cleans_only_owned_group_and_records_exit(self):
        self.smoke_reports("hha")
        proc = Mock(pid=123456)
        proc.poll.return_value = None
        def stop_owned(process, identities):
            self.assertIs(process, proc)
            self.assertEqual(identities, {123456: "original-starttime"})
            proc.poll.return_value = -15
            return {"remaining_members": [], "signals": ["SIGTERM"]}
        with patch.object(runner.guard, "preflight"), \
             patch.object(runner.guard, "free_port", return_value=12345), \
             patch.object(runner.guard, "digest", return_value="c" * 64), \
             patch.object(runner.guard, "process_start", return_value="original-starttime"), \
             patch.object(runner.os, "getpgid", return_value=123456), \
             patch.object(runner.guard, "resources", return_value={"compute_processes": [{"pid": 98765}]}), \
             patch.object(runner.guard, "remember_group_members"), \
             patch.object(runner.guard, "check_running", side_effect=runner.guard.StopProbe(
                 "BLOCKED", "FOREIGN_GPU_PROCESS_APPEARED", {"pid": 98765})), \
             patch.object(runner.guard, "stop_own_group", side_effect=stop_owned) as cleanup, \
             patch.object(runner.subprocess, "Popen", return_value=proc) as popen:
            with self.assertRaisesRegex(runner.SuiteFailure, "FOREIGN_GPU_PROCESS"):
                runner.run_arm(self.suite, self.suite_path, "hha", "train")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        cleanup.assert_called_once()
        directory = Path(self.suite["output_root"]) / "runs/hha"
        self.assertEqual((directory / "exitcode").read_text().strip(), "-15")
        self.assertEqual(json.loads((directory / "arm_status.json").read_text())["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
