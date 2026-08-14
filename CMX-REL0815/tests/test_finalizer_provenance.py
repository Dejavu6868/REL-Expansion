import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.finalize_run import (
    validate_code_manifest,
    validate_initialization_report,
    validate_resume_evidence,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FinalizerProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.repo = root / "repo"
        self.run = root / "run"
        self.repo.mkdir()
        self.run.mkdir()
        (self.run / "status").mkdir()
        (self.repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.run / "code_manifest.sha256").write_text(
            "{}  ./module.py\n".format(_sha(self.repo / "module.py")),
            encoding="utf-8",
        )
        self.checkpoint = root / "mit_b2.pth"
        self.checkpoint.write_bytes(b"pretrained")
        self.config = {
            "pretrained_model": str(self.checkpoint),
            "pretrained_sha256": _sha(self.checkpoint),
        }
        base_keys = ["block1.{}.weight".format(index) for index in range(332)]
        loaded_keys = base_keys + [
            key.replace("block", "extra_block") for key in base_keys
        ]
        missing_keys = ["FRMs.{}.weight".format(index) for index in range(32)] + [
            "FFMs.{}.weight".format(index) for index in range(116)
        ]
        self.initialization = {
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": _sha(self.checkpoint),
            "loading_module": (
                "dual MiT backbone: identical weights copied to RGB and REL+ encoders"
            ),
            "loaded_tensor_count": 664,
            "loaded_parameter_count": 48392576,
            "model_state_parameter_count": 64988560,
            "loaded_parameter_ratio": 48392576 / 64988560.0,
            "loaded_keys": loaded_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": [],
            "checkpoint_unmapped_keys": ["head.bias", "head.weight"],
            "shape_mismatch": [],
            "strict": False,
            "decoder_and_fusion_initialization": (
                "unchanged CMX defaults; decoder Kaiming, fusion module constructors"
            ),
        }
        (self.run / "initialization_report.json").write_text(
            json.dumps(self.initialization), encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_code_manifest_passes_then_detects_content_and_file_set_changes(self):
        self.assertEqual(
            validate_code_manifest(self.run, self.repo)["status"], "passed"
        )
        (self.repo / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content changed"):
            validate_code_manifest(self.run, self.repo)
        (self.repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.repo / "added.py").write_text("VALUE = 3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file set changed"):
            validate_code_manifest(self.run, self.repo)

    def test_initialization_must_match_rel_default_backbone_evidence(self):
        evidence = validate_initialization_report(self.run, self.config)
        self.assertEqual(evidence["report"]["loaded_tensor_count"], 664)
        self.initialization["checkpoint_sha256"] = "0" * 64
        (self.run / "initialization_report.json").write_text(
            json.dumps(self.initialization), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_initialization_report(self.run, self.config)
        self.initialization["checkpoint_sha256"] = self.config["pretrained_sha256"]
        (self.run / "initialization_report.json").write_text(
            json.dumps(self.initialization), encoding="utf-8"
        )
        self.checkpoint.write_bytes(b"mutated-pretrained")
        with self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"):
            validate_initialization_report(self.run, self.config)

    def test_resume_evidence_is_bound_to_current_run_checkpoint(self):
        fresh_command = {"argv": [str(index) for index in range(11)]}
        self.assertEqual(
            validate_resume_evidence(self.run, fresh_command), {"resumed": False}
        )

        checkpoints = self.run / "checkpoints"
        checkpoints.mkdir()
        checkpoint = checkpoints / "epoch-4.pth"
        checkpoint.write_bytes(b"resume")
        digest = _sha(checkpoint)
        report = {
            "status": "passed",
            "exit_code": 0,
            "checkpoint": str(checkpoint),
            "sha256": digest,
            "epoch": 4,
            "iteration": 4408,
            "model_entry_count": 1,
            "optimizer_keys": ["param_groups", "state"],
        }
        (self.run / "status" / "resume_checkpoint_validation.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (self.run / "status" / "resume_checkpoint.sha256").write_text(
            "{}  {}\n".format(digest, checkpoint), encoding="utf-8"
        )
        resume_command = {
            "argv": [str(index) for index in range(11)] + ["-c", str(checkpoint)]
        }
        evidence = validate_resume_evidence(self.run, resume_command)
        self.assertTrue(evidence["resumed"])
        self.assertEqual(evidence["epoch"], 4)

        report["epoch"] = 8
        (self.run / "status" / "resume_checkpoint_validation.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "filename mismatch"):
            validate_resume_evidence(self.run, resume_command)


if __name__ == "__main__":
    unittest.main()
