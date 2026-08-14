import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


class _FakeImage:
    shape = (480, 480, 3)
    dtype = "uint8"


fake_cv2 = types.SimpleNamespace(
    IMREAD_UNCHANGED=-1,
    imread=lambda path, mode: _FakeImage() if Path(path).read_bytes().startswith(b"PNG") else None,
    setNumThreads=lambda count: None,
)
old_cv2 = sys.modules.get("cv2")
sys.modules["cv2"] = fake_cv2
spec = importlib.util.spec_from_file_location(
    "cache_validator_under_test",
    Path(__file__).parents[1] / "scripts" / "validate_relplus_cache.py",
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
if old_cv2 is None:
    del sys.modules["cv2"]
else:
    sys.modules["cv2"] = old_cv2


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CacheValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.dataset = root / "dataset"
        self.run = root / "run"
        (self.run / "data_reports").mkdir(parents=True)
        (self.run / "relplus_cache" / "area_1").mkdir(parents=True)
        (self.run / "relplus_cache" / "area_5").mkdir(parents=True)
        self.dataset.mkdir()
        self.train = self.dataset / "train.txt"
        self.test = self.dataset / "test.txt"
        self.train.write_text("area_1/a\n", encoding="utf-8")
        self.test.write_text("area_5/b\n", encoding="utf-8")
        representation_spec = {
            "model_name": "cmx_rel+",
            "config_name": "cmx_rel+",
            "representation_semantics": "REL-default",
            "representation_version": "relplus_rel_default_v2",
            "point_frame": "camera_centered_world_axes",
            "translation_in_red_loa": False,
            "channel_order": ["ReD", "EGVIA", "LOA"],
            "depth_definition": "camera-z; metres=(uint16+1)/512; 65535 invalid",
            "pixel_origin": 1,
            "normal_estimator": "REL square-support algebraic plane fit",
            "normal_radius_native_pixels": 3,
            "alpha_degrees": 45.0,
            "lambda": 0.5,
            "red_height_normalization": "valid-image min-max to uint8",
            "angle_normalization": "zero-to-pi linearly mapped to uint8",
            "invalid_pixel": [255, 255, 255],
            "native_depth_shape": [1080, 1080],
            "cache_shape": [480, 480, 3],
            "cache_resize": "bilinear channels; nearest validity mask",
            "output_dtype": "uint8",
            "intrinsics_usage": "K backprojects camera-z depth into camera coordinates",
            "extrinsics_usage": (
                "R rotates points and normals into world axes; t/C are validated and retained "
                "for provenance but do not enter ReD, EGVIA, or LOA"
            ),
        }
        spec_path = self.run / "data_reports" / "relplus_representation_spec.json"
        spec_path.write_text(
            json.dumps(representation_spec, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        spec_sha256 = _sha(spec_path)
        generator_sha256 = _sha(
            Path(__file__).parents[1] / "scripts" / "prepare_relplus.py"
        )
        generator_files_sha256 = {
            relative: _sha(Path(__file__).parents[1] / relative)
            for relative in (
                "relplus/__init__.py",
                "relplus/geometry.py",
                "relplus/io.py",
                "relplus/representation.py",
                "relplus/spec.py",
                "scripts/prepare_relplus.py",
            )
        }
        generator_bundle_sha256 = hashlib.sha256(
            (
                json.dumps(
                    generator_files_sha256, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        first = self.run / "relplus_cache" / "area_1" / "a.png"
        second = self.run / "relplus_cache" / "area_5" / "b.png"
        first.write_bytes(b"PNG-first")
        second.write_bytes(b"PNG-second")
        audit = {
            "dataset_root": str(self.dataset),
            "train_count": 1,
            "test_count": 1,
            "overlap_count": 0,
            "train_sha256": _sha(self.train),
            "test_sha256": _sha(self.test),
            "representation_semantics": "REL-default",
            "representation_version": "relplus_rel_default_v2",
            "point_frame": "camera_centered_world_axes",
            "translation_in_red_loa": False,
            "normal_radius_native_pixels": 3,
            "alpha_degrees": 45.0,
            "lambda": 0.5,
            "representation_spec_sha256": spec_sha256,
            "representation_generator_sha256": generator_sha256,
            "representation_generator_files_sha256": generator_files_sha256,
            "representation_generator_bundle_sha256": generator_bundle_sha256,
        }
        (self.run / "data_reports" / "data_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        rows = [
            {
                "sample_id": "area_1/a",
                "status": "generated",
                "sha256": _sha(first),
                "valid_pixels": 230400,
                "invalid_pixels": 0,
                "representation_version": "relplus_rel_default_v2",
                "representation_spec_sha256": spec_sha256,
                "representation_generator_sha256": generator_sha256,
                "representation_generator_bundle_sha256": generator_bundle_sha256,
            },
            {
                "sample_id": "area_5/b",
                "status": "generated",
                "sha256": _sha(second),
                "valid_pixels": 230400,
                "invalid_pixels": 0,
                "representation_version": "relplus_rel_default_v2",
                "representation_spec_sha256": spec_sha256,
                "representation_generator_sha256": generator_sha256,
                "representation_generator_bundle_sha256": generator_bundle_sha256,
            },
        ]
        (self.run / "data_reports" / "cache_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        statistics = {
            "sample_count": 2,
            "generated_count": 2,
            "skipped_count": 0,
            "invalid_pixels": 0,
            "invalid_ratio": 0.0,
            "channel_order": ["ReD", "EGVIA", "LOA"],
            "representation_semantics": "REL-default",
            "representation_version": "relplus_rel_default_v2",
            "point_frame": "camera_centered_world_axes",
            "translation_in_red_loa": False,
            "normal_radius_native_pixels": 3,
            "alpha_degrees": 45.0,
            "lambda": 0.5,
            "representation_spec_sha256": spec_sha256,
            "representation_generator_sha256": generator_sha256,
            "representation_generator_files_sha256": generator_files_sha256,
            "representation_generator_bundle_sha256": generator_bundle_sha256,
            "ReD": {"valid_pixels": 460800},
            "EGVIA": {"valid_pixels": 460800},
            "LOA": {"valid_pixels": 460800},
        }
        (self.run / "data_reports" / "relplus_statistics.json").write_text(
            json.dumps(statistics), encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _main(self):
        with mock.patch.object(
            sys, "argv", ["validate_relplus_cache.py", "--run-dir", str(self.run), "--workers", "2", "--progress-every", "0"]
        ):
            return validator.main()

    def test_complete_cache_passes_and_writes_atomic_evidence(self):
        self.assertEqual(self._main(), 0)
        report = json.loads(
            (self.run / "data_reports" / "cache_validation.json").read_text()
        )
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["manifest_count"], 2)
        self.assertEqual(report["png_count"], 2)
        self.assertEqual(report["validated_files"], 2)
        self.assertTrue(report["all_sha256_match"])
        self.assertTrue(report["all_decodable"])
        self.assertTrue(report["representation_semantics_valid"])
        self.assertEqual(
            (self.run / "status" / "cache_validation.exitcode").read_text(), "0\n"
        )

    def test_sha_mismatch_fails_closed_with_nonzero_status(self):
        (self.run / "relplus_cache" / "area_1" / "a.png").write_bytes(b"PNG-mutated")
        self.assertEqual(self._main(), 1)
        report = json.loads(
            (self.run / "data_reports" / "cache_validation.json").read_text()
        )
        self.assertFalse(report["all_sha256_match"])
        self.assertEqual(report["validated_files"], 1)
        self.assertEqual(
            (self.run / "status" / "cache_validation.exitcode").read_text(), "1\n"
        )

    def test_absolute_world_semantics_fail_closed(self):
        statistics_path = self.run / "data_reports" / "relplus_statistics.json"
        statistics = json.loads(statistics_path.read_text())
        statistics["representation_version"] = "relplus_absolute_world_v1"
        statistics["point_frame"] = "absolute_world"
        statistics["translation_in_red_loa"] = True
        statistics_path.write_text(json.dumps(statistics), encoding="utf-8")

        self.assertEqual(self._main(), 1)
        report = json.loads(
            (self.run / "data_reports" / "cache_validation.json").read_text()
        )
        self.assertFalse(report["representation_semantics_valid"])
        self.assertEqual(
            (self.run / "status" / "cache_validation.exitcode").read_text(), "1\n"
        )

    def test_representation_math_hash_mismatch_fails_closed(self):
        statistics_path = self.run / "data_reports" / "relplus_statistics.json"
        statistics = json.loads(statistics_path.read_text())
        statistics["representation_generator_bundle_sha256"] = "0" * 64
        statistics_path.write_text(json.dumps(statistics), encoding="utf-8")

        self.assertEqual(self._main(), 1)
        report = json.loads(
            (self.run / "data_reports" / "cache_validation.json").read_text()
        )
        self.assertFalse(report["representation_generator_valid"])

    def test_descendant_cache_symlink_fails_closed(self):
        cached = self.run / "relplus_cache" / "area_1" / "a.png"
        external = self.run / "external.png"
        cached.replace(external)
        os.symlink(external, cached)

        self.assertEqual(self._main(), 1)
        report = json.loads(
            (self.run / "data_reports" / "cache_validation.json").read_text()
        )
        self.assertFalse(report["cache_tree_symlink_free"])


if __name__ == "__main__":
    unittest.main()
