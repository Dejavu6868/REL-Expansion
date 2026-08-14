import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import SourceAlignedRELTrainPre
from rel_source_aligned.adapters.stanford2d3d_perspective_adapter import (
    PerspectiveInputAdapter,
)
from rel_source_aligned.cmx.rel_dataset_adapter import apply_shared_spatial_transform
from rel_source_aligned.reference.official_rel_core import (
    OFFICIAL_CHANNEL_ORDER,
    SourceExactRELCore,
    erp_azimuth,
)
from rel_source_aligned.reference.reference_loader import load_official_rel_module
from tools.rel_source_aligned_preflight import run_one_batch_preflight


AUTHORITY_ROOT = Path(
    os.environ.get(
        "REL_AUTHORITY_ROOT",
        "/home/zhuzhaoziao/rel_exp/"
        "REL-SF4PASS_authority_16c1267608171d67b34ecc3d0190920a06f1017e",
    )
)


def _erp_depth(height=24, width=48):
    rows, columns = np.indices((height, width), dtype=np.float32)
    return 1.5 + 0.003 * rows + 0.002 * columns


def _perspective_fixture(height=32, width=40):
    rows, columns = np.indices((height, width), dtype=np.float32)
    raw_depth = np.rint((1.5 + rows * 0.002 + columns * 0.003) * 512.0).astype(
        np.uint16
    )
    raw_depth[0, 0] = 0
    raw_depth[-1, -1] = 65535
    k = np.array(
        [[36.0, 0.0, width / 2.0], [0.0, 36.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return raw_depth, k


def _write_tiny_dataset(root):
    height, width = 32, 40
    raw_depth, k = _perspective_fixture(height, width)
    rows, columns = np.indices((height, width), dtype=np.uint8)
    rgb = np.stack([columns, rows, (rows + columns) % 255], axis=2)
    label = ((rows.astype(np.uint16) + columns.astype(np.uint16)) % 13 + 1).astype(
        np.uint8
    )
    for directory in ("RGB", "Label", "Depth16", "Pose"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "RGB" / "sample.png"), rgb)
    cv2.imwrite(str(root / "Label" / "sample.png"), label)
    cv2.imwrite(str(root / "Depth16" / "sample.png"), raw_depth)
    (root / "Pose" / "sample.json").write_text(
        json.dumps({"camera_k_matrix": k.tolist()}), encoding="utf-8"
    )
    (root / "train.txt").write_text("sample\n", encoding="utf-8")
    (root / "test.txt").write_text("sample\n", encoding="utf-8")
    return raw_depth, k


def _tiny_setting(root):
    return {
        "rgb_root": str(root / "RGB"),
        "rgb_format": ".png",
        "gt_root": str(root / "Label"),
        "gt_format": ".png",
        "transform_gt": True,
        "x_root": str(root / "unused"),
        "x_format": ".png",
        "x_single_channel": False,
        "x_mode": "rel_source_aligned",
        "rel_impl": "official_source",
        "depth_root": str(root / "Depth16"),
        "depth_format": ".png",
        "pose_root": str(root / "Pose"),
        "pose_format": ".json",
        "train_source": str(root / "train.txt"),
        "eval_source": str(root / "test.txt"),
        "class_names": [str(index) for index in range(13)],
    }


class SourceAlignedRELTest(unittest.TestCase):
    def test_reference_rel_runs(self):
        reference = load_official_rel_module(AUTHORITY_ROOT)
        output = reference.getREL(_erp_depth())
        self.assertEqual(output.shape, (24, 48, 3))
        self.assertEqual(output.dtype, np.uint8)
        self.assertGreaterEqual(int(output.min()), 0)
        self.assertLessEqual(int(output.max()), 255)

    def test_reference_and_new_rel_match(self):
        depth = _erp_depth()
        missing = depth == 0
        reference = load_official_rel_module(AUTHORITY_ROOT)
        expected = reference.getREL(depth.copy())
        points, normals, _ = reference.processDepthImage_ERP(depth * 100, missing)
        actual = SourceExactRELCore().encode(
            points, normals, erp_azimuth(depth.shape), missing
        ).rel
        np.testing.assert_array_equal(actual, expected)

    def test_channel_order_and_range(self):
        depth = _erp_depth(12, 24)
        reference = load_official_rel_module(AUTHORITY_ROOT)
        points, normals, _ = reference.processDepthImage_ERP(
            depth * 100, np.zeros_like(depth, dtype=bool)
        )
        encoded = SourceExactRELCore().encode(
            points,
            normals,
            erp_azimuth(depth.shape),
            np.zeros_like(depth, dtype=bool),
        )
        self.assertEqual(encoded.channel_order, OFFICIAL_CHANNEL_ORDER)
        self.assertEqual(
            encoded.channel_order,
            ("EGVIA_source_code", "LOA_source_code", "ReD_source_code"),
        )
        self.assertEqual(encoded.rel.dtype, np.uint8)
        self.assertTrue(np.isfinite(encoded.rel).all())
        self.assertGreaterEqual(int(encoded.rel.min()), 0)
        self.assertLessEqual(int(encoded.rel.max()), 255)

    def test_invalid_mask(self):
        rows, columns = np.indices((5, 7), dtype=np.float32)
        points = np.stack([columns + 1, rows + 2, rows - 2], axis=2)
        normals = np.zeros_like(points)
        normals[:, :, 2] = -1
        missing = np.zeros((5, 7), dtype=bool)
        missing[2, 3] = True
        encoded = SourceExactRELCore().encode(
            points, normals, np.zeros((5, 7), dtype=np.float64), missing
        )
        np.testing.assert_array_equal(encoded.rel[2, 3], [255, 255, 255])

    def test_rel_generated_before_random_augmentation(self):
        raw_depth, k = _perspective_fixture()
        adapter = PerspectiveInputAdapter(AUTHORITY_ROOT)
        preprocess = SourceAlignedRELTrainPre(
            np.array([0.485, 0.456, 0.406]),
            np.array([0.229, 0.224, 0.225]),
            adapter=adapter,
            target_size=(20, 24),
            scale_array=[0.75],
            horizontal_flip=False,
        )
        rgb = np.zeros((32, 40, 3), dtype=np.uint8)
        label = np.zeros((32, 40), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            pose = Path(directory) / "pose.json"
            pose.write_text(json.dumps({"camera_k_matrix": k.tolist()}), encoding="utf-8")
            preprocess.apply_with_parameters(
                rgb, label, raw_depth, str(pose), scale=0.75, crop_pos=(1, 2)
            )
        self.assertEqual(
            preprocess.last_stage_order,
            ("source_aligned_rel_generation", "shared_spatial_transform", "normalization"),
        )
        self.assertEqual(preprocess.last_generation_shape, raw_depth.shape)

    def test_rgb_rel_label_share_spatial_transform(self):
        grid = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
        rgb = np.repeat(grid[:, :, None], 3, axis=2)
        rel = rgb.copy()
        transformed_rgb, transformed_label, transformed_rel = apply_shared_spatial_transform(
            rgb,
            grid,
            rel,
            scale=1.0,
            crop_size=(4, 5),
            crop_pos=(1, 2),
            mirror=True,
        )
        np.testing.assert_array_equal(transformed_rgb[:, :, 0], transformed_label)
        np.testing.assert_array_equal(transformed_rel[:, :, 0], transformed_label)

    def test_new_cmx_mode_uses_source_aligned_rel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tiny_dataset(root)
            preprocess = SourceAlignedRELTrainPre(
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]),
                adapter=PerspectiveInputAdapter(AUTHORITY_ROOT),
                target_size=(24, 24),
                scale_array=[1.0],
                horizontal_flip=False,
            )
            sample = RGBXDataset(_tiny_setting(root), "train", preprocess)[0]
        self.assertEqual(sample["modal_x"].shape, (3, 24, 24))
        self.assertEqual(preprocess.last_impl, "official_source")

    def test_legacy_rel_is_not_called_in_new_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tiny_dataset(root)
            preprocess = SourceAlignedRELTrainPre(
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]),
                adapter=PerspectiveInputAdapter(AUTHORITY_ROOT),
                target_size=(24, 24),
                scale_array=[1.0],
                horizontal_flip=False,
            )
            with mock.patch(
                "dataloader.dataloader.generate_relplus_from_depth",
                side_effect=AssertionError("legacy pose REL was called"),
            ), mock.patch(
                "dataloader.dataloader.generate_relplus_from_depth_local",
                side_effect=AssertionError("legacy local REL was called"),
            ):
                sample = RGBXDataset(_tiny_setting(root), "train", preprocess)[0]
        self.assertEqual(sample["modal_x"].shape, (3, 24, 24))

    def test_source_aligned_training_is_blocked_pending_user_approval(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment["CMX_RUN_DIR"] = directory
            completed = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "tools" / "run_with_config.py"),
                    "--config",
                    "configs.rel_source_aligned",
                    "train.py",
                ],
                cwd=str(repo_root),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("training_authorized=false", completed.stdout)

    @unittest.skipUnless(
        os.environ.get("RUN_CMX_ONE_BATCH") == "1",
        "set RUN_CMX_ONE_BATCH=1 for the bounded real-data/GPU preflight",
    )
    def test_one_batch_forward(self):
        report = run_one_batch_preflight(
            repo_root=Path(__file__).resolve().parents[1],
            authority_root=AUTHORITY_ROOT,
            dataset_root=Path(
                os.environ.get(
                    "CMX_DATASET_ROOT", "/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480"
                )
            ),
            device=os.environ.get("CMX_PREFLIGHT_DEVICE", "cuda:0"),
        )
        self.assertTrue(report["dataset_item_pass"])
        self.assertTrue(report["collation_pass"])
        self.assertTrue(report["forward_pass"])
        self.assertTrue(report["loss_finite"])
        self.assertTrue(report["x_input_changed"])
        self.assertTrue(report["logits_changed"])
        self.assertTrue(report["rgb_input_unchanged"])
        self.assertFalse(report["backward_called"])
        self.assertFalse(report["optimizer_step_called"])
        self.assertFalse(report["checkpoint_written"])


if __name__ == "__main__":
    unittest.main()
