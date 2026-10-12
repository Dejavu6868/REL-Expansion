"""check_egvia_rules.py: the rules themselves, and a run on the synthetic room."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_egvia_rules as er  # noqa: E402
import cross_projection as cp  # noqa: E402
import eval_pano_transfer as ev  # noqa: E402
from test_cross_projection_synthetic import (  # noqa: E402
    ROOM_MAX,
    ROOM_MIN,
    TABLE_MAX,
    TABLE_MIN,
    render_erp_raw_depth,
)

CLASSES = ev.EXPECTED_CLASSES[1:]
CEILING, FLOOR, TABLE, WALL = (CLASSES.index(name) for name in ("ceiling", "floor", "table", "wall"))
METRIC_RANGE = (-200.0, 200.0)


def sweep(height):
    angle = np.linspace(0.0, 255.0, 25501, dtype=np.float32)
    height = np.full_like(angle, height)
    return er.egvia_rules(angle, height, np.zeros_like(angle), METRIC_RANGE)


@pytest.mark.parametrize("height", [0.0, 128.0, 255.0])
def test_only_the_gated_rules_jump(height):
    steps = {rule: np.abs(np.diff(egvia)).max() for rule, egvia in sweep(height).items()}
    for rule in ("v3_0", "banded", "banded_metric", "v3_0_metric"):
        assert steps[rule] < 0.05, rule  # 0.0072 degrees per step
    if height != 0.0:  # v2_1 and paper_eq4 jump by (1 - lambda) |A - H| at 45 degrees
        assert steps["v2_1"] > 20 and steps["paper_eq4"] > 20


def test_banded_keeps_flat_orientations_apart():
    angle = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    for height in (0.0, 255.0):
        egvia = er.egvia_rules(angle, np.full(3, height, np.float32), np.zeros(3), METRIC_RANGE)["banded"]
        assert np.allclose(egvia, np.array([0.0, 85.0, 170.0]) + height / 3.0)


def test_metric_height_ignores_what_else_is_in_view():
    height_cm = np.array([-140.0, -65.0, 160.0])
    assert np.allclose(er.metric_height(height_cm, METRIC_RANGE), (height_cm + 200.0) * 255.0 / 400.0)
    assert er.metric_height(np.array([-500.0, 500.0]), METRIC_RANGE).tolist() == [0.0, 255.0]


@pytest.fixture(scope="module")
def room():
    raw = render_erp_raw_depth(256, 512)
    return raw, cp.decode_erp_depth(raw)


def test_rules_reproduce_the_frozen_erp_encoder(room):
    frozen = cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)
    er._JOB["processDepthImage_ERP"] = frozen["processDepthImage_ERP"]
    depth = room[1]
    angle, height, height_cm = er.erp_angle_and_height(depth)
    rules = er.egvia_rules(angle, height, height_cm, METRIC_RANGE)
    valid = depth != 0
    for rule, alpha in (("v2_1", 45), ("v3_0", -1)):
        assert np.array_equal(er.to_byte(rules[rule])[valid], frozen["getREL"](depth, alpha=alpha)[..., 0][valid])


def erp_labels(depth):
    points = cp.erp_directions(*depth.shape) * depth[..., None]
    label = np.full(depth.shape, WALL, dtype=np.uint8)
    label[np.isclose(points[..., 2], ROOM_MIN[2], atol=0.01)] = FLOOR
    label[np.isclose(points[..., 2], ROOM_MAX[2], atol=0.01)] = CEILING
    label[np.all((points >= TABLE_MIN - 0.01) & (points <= TABLE_MAX + 0.01), axis=2)] = TABLE
    return label


def write_frame(root, sample_id, depth, label, yaw, pitch, fov):
    k_json, camera_to_world, rows, columns, raw = cp.render_crop(depth, yaw, pitch, 1080, fov)
    cv2.imwrite(str(root / (sample_id + "_depth.png")), raw)
    (root / (sample_id + ".json")).write_text(json.dumps({
        "camera_k_matrix": k_json.tolist(),
        "camera_rt_matrix": np.hstack([camera_to_world.T, np.zeros((3, 1))]).tolist(),
        "camera_location": [0.0, 0.0, 0.0],
    }))
    stored = cv2.resize(label[rows, columns] + 1, (480, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(root / "Label" / (sample_id + ".png")), stored)
    return "{},train,{},{}\n".format(sample_id, root / (sample_id + "_depth.png"), root / (sample_id + ".json"))


def run_tool(tmp_path, raw, depth, frames):
    label = erp_labels(depth)
    (tmp_path / "Label").mkdir()
    manifest = "sample_id,split,depth_path,camera_metadata_path\n"
    for sample_id, yaw, pitch, fov in frames:
        manifest += write_frame(tmp_path, sample_id, depth, label, yaw, pitch, fov)
    (tmp_path / "manifest.csv").write_text(manifest)
    (tmp_path / "config.json").write_text(json.dumps(
        {"class_names": CLASSES, "gt_root_folder": str(tmp_path / "Label"), "gt_format": ".png"}
    ))
    names = ["<UNK>_0"] + [name + "_1" for name in ev.EXPECTED_CLASSES[1:]]
    (tmp_path / "semantic_labels.json").write_text(json.dumps(names))
    index = label.astype(np.uint32) + 1
    semantic = np.dstack([index % 256, index // 256 % 256, index // 65536]).astype(np.uint8)
    for area in er.TRAIN_AREAS:
        folder = tmp_path / "s2d" / area / "pano"
        for kind, image in (("rgb", np.zeros(semantic.shape, np.uint8)), ("depth", raw), ("semantic", semantic)):
            (folder / kind).mkdir(parents=True)
            cv2.imwrite(str(folder / kind / ("p_{}.png".format(kind))), image)
    subprocess.run(
        [sys.executable, str(TOOLS / "check_egvia_rules.py"), "--manifest", str(tmp_path / "manifest.csv"),
         "--config", str(tmp_path / "config.json"), "--stanford-root", str(tmp_path / "s2d"),
         "--semantic-labels", str(tmp_path / "semantic_labels.json"), "--output", str(tmp_path / "out"),
         "--height", "256", "--width", "512", "--workers", "2"],
        check=True, capture_output=True,
    )
    return json.loads((tmp_path / "out" / "egvia_rules.json").read_text())


def test_synthetic_room_run(tmp_path, room):
    report = run_tool(tmp_path, *room, [
        ("both", 35.0, -20.0, 62.48),  # table and floor in view
        ("table", 35.0, -14.0, 12.0),  # table top and wall, no floor: per-image height puts the table lowest
    ])
    assert report["pinhole_frames"] == 2 and report["panoramas"] == 5
    # Table labels cover the vertical table sides too, which share EGVIA with walls under every rule.
    for subset in ("pinhole", "panorama"):
        pairs = report["subsets"][subset]["egvia_overlap_key_pairs"]
        assert pairs["floor-table"]["v2_1"] > 3 * pairs["floor-table"]["banded_metric"]  # no height on horizontals
    pinhole = report["subsets"]["pinhole"]
    assert pinhole["egvia_overlap_key_pairs"]["table-wall"]["banded_metric"] < 0.6 * pinhole["egvia_overlap_key_pairs"]["table-wall"]["v3_0"]
    mean = pinhole["egvia_overlap_mean_over_class_pairs"]
    assert mean["banded_metric"] < mean["banded"] < mean["v3_0"] < mean["v2_1"]
    assert mean["v3_0_metric"] < mean["v3_0"]
    assert pinhole["share_within_5_deg_of_switch"]["floor"] > 0  # floor next to walls and table: the normal window straddles the corner
    heights = report["subsets"]["panorama"]["camera_relative_height_cm"]
    assert heights["floor"]["median"] == pytest.approx(ROOM_MIN[2] * 100, abs=5)
    assert heights["table"]["p95"] == pytest.approx(TABLE_MAX[2] * 100, abs=5)
    rows = (tmp_path / "out" / "egvia_overlap.csv").read_text().splitlines()
    assert len(rows) == 1 + 2 * len(er.RULES) * len(CLASSES)
