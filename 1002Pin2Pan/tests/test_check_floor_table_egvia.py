"""check_floor_table_egvia.py on a synthetic room: one table, floor at -1.4 m, camera at the origin."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_floor_table_egvia as ft  # noqa: E402
import cross_projection as cp  # noqa: E402
import eval_pano_transfer as ev  # noqa: E402
from test_cross_projection_synthetic import ROOM_MIN, TABLE_MAX, TABLE_MIN, render_erp_raw_depth  # noqa: E402

CLASSES = ev.EXPECTED_CLASSES[1:]
FLOOR, TABLE, WALL = (CLASSES.index(name) for name in ("floor", "table", "wall"))


def erp_labels(depth):
    points = cp.erp_directions(*depth.shape) * depth[..., None]
    label = np.full(depth.shape, WALL, dtype=np.uint8)
    label[np.isclose(points[..., 2], ROOM_MIN[2], atol=0.01)] = FLOOR
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


def test_v3_0_separates_table_tops_from_floor(tmp_path):
    raw = render_erp_raw_depth(512, 1024)
    depth = cp.decode_erp_depth(raw)
    label = erp_labels(depth)
    (tmp_path / "Label").mkdir()
    manifest = "sample_id,split,depth_path,camera_metadata_path\n"
    manifest += write_frame(tmp_path, "both", depth, label, 35.0, -20.0, 62.48)  # table and floor in view
    manifest += write_frame(tmp_path, "floor", depth, label, 215.0, -20.0, 62.48)  # no table
    manifest += write_frame(tmp_path, "table", depth, label, 35.0, -14.0, 12.0)  # table top and wall, no floor
    (tmp_path / "manifest.csv").write_text(manifest)
    (tmp_path / "config.json").write_text(json.dumps(
        {"class_names": CLASSES, "gt_root_folder": str(tmp_path / "Label"), "gt_format": ".png"}
    ))
    names = ["<UNK>_0"] + [name + "_1" for name in ev.EXPECTED_CLASSES[1:]]
    (tmp_path / "semantic_labels.json").write_text(json.dumps(names))
    index = label.astype(np.uint32) + 1
    semantic = np.dstack([index % 256, index // 256 % 256, index // 65536]).astype(np.uint8)
    for area in ft.TRAIN_AREAS:
        folder = tmp_path / "s2d" / area / "pano"
        for kind, image in (("rgb", np.zeros(semantic.shape, np.uint8)), ("depth", raw), ("semantic", semantic)):
            (folder / kind).mkdir(parents=True)
            cv2.imwrite(str(folder / kind / ("p_{}.png".format(kind))), image)

    subprocess.run(
        [sys.executable, str(TOOLS / "check_floor_table_egvia.py"), "--manifest", str(tmp_path / "manifest.csv"),
         "--config", str(tmp_path / "config.json"), "--stanford-root", str(tmp_path / "s2d"),
         "--semantic-labels", str(tmp_path / "semantic_labels.json"), "--output", str(tmp_path / "out"),
         "--height", "256", "--width", "512", "--workers", "2"],
        check=True, capture_output=True,
    )
    report = json.loads((tmp_path / "out" / "floor_table_egvia.json").read_text())
    assert report["pinhole_frames"] == 3 and report["panoramas"] == 5
    assert report["pinhole_table_frames_with_floor_in_view"] == 1
    assert report["pinhole_table_frames_without_floor_in_view"] == 1
    for subset in ("pinhole_floor_in_view", "panorama"):
        v2_1, v3_0 = (report["subsets"][subset][variant] for variant in ("v2_1", "v3_0"))
        assert v2_1["table_above_floor"] < 0.8  # v2.1: edge pixels only
        assert v3_0["table_above_floor"] > 0.98
        assert v3_0["table_top_median"] - v3_0["floor_median"] >= 20
    assert report["pinhole_v3_0_table_minus_floor_median_gap"]["median"] >= 20
    bottom = report["subsets"]["panorama_bottom_band"]["v3_0"]
    assert bottom["floor_pixels"] > 0 and bottom["table_top_pixels"] == 0  # the table is above -60 degrees
    rows = (tmp_path / "out" / "floor_table_egvia_histograms.csv").read_text().splitlines()
    assert rows[0].split(",")[:4] == ["subset", "variant", "surface", "egvia_0"] and len(rows) == 1 + 5 * 4
