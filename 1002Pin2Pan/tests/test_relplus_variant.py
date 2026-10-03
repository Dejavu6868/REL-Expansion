"""REL+ height-everywhere variant: encoders, eval gate and the cache wrapper."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_projection as cp  # noqa: E402
import relplus_variant as rv  # noqa: E402
import relplus_variant_cache as rvc  # noqa: E402
from test_cross_projection_synthetic import (  # noqa: E402
    ROOM_MAX,
    ROOM_MIN,
    TABLE_MAX,
    render_erp_raw_depth,
)


@pytest.fixture(scope="module")
def frozen():
    return cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)


@pytest.fixture
def restore_v2_1(frozen):
    yield
    rv.apply_variant("v2_1")


@pytest.fixture(scope="module")
def room():
    depth = cp.decode_erp_depth(render_erp_raw_depth(512, 1024))
    height = cp.erp_directions(512, 1024)[..., 2] * depth
    return depth, height


def surfaces(height):
    return {
        name: np.isclose(height, value, atol=0.01)
        for name, value in (("floor", ROOM_MIN[2]), ("table", TABLE_MAX[2]), ("ceiling", ROOM_MAX[2]))
    }


def check_variant(v2_1, variant, masks):
    """Only EGVIA moves; table tops leave the floor value, floor and ceiling stay put."""
    assert np.array_equal(v2_1[..., 1:], variant[..., 1:])
    median = lambda image, name: np.median(image[..., 0][masks[name]])
    assert median(v2_1, "table") == pytest.approx(median(v2_1, "floor"), abs=3)
    assert median(variant, "table") - median(variant, "floor") >= 25
    assert median(variant, "floor") <= 3
    assert not masks["ceiling"].any() or median(variant, "ceiling") >= 250


def test_erp_getrel_variant_separates_table_tops(frozen, room, restore_v2_1):
    depth, height = room
    local = dict(frozen)
    rv.apply_variant("v2_1", local)
    v2_1 = local["getREL"](depth)
    assert np.array_equal(v2_1, frozen["getREL"](depth))
    rv.apply_variant("height_everywhere", local)
    check_variant(v2_1, local["getREL"](depth), surfaces(height))


def test_crop_generator_variant_separates_table_tops(frozen, room, restore_v2_1):
    depth, height = room
    k_json, camera_to_world, rows, columns, raw = cp.render_crop(depth, 60.0, -20.0, 240, 62.48)
    camera = frozen["CameraGeometry"].from_json_k(k_json, (240, 240), camera_to_world.T)
    v2_1 = frozen["generate_rel_plus_v2_1"](raw, camera)
    rv.apply_variant("height_everywhere")
    variant = frozen["generate_rel_plus_v2_1"](raw, camera)
    check_variant(v2_1, variant, surfaces(height[rows, columns]))


def eval_config(tmp_path, x_mode="rel_plus_v2_1", marker=None):
    cache_root = tmp_path / "cache"
    (cache_root / "RELPlus").mkdir(parents=True)
    if marker is not None:
        (cache_root / rv.MARKER_NAME).write_text(json.dumps({"variant": marker}), encoding="utf-8")
    return EasyDict(x_mode=x_mode, x_root_folder=str(cache_root / "RELPlus"))


def test_eval_variant_must_match_the_training_cache(frozen, tmp_path, restore_v2_1):
    assert rv.select_for_eval("v2_1", eval_config(tmp_path / "a"), dict(frozen)) == "v2_1"
    with pytest.raises(ValueError, match="trained on the v2_1 cache"):
        rv.select_for_eval("height_everywhere", eval_config(tmp_path / "b"), dict(frozen))
    marked = eval_config(tmp_path / "c", marker="height_everywhere")
    with pytest.raises(ValueError, match="trained on the height_everywhere cache"):
        rv.select_for_eval("v2_1", marked, dict(frozen))
    local = dict(frozen)
    assert rv.select_for_eval("height_everywhere", marked, local) == "height_everywhere"
    assert local["getREL"].keywords == {"alpha": -1.0}
    hha = EasyDict(x_mode="hha_frozen_cache", x_root_folder=str(tmp_path / "hha" / "HHA"))
    assert rv.select_for_eval("v2_1", hha, dict(frozen)) is None
    with pytest.raises(ValueError, match="REL\\+ arm only"):
        rv.select_for_eval("height_everywhere", hha, dict(frozen))


def test_output_root_keeps_one_variant(tmp_path):
    root = tmp_path / "cache"
    rvc.prepare_output_root(root, "height_everywhere", cp.DEFAULT_SOURCE_ROOT)
    assert rv.read_marker(root)["egvia_alpha"] == -1.0
    (root / "RELPlus").mkdir()
    rvc.prepare_output_root(root, "height_everywhere", cp.DEFAULT_SOURCE_ROOT)  # resume
    with pytest.raises(FileExistsError, match="height_everywhere variant"):
        rvc.prepare_output_root(root, "v2_1", cp.DEFAULT_SOURCE_ROOT)
    unmarked = tmp_path / "frozen_cache"
    (unmarked / "RELPlus").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="without"):
        rvc.prepare_output_root(unmarked, "height_everywhere", cp.DEFAULT_SOURCE_ROOT)


def write_frame(root, sample_id, depth, yaw, pitch):
    """A synthetic S2D frame: native 1080 Depth16 PNG plus a pose JSON at the origin."""
    k_json, camera_to_world, _, _, raw = cp.render_crop(depth, yaw, pitch, 1080, 62.48)
    depth_path = root / (sample_id + "_depth.png")
    pose_path = root / (sample_id + ".json")
    cv2.imwrite(str(depth_path), raw)
    rt = np.hstack([camera_to_world.T, np.zeros((3, 1))])
    pose_path.write_text(json.dumps({
        "camera_k_matrix": k_json.tolist(),
        "camera_rt_matrix": rt.tolist(),
        "camera_location": [0.0, 0.0, 0.0],
    }), encoding="utf-8")
    return {
        "sample_id": sample_id,
        "protocol_id": "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT",
        "split": "train",
        "depth_path": str(depth_path),
        "camera_metadata_path": str(pose_path),
    }


def test_cache_workers_write_the_variant(frozen, room, tmp_path, restore_v2_1):
    """The frozen cache tool, run through the wrapper with two worker processes."""
    from rel_plus.profiles import STANFORD_S2D_PROFILE
    from rel_plus.stanford_s2d import load_canonical_frame

    rows = [write_frame(tmp_path, "frame{}".format(i), room[0], 60.0 * i, -20.0) for i in range(2)]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "cache"
    subprocess.run(
        [sys.executable, str(TOOLS / "relplus_variant_cache.py"), "generate",
         "--manifest", str(manifest), "--output-root", str(output), "--workers", "2", "--limit", "2"],
        check=True, capture_output=True,
    )
    assert rv.read_marker(output)["variant"] == "height_everywhere"
    for row in rows:
        raw, camera, _ = load_canonical_frame(
            row["depth_path"], row["camera_metadata_path"], dataset_profile=STANFORD_S2D_PROFILE
        )
        rv.apply_variant("v2_1")
        v2_1 = frozen["generate_rel_plus_v2_1"](raw, camera)
        rv.apply_variant("height_everywhere")
        expected = frozen["generate_rel_plus_v2_1"](raw, camera)
        cached = cv2.imread(str(output / "RELPlus" / (row["sample_id"] + ".png")), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(cached, expected)
        assert not np.array_equal(cached[..., 0], v2_1[..., 0])
