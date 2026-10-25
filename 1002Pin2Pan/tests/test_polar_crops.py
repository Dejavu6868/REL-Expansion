"""Polar crop generator and its CPU checks on a synthetic box room with a table."""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_polar_crops as check  # noqa: E402
import cross_projection as cp  # noqa: E402
import eval_pano_transfer as ev  # noqa: E402
import make_polar_crops as mpc  # noqa: E402

ROOM_MIN = np.array([-2.5, -2.0, -1.4])  # camera 1.4 m above the floor
ROOM_MAX = np.array([3.5, 4.0, 1.6])
TABLE_MIN = np.array([0.6, 1.2, -1.4])
TABLE_MAX = np.array([1.8, 2.2, -0.65])
CLASS = {name: index for index, name in enumerate(ev.EXPECTED_CLASSES)}
# Panorama frame relative to the dataset world: a yaw and a small tilt, as a real capture might have.
ERP_FROM_WORLD = check.rotation(37.0, 0.4, -0.3)
STEM = "camera_0123abcd_office_1_frame_equirectangular_domain_"


def trace(directions):
    """Range (m), stored class and BGR colour for world directions from the camera."""
    with np.errstate(divide="ignore", invalid="ignore"):
        bounds = np.where(directions > 0, ROOM_MAX, ROOM_MIN)
        hits = np.where(directions != 0, bounds / directions, np.inf)
        axis = hits.argmin(axis=-1)
        room = hits.min(axis=-1)
        t1, t2 = TABLE_MIN / directions, TABLE_MAX / directions
    near = np.nanmax(np.minimum(t1, t2), axis=-1)
    far = np.nanmin(np.maximum(t1, t2), axis=-1)
    table = (far >= near) & (near > 0) & (near < room)
    distance = np.where(table, near, room)
    label = np.full(distance.shape, CLASS["wall"], np.uint8)
    label[(axis == 2) & (directions[..., 2] < 0)] = CLASS["floor"]
    label[(axis == 2) & (directions[..., 2] > 0)] = CLASS["ceiling"]
    label[table] = CLASS["table"]
    point = directions * distance[..., None]
    checker = (np.floor(point * 2.0).astype(np.int64).sum(axis=-1) % 2).astype(np.float64)
    colour = 40.0 + 15.0 * label[..., None] + 80.0 * checker[..., None] * np.array([1.0, 0.6, 0.3])
    return distance, label, np.clip(colour, 0, 255).astype(np.uint8)


def write_panorama(root, height=512, width=1024, area="area_1"):
    world = cp.erp_directions(height, width) @ ERP_FROM_WORLD  # ERP-frame rays back to world
    distance, label, colour = trace(world)
    pano = root / area / "pano"
    for folder in ("rgb", "depth", "semantic"):
        (pano / folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pano / "rgb" / (STEM + "rgb.png")), colour)
    cv2.imwrite(str(pano / "depth" / (STEM + "depth.png")),
                np.clip(np.rint(distance * 512.0) - 1, 0, 65534).astype(np.uint16))
    semantic = np.zeros(label.shape + (3,), np.uint8)
    semantic[..., 0] = label  # instance i is class i in the label file below
    cv2.imwrite(str(pano / "semantic" / (STEM + "semantic.png")), semantic)


def write_semantic_labels(path):
    path.write_text(json.dumps(["{}_{}_x_{}".format(name, i, i) for i, name in enumerate(ev.EXPECTED_CLASSES)]))


@pytest.fixture(scope="module")
def frozen():
    return cp.import_frozen_source(cp.DEFAULT_SOURCE_ROOT)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("stanford")
    write_panorama(root)
    labels = root / "semantic_labels.json"
    write_semantic_labels(labels)
    return root, labels


@pytest.fixture(scope="module")
def crops(dataset, frozen):
    root, labels = dataset
    output = root / "crops"
    assert mpc.main(["--stanford-root", str(root), "--semantic-labels", str(labels), "--output-root", str(output),
                     "--areas", "area_1", "--total-crops", "6", "--workers", "1"]) == 0
    return output


def test_area_5_is_refused(dataset, tmp_path):
    root, labels = dataset
    with pytest.raises(SystemExit):
        mpc.main(["--stanford-root", str(root), "--semantic-labels", str(labels),
                  "--output-root", str(tmp_path / "out"), "--areas", "area_1", "area_5a"])


def test_crop_plan_is_balanced_steep_and_deterministic():
    panoramas = [{"sample_id": "area_1/p{}".format(i)} for i in range(5)]
    plans = mpc.crop_plan(panoramas, 23, 12345, 55.0, 90.0, (45.0, 75.0))
    assert [len(c) for _, c in plans] == [5, 5, 5, 4, 4]
    pitches = np.array([c[2] for _, crops in plans for c in crops])
    assert np.all(np.abs(pitches) >= 55.0) and np.all(np.abs(pitches) <= 90.0)

    assert abs(np.sum(pitches > 0) - np.sum(pitches < 0)) <= 3
    for _, crops in plans:  # each panorama alternates up and down
        assert np.all(np.diff(np.sign([c[2] for c in crops])) != 0)
    assert plans == mpc.crop_plan(panoramas, 23, 12345, 55.0, 90.0, (45.0, 75.0))


def test_straight_down_is_refused_by_the_frozen_generator_and_by_the_tool(frozen, dataset, tmp_path):
    from rel_plus.geometry import GravityAlignmentSingularity, align_points_and_normals_to_gravity

    straight_down = cp.crop_camera_to_world(0.0, -90.0).T @ np.array([0.0, 0.0, -1.0])
    with pytest.raises(GravityAlignmentSingularity):
        align_points_and_normals_to_gravity(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), straight_down)
    root, labels = dataset
    with pytest.raises(SystemExit):
        mpc.main(["--stanford-root", str(root), "--semantic-labels", str(labels),
                  "--output-root", str(tmp_path / "out"), "--areas", "area_1", "--max-pitch", "90"])


def test_outputs_follow_the_s2d_layout(crops):
    with (crops / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert (crops / "train.txt").read_text().split() == [r["sample_id"] for r in rows]
    for row in rows:
        assert row["sample_id"].startswith("area_1/camera_0123abcd_office_1_frame_10")
        assert (row["split"], row["protocol_id"]) == ("train", mpc.PROTOCOL_ID)
        assert cv2.imread(row["rgb_path"]).shape == (480, 480, 3)
        assert cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE).shape == (480, 480)
        depth = cv2.imread(row["depth_path"], cv2.IMREAD_UNCHANGED)
        assert depth.shape == (1080, 1080) and depth.dtype == np.uint16
    summary = json.loads((crops / "crops.json").read_text())["summary"]
    assert summary["up"]["crops"] == 3 and summary["down"]["crops"] == 3
    assert summary["up"]["class_pixel_share"]["ceiling"] > 0.3
    assert summary["down"]["class_pixel_share"]["floor"] + summary["down"]["class_pixel_share"]["table"] > 0.5


def test_crop_pixels_match_the_scene(crops):
    """Label, depth and RGB of each crop equal the scene seen through its own pose."""
    for crop in json.loads((crops / "crops.json").read_text())["crops"]:
        sample_id = crop["sample_id"]
        pose = json.loads((crops / "Pose" / (sample_id + ".json")).read_text())
        erp_from_camera = np.asarray(pose["camera_rt_matrix"])[:, :3].T
        world_from_camera = ERP_FROM_WORLD.T @ erp_from_camera
        k_small = cp.crop_intrinsics(480, crop["hfov_deg"])
        rays_camera, rays_world = cp.crop_rays(480, k_small, world_from_camera)
        distance, label, colour = trace(rays_world)
        stored = cv2.imread(str(crops / "Label" / (sample_id + ".png")), cv2.IMREAD_GRAYSCALE)
        assert np.mean(stored == label) > 0.97
        rgb = cv2.imread(str(crops / "RGB" / (sample_id + ".png")))
        assert np.median(np.abs(rgb.astype(int) - colour.astype(int))) <= 2
        depth = cv2.imread(str(crops / "Depth16" / (sample_id + ".png")), cv2.IMREAD_UNCHANGED)
        small = cv2.resize(depth, (480, 480), interpolation=cv2.INTER_NEAREST) / 512.0
        z = distance / np.linalg.norm(rays_camera, axis=2)
        assert np.median(np.abs(small - z) / z) < 0.01


def test_frozen_generator_reads_a_straight_down_crop(crops, frozen, tmp_path):
    """The frozen loader accepts the files and puts the floor 140 cm below the camera."""
    from rel_plus.profiles import STANFORD_S2D_PROFILE
    from rel_plus.stanford_s2d import load_canonical_frame

    rgb = np.zeros((480, 480, 3), np.uint8)
    label = np.zeros((480, 480), np.uint8)
    range_m = np.full((512, 1024), 0.0, np.float32)
    world = cp.erp_directions(512, 1024)
    room, _, _ = trace(world)  # ERP taken as level here
    range_m[:] = room
    camera_to_world = cp.crop_camera_to_world(30.0, -mpc.MAX_PITCH_DEG)
    _, _, depth, k_depth = mpc.render(rgb, label, range_m, camera_to_world, 60.0)
    cv2.imwrite(str(tmp_path / "d.png"), depth)
    (tmp_path / "p.json").write_text(json.dumps(mpc.pose_json(k_depth, camera_to_world, {})))
    raw, camera, _ = load_canonical_frame(str(tmp_path / "d.png"), str(tmp_path / "p.json"),
                                          dataset_profile=STANFORD_S2D_PROFILE)
    _, debug = frozen["generate_rel_plus_v2_1"](raw, camera, return_debug=True)
    valid = np.asarray(debug["encoding_valid_mask"], bool)
    floor = valid & (np.abs(debug["height_raw_cm"] + 140.0) < 5.0)
    assert floor.mean() > 0.5  # the table covers part of the view


def write_pinhole(root, frame, yaw, pitch):
    """A dataset pinhole rendered in world coordinates, in the S2D layout; returns its manifest row."""
    sample_id = "area_1/camera_0123abcd_office_1_frame_{}".format(frame)
    world_from_camera = cp.crop_camera_to_world(yaw, pitch)
    k_native = cp.crop_intrinsics(1080, 62.5)
    rays_camera, rays_world = cp.crop_rays(1080, k_native, world_from_camera)
    distance, _, _ = trace(rays_world)
    z = distance / np.linalg.norm(rays_camera, axis=2)
    _, small_world = cp.crop_rays(480, cp.crop_intrinsics(480, 62.5), world_from_camera)
    _, label, colour = trace(small_world)
    paths = {name: root / folder / (sample_id + ext) for name, folder, ext in (
        ("rgb_path", "RGB", ".png"), ("label_path", "Label", ".png"),
        ("depth_path", "Depth16", ".png"), ("camera_metadata_path", "Pose", ".json"))}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(paths["rgb_path"]), colour)
    cv2.imwrite(str(paths["label_path"]), label)
    cv2.imwrite(str(paths["depth_path"]), np.clip(np.rint(z * 512), 1, 65534).astype(np.uint16))
    paths["camera_metadata_path"].write_text(json.dumps(mpc.pose_json(k_native, world_from_camera, {})))
    return dict(sample_id=sample_id, split="train", area="area_1", room="office_1", camera="0123abcd",
                **{k: str(v) for k, v in paths.items()})


def run_reproduce(dataset, tmp_path, poses, examples):
    root, labels = dataset
    rows = [write_pinhole(tmp_path / "s2d", frame, yaw, pitch) for frame, (yaw, pitch) in enumerate(poses)]
    manifest = tmp_path / "full_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "reproduce.json"
    assert check.main(["reproduce", "--output", str(output), "--stanford-root", str(root), "--semantic-labels",
                       str(labels), "--manifest", str(manifest), "--workers", "1",
                       "--examples", str(examples)]) == 0
    return json.loads(output.read_text())


def test_reproduce_recovers_the_panorama_tilt(dataset, tmp_path):
    report = run_reproduce(dataset, tmp_path, ((10.0, 15.0), (200.0, -20.0)), examples=1)
    expected_tilt = np.degrees(np.arccos((ERP_FROM_WORLD @ [0, 0, 1])[2]))
    for frame in report["frames"]:
        assert frame["status"] == "PASS", frame
        assert abs(frame["tilt_deg"] - expected_tilt) < 0.15
        assert frame["fit"]["mirror"] == "none"
        assert frame["rgb_mean_abs_diff"] < 10
    assert report["decision"]["reproduce"] == "PASS"
    assert report["decision"]["level_assumption"] == "OK"  # 0.5 degrees is under the 1 degree bar
    assert len(list((tmp_path / "reproduce_examples").glob("*.png"))) == 1


def test_reproduce_fails_a_mislabelled_panorama(dataset, tmp_path, monkeypatch):
    shifted = ev.label_lookup(dataset[1]).copy()
    shifted[CLASS["wall"]] = CLASS["window"]  # the panorama's walls now disagree with the frame
    monkeypatch.setattr(ev, "label_lookup", lambda path: shifted)
    report = run_reproduce(dataset, tmp_path, ((90.0, 0.0),), examples=0)
    assert report["frames"][0]["status"] == "FAIL"
    assert report["decision"]["reproduce"] == "FAIL"


def test_geometry_keeps_the_failing_quantity(dataset, tmp_path):
    root, _ = dataset
    output = tmp_path / "geometry.json"
    assert check.main(["geometry", "--output", str(output), "--stanford-root", str(root), "--count", "1",
                       "--erp-size", "256", "512", "--areas", "area_1", "--workers", "1"]) == 0
    report = json.loads(output.read_text())
    summary = report["summary"]
    assert set(summary) == {"phase0", "polar"}
    for name in ("phase0", "polar"):
        failures = report["panoramas"][0][name]["failures"]
        counted = sum(summary[name]["failing_crops_by_quantity"].values())
        assert counted == len(failures)
        assert all(" median " in message for message in failures)


def test_contact_sheet(crops, tmp_path):
    output = tmp_path / "sheet.png"
    assert check.main(["sheet", "--output", str(output), "--crop-root", str(crops), "--count", "6"]) == 0
    assert cv2.imread(str(output)).shape == (2 * 240, 4 * 480, 3)


def test_frozen_cache_tool_encodes_the_crops_as_v3_1(crops, tmp_path):
    import subprocess

    import relplus_variant as rv

    pytest.importorskip("easydict")  # the frozen config module needs it

    output = tmp_path / "cache"
    subprocess.run(
        [sys.executable, str(Path(mpc.__file__).with_name("relplus_variant_cache.py")), "generate",
         "--manifest", str(crops / "manifest.csv"), "--output-root", str(output), "--workers", "2",
         "--authorize-full-cache", "--variant", "v3_1"],
        check=True, capture_output=True,
    )
    summary = json.loads((output / "cache_generation_summary.json").read_text())
    assert summary["status"] == "PASS" and summary["generated_or_resumed_count"] == 6
    assert rv.read_marker(output)["variant"] == "v3_1"
    assert (output / "train.txt").read_text() == (crops / "train.txt").read_text()
    for sample_id in (crops / "train.txt").read_text().split():
        rel = cv2.imread(str(output / "RELPlus" / (sample_id + ".png")), cv2.IMREAD_UNCHANGED)
        assert rel.shape == (480, 480, 3)
