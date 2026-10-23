"""Training on a view subset: the subset lists, the config the run trains with, and a CPU run."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_view_subset as mv  # noqa: E402
import train_view_subset as tv  # noqa: E402
from test_audit_view_coverage import write_pose  # noqa: E402

CONFIGS = TOOLS.parents[1] / "0927调参结果" / "configs"


def write_manifest(tmp_path):
    """Location aa: 24 level views (redundant) and two steep ones; location bb: four 90-degree views."""
    views = [("aa", yaw, 0, 60.0, [0, 0, 1.5]) for yaw in range(0, 360, 15)]
    views += [("aa", 0, 25, 60.0, [0, 0, 1.5]), ("aa", 180, -25, 60.0, [0, 0, 1.5])]
    views += [("bb", yaw, 0, 90.0, [4, 1, 1.5]) for yaw in (0, 90, 180, 270)]
    rows = []
    for frame, (uuid, yaw, pitch, fov, location) in enumerate(views):
        pose = tmp_path / "camera_{}_office_1_frame_{}_domain_pose.json".format(uuid, frame)
        write_pose(pose, yaw, pitch, fov, location)
        rows.append({"sample_id": "area_1/{}_{}".format(uuid, frame), "split": "train",
                     "camera_metadata_path": str(pose)})
    rows.append({"sample_id": "area_5a/test_0", "split": "test", "camera_metadata_path": "unused"})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, [row["sample_id"] for row in rows[:-1]]


def make_subset(monkeypatch, manifest, output, *extra):
    monkeypatch.setattr(sys, "argv", ["make_view_subset.py", "--manifest", str(manifest), "--output", str(output),
                                      "--drop-fraction", "0.2", "--directions", "2048"] + list(extra))
    return mv.main()


def test_greedy_subset_keeps_coverage_and_manifest_order(tmp_path, monkeypatch):
    manifest, train_ids = write_manifest(tmp_path)
    output = tmp_path / "greedy80.txt"
    assert make_subset(monkeypatch, manifest, output, "--rule", "greedy") == 0
    kept = output.read_text().split()
    assert len(kept) == 24 and kept == [sample for sample in train_ids if sample in kept]
    assert all(sample in kept for sample in train_ids[-6:])  # both steep views and all of location bb
    report = json.loads((tmp_path / "greedy80.txt.json").read_text())
    assert report["kept"] == 24 and report["dropped"] == 6 and report["rule"] == "greedy"
    assert report["lost_share_by_level"]["1"]["whole_sphere"] == 0.0
    assert report["subset_sha256"] == mv.sha256(output)
    with pytest.raises(FileExistsError):
        make_subset(monkeypatch, manifest, output, "--rule", "greedy")


def test_random_subset_needs_a_seed_and_follows_it(tmp_path, monkeypatch):
    manifest, _ = write_manifest(tmp_path)
    with pytest.raises(SystemExit):
        make_subset(monkeypatch, manifest, tmp_path / "r.txt", "--rule", "random")
    lists = []
    for name, seed in (("a", 1), ("b", 1), ("c", 2)):
        assert make_subset(monkeypatch, manifest, tmp_path / (name + ".txt"), "--rule", "random", "--seed", str(seed)) == 0
        lists.append((tmp_path / (name + ".txt")).read_text())
    assert lists[0] == lists[1] != lists[2]
    assert len(lists[0].split()) == 24


def test_config_changes_only_names_outputs_and_epoch_length(tmp_path):
    arm = json.loads((CONFIGS / "relplus.json").read_text(encoding="utf-8"))
    subset = tmp_path / "greedy85.txt"
    subset.write_text("x\n")
    config = tv.subset_config(arm, subset, 44968, tmp_path / "run", tmp_path / "source")
    assert config["niters_per_epoch"] == 44968 // 56 + 1 == 804
    assert config["logical_samples_per_epoch"] == 56 * 804
    assert config["view_subset_count"] == 44968 and config["experiment_name"].endswith("_greedy85")
    changed = {key for key in arm if arm[key] != config[key]}
    allowed = {"experiment_name", "root_dir", "abs_dir", "output_dir", "log_dir", "tb_dir", "checkpoint_dir",
               "log_dir_link", "ddp_smoke_report", "niters_per_epoch", "logical_samples_per_epoch"} | set(tv.tr.LOG_FILES)
    assert changed <= allowed, changed - allowed
    assert config["train_source"] == arm["train_source"] and config["num_train_imgs"] == arm["num_train_imgs"] == 52903
    smoke = tv.subset_config(arm, subset, 44968, tmp_path / "run", tmp_path / "source", smoke_iterations=3)
    assert smoke["nepochs"] == 1 and smoke["niters_per_epoch"] == 3
    with pytest.raises(ValueError):
        tv.subset_config(arm, subset, 52903, tmp_path / "run", tmp_path / "source")
    with pytest.raises(ValueError):
        tv.subset_config(dict(arm, architecture="RGB-only (CMX RGB stream)"), subset, 10, tmp_path / "r", tmp_path)


def test_subset_dataset_filters_only_the_training_split():
    class Base(object):
        def __init__(self, split):
            self.names = self._get_file_names(split)

        def _get_file_names(self, split_name):
            return ["a", "b", "c", "d"] if split_name == "train" else ["t1", "t2"]

    subset = tv.subset_dataset(Base, ["d", "b"])
    assert subset("train").names == ["b", "d"] and subset("val").names == ["t1", "t2"]
    with pytest.raises(RuntimeError, match="1 subset IDs"):
        tv.subset_dataset(Base, ["b", "z"])("train")


def test_read_ids_rejects_duplicates_and_empty_lists(tmp_path):
    (tmp_path / "dup.txt").write_text("a\nb\na\n")
    (tmp_path / "empty.txt").write_text("\n")
    for name in ("dup.txt", "empty.txt"):
        with pytest.raises(ValueError):
            tv.read_ids(tmp_path / name)


# --- one epoch of CMX training on CPU ------------------------------------------------------------


def write_launcher(path):
    """Run train_view_subset.py at world size 1 with the server-side data reports stubbed.

    Every loaded training sample's name is appended to <output>/loaded.txt.
    """
    path.write_text(
        "import sys\n"
        "sys.path.insert(0, {!r})\n"
        "sys.path.insert(0, {!r})\n".format(str(TOOLS), str(tv.DEFAULT_SOURCE_ROOT))
        + "import utils.training_protocol as protocol\n"
        "protocol.assert_training_ready = lambda config: {}\n"
        "protocol.assert_runtime_dataset_contract = lambda config, **_: {}\n"
        "from dataloader.RGBXDataset import RGBXDataset\n"
        "import train_view_subset as tv\n"
        "tv.tr.WORLD_SIZE = 1\n"
        "record = sys.argv[sys.argv.index('--output') + 1] + '/loaded.txt'\n"
        "original = RGBXDataset.__getitem__\n"
        "def getitem(self, index):\n"
        "    with open(record, 'a') as handle:\n"
        "        handle.write(self._file_names[index] + '\\n')\n"
        "    return original(self, index)\n"
        "RGBXDataset.__getitem__ = getitem\n"
        "raise SystemExit(tv.main())\n",
        encoding="utf-8",
    )
    return path


def test_trains_one_epoch_on_the_subset_only(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorboardX")
    pytest.importorskip("tqdm")
    import cv2
    import eval_pano_transfer as ev

    ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)
    from models.encoders.dual_segformer import mit_b0

    rng = np.random.default_rng(0)
    cache, ids = tmp_path / "cache", ["s{}".format(i) for i in range(6)]
    for folder, make in (("RGB", lambda: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)),
                         ("Label", lambda: rng.integers(0, 14, (64, 64), dtype=np.uint8)),
                         ("HHA", lambda: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))):
        (cache / folder).mkdir(parents=True)
        for sample_id in ids:
            cv2.imwrite(str(cache / folder / (sample_id + ".png")), make())
    (cache / "train.txt").write_text("\n".join(ids) + "\n")
    torch.save(mit_b0().state_dict(), tmp_path / "mit_b0.pth")
    config = json.loads((CONFIGS / "hha.json").read_text(encoding="utf-8"))
    config.update(
        backbone="mit_b0", decoder_embed_dim=32, image_height=64, image_width=64, train_scale_array=[0.5, 1.0],
        num_workers=2, batch_size=2, nepochs=1, warm_up_epoch=1, num_train_imgs=6, num_eval_imgs=6,
        logical_samples_per_epoch=8, niters_per_epoch=4,
        rgb_root_folder=str(cache / "RGB"), gt_root_folder=str(cache / "Label"), x_root_folder=str(cache / "HHA"),
        train_source=str(cache / "train.txt"), eval_source=str(cache / "train.txt"),
        pretrained_model=str(tmp_path / "mit_b0.pth"),
    )
    (tmp_path / "hha.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "subset.txt").write_text("s1\ns2\ns4\ns5\n")
    (tmp_path / "bad.txt").write_text("s1\nzz\n")
    launcher = write_launcher(tmp_path / "launch.py")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")

    def train(subset, output):
        return subprocess.run([sys.executable, str(launcher), "--config", str(tmp_path / "hha.json"),
                               "--subset", str(tmp_path / subset), "--output", str(tmp_path / output)],
                              capture_output=True, text=True, env=environment)

    bad = train("bad.txt", "bad")
    assert bad.returncode != 0 and "1 subset IDs are not in the training list" in bad.stderr

    result = train("subset.txt", "run")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    run = tmp_path / "run"
    loaded = (run / "loaded.txt").read_text().split()
    assert set(loaded) == {"s1", "s2", "s4", "s5"} and len(loaded) == 6  # 3 iterations of 2, two repeats
    resolved = json.loads((run / "resolved_config.json").read_text())
    record = json.loads((run / "view_subset.json").read_text())
    assert resolved["niters_per_epoch"] == 3 and resolved["logical_samples_per_epoch"] == 6
    assert record["subset_count"] == 4 and record["full_train_count"] == 6 and record["niters_per_epoch"] == 3
    assert (run / "checkpoints" / "epoch-1.pth").is_file()
