"""Single-channel REL+: the replicated input, the config it trains with, and a CPU run to evaluation."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import single_channel as sc  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[2] / "0927调参结果" / "configs"


def relplus_config():
    return json.loads((CONFIGS / "relplus.json").read_text(encoding="utf-8"))


def test_replicate_copies_one_channel_into_all_three():
    x = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    for index, name in enumerate(sc.CHANNELS):
        out = sc.replicate(x, name)
        assert out.shape == x.shape and out.dtype == np.uint8
        for channel in range(3):
            assert np.array_equal(out[:, :, channel], x[:, :, index])
    with pytest.raises(ValueError):
        sc.replicate(x[:, :, 0], "LOA")
    assert sc.channel_of({}) is None and sc.channel_of({"single_channel": "ReD"}) == "ReD"
    with pytest.raises(ValueError, match="single_channel"):
        sc.channel_of({"single_channel": "Height"})


def test_get_rel_wrapper_replicates_the_panorama_encoding():
    encoded = np.random.default_rng(0).integers(0, 256, (4, 8, 3), dtype=np.uint8)
    wrapped = sc.wrap_get_rel(lambda depth: encoded, "EGVIA")
    assert np.array_equal(wrapped(None), np.repeat(encoded[:, :, :1], 3, axis=2))


def test_loader_replicates_only_the_relplus_images(tmp_path):
    from easydict import EasyDict

    rng = np.random.default_rng(0)
    for folder in ("RELPlus", "ValidMask", "RGB"):
        (tmp_path / folder).mkdir()
    x = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    for folder in ("RELPlus", "RGB"):
        cv2.imwrite(str(tmp_path / folder / "a.png"), x)
    cv2.imwrite(str(tmp_path / "ValidMask" / "a.png"), np.ones((6, 6), np.uint8))

    class Dataset:
        @staticmethod
        def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
            return np.asarray(cv2.imread(filepath, mode), dtype=dtype)

    config = EasyDict(relplus_config(), x_root_folder=str(tmp_path / "RELPlus"))
    assert sc.install_loader(config, Dataset) is None  # full arm: loader untouched
    config.single_channel = "LOA"
    assert sc.install_loader(config, Dataset) == "LOA"
    read = Dataset._open_image(str(tmp_path / "RELPlus" / "a.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(read, np.repeat(x[:, :, 1:2], 3, axis=2))
    assert np.array_equal(Dataset._open_image(str(tmp_path / "RGB" / "a.png"), cv2.IMREAD_UNCHANGED), x)
    assert Dataset._open_image(str(tmp_path / "ValidMask" / "a.png"), cv2.IMREAD_UNCHANGED).ndim == 2
    with pytest.raises(RuntimeError, match="already"):
        sc.install_loader(config, Dataset)
    with pytest.raises(ValueError, match="rel_plus_v2_1"):
        sc.install_loader(EasyDict(config, x_mode="hha_frozen_cache"), type("D", (), {"_open_image": None}))


def test_config_changes_only_the_channel_name_and_outputs(tmp_path):
    pytest.importorskip("torch")
    import relplus_variant as rv
    import train_single_channel as ts

    relplus = relplus_config()
    output = tmp_path / "run"
    config = ts.single_channel_config(relplus, "LOA", output, tmp_path / "source")
    changed = {key for key in config if relplus.get(key) != config[key]}
    assert set(config) == set(relplus) | {"single_channel"}
    assert changed == {
        "single_channel", "arm_name", "experiment_name", "root_dir", "abs_dir", "output_dir", "log_dir",
        "tb_dir", "checkpoint_dir", "log_dir_link", "log_file", "link_log_file", "val_log_file",
        "link_val_log_file", "ddp_smoke_report",
    }
    assert config["single_channel"] == "LOA" and config["arm_name"] == "CMX-REL+ LOA only"
    assert config["seed"] == relplus["seed"] and config["architecture"] == "Original CMX"
    with pytest.raises(ValueError, match="already"):
        ts.single_channel_config(config, "ReD", output, tmp_path)
    with pytest.raises(ValueError, match="REL\\+"):
        ts.single_channel_config(json.loads((CONFIGS / "hha.json").read_text()), "LOA", output, tmp_path)

    cache = tmp_path / "cache"
    (cache / "RELPlus").mkdir(parents=True)
    path = tmp_path / "relplus.json"
    path.write_text(json.dumps(dict(relplus, x_root_folder=str(cache / "RELPlus"))))
    with pytest.raises(ValueError, match="v3_1new but the config's cache .* is v2_1"):
        ts.main(["--config", str(path), "--channel", "LOA", "--output", str(tmp_path / "refused")])
    (cache / rv.MARKER_NAME).write_text(json.dumps({"variant": "v3_1"}))
    with pytest.raises(ValueError, match="is v3_1$"):
        ts.main(["--config", str(path), "--channel", "LOA", "--output", str(tmp_path / "refused")])


# --- training to evaluation on CPU -----------------------------------------------------------------


def write_launcher(path):
    """Run train_single_channel.py at world size 1 with the server-side data reports stubbed.

    Every training forward appends a hash of its (rgb, modal_x, label) batch to <output>/batches.txt.
    """
    path.write_text(
        "import hashlib, sys\n"
        "sys.path.insert(0, {!r})\n"
        "sys.path.insert(0, {!r})\n".format(str(TOOLS), str(sc_source_root()))
        + "import utils.training_protocol as protocol\n"
        "protocol.assert_training_ready = lambda config: {}\n"
        "protocol.assert_runtime_dataset_contract = lambda config, **_: {}\n"
        "import train_single_channel as ts\n"
        "import utils.training_runtime as runtime\n"
        "from models.builder import EncoderDecoder\n"
        "ts.WORLD_SIZE = 1\n"
        "record = sys.argv[sys.argv.index('--output') + 1] + '/batches.txt'\n"
        "class Recording(EncoderDecoder):\n"
        "    def forward(self, rgb, modal_x, label=None):\n"
        "        with open(record, 'a') as handle:\n"
        "            handle.write(hashlib.sha256(b''.join(t.numpy().tobytes() for t in (rgb, modal_x, label)))"
        ".hexdigest() + '\\n')\n"
        "        return super().forward(rgb, modal_x, label)\n"
        "build = runtime.build_training_runtime\n"
        "runtime.build_training_runtime = lambda *a, **k: build(*a, model_class=Recording, **k)\n"
        "raise SystemExit(ts.main())\n",
        encoding="utf-8",
    )
    return path


def sc_source_root():
    import eval_pano_transfer as ev

    return ev.DEFAULT_SOURCE_ROOT


def test_trains_and_evaluates_like_a_replicated_cache(monkeypatch, tmp_path):
    """Replicating at load time gives the same batches and pinhole scores as a cache of replicated PNGs.

    Frames are 480 px because the frozen pinhole evaluator only takes full frames.
    """
    torch = pytest.importorskip("torch")
    import eval_pano_transfer as ev
    import eval_pinhole as ep
    import relplus_variant as rv

    ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)  # puts the frozen source on sys.path
    from models.encoders.dual_segformer import mit_b0
    from test_cross_projection_synthetic import render_erp_raw_depth
    from test_gated_relplus import write_launcher as gated_launcher
    from test_rgb_only import run

    rng = np.random.default_rng(0)
    cache, ids = tmp_path / "cache", ["s{}".format(i) for i in range(4)]
    replicated = tmp_path / "replicated"
    for folder, make in (("RGB", lambda: rng.integers(0, 256, (480, 480, 3), dtype=np.uint8)),
                         ("Label", lambda: rng.integers(0, 14, (480, 480), dtype=np.uint8)),
                         ("RELPlus", lambda: rng.integers(0, 256, (480, 480, 3), dtype=np.uint8)),
                         ("ValidMask", lambda: np.ones((480, 480), dtype=np.uint8))):
        (cache / folder).mkdir(parents=True)
        (replicated / folder).mkdir(parents=True)
        for sample_id in ids:
            image = make()
            cv2.imwrite(str(cache / folder / (sample_id + ".png")), image)
            if folder == "RELPlus":  # cv2 writes BGR as stored; LOA is index 1 either way
                image = np.repeat(image[:, :, 1:2], 3, axis=2)
            cv2.imwrite(str(replicated / folder / (sample_id + ".png")), image)
    for root in (cache, replicated):
        (root / "train.txt").write_text("\n".join(ids) + "\n")
        (root / rv.MARKER_NAME).write_text(json.dumps({"variant": "v3_1new"}))
    torch.save(mit_b0().state_dict(), tmp_path / "mit_b0.pth")

    def write_config(name, root):
        config = relplus_config()
        config.update(
            backbone="mit_b0", decoder_embed_dim=32, image_height=480, image_width=480, train_scale_array=[0.5, 1.0],
            num_workers=2, batch_size=2, num_train_imgs=4, num_eval_imgs=4, logical_samples_per_epoch=4,
            niters_per_epoch=2, rgb_root_folder=str(root / "RGB"), gt_root_folder=str(root / "Label"),
            x_root_folder=str(root / "RELPlus"), x_valid_root_folder=str(root / "ValidMask"),
            train_source=str(root / "train.txt"), eval_source=str(root / "train.txt"),
            pretrained_model=str(tmp_path / "mit_b0.pth"),
        )
        (tmp_path / name).write_text(json.dumps(config))
        return tmp_path / name

    single = run([str(write_launcher(tmp_path / "single.py")), "--config", str(write_config("relplus.json", cache)),
                  "--channel", "LOA", "--smoke-iterations", "2", "--output", str(tmp_path / "single")])
    assert single.returncode == 0, single.stdout[-3000:] + single.stderr[-3000:]
    assert "X input: REL+ LOA only" in single.stdout
    vanilla = run([str(gated_launcher(tmp_path / "cmx.py", "cmx")), "--config",
                   str(write_config("replicated.json", replicated)), "--smoke-iterations", "2",
                   "--output", str(tmp_path / "cmx")])
    assert vanilla.returncode == 0, vanilla.stdout[-3000:] + vanilla.stderr[-3000:]
    batches = (tmp_path / "single" / "batches.txt").read_text().split()
    assert len(batches) == 2 and batches == (tmp_path / "cmx" / "batches.txt").read_text().split()

    run_dir = tmp_path / "single"
    record = json.loads((run_dir / "single_channel.json").read_text())
    assert record["single_channel"] == "LOA" and record["relplus_variant"] == "v3_1new" and record["seed"] == 12345
    checkpoint = run_dir / "checkpoints" / "epoch-1.pth"
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    # Pinhole eval: the single-channel config on the original cache equals the plain config on the replicated one.
    import dataloader.RGBXDataset as dataset_module
    import utils.training_protocol as protocol

    original_open = dataset_module.RGBXDataset.__dict__["_open_image"]
    monkeypatch.setattr(protocol, "assert_runtime_dataset_contract", lambda config, **_: {})
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    (tmp_path / "plain.json").write_text(json.dumps(dict(
        {key: value for key, value in resolved.items() if key != "single_channel"},
        **{key: json.loads((tmp_path / "replicated.json").read_text())[key]
           for key in ("rgb_root_folder", "gt_root_folder", "x_root_folder", "x_valid_root_folder", "eval_source")})))
    confusions = {}
    for name, config_path in (("single", run_dir / "resolved_config.json"), ("plain", tmp_path / "plain.json")):
        monkeypatch.setattr(sys, "argv", [
            "eval_pinhole.py", "--config", str(config_path), "--checkpoint", str(checkpoint), "--expected-epoch", "1",
            "--output", str(tmp_path / ("pinhole_" + name)), "--device", "cpu", "--eight-class-predictions",
        ])
        try:
            assert ep.main() == 0
        finally:
            dataset_module.RGBXDataset._open_image = original_open
        report = json.loads((tmp_path / ("pinhole_" + name) / "eval_pinhole.json").read_text())
        assert report["single_channel"] == (None if name == "plain" else "LOA")
        confusions[name] = (tmp_path / ("pinhole_" + name) / "evaluation" / "confusion_matrix.csv").read_text()
    assert confusions["single"] == confusions["plain"]

    # Panorama eval: getREL output replicated, reported in metrics.json.
    names = ["<UNK>_0"] + [name + "_1" for name in ev.EXPECTED_CLASSES[1:]]
    (tmp_path / "semantic_labels.json").write_text(json.dumps(names))
    folder = tmp_path / "s2d" / "area_5a" / "pano"
    for kind in ("rgb", "depth", "semantic"):
        (folder / kind).mkdir(parents=True)
    indices = rng.integers(0, len(names), size=(64, 128)).astype(np.uint32)
    cv2.imwrite(str(folder / "rgb" / "c_rgb.png"), rng.integers(0, 256, (64, 128, 3), dtype=np.uint8))
    cv2.imwrite(str(folder / "depth" / "c_depth.png"), render_erp_raw_depth(64, 128))
    cv2.imwrite(str(folder / "semantic" / "c_semantic.png"),
                np.dstack([indices % 256, indices // 256 % 256, indices // 65536]).astype(np.uint8))
    result = run([str(TOOLS / "eval_pano_transfer.py"), "--stanford-root", str(tmp_path / "s2d"),
                  "--semantic-labels", str(tmp_path / "semantic_labels.json"), "--areas", "area_5a",
                  "--config", str(run_dir / "resolved_config.json"), "--checkpoint", str(checkpoint),
                  "--expected-epoch", "1", "--height", "64", "--width", "128", "--wrap-pad", "16",
                  "--device", "cpu", "--eight-class-predictions", "--relplus-variant", "v3_1new",
                  "--output", str(tmp_path / "pano")])
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    metrics = json.loads((tmp_path / "pano" / "metrics.json").read_text())
    assert metrics["single_channel"] == "LOA" and metrics["relplus_variant"] == "v3_1new"
    assert metrics["panorama_x_input"].endswith("LOA only, copied into all three channels")
