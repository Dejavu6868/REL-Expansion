"""RGB-only baseline: the network, the config it trains with, and a CPU run from training to evaluation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pano_transfer as ev  # noqa: E402
import eval_pinhole as ep  # noqa: E402
import rgb_only as ro  # noqa: E402
import train_rgb_only as tr  # noqa: E402

CONFIGS = ev.REPO_ROOT / "0927调参结果" / "configs"


def hha_config():
    return json.loads((CONFIGS / "hha.json").read_text(encoding="utf-8"))


def small_config(path, **changes):
    config = hha_config()
    config.update(backbone="mit_b0", decoder_embed_dim=32, **changes)
    path.write_text(json.dumps(config), encoding="utf-8")
    return ev.load_config(path)


@pytest.fixture(scope="module")
def frozen():
    return ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)


def test_network_is_the_cmx_rgb_stream_without_x_or_fusion(frozen, tmp_path):
    config = small_config(tmp_path / "config.json", architecture=ro.ARCHITECTURE)
    torch.manual_seed(3)
    cmx = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    torch.manual_seed(3)
    rgb = ro.model_class()(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    ours, theirs = rgb.state_dict(), cmx.state_dict()
    assert set(ours) == {key for key in theirs if not key.startswith(
        tuple("backbone." + name for name in ro.REMOVED_MODULES))}
    assert all(torch.equal(ours[key], theirs[key]) for key in ours)  # same draws at the same seed

    rgb.eval()
    image = torch.randn(1, 3, 64, 96)
    logits = rgb(image, torch.zeros(1, 3, 64, 96))
    assert logits.shape == (1, 13, 64, 96)
    assert torch.equal(logits, rgb(image, torch.randn(1, 3, 64, 96)))
    assert ro.is_rgb_only(config) and not ro.is_rgb_only(ev.load_config(CONFIGS / "hha.json"))


def test_full_size_parameter_counts(frozen):
    config = ev.load_config(CONFIGS / "hha.json")
    count = lambda network: sum(parameter.numel() for parameter in network.parameters())  # noqa: E731
    rgb = ro.model_class()(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    cmx = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    assert (count(rgb), count(rgb.backbone), count(cmx)) == (25779405, 24196288, 66567573)


def test_pretrained_weights_reach_the_rgb_stream_as_in_cmx(frozen, tmp_path):
    from models.encoders.dual_segformer import mit_b0

    torch.manual_seed(5)
    pretrained = {key: value + torch.randn_like(value) for key, value in mit_b0().state_dict().items()
                  if not key.startswith(ro.REMOVED_MODULES)}
    torch.save(pretrained, tmp_path / "mit_b0.pth")
    config = small_config(tmp_path / "config.json", pretrained_model=str(tmp_path / "mit_b0.pth"))
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=255)
    rgb = ro.model_class()(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    cmx = frozen["EncoderDecoder"](cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    for key, value in pretrained.items():
        assert torch.equal(rgb.backbone.state_dict()[key], value)
        assert torch.equal(cmx.backbone.state_dict()[key], value)


def test_config_changes_only_the_architecture_name_and_outputs(tmp_path):
    hha = hha_config()
    output = tmp_path / "run"
    config = tr.rgb_only_config(hha, output, tmp_path / "source")
    changed = {key for key in hha if hha[key] != config[key]}
    assert set(config) == set(hha)
    assert changed == {
        "architecture", "arm_name", "experiment_name", "root_dir", "abs_dir", "output_dir", "log_dir",
        "tb_dir", "checkpoint_dir", "log_dir_link", "log_file", "link_log_file", "val_log_file",
        "link_val_log_file", "ddp_smoke_report",
    }
    assert all(str(output) in config[key] for key in changed - {
        "architecture", "arm_name", "experiment_name", "root_dir", "abs_dir", "ddp_smoke_report"})
    assert ro.is_rgb_only(config) and config["seed"] == hha["seed"] and config["comparison_arm"] == "hha"
    smoke = tr.rgb_only_config(hha, output, tmp_path / "source", smoke_iterations=3)
    assert (smoke["nepochs"], smoke["niters_per_epoch"]) == (1, 3)
    with pytest.raises(ValueError, match="HHA"):
        tr.rgb_only_config(json.loads((CONFIGS / "relplus.json").read_text()), output, tmp_path)
    (output / "logs").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        tr.main(["--config", str(CONFIGS / "hha.json"), "--output", str(output)])


def test_pinhole_eval_builds_the_rgb_only_network(monkeypatch, tmp_path):
    from test_eval_pinhole import _TinyDataset, _frozen

    frozen_eval, core = _frozen()
    import dataloader.RGBXDataset as dataset_module
    import dataloader.data_setting as data_setting
    import models.builder as builder
    import utils.training_protocol as protocol

    for target, name in ((core, "prepare_eval_sample"), (core, "evaluate_prepared_sample"),
                         (frozen_eval, "_unbatch")):
        monkeypatch.setattr(target, name, getattr(target, name))  # restored after the test
    monkeypatch.setitem(sys.modules, "models.builder", builder)
    monkeypatch.setattr(protocol, "assert_runtime_dataset_contract", lambda config, **_: {})
    monkeypatch.setattr(data_setting, "build_data_setting", lambda config, split: {"x_mode": config.x_mode})
    monkeypatch.setattr(dataset_module, "RGBXDataset", _TinyDataset)
    config = small_config(tmp_path / "config.json", architecture=ro.ARCHITECTURE, num_eval_imgs=3,
                          num_workers=0, eval_crop_size=[64, 64])
    network = ro.model_class()(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    torch.save({"epoch": 200, "model": network.state_dict()}, tmp_path / "checkpoint.pth")
    monkeypatch.setattr(sys, "argv", [
        "eval_pinhole.py", "--config", str(tmp_path / "config.json"), "--checkpoint",
        str(tmp_path / "checkpoint.pth"), "--output", str(tmp_path / "eval"), "--device", "cpu",
        "--eight-class-predictions",
    ])
    assert ep.main() == 0
    report = json.loads((tmp_path / "eval" / "eval_pinhole.json").read_text())
    assert report["architecture"] == ro.ARCHITECTURE and report["hha_input_adapter"] is not None
    confusion = np.loadtxt(tmp_path / "eval" / "evaluation" / "confusion_matrix.csv", delimiter=",",
                           skiprows=1, dtype=np.int64)[:, 1:]
    assert confusion.sum() == 3 * 60 * 64 and confusion[:, [0, 1, 2, 5, 6]].sum() == 0


# --- training to panorama evaluation on CPU ------------------------------------------------------


def run(arguments):
    # Engine checkpoints hold numpy RNG states, which torch >= 2.6 refuses by default (the server runs 1.8).
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    return subprocess.run([sys.executable] + arguments, capture_output=True, text=True, env=environment)


def write_launcher(path, world_size, network="rgb_only"):
    """Run train_rgb_only.py with the server-side data reports stubbed and the world size of this test.

    Every training forward appends a hash of its (rgb, modal_x, label) batch to <output>/batches.txt.
    network="cmx" trains the frozen CMX network through the same runner, for comparing batches.
    """
    path.write_text(
        "import hashlib, sys\n"
        "sys.path.insert(0, {!r})\n"
        "sys.path.insert(0, {!r})\n".format(str(TOOLS), str(ev.DEFAULT_SOURCE_ROOT))
        + "import utils.training_protocol as protocol\n"
        "protocol.assert_training_ready = lambda config: {}\n"
        "protocol.assert_runtime_dataset_contract = lambda config, **_: {}\n"
        "import train_rgb_only as tr\n"
        "from models.builder import EncoderDecoder\n"
        "tr.WORLD_SIZE = " + str(world_size) + "\n"
        "base = EncoderDecoder if " + repr(network) + " == 'cmx' else tr.ro.model_class()\n"
        "record = sys.argv[sys.argv.index('--output') + 1] + '/batches.txt'\n"
        "class Recording(base):\n"
        "    def forward(self, rgb, modal_x, label=None):\n"
        "        with open(record, 'a') as handle:\n"
        "            handle.write(hashlib.sha256(b''.join(t.numpy().tobytes() for t in (rgb, modal_x, label)))"
        ".hexdigest() + '\\n')\n"
        "        return super().forward(rgb, modal_x, label)\n"
        "tr.ro.model_class = lambda: Recording\n"
        "raise SystemExit(tr.main())\n",
        encoding="utf-8",
    )
    return path


def test_trains_and_evaluates_end_to_end(frozen, monkeypatch, tmp_path):
    pytest.importorskip("tensorboardX")  # the frozen train.py imports both
    pytest.importorskip("tqdm")
    from models.encoders.dual_segformer import mit_b0
    from test_cross_projection_synthetic import render_erp_raw_depth

    rng = np.random.default_rng(0)
    cache, ids = tmp_path / "cache", ["s{}".format(i) for i in range(4)]
    for folder, make in (("RGB", lambda: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)),
                         ("Label", lambda: rng.integers(0, 14, (64, 64), dtype=np.uint8)),
                         ("HHA", lambda: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))):
        (cache / folder).mkdir(parents=True)
        for sample_id in ids:
            cv2.imwrite(str(cache / folder / (sample_id + ".png")), make())
    (cache / "train.txt").write_text("\n".join(ids) + "\n")
    torch.save({key: value for key, value in mit_b0().state_dict().items()
                if not key.startswith(ro.REMOVED_MODULES)}, tmp_path / "mit_b0.pth")
    small_config(
        tmp_path / "hha.json", image_height=64, image_width=64, train_scale_array=[0.5, 1.0],
        num_workers=2, batch_size=2, num_train_imgs=4, num_eval_imgs=4, logical_samples_per_epoch=4, niters_per_epoch=2,
        rgb_root_folder=str(cache / "RGB"), gt_root_folder=str(cache / "Label"), x_root_folder=str(cache / "HHA"),
        train_source=str(cache / "train.txt"), eval_source=str(cache / "train.txt"),
        pretrained_model=str(tmp_path / "mit_b0.pth"),
    )
    train = ["--config", str(tmp_path / "hha.json"), "--smoke-iterations", "2"]

    refused = run([str(write_launcher(tmp_path / "launch8.py", 8))] + train + ["--output", str(tmp_path / "x")])
    assert refused.returncode != 0 and "world size 1" in refused.stderr
    for network in ("rgb_only", "cmx"):
        launcher = write_launcher(tmp_path / (network + ".py"), 1, network)
        result = run([str(launcher)] + train + ["--output", str(tmp_path / network)])
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    batches = (tmp_path / "rgb_only" / "batches.txt").read_text().split()
    assert len(batches) == 2 and batches == (tmp_path / "cmx" / "batches.txt").read_text().split()

    run_dir = tmp_path / "rgb_only"
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    record = json.loads((run_dir / "rgb_only.json").read_text())
    assert resolved["architecture"] == ro.ARCHITECTURE and resolved["checkpoint_dir"] == str(run_dir / "checkpoints")
    assert record["seed"] == 12345 and record["smoke_iterations"] == 2
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    network = ro.model_class()(cfg=ev.load_config(run_dir / "resolved_config.json"), criterion=None,
                               norm_layer=nn.BatchNorm2d)
    assert frozen["load_checkpoint_once"](network, run_dir / "checkpoints" / "epoch-1.pth", expected_epoch=1) == 1
    assert record["parameters"] == sum(parameter.numel() for parameter in network.parameters())

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
                  "--config", str(run_dir / "resolved_config.json"), "--checkpoint",
                  str(run_dir / "checkpoints" / "epoch-1.pth"), "--expected-epoch", "1", "--height", "64",
                  "--width", "128", "--wrap-pad", "16", "--device", "cpu", "--eight-class-predictions",
                  "--output", str(tmp_path / "pano")])
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    metrics = json.loads((tmp_path / "pano" / "metrics.json").read_text())
    assert metrics["architecture"] == ro.ARCHITECTURE and metrics["hha_recipe"] is None
    assert metrics["panorama_x_input"].startswith("none")
