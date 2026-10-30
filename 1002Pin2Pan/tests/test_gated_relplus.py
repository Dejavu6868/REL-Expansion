"""Gated REL+: the gate against Max's code, the network against CMX, the trainer, and a CPU run to evaluation."""

import json
import math
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
import gated_relplus as gr  # noqa: E402
import relplus_variant as rv  # noqa: E402
import train_gated_relplus as tg  # noqa: E402
from test_rgb_only import run  # noqa: E402

CONFIGS = ev.REPO_ROOT / "0927调参结果" / "configs"
# backbone.Gate.* of Max's gated checkpoint (smmf_ckpt_keys.json, model-100.pth of log_rebut/..._rle_gate_mit_b2)
MAX_GATE_KEYS = {
    "conv.0.weight": [8, 128, 5, 5], "conv.0.bias": [8], "conv.1.weight": [8], "conv.1.bias": [8],
    "conv.1.running_mean": [8], "conv.1.running_var": [8], "conv.1.num_batches_tracked": [],
    "conv.3.weight": [8, 8, 5, 5], "conv.3.bias": [8], "conv.4.weight": [8], "conv.4.bias": [8],
    "conv.4.running_mean": [8], "conv.4.running_var": [8], "conv.4.num_batches_tracked": [],
    "fc.weight": [5, 8, 1, 1],
}


def relplus_config():
    return json.loads((CONFIGS / "relplus.json").read_text(encoding="utf-8"))


def small_config(path, **changes):
    config = relplus_config()
    config.update(backbone="mit_b0", decoder_embed_dim=32, architecture=gr.ARCHITECTURE, **changes)
    path.write_text(json.dumps(config), encoding="utf-8")
    return ev.load_config(path)


@pytest.fixture(scope="module")
def frozen():
    return ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)


class ExpDecayTemp:  # Max's models/net_utils.py, verbatim
    def __init__(self, start_t, end_t, time_len, start_epoch):
        self.start_t = start_t
        self.end_t = end_t
        self.time_len = time_len
        self.start_epoch = start_epoch
        if time_len == 0:
            self.b = 1
        else:
            self.b = math.exp(1 / time_len * math.log(end_t / start_t))

    def get_t(self, epoch):
        if epoch <= self.start_epoch:
            return self.start_t
        return self.start_t * self.b ** (epoch - self.start_epoch)


def test_schedule_is_maxs():
    temp = ExpDecayTemp(start_t=1.0, end_t=0.01, time_len=100 - 1, start_epoch=1)  # his config.py and train.py
    for epoch in range(1, 201):
        assert gr.schedule(epoch) == (epoch < 100, temp.get_t(epoch))
    assert gr.schedule(100)[1] == pytest.approx(0.01)


def test_full_size_network_is_cmx_plus_maxs_gate(frozen):
    config = ev.load_config(CONFIGS / "relplus.json")
    gated = gr.model_class()(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    cmx = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    shapes = {key: list(value.shape) for key, value in gated.state_dict().items()}
    gate = {key[len("backbone.Gate."):]: shape for key, shape in shapes.items() if key.startswith("backbone.Gate.")}
    assert gate == MAX_GATE_KEYS
    assert {key: shape for key, shape in shapes.items() if not key.startswith("backbone.Gate.")} == {
        key: list(value.shape) for key, value in cmx.state_dict().items()}
    count = lambda network: sum(parameter.numel() for parameter in network.parameters())  # noqa: E731
    assert count(gated) - count(cmx) == count(gated.backbone.Gate) == 27288


def test_cmx_weights_start_where_the_vanilla_arms_do(frozen, tmp_path):
    config = small_config(tmp_path / "config.json")
    torch.manual_seed(3)
    cmx = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    torch.manual_seed(3)
    gated = gr.model_class()(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    theirs = cmx.state_dict()
    assert all(torch.equal(value, theirs[key]) for key, value in gated.state_dict().items()
               if not key.startswith("backbone.Gate."))
    assert gr.is_gated(config) and not gr.is_gated(ev.load_config(CONFIGS / "relplus.json"))


def force_choices(gated, choices):
    """Make the gate return one-hot weights for the given per-sample branches."""
    def forward(rgb, depth, temp=1.0, hard=False):
        return nn.functional.one_hot(torch.tensor(choices), gr.BRANCHES).float()
    gated.backbone.Gate.forward = forward


def rgb_after(cmx, stages):
    """CMX with the FFM output replaced by the rectified RGB feature from stage ``stages`` on."""
    for stage in range(stages, 4):
        cmx.backbone.FFMs[stage].forward = lambda x_rgb, x_e: x_rgb


def test_evaluation_sends_ffm_up_to_the_chosen_stage(frozen, tmp_path):
    config = small_config(tmp_path / "config.json")
    torch.manual_seed(0)
    gated = gr.model_class()(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d).eval()
    rgb, modal_x = torch.randn(3, 3, 64, 96), torch.randn(3, 3, 64, 96)
    expected = []
    for choice in range(gr.BRANCHES):
        cmx = frozen["EncoderDecoder"](cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
        cmx.load_state_dict({k: v for k, v in gated.state_dict().items() if not k.startswith("backbone.Gate.")})
        rgb_after(cmx.eval(), choice)
        force_choices(gated, [choice] * 3)
        with torch.no_grad():
            ours, theirs = gated(rgb, modal_x), cmx(rgb, modal_x)
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)  # g = 4 is plain CMX
        expected.append(theirs)
    gated.gate_log.clear()
    force_choices(gated, [4, 0, 2])  # each window in a batch gets its own choice
    with torch.no_grad():
        mixed = gated(rgb, modal_x)
    for index, choice in enumerate([4, 0, 2]):
        torch.testing.assert_close(mixed[index], expected[choice][index], rtol=1e-5, atol=1e-5)
    assert gated.gate_log == [4, 0, 2]

    del gated.backbone.Gate.forward  # the real gate: argmax per sample, logged in batch order
    gated.gate_log.clear()
    with torch.no_grad():
        gated(rgb, modal_x)
    assert len(gated.gate_log) == 3 and all(0 <= g < gr.BRANCHES for g in gated.gate_log)


def test_training_blends_by_maxs_weights_and_trains_the_gate(tmp_path):
    config = small_config(tmp_path / "config.json", pretrained_model=None)
    torch.manual_seed(1)
    gated = gr.model_class()(cfg=config, criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=255),
                             norm_layer=nn.BatchNorm2d).train()
    seen = {}
    gated.backbone.Gate.register_forward_hook(lambda module, inputs, weight: seen.update(weight=weight))
    for stage, ffm in enumerate(gated.backbone.FFMs):
        ffm.register_forward_hook(lambda module, inputs, out, stage=stage: seen.update({stage: (inputs[0], out)}))
    decode = gated.decode_head.forward  # called directly, as the frozen encode_decode does, so no hook fires

    def capture(outs):
        seen.update(outs=outs)
        return decode(outs)

    gated.decode_head.forward = capture
    rgb, modal_x = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    label = torch.randint(0, 13, (2, 64, 64))
    for soft, temp in ((True, 1.0), (False, 0.05)):
        gated.soft_gate, gated.temp = soft, temp
        gated.zero_grad()
        gated(rgb, modal_x, label).mean().backward()
        weight = seen["weight"]
        assert torch.allclose(weight.sum(1), torch.ones(2))
        if not soft:
            assert set(weight.detach().flatten().tolist()) <= {0.0, 1.0}
        for stage in range(4):
            x_rgb, fused = seen[stage]
            ffm = weight[:, stage + 1:].sum(1).view(-1, 1, 1, 1)  # P(g > stage), the eval rule in expectation
            torch.testing.assert_close(seen["outs"][stage], (1 - ffm) * x_rgb + ffm * fused, rtol=1e-5, atol=1e-6)
        assert gated.backbone.Gate.fc.weight.grad.abs().sum() > 0  # straight-through keeps the gate learning


def test_pinhole_eval_builds_the_gated_network(monkeypatch, tmp_path):
    import eval_pinhole as ep
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
    config = small_config(tmp_path / "config.json", num_eval_imgs=3, num_workers=0, eval_crop_size=[64, 64])
    network = gr.model_class()(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    torch.save({"epoch": 200, "model": network.state_dict()}, tmp_path / "checkpoint.pth")
    monkeypatch.setattr(sys, "argv", [
        "eval_pinhole.py", "--config", str(tmp_path / "config.json"), "--checkpoint",
        str(tmp_path / "checkpoint.pth"), "--output", str(tmp_path / "eval"), "--device", "cpu",
        "--eight-class-predictions",
    ])
    assert ep.main() == 0
    report = json.loads((tmp_path / "eval" / "eval_pinhole.json").read_text())
    assert report["architecture"] == gr.ARCHITECTURE and report["x_mode"] == "rel_plus_v2_1"
    assert sys.modules["models.builder"].EncoderDecoder is gr.model_class()


def test_config_changes_only_the_architecture_name_and_outputs(tmp_path):
    relplus = relplus_config()
    output = tmp_path / "run"
    config = tg.gated_config(relplus, output, tmp_path / "source")
    changed = {key for key in relplus if relplus[key] != config[key]}
    assert set(config) == set(relplus)
    assert changed == {
        "architecture", "arm_name", "experiment_name", "root_dir", "abs_dir", "output_dir", "log_dir",
        "tb_dir", "checkpoint_dir", "log_dir_link", "log_file", "link_log_file", "val_log_file",
        "link_val_log_file", "ddp_smoke_report",
    }
    assert gr.is_gated(config) and config["seed"] == relplus["seed"] and config["x_mode"] == "rel_plus_v2_1"
    smoke = tg.gated_config(relplus, output, tmp_path / "source", smoke_iterations=3)
    assert (smoke["nepochs"], smoke["niters_per_epoch"]) == (1, 3)
    with pytest.raises(ValueError, match="REL\\+"):
        tg.gated_config(json.loads((CONFIGS / "hha.json").read_text()), output, tmp_path)

    cache = tmp_path / "cache"
    (cache / "RELPlus").mkdir(parents=True)
    path = tmp_path / "relplus.json"
    path.write_text(json.dumps(dict(relplus, x_root_folder=str(cache / "RELPlus"))))
    assert tg.cache_variant(json.loads(path.read_text())) == "v2_1"
    with pytest.raises(ValueError, match="v3_1new but the config's cache .* is v2_1"):
        tg.main(["--config", str(path), "--output", str(tmp_path / "refused")])
    (cache / rv.MARKER_NAME).write_text(json.dumps({"variant": "v3_1"}))
    with pytest.raises(ValueError, match="is v3_1$"):
        tg.main(["--config", str(path), "--output", str(tmp_path / "refused")])


def test_set_epoch_also_sets_the_gate_schedule():
    class Sampler:
        epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    sampler, model = Sampler(), type("Model", (), {})()
    tg.follow_schedule(sampler, model)
    for epoch in (1, 99, 100, 200):
        sampler.set_epoch(epoch)
        assert (model.soft_gate, model.temp) == gr.schedule(epoch)
    assert sampler.epochs == [1, 99, 100, 200]


def test_gate_rows_follow_the_predictors_feed_order():
    windows = ev.erp_tile_windows(2048, 4096, 3, 7, 1080, 1080, column_stride=720, column_offset=-652)
    assert ev.gate_rows(None) == ["whole"]
    assert ev.gate_rows(windows) == ["row 0"] * 7 + ["row 484"] * 7 + ["row 968"] * 7
    assert ev.gate_rows(windows, {"scale": 0.5}) == ["row 0 (x0.5)"] * 7 + ["row 968 (x0.5)"] * 7 + ["row 484"] * 7


def test_trains_and_evaluates_end_to_end(frozen, monkeypatch, tmp_path):
    from models.encoders.dual_segformer import mit_b0
    from test_cross_projection_synthetic import render_erp_raw_depth

    rng = np.random.default_rng(0)
    cache, ids = tmp_path / "cache", ["s{}".format(i) for i in range(4)]
    for folder, make in (("RGB", lambda: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)),
                         ("Label", lambda: rng.integers(0, 14, (64, 64), dtype=np.uint8)),
                         ("RELPlus", lambda: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)),
                         ("ValidMask", lambda: np.ones((64, 64), dtype=np.uint8))):
        (cache / folder).mkdir(parents=True)
        for sample_id in ids:
            cv2.imwrite(str(cache / folder / (sample_id + ".png")), make())
    (cache / "train.txt").write_text("\n".join(ids) + "\n")
    (cache / rv.MARKER_NAME).write_text(json.dumps({"variant": "v3_1new"}))
    torch.save(mit_b0().state_dict(), tmp_path / "mit_b0.pth")
    config = relplus_config()
    config.update(
        backbone="mit_b0", decoder_embed_dim=32, image_height=64, image_width=64, train_scale_array=[0.5, 1.0],
        num_workers=2, batch_size=2, num_train_imgs=4, num_eval_imgs=4, logical_samples_per_epoch=4,
        niters_per_epoch=2, rgb_root_folder=str(cache / "RGB"), gt_root_folder=str(cache / "Label"),
        x_root_folder=str(cache / "RELPlus"), x_valid_root_folder=str(cache / "ValidMask"),
        train_source=str(cache / "train.txt"), eval_source=str(cache / "train.txt"),
        pretrained_model=str(tmp_path / "mit_b0.pth"),
    )
    (tmp_path / "relplus.json").write_text(json.dumps(config))
    train = ["--config", str(tmp_path / "relplus.json"), "--smoke-iterations", "2"]
    outputs = {}
    for network in ("gated", "cmx"):
        result = run([str(write_launcher(tmp_path / (network + ".py"), network))] + train
                     + ["--output", str(tmp_path / network)])
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        outputs[network] = result.stdout
    batches = (tmp_path / "gated" / "batches.txt").read_text().split()
    assert len(batches) == 2 and batches == (tmp_path / "cmx" / "batches.txt").read_text().split()
    assert "Gate epoch 1: soft=True temp=1" in outputs["gated"]

    run_dir = tmp_path / "gated"
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    record = json.loads((run_dir / "gated_relplus.json").read_text())
    assert resolved["architecture"] == gr.ARCHITECTURE and record["relplus_variant"] == "v3_1new"
    assert record["seed"] == 12345 and record["smoke_iterations"] == 2
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    network = gr.model_class()(cfg=ev.load_config(run_dir / "resolved_config.json"), criterion=None,
                               norm_layer=nn.BatchNorm2d)
    assert frozen["load_checkpoint_once"](network, run_dir / "checkpoints" / "epoch-1.pth", expected_epoch=1) == 1
    assert record["parameters"] == sum(parameter.numel() for parameter in network.parameters())

    names = ["<UNK>_0"] + [name + "_1" for name in ev.EXPECTED_CLASSES[1:]]
    (tmp_path / "semantic_labels.json").write_text(json.dumps(names))
    folder = tmp_path / "s2d" / "area_5a" / "pano"
    for kind in ("rgb", "depth", "semantic"):
        (folder / kind).mkdir(parents=True)
    for stem in ("c", "d"):  # windows of 136 px: the gate's two 5x5 stride-2 convs need 68 px at x0.5
        indices = rng.integers(0, len(names), size=(256, 512)).astype(np.uint32)
        cv2.imwrite(str(folder / "rgb" / (stem + "_rgb.png")), rng.integers(0, 256, (256, 512, 3), dtype=np.uint8))
        cv2.imwrite(str(folder / "depth" / (stem + "_depth.png")), render_erp_raw_depth(256, 512))
        cv2.imwrite(str(folder / "semantic" / (stem + "_semantic.png")),
                    np.dstack([indices % 256, indices // 256 % 256, indices // 65536]).astype(np.uint8))
    common = ["--stanford-root", str(tmp_path / "s2d"), "--semantic-labels", str(tmp_path / "semantic_labels.json"),
              "--areas", "area_5a", "--config", str(run_dir / "resolved_config.json"), "--checkpoint",
              str(run_dir / "checkpoints" / "epoch-1.pth"), "--expected-epoch", "1", "--height", "256",
              "--width", "512", "--wrap-pad", "16", "--device", "cpu", "--eight-class-predictions",
              "--relplus-variant", "v3_1new"]
    smmf = ["--erp-tiles", "3", "7", "--erp-tile-size", "136", "136", "--erp-tile-column-stride", "90",
            "--erp-tile-column-offset", "-54", "--erp-tile-exp-sum"]
    for name, extra in (("whole", []), ("smmf", smmf), ("p47", smmf + ["--erp-tile-polar-scale", "0.5"])):
        result = run([str(TOOLS / "eval_pano_transfer.py")] + common + extra + ["--output", str(tmp_path / name)])
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    counts = {name: json.loads((tmp_path / name / "metrics.json").read_text())["gate_counts_per_window_row"]
              for name in ("whole", "smmf", "p47")}
    assert {row: sum(c) for row, c in counts["whole"].items()} == {"whole": 2}
    assert {row: sum(c) for row, c in counts["smmf"].items()} == {"row 0": 14, "row 60": 14, "row 120": 14}
    assert {row: sum(c) for row, c in counts["p47"].items()} == {"row 0 (x0.5)": 14, "row 120 (x0.5)": 14,
                                                                 "row 60": 14}
    assert counts["p47"]["row 60"] == counts["smmf"]["row 60"]  # the same native middle windows

    refused = run([str(TOOLS / "eval_pano_transfer.py")] + common + ["--polar-strips", "8", "--output",
                                                                      str(tmp_path / "strips")])
    assert refused.returncode != 0 and "whole pass or ERP tiles" in refused.stderr


def write_launcher(path, network):
    """Run train_gated_relplus.py at world size 1 with the server-side data reports stubbed.

    Every training forward appends a hash of its (rgb, modal_x, label) batch to <output>/batches.txt.
    network="cmx" trains the frozen CMX network through the same runner, for comparing batches.
    """
    path.write_text(
        "import hashlib, sys\n"
        "sys.path.insert(0, {!r})\n"
        "sys.path.insert(0, {!r})\n".format(str(TOOLS), str(ev.DEFAULT_SOURCE_ROOT))
        + "import torch.nn as nn\n"
        "import utils.training_protocol as protocol\n"
        "protocol.assert_training_ready = lambda config: {}\n"
        "protocol.assert_runtime_dataset_contract = lambda config, **_: {}\n"
        "import train_gated_relplus as tg\n"
        "from models.builder import EncoderDecoder\n"
        "tg.WORLD_SIZE = 1\n"
        "base = EncoderDecoder if " + repr(network) + " == 'cmx' else tg.gr.model_class()\n"
        "record = sys.argv[sys.argv.index('--output') + 1] + '/batches.txt'\n"
        "class Recording(base):\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        super().__init__(*args, **kwargs)\n"
        "        if not hasattr(self.backbone, 'Gate'):\n"
        "            self.backbone.Gate = nn.Identity()  # plain CMX: no gate parameters to count\n"
        "    def forward(self, rgb, modal_x, label=None):\n"
        "        with open(record, 'a') as handle:\n"
        "            handle.write(hashlib.sha256(b''.join(t.numpy().tobytes() for t in (rgb, modal_x, label)))"
        ".hexdigest() + '\\n')\n"
        "        return super().forward(rgb, modal_x, label)\n"
        "tg.gr.model_class = lambda: Recording\n"
        "raise SystemExit(tg.main())\n",
        encoding="utf-8",
    )
    return path
