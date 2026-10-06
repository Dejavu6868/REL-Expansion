"""Pin2Pan adaptation: the Trans4PASS ports against the original code, and an end-to-end CPU run."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pin2pan_adapt as pa  # noqa: E402


def trans4pass_feat_kl_loss(feats, labels, feats_mem):
    """Trans4PASS adaptations/utils/loss.py at 758f2301, verbatim apart from the unused CLS."""
    B, C, H, W = feats.shape
    _, H_org, W_org = labels.shape
    labels = F.interpolate(labels.unsqueeze(1).float(), (H, W), mode='nearest')

    select_feat = torch.clone(feats)
    feats = feats.permute(0, 2, 3, 1).contiguous().view(-1, C)
    select_feat = select_feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)

    feats_mem = feats_mem.squeeze(1)
    batch_feats_mem = torch.zeros_like(feats_mem)

    ignore_index = 255
    for c in labels.unique():
        if c == ignore_index: continue
        c = c.item()
        feats_cls = feats[labels == c].mean(0)
        batch_feats_mem[int(c)] = feats_cls

        m = labels == c
        m = m[..., None].repeat(1, C)
        feat_temp = feats_mem[int(c)][None, ...].expand(labels.shape[0], -1)
        select_feat = torch.where(m, feat_temp, select_feat)
    feats = feats.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    select_feat = select_feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    T = 20
    alpha = 0.9
    loss_kl = nn.KLDivLoss()(
        F.log_softmax(feats / T, dim=1),
        F.softmax(select_feat / T, dim=1)) * (alpha * T * T) + F.cross_entropy(feats, torch.argmax(select_feat, dim=1).long()) * (1. - alpha)

    return loss_kl, batch_feats_mem, select_feat


def test_feat_kl_loss_matches_trans4pass():
    generator = torch.Generator().manual_seed(0)
    feats = torch.randn(2, 8, 6, 10, generator=generator) * 30
    labels = torch.randint(0, 4, (2, 24, 40), generator=generator)
    labels[:, :8] = 255
    labels[labels == 2] = 255  # class 2 absent; class 4 never labelled
    memory = torch.randn(5, 8, generator=generator) * 30

    ours = feats.clone().requires_grad_(True)
    loss, means, present = pa.feat_kl_loss(ours, labels, memory)
    loss.backward()
    theirs = feats.clone().requires_grad_(True)
    expected, expected_means, _ = trans4pass_feat_kl_loss(theirs, labels, memory[:, None])
    expected.backward()

    assert torch.allclose(loss, expected, rtol=1e-5)
    assert torch.allclose(ours.grad, theirs.grad, rtol=1e-4, atol=1e-7)
    assert present.tolist() == [True, True, False, True, False]
    assert torch.allclose(means, expected_means.detach(), atol=1e-5)


def trans4pass_pseudo_labels(predicted_label, predicted_prob, num_classes):
    """Trans4PASS adaptations/gen_pseudo_label.py at 758f2301 (np.int is int)."""
    thres = []
    for i in range(num_classes):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[int(np.round(len(x) * 0.5))])
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9
    labels = predicted_label.copy()
    for index in range(len(labels)):
        label = labels[index]
        prob = predicted_prob[index]
        for i in range(num_classes):
            label[(prob < thres[i]) * (label == i)] = 255
    return thres, labels


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_thresholds_match_trans4pass(seed):
    rng = np.random.default_rng(seed)
    num_classes = 6
    prediction = rng.integers(0, 5, size=(3, 7, 9)).astype(np.uint8)  # class 5 never predicted
    prediction[0, 0, :3] = 4
    prediction[prediction == 4] = rng.choice([4, 0], size=int((prediction == 4).sum()))
    probability = rng.uniform(0.2, 1.0, size=prediction.shape)
    probability[prediction == 1] = rng.uniform(0.92, 1.0, size=int((prediction == 1).sum()))  # hits the cap
    confidence = pa.quantize_confidence(probability)

    histograms = np.stack([
        np.bincount(confidence[prediction == c], minlength=pa.CONFIDENCE_LEVELS + 1) for c in range(num_classes)
    ])
    thresholds = pa.class_thresholds(histograms)
    expected, expected_labels = trans4pass_pseudo_labels(
        prediction, confidence / pa.CONFIDENCE_LEVELS, num_classes
    )
    assert np.array_equal(thresholds, expected)
    assert thresholds[1] == 0.9 and thresholds[5] == 0
    assert np.array_equal(pa.apply_thresholds(prediction, confidence, thresholds), expected_labels)


def test_memory_moves_present_classes_toward_their_channel_means():
    memory = pa.PrototypeMemory(torch.arange(12.0).reshape(3, 4), momentum=0.9, update_every=100)
    first = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], [5.0, 5.0, 5.0, 5.0]])
    second = torch.tensor([[3.0, 4.0, 5.0, 6.0], [0, 0, 0, 0], [0, 0, 0, 0]])
    memory.collect(first, torch.tensor([True, False, True]))
    assert not memory.maybe_update(0)
    memory.collect(second, torch.tensor([True, False, False]))
    assert memory.maybe_update(100)
    expected = torch.arange(12.0).reshape(3, 4)
    expected[0] = expected[0] * 0.9 + torch.tensor([2.0, 3.0, 4.0, 5.0]) * 0.1
    expected[2] = expected[2] * 0.9 + 5.0 * 0.1
    assert torch.allclose(memory.memory, expected)
    assert memory.counts.sum() == 0 and not memory.maybe_update(150)
    assert memory.maybe_update(200)  # nothing collected since: no class moves
    assert torch.allclose(memory.memory, expected)


def test_initial_memory_averages_correct_pixels_over_domains():
    def batch(classes, wrong_class=None):
        label = torch.tensor([classes]).repeat_interleave(2, dim=1)[:, None].repeat(1, 2, 1)  # 1 x 2 x 2n
        feats = label[:, None].float().repeat(1, 3, 1, 1) * 10 + torch.tensor([0.0, 1.0, 2.0])[None, :, None, None]
        logits = F.one_hot(label, 4).permute(0, 3, 1, 2).float()
        if wrong_class is not None:
            logits[:, :, :, -1] = F.one_hot(torch.tensor(wrong_class), 4).float()[:, None]
        return {"label": label}, feats, logits

    source = [batch([0, 1]), batch([1, 1], wrong_class=3)]
    target = [batch([2, 1])]
    lookup = {id(b[0]): b[1:] for b in source + target}
    memory, source_counts, target_counts = pa.initial_memory(
        [b[0] for b in source], [b[0] for b in target], lambda b: lookup[id(b)], 4, 3, "cpu"
    )
    assert source_counts == [1, 2, 0, 0] and target_counts == [0, 1, 1, 0]
    assert torch.allclose(memory, torch.tensor([[0.0, 1, 2], [10, 11, 12], [20, 21, 22], [0, 0, 0]]))


def test_target_crops_wrap_around_and_keep_full_height(tmp_path):
    height, width = 4, 16
    columns = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    for folder, image in (("RGB", np.dstack([columns] * 3)), ("X", np.dstack([columns * 2] * 3)),
                          ("Pseudo", columns + 100)):
        (tmp_path / folder / "area_1").mkdir(parents=True)
        cv2.imwrite(str(tmp_path / folder / "area_1" / "p.png"), image)
    unlabelled = pa.TargetPanoramas(tmp_path, ["area_1/p"], 6, lambda image: image.astype(np.float64))
    labelled = pa.TargetPanoramas(tmp_path, ["area_1/p"], 6, lambda image: image.astype(np.float64),
                                  tmp_path / "Pseudo")
    starts = set()
    for _ in range(40):
        sample = labelled[0]
        crop = sample["data"][0].long()
        assert crop.shape == (height, 6) and (crop == crop[0]).all()
        assert torch.equal((crop[0] - crop[0, 0]) % width, torch.arange(6))
        assert torch.equal(sample["modal_x"][0].long(), crop * 2)
        assert torch.equal(sample["label"], crop + 100)
        starts.add(int(crop[0, 0]))
    assert len(starts) > 8 and any(start > width - 6 for start in starts)  # some crops cross the seam
    assert (unlabelled[0]["label"] == 255).all()


# --- end to end on a tiny network, CPU ---------------------------------------------------------


def free_port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def run(arguments, torchrun=False):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    prefix = [sys.executable]
    if torchrun:
        prefix += ["-m", "torch.distributed.launch", "--nproc_per_node=2", "--master_port={}".format(free_port())]
    result = subprocess.run(prefix + arguments, capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


def run_with_discriminator_sync_check(arguments, root):
    """Check the actual training loop starts each step with identical discriminator weights."""
    wrapper = root / "check_discriminator_sync.py"
    wrapper.write_text(
        "import sys\n"
        "sys.path.insert(0, {!r})\n".format(str(TOOLS))
        + """import torch
import torch.distributed as dist
import train_pin2pan as training

original_average_gradients = training.average_gradients

def check_and_average(parameters, world_size):
    actual = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    expected = actual.clone()
    dist.broadcast(expected, src=0)
    identical = torch.tensor(int(torch.equal(actual, expected)))
    dist.all_reduce(identical, op=dist.ReduceOp.MIN)
    assert bool(identical), 'discriminator weights differ across ranks before optimizer step'
    original_average_gradients(parameters, world_size)

training.average_gradients = check_and_average
raise SystemExit(training.main())
""",
        encoding="utf-8",
    )
    return run([str(wrapper)] + arguments[1:], torchrun=True)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import eval_pano_transfer as ev
    from test_cross_projection_synthetic import render_erp_raw_depth

    root = tmp_path_factory.mktemp("pin2pan")
    rng = np.random.default_rng(0)
    names = ["<UNK>_0"] + [name + "_1" for name in ev.EXPECTED_CLASSES[1:]]
    (root / "semantic_labels.json").write_text(json.dumps(names))
    for area, stems in (("area_1", ["a", "b"]), ("area_5a", ["c"])):
        for stem in stems:
            for kind in ("rgb", "depth", "semantic"):
                (root / "s2d" / area / "pano" / kind).mkdir(parents=True, exist_ok=True)
            indices = rng.integers(0, len(names), size=(64, 128)).astype(np.uint32)
            semantic = np.dstack([indices % 256, indices // 256 % 256, indices // 65536]).astype(np.uint8)
            folder = root / "s2d" / area / "pano"
            cv2.imwrite(str(folder / "rgb" / (stem + "_rgb.png")), rng.integers(0, 256, (64, 128, 3), dtype=np.uint8))
            cv2.imwrite(str(folder / "depth" / (stem + "_depth.png")), render_erp_raw_depth(64, 128))
            cv2.imwrite(str(folder / "semantic" / (stem + "_semantic.png")), semantic)

    source = root / "source_cache"
    ids = ["s{}".format(i) for i in range(4)]
    for folder, make in (
        ("RGB", lambda: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)),
        ("Label", lambda: rng.integers(0, 14, (64, 64), dtype=np.uint8)),
        ("RELPlus", lambda: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)),
        ("ValidMask", lambda: np.ones((64, 64), dtype=np.uint8)),
    ):
        (source / folder).mkdir(parents=True)
        for sample_id in ids:
            cv2.imwrite(str(source / folder / (sample_id + ".png")), make())
    (source / "train.txt").write_text("\n".join(ids) + "\n")
    config = json.loads((ev.REPO_ROOT / "0927调参结果" / "configs" / "relplus.json").read_text(encoding="utf-8"))
    config.update(
        backbone="mit_b0", decoder_embed_dim=32, image_height=64, image_width=64,
        train_scale_array=[0.5, 1.0], num_workers=0, logical_samples_per_epoch=4,
        rgb_root_folder=str(source / "RGB"), gt_root_folder=str(source / "Label"),
        x_root_folder=str(source / "RELPlus"), x_valid_root_folder=str(source / "ValidMask"),
        train_source=str(source / "train.txt"), eval_source=str(source / "train.txt"),
    )
    (root / "config.json").write_text(json.dumps(config))

    frozen = ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)
    torch.manual_seed(0)
    network = frozen["EncoderDecoder"](cfg=ev.load_config(root / "config.json"), criterion=None,
                                       norm_layer=nn.BatchNorm2d)
    torch.save({"epoch": 200, "model": network.state_dict()}, root / "source.pth")
    return root, frozen


def test_adaptation_runs_end_to_end(world):
    import build_target_cache as tc
    import eval_pano_transfer as ev

    root, frozen = world
    common = ["--config", str(root / "config.json")]
    cache = ["--stanford-root", str(root / "s2d"), "--semantic-labels", str(root / "semantic_labels.json"),
             "--height", "64", "--width", "128", "--workers", "1"] + common
    refused = subprocess.run(
        [sys.executable, str(TOOLS / "build_target_cache.py"), "--output", str(root / "refused"),
         "--areas", "area_1", "area_5a"] + cache, capture_output=True, text=True,
    )
    assert refused.returncode == 2 and "test set" in refused.stderr
    run([str(TOOLS / "build_target_cache.py"), "--output", str(root / "target"), "--areas", "area_1"] + cache)
    manifest = json.loads((root / "target" / "manifest.json").read_text())
    assert manifest["samples"] == ["area_1/a", "area_1/b"] and manifest["relplus_variant"] == "v2_1"
    rgb = cv2.imread(str(root / "target" / "RGB" / "area_1" / "a.png"), cv2.IMREAD_UNCHANGED)
    assert rgb.shape == (64, 128, 3)

    run([str(TOOLS / "gen_pseudo_labels.py"), "--target-cache", str(root / "target"), "--checkpoint",
         str(root / "source.pth"), "--output", str(root / "pseudo0"), "--device", "cpu", "--wrap-pad", "16"] + common)
    report = json.loads((root / "pseudo0" / "pseudo_labels.json").read_text())
    pseudo = cv2.imread(str(root / "pseudo0" / "PseudoLabel" / "area_1" / "b.png"), cv2.IMREAD_UNCHANGED)
    assert pseudo.shape == (64, 128) and (pseudo == 255).any() and (pseudo < 13).any()
    assert 0 < report["pseudo_labels"]["coverage_percent"] < 100
    assert not (root / "pseudo0" / "_work").exists()

    train = [str(TOOLS / "train_pin2pan.py"), "--checkpoint", str(root / "source.pth"), "--target-cache",
             str(root / "target"), "--device", "cpu", "--iterations", "2", "--source-batch", "1",
             "--target-crop-width", "64", "--memory-source-images", "2", "--num-workers", "0",
             "--log-every", "1"] + common
    run(train + ["--stage", "warmup", "--output", str(root / "warmup")])
    run([str(TOOLS / "gen_pseudo_labels.py"), "--target-cache", str(root / "target"), "--checkpoint",
         str(root / "warmup" / "checkpoint.pth"), "--output", str(root / "pseudo1"), "--device", "cpu",
         "--wrap-pad", "16"] + common)
    run_with_discriminator_sync_check(
        train + ["--stage", "mpa", "--pseudo-labels", str(root / "pseudo1"), "--output", str(root / "mpa")],
        root,
    )

    for stage, world_size in (("warmup", 1), ("mpa", 2)):
        network = frozen["EncoderDecoder"](cfg=ev.load_config(root / "config.json"), criterion=None,
                                           norm_layer=nn.BatchNorm2d)
        assert frozen["load_checkpoint_once"](network, root / stage / "checkpoint.pth", expected_epoch=200) == 200
        metadata = json.loads((root / stage / "pin2pan.json").read_text())
        assert metadata["stage"] == stage and metadata["world_size"] == world_size
        log = [json.loads(line) for line in (root / stage / "train_log.jsonl").read_text().splitlines()]
        assert [record["iteration"] for record in log] == [1, 2]
    assert sorted(log[-1]) == sorted(["iteration", "lr", "seconds", "source_focal", "adversarial", "discriminator",
                                      "pseudo_focal", "prototype_source", "prototype_target"])
    state = torch.load(str(root / "mpa" / "adaptation_state.pth"))
    assert state["memory"].shape == (13, 32)
    assert metadata["memory_init_counts"]["target_batches"] != [0] * 13

    hha = ev.load_config(root / "config.json")
    hha.x_mode = "hha_frozen_cache"
    with pytest.raises(ValueError, match="hha_frozen_cache"):
        tc.load_manifest(root / "target", hha)
