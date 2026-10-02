import importlib.util
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.loss_opr import FocalLoss2d
from utils.training_protocol import (
    assert_training_ready,
    build_author_criterion,
    build_author_optimizer,
    configure_author_cudnn,
    set_author_seed,
)
from utils.lr_policy import WarmUpPolyLR


AUTHOR_ROOT = Path(
    os.environ.get(
        "REL_SF4PASS_AUTHOR_ROOT",
        "/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference",
    )
)


def _load_author(relative_path, module_name):
    path = AUTHOR_ROOT / relative_path
    if not path.is_file():
        pytest.skip("author REL source is unavailable: {}".format(path))
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_focal_loss_matches_author_map_scalar_and_gradients():
    author = _load_author("utils/loss_opr.py", "author_rel_loss")
    torch.manual_seed(17)
    base_logits = torch.randn(2, 4, 3, 5, dtype=torch.float32)
    labels = torch.tensor(
        [
            [[0, 1, 2, 3, 255], [1, 2, 3, 0, 1], [2, 3, 0, 1, 2]],
            [[3, 2, 1, 0, 255], [2, 1, 0, 3, 2], [1, 0, 3, 2, 1]],
        ],
        dtype=torch.long,
    )
    ours_logits = base_logits.clone().requires_grad_(True)
    author_logits = base_logits.clone().requires_grad_(True)
    ours = FocalLoss2d(gamma=2, reduction="none", ignore_index=255)
    reference = author.FocalLoss2d(gamma=2, reduction="none", ignore_index=255)
    ours_map = ours(ours_logits, labels)
    author_map = reference(author_logits, labels)
    torch.testing.assert_allclose(ours_map, author_map, rtol=1e-6, atol=1e-7)
    torch.testing.assert_allclose(ours_map.mean(), author_map.mean(), rtol=1e-6, atol=1e-7)
    ours_map.mean().backward()
    author_map.mean().backward()
    torch.testing.assert_allclose(ours_logits.grad, author_logits.grad, rtol=1e-6, atol=1e-7)

    conv_a = nn.Conv2d(2, 4, 1, bias=False)
    conv_b = nn.Conv2d(2, 4, 1, bias=False)
    conv_b.load_state_dict(conv_a.state_dict())
    features = torch.randn(2, 2, 3, 5)
    ours(conv_a(features), labels).mean().backward()
    reference(conv_b(features), labels).mean().backward()
    torch.testing.assert_allclose(conv_a.weight.grad, conv_b.weight.grad, rtol=1e-6, atol=1e-7)


def test_focal_gamma_is_live_and_formal_criterion_is_unreduced():
    logits = torch.tensor([[[[2.0]], [[0.5]], [[-1.0]]]])
    labels = torch.tensor([[[0]]])
    gamma_one = FocalLoss2d(gamma=1, reduction="none")(logits, labels)
    gamma_two = FocalLoss2d(gamma=2, reduction="none")(logits, labels)
    assert not torch.allclose(gamma_one, gamma_two)

    criterion = build_author_criterion(
        SimpleNamespace(criterion="Focal", focal_gamma=2, background=255)
    )
    assert isinstance(criterion, FocalLoss2d)
    assert criterion.gamma == 2
    assert criterion.loss.reduction == "none"


def test_scheduler_matches_author_first_300_iterations():
    author = _load_author("utils/lr_policy.py", "author_rel_lr")
    ours = WarmUpPolyLR(6e-5, 0.9, 200 * 6613, 10 * 6613)
    reference = author.WarmUpPolyLR(6e-5, 0.9, 200 * 6613, 10 * 6613)
    actual = np.array([ours.get_lr(index) for index in range(300)])
    expected = np.array([reference.get_lr(index) for index in range(300)])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    actual_group_lrs = np.stack([actual, actual], axis=0)
    expected_group_lrs = np.stack([expected, expected], axis=0)
    np.testing.assert_allclose(
        actual_group_lrs, expected_group_lrs, rtol=0.0, atol=0.0
    )


def test_author_seed_formula_and_cudnn_contract_are_explicit():
    first = set_author_seed(12345, epoch=7, local_rank=3, distributed=True)
    values_a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    second = set_author_seed(12345, epoch=7, local_rank=3, distributed=True)
    values_b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second == 15352
    assert values_a == values_b
    assert os.environ["PYTHONHASHSEED"] == "15352"

    report = configure_author_cudnn()
    assert report == {"benchmark": False, "deterministic": False}
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is False


def test_author_optimizer_groups_and_defaults():
    model = nn.Sequential(nn.Conv2d(3, 4, 1, bias=True), nn.BatchNorm2d(4))
    config = SimpleNamespace(
        optimizer="AdamW", lr=6e-5, weight_decay=0.01, momentum=0.9
    )
    optimizer = build_author_optimizer(model, nn.BatchNorm2d, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_training_and_data_authorization_are_independent_fail_closed_gates():
    with pytest.raises(RuntimeError, match="Formal training is not authorized"):
        assert_training_ready(
            SimpleNamespace(training_authorized=False, data_ready=False)
        )
    with pytest.raises(RuntimeError, match="Formal data is not ready"):
        assert_training_ready(
            SimpleNamespace(training_authorized=True, data_ready=False)
        )
    assert_training_ready(SimpleNamespace(training_authorized=True, data_ready=True))
