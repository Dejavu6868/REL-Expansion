"""decoder_fold: the fuse convolution run before the upsampling returns the original decoder's logits."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_fold  # noqa: E402
import eval_pano_transfer as ev  # noqa: E402


def trained_like_head():
    ev.import_frozen(ev.DEFAULT_SOURCE_ROOT)
    from models.decoders.MLPDecoder import DecoderHead

    torch.manual_seed(0)
    head = DecoderHead(in_channels=[8, 16, 24, 32], num_classes=13, norm_layer=nn.BatchNorm2d, embed_dim=16).eval()
    norm = head.linear_fuse[1]  # a fresh BatchNorm is the identity and fresh biases are small: make them count
    norm.running_mean.normal_(0, 0.5), norm.running_var.uniform_(0.5, 2.0)
    norm.weight.data.uniform_(0.5, 1.5), norm.bias.data.normal_(0, 0.5)
    for module in (head.linear_c1, head.linear_c2, head.linear_c3, head.linear_c4):
        module.proj.bias.data.normal_(0, 0.5)
    head.linear_fuse[0].bias.data.normal_(0, 0.5)
    return head


def features(batch=2):  # odd sizes: the three resizes are not by whole factors
    return [torch.randn(batch, channels, height, width)
            for channels, (height, width) in zip((8, 16, 24, 32), ((23, 19), (12, 10), (6, 5), (3, 3)))]


def test_folded_decoder_returns_the_original_logits_and_checks_itself():
    head = trained_like_head()
    network = SimpleNamespace(decode_head=head)
    folded = decoder_fold.fold_decoder(network, check_inputs=3)
    assert network.decode_head is folded and folded.reference is head
    with torch.no_grad():
        inputs = features()
        out = folded(inputs)
        assert out.shape == (2, 13, 23, 19) and torch.allclose(out, head(inputs), atol=1e-5)
        assert folded.check["network_inputs_compared"] == 2 and folded.reference is head  # 2 of 3 compared so far
        folded(features())
        assert folded.check["network_inputs_compared"] == 4 and folded.reference is None  # the check is over
        done = dict(folded.check)
        assert done["positions"] == 4 * 23 * 19 and done["positions_with_another_class"] == 0
        assert 0 < done["largest_logit_difference"] < 1e-5 < done["largest_logit"]
        inputs = features(1)
        assert torch.allclose(folded(inputs), head(inputs), atol=1e-5) and folded.check == done
    assert sum(layer.weight.numel() for layer in folded.projections) == 16 * (8 + 16 + 24 + 32)  # no 4E -> E layer


def test_a_wrong_fold_is_caught_by_the_check():
    folded = decoder_fold.FoldedDecoderHead(trained_like_head())
    folded.bias.data += 1
    with torch.no_grad(), pytest.raises(RuntimeError, match="differs from the original"):
        folded(features())


def test_only_the_mlp_head_in_evaluation_mode_is_folded():
    with pytest.raises(ValueError, match="MLP decoder head"):
        decoder_fold.FoldedDecoderHead(nn.Conv2d(3, 3, 1))
    folded = decoder_fold.FoldedDecoderHead(trained_like_head(), check_inputs=0)
    assert folded.reference is None
    with pytest.raises(RuntimeError, match="evaluation only"):
        folded.train()(features())
