"""CMX's MLP decoder head with its fuse convolution run before the upsampling: the same function, fewer MACs.

``DecoderHead`` projects the four encoder maps to E channels, resizes three of them to the 1/4-size map,
concatenates the four and fuses them with a 1 x 1 convolution over 4E channels at 1/4 size. That convolution
alone is 65,536 multiply-adds per input pixel (E = 512), 28% of the network at a 445 x 295 input.

A 1 x 1 convolution mixes channels and a bilinear resize mixes positions, so the two commute:

    fuse(cat(up(L4 c4), up(L3 c3), up(L2 c2), L1 c1)) = up(A4 c4) + up(A3 c3) + up(A2 c2) + A1 c1 + b

with ``A_i = W_i L_i`` (``W_i`` is the block of the fuse weight that reads map i, ``L_i`` the map's projection)
and ``b`` the fuse bias plus every ``W_i`` times its projection's bias (a resize leaves a constant unchanged).
Each ``A_i`` runs at its own map's size, so the 4E-channel convolution at 1/4 size is gone. Nothing is retrained
and no weight file changes; the outputs differ only by float rounding.

``fold_decoder(network)`` swaps the head of a built network. For the first ``check_inputs`` network inputs the
original head runs as well and the two outputs are compared; the result is in ``head.check``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECK_INPUTS = 16      # network inputs on which the original head also runs
TOLERANCE = 1e-3       # largest logit difference allowed in that check (float rounding is about 1e-5)


class FoldedDecoderHead(nn.Module):
    def __init__(self, head, check_inputs=CHECK_INPUTS):
        super().__init__()
        fuse = getattr(head, "linear_fuse", None)
        maps = [getattr(head, name, None) for name in ("linear_c4", "linear_c3", "linear_c2", "linear_c1")]
        if (not isinstance(fuse, nn.Sequential) or len(fuse) != 3 or not isinstance(fuse[0], nn.Conv2d)
                or fuse[0].kernel_size != (1, 1) or fuse[0].groups != 1 or not isinstance(fuse[2], nn.ReLU)
                or any(not isinstance(getattr(mlp, "proj", None), nn.Linear) for mlp in maps)
                or fuse[0].in_channels != sum(mlp.proj.out_features for mlp in maps)):
            raise ValueError("only the MLP decoder head (four projections, a 1 x 1 fuse convolution, a norm and a "
                             "ReLU) can be folded")
        conv, self.norm, self.linear_pred = fuse[0], fuse[1], head.linear_pred
        self.align_corners = head.align_corners
        weight = conv.weight.detach().double()[:, :, 0, 0]  # (E, 4E): the blocks read c4, c3, c2, c1 in this order
        bias = torch.zeros(conv.out_channels, dtype=torch.float64, device=weight.device)
        if conv.bias is not None:
            bias = bias + conv.bias.detach().double()
        self.projections = nn.ModuleList()
        start = 0
        for mlp in maps:
            block = weight[:, start:start + mlp.proj.out_features]
            start += mlp.proj.out_features
            layer = nn.Conv2d(mlp.proj.in_features, conv.out_channels, kernel_size=1, bias=False)
            layer.weight.data = (block @ mlp.proj.weight.detach().double()).float()[:, :, None, None]
            if mlp.proj.bias is not None:
                bias = bias + block @ mlp.proj.bias.detach().double()
            self.projections.append(layer)
        self.bias = nn.Parameter(bias.float(), requires_grad=False)
        self.train(head.training)
        self.reference = head if check_inputs else None
        self.check_inputs = check_inputs
        self.check = {"network_inputs_compared": 0, "largest_logit": 0.0, "largest_logit_difference": 0.0,
                      "positions": 0, "positions_with_another_class": 0}

    def folded(self, inputs):
        c1, c2, c3, c4 = inputs
        fused = self.projections[3](c1)
        for layer, feature in zip(self.projections[:3], (c4, c3, c2)):
            fused = fused + F.interpolate(layer(feature), size=c1.shape[2:], mode="bilinear",
                                          align_corners=self.align_corners)
        return self.linear_pred(F.relu(self.norm(fused + self.bias[None, :, None, None])))

    def forward(self, inputs):
        if self.training:
            raise RuntimeError("the folded decoder is for evaluation only")
        out = self.folded(inputs)
        if self.reference is not None:
            original = self.reference(inputs)
            difference = float((out - original).abs().max())
            check = self.check
            check["network_inputs_compared"] += len(out)
            check["largest_logit"] = max(check["largest_logit"], float(original.abs().max()))
            check["largest_logit_difference"] = max(check["largest_logit_difference"], difference)
            check["positions"] += out[:, 0].numel()
            check["positions_with_another_class"] += int((out.argmax(1) != original.argmax(1)).sum())
            if difference > TOLERANCE:
                raise RuntimeError("the folded decoder differs from the original by {} in a logit".format(difference))
            if check["network_inputs_compared"] >= self.check_inputs:
                self.reference = None
        return out


def fold_decoder(network, check_inputs=CHECK_INPUTS):
    """Replace ``network.decode_head`` by its folded form (weights must be loaded already); returns the new head."""
    network.decode_head = FoldedDecoderHead(network.decode_head, check_inputs)
    return network.decode_head
