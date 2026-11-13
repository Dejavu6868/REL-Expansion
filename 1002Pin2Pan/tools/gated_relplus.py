"""Gated REL+: the frozen CMX network with SMMF's GlobalGate, for the gated control arm.

The gate is Max's (``Pano_MA_Seg/models/encoders/dual_gate_segformer.py``: ``GlobalGate``, ``DiffSoftmax`` and
the gated ``forward_features``), copied without changes except that the gate's input width follows the backbone
(128 for MiT-B2, as in his code). On the first patch embeddings of both streams it scores five branches. Branch g
sends FFM(RGB, X) to the decoder at stages 1..g and the FRM-rectified RGB feature at the other stages. Both
streams and the FRMs always run.

- Training (``self.training``): every stage blends the two by the gate weights, as Max's train mode does. The RGB
  feature gets w0 at stage 1 and w0 + w1 at stage 2, and the FFM gets w3 + w4 at stage 3 and w4 at stage 4. The
  weights are ``DiffSoftmax`` at temperature ``temp``: soft before ``EPOCH_HARD``, straight-through one-hot from
  then on. ``schedule(epoch)`` gives both values, from Max's config (temperature 1.0 to 0.01 from epoch 1 to 100,
  then ``ExpDecayTemp`` keeps decaying).
- Evaluation: g = argmax per sample, so each window in a batch gets its own choice, and a stage's FFM runs only
  for the windows that use it (so the cost per choice is that of Max's forward). The choices are appended to
  ``gate_log`` in batch order.

The gate is built after the frozen network, so at the same seed the CMX weights start where the vanilla arm's do.
``forward(rgb, modal_x, label=None)`` keeps the CMX signature, so the frozen training loop and the evaluators run
it unchanged. The frozen source must be on ``sys.path`` before ``model_class()`` is called.
"""

import functools
import math

ARCHITECTURE = "Gated CMX (SMMF GlobalGate, 5 branches) + MLPDecoder"
BRANCHES = 5
EPOCH_HARD = 100  # Max's C.epoch_hard: soft gate for epochs < 100
START_TEMP, END_TEMP, TEMP_START_EPOCH = 1.0, 0.01, 1  # C.start_temp, C.end_temp, C.tempsched_start_epoch


def is_gated(config):
    return config.get("architecture") == ARCHITECTURE


def schedule(epoch):
    """(soft_gate, temp) for a 1-based training epoch, as Max's train.py sets them (ExpDecayTemp)."""
    time_len = EPOCH_HARD - TEMP_START_EPOCH
    b = math.exp(1 / time_len * math.log(END_TEMP / START_TEMP))
    temp = START_TEMP if epoch <= TEMP_START_EPOCH else START_TEMP * b ** (epoch - TEMP_START_EPOCH)
    return epoch < EPOCH_HARD, temp


@functools.lru_cache(maxsize=None)
def model_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from models.builder import EncoderDecoder

    def diff_softmax(logits, tau=1.0, hard=False, dim=-1):
        y_soft = (logits / tau).softmax(dim)
        if hard:  # straight through
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    class GlobalGate(nn.Module):
        def __init__(self, in_channels=128, branch_num=BRANCHES, hidden_dim=8):
            super().__init__()
            self.bnum = branch_num
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=5, stride=2),
                nn.BatchNorm2d(hidden_dim),
                nn.Tanh(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=2),
                nn.BatchNorm2d(hidden_dim),
                nn.Tanh())
            self.fc = nn.Conv2d(hidden_dim, self.bnum, kernel_size=1, bias=False)

        def forward(self, rgb, depth, temp=1.0, hard=False):
            y = self.conv(torch.cat([rgb, depth], dim=1))
            y = F.adaptive_avg_pool2d(y, 1)
            y = self.fc(y)
            y = diff_softmax(y, tau=temp, hard=hard, dim=1)
            return y.squeeze(-1).squeeze(-1)

    class GatedEncoderDecoder(EncoderDecoder):
        def __init__(self, cfg, criterion, norm_layer):
            if not str(cfg.backbone).startswith("mit_b") or cfg.decoder != "MLPDecoder":
                raise ValueError("gated REL+ needs a MiT backbone with the MLP decoder")
            super().__init__(cfg=cfg, criterion=criterion, norm_layer=norm_layer)
            if self.aux_head:
                raise ValueError("gated REL+ has no auxiliary head")
            self.backbone.Gate = GlobalGate(2 * self.channels[0])
            self.soft_gate, self.temp = schedule(1)
            self.gate_log = []

        def encode_decode(self, rgb, modal_x):
            """The frozen forward_features with Max's gate on the decoder inputs."""
            b = self.backbone
            batch, outs = rgb.shape[0], []
            x_rgb, x_e = rgb, modal_x
            for stage in range(4):
                x_rgb, height, width = getattr(b, "patch_embed{}".format(stage + 1))(x_rgb)
                x_e, _, _ = getattr(b, "extra_patch_embed{}".format(stage + 1))(x_e)
                if stage == 0:
                    weight = b.Gate(
                        x_rgb.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous(),
                        x_e.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous(),
                        temp=self.temp, hard=not self.soft_gate if self.training else True,
                    )
                    gate = torch.argmax(weight, dim=1)
                    if not self.training:
                        self.gate_log.extend(gate.tolist())
                for block in getattr(b, "block{}".format(stage + 1)):
                    x_rgb = block(x_rgb, height, width)
                for block in getattr(b, "extra_block{}".format(stage + 1)):
                    x_e = block(x_e, height, width)
                x_rgb = getattr(b, "norm{}".format(stage + 1))(x_rgb)
                x_e = getattr(b, "extra_norm{}".format(stage + 1))(x_e)
                x_rgb = x_rgb.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
                x_e = x_e.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
                x_rgb, x_e = b.FRMs[stage](x_rgb, x_e)
                if self.training:
                    fused = b.FFMs[stage](x_rgb, x_e)
                    if stage < 2:  # stages 1, 2: the RGB feature weighted by w0 (+ w1)
                        w = weight[:, :stage + 1].sum(1).view(-1, 1, 1, 1)
                        outs.append(w * x_rgb + (1 - w) * fused)
                    else:  # stages 3, 4: the FFM weighted by w3 + w4, then w4
                        w = weight[:, stage + 1:].sum(1).view(-1, 1, 1, 1)
                        outs.append((1 - w) * x_rgb + w * fused)
                    continue
                use = gate > stage  # the FFM runs only for the samples that use it, as in Max's batch-1 forward
                if bool(use.all()):
                    outs.append(b.FFMs[stage](x_rgb, x_e))
                elif bool(use.any()):
                    out = x_rgb.clone()
                    out[use] = b.FFMs[stage](x_rgb[use], x_e[use])
                    outs.append(out)
                else:
                    outs.append(x_rgb)
            out = self.decode_head.forward(outs)
            return F.interpolate(out, size=rgb.shape[2:], mode="bilinear", align_corners=False)

    return GatedEncoderDecoder
