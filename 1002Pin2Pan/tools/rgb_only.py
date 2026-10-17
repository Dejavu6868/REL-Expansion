"""RGB-only baseline: the RGB stream of the frozen CMX network, with no X stream and no fusion.

``model_class()`` returns a subclass of the frozen ``EncoderDecoder``. It builds the dual
MiT encoder and the MLP decoder exactly as the CMX arms do, then removes the X stream
(``extra_*``) and the fusion modules (``FRMs``, ``FFMs``); the four RGB stage outputs go
straight to the decoder, as in SegFormer. Construction is unchanged, so the random draws are
the same as CMX's: at the same seed the RGB encoder and the decoder start from the same
weights as the HHA arm. The decoder keeps CMX's 512 channels (SegFormer-B2's own head has
768), so the X stream and fusion are the only difference.

``forward(rgb, modal_x, label=None)`` keeps the CMX signature and ignores ``modal_x``, so the
frozen training loop and evaluators run it unchanged on the HHA arm's data. The frozen
source must be on ``sys.path`` before ``model_class()`` is called.
"""

import functools

ARCHITECTURE = "RGB-only MiT + MLPDecoder (CMX RGB stream, no X stream or fusion)"
REMOVED_MODULES = tuple(
    "extra_{}{}".format(name, stage)
    for name in ("patch_embed", "block", "norm")
    for stage in range(1, 5)
) + ("FRMs", "FFMs")


def is_rgb_only(config):
    return config.get("architecture") == ARCHITECTURE


@functools.lru_cache(maxsize=None)
def model_class():
    import torch.nn.functional as F
    from models.builder import EncoderDecoder

    class RGBOnlyEncoderDecoder(EncoderDecoder):
        def __init__(self, cfg, criterion, norm_layer):
            if not str(cfg.backbone).startswith("mit_b") or cfg.decoder != "MLPDecoder":
                raise ValueError("RGB-only needs a MiT backbone with the MLP decoder")
            super().__init__(cfg=cfg, criterion=criterion, norm_layer=norm_layer)
            for name in REMOVED_MODULES:
                delattr(self.backbone, name)

        def encode_decode(self, rgb, modal_x):
            """The frozen forward_features with the X stream and fusion left out."""
            del modal_x
            x, batch, outs = rgb, rgb.shape[0], []
            for stage in range(1, 5):
                x, height, width = getattr(self.backbone, "patch_embed{}".format(stage))(x)
                for block in getattr(self.backbone, "block{}".format(stage)):
                    x = block(x, height, width)
                x = getattr(self.backbone, "norm{}".format(stage))(x)
                x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(x)
            out = self.decode_head.forward(outs)
            return F.interpolate(out, size=rgb.shape[2:], mode="bilinear", align_corners=False)

    return RGBOnlyEncoderDecoder
