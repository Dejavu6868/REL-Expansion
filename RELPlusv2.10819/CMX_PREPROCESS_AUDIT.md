# CMX preprocessing audit for REL+ v2.1

Protocol: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`.

## Sources inspected

- Public/server-clean upstream copy:
  `/home/zhuzhaoziao/RELPlus/CMX`, branch `main`, commit
  `e251d860aebc2f583a6c4919877e6bebe7f1aff3`, “Update download link for
  ZJU-RGB-P”, 2024-09-02.
- Existing server REL fork:
  `/home/zhuzhaoziao/RELPlus/CMX-REL`; it has preserved uncommitted
  REL/config/tool work and was read only.
- Current three-arm reproductions:
  `CMX-RGBD`, `CMX-HHA`, and `CMX-REL`; all were read only.
- Safe v2.1 copy: `cmx_integration/`, derived from CMX-REL without its Git
  metadata, caches or runtime products.

## Observed fork differences

Relative to clean upstream, CMX-REL adds `x_mode`, reads `rel_original` with
`cv2.IMREAD_UNCHANGED`, checks three channels, propagates `x_mode` through
the loader setting, and raises a clear missing-file error. Its train entry also
adds reproducible seeds, architecture logging and selectable focal loss.
Existing model files are not modified by the v2.1 integration.

## RGB behavior

All four inspected CMX trees call
`_open_image(rgb_path, cv2.COLOR_BGR2RGB)`. In this code the second argument
is passed directly to `cv2.imread`; there is no `cv2.cvtColor` call.
Therefore the misleading constant name does not perform a BGR-to-RGB
conversion. v2.1 preserves these returned bytes. The executable RGB sentinel
checks the pre-normalization order and the model tensor order.

## Modal and spatial behavior

- Upstream generic three-channel X also passes `cv2.COLOR_BGR2RGB` as an
  imread flag.
- Existing REL mode uses `cv2.IMREAD_UNCHANGED`.
- v2.1 adds only `x_mode="rel_plus_v2_1"`, reads REL+ unchanged, validates
  three channels and loads a single-channel diagnostic valid mask.
- Existing CMX scales RGB and X with `INTER_LINEAR`, labels with
  `INTER_NEAREST`; normalizes, chooses one crop, centre-pads and transposes
  HWC to CHW.
- v2.1 calls the tested shared adapter for that same order, adds nearest mask
  scaling and explicit source/scaled-shape checks, and returns float32 tensors.
- Existing `TrainPre` calls `random_mirror()` unconditionally. v2.1 exposes
  `train_horizontal_flip=False`, validates it at `TrainPre` construction and
  never calls mirror in this mode.

The mask remains a dataset/debug field and is intentionally absent from the
model call. Gate, DyMM, SMMF and SGA remain off; Original CMX MiT-B2,
dual encoder and MLPDecoder mathematics are unchanged.
