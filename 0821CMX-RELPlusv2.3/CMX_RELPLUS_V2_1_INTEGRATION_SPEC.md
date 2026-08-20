# CMX-REL+ v2.1 Integration Spec

## Frozen representation

- Integration protocol: `CMX_RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`.
- Representation: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`.
- Generator source remains frozen under `/home/zhuzhaoziao/RELPlus/RELPlusv2.1`.
- Stored value is `uint8 HxWx3` in the exact order `(EGVIA, LOA, ReD)`.
- Generation is performed on the complete deterministic canonical 480 image;
  REL+ is not recomputed after a random crop.
- The byte regression report records zero changed pixels, zero changed
  channels and zero maximum difference. No file hash is used.

## Dataset contract

`x_mode` is explicitly `rel_plus_v2_1`. `dataloader/data_setting.py` is the
sole builder for train and validation settings. A REL+ config fails closed if
`x_mode`, `x_valid_root_folder` or `x_valid_format` is absent.

`RGBXDataset` reads REL+ with `cv2.IMREAD_UNCHANGED`, requires `uint8 HxWx3`,
and performs no BGR/RGB conversion on it. The mask must be a shape-matched
single-channel `uint8` or `bool` image and is converted to `bool`.

Public RGB loading retains the executable Original CMX behavior. No
variable-name-based RGB/BGR rewrite was made.

## TrainPre and ValPre

`S2D_RELPLUS_COMPARISON_NO_FLIP` samples one Python-random spatial transform
per sample and applies its scale/crop/center-pad coordinates to RGB, X, label
and mask. Interpolation is linear for RGB/X and nearest for label/mask. The
order remains resize, normalize, crop, pad, HWC-to-CHW.

The shared profile is available to future RGBD, HHA and REL+ arms. With the
same seed, epoch, rank and sample order, all three produce identical transform
traces. No comparison-arm training is part of this delivery.

`ValPre(x_mode="rel_plus_v2_1")` requires the fourth mask argument and checks
its shape. Standard mode retains the historical three-value interface.

## Invalid policy

`SOURCE_COMPAT_STORAGE_255` is frozen: stored invalid values remain 255, X is
linearly resized and normally ImageNet-normalized, and invalid values are not
zeroed. The mask is used only for synchronized augmentation, diagnostics,
visualization and audit; it is not a model input or loss mask.

## No-flip and architecture

Train and evaluation flip are false. REL+ startup validates horizontal flip,
vertical flip, arbitrary rotation and perspective warp. Original CMX remains
MiT-B2 dual encoders plus MLPDecoder with Gate, SMMF, DyMM and SGA all off.
