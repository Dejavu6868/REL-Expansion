# CMX-REL+ 2D audit

Audit date: 2026-07-23 (Asia/Shanghai). This document freezes the REL-default rerun implementation and comparison protocol before formal training.

## Repositories and provenance

- Target only: `/home/zhuzhaoziao/rel_exp/cmx_rel+`, branch `cmx-relplus-2d`, initially copied from the verified 2D CMX tree without modifying the source.
- 2D CMX source: `/home/zhuzhaoziao/rel_exp/cmx`, commit `e251d860aebc2f583a6c4919877e6bebe7f1aff3`. At audit it had only pre-existing untracked `configs/`, `scripts/`, and `tools/`.
- Local REL formula source: `/data/bxh_copy/Pano_MA_Seg`. It has no resolvable Git ref, so branch and commit are recorded as `unknown-no-git-ref`; this is a real provenance limitation.
- Independent panoramic CMX-REL reproduction: `/home/zhuzhaoziao/rel_exp/cmx_rel`, clean commit `9d614e2a7622110e7615b2e98f9dadf5b1f11d0e`. It is not a 2D comparison baseline.
- Historical perspective REL+ reference: `/home/zhuzhaoziao/rel_exp/rel_plus`, no Git provenance. It is audited as a conflicting reference and is not copied.

The source repositories, original data, previous checkpoints, and previous output directories are read-only for this experiment.

## Dataset and split

- Dataset: `/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480`.
- Train: `/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480/train.txt`, 52,903 samples, Areas 1/2/3/4/6, SHA-256 `96788184f2a1b318a05395a2c6b3867759526e0adb612a90eb6af59b1491b011`.
- Test: `/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480/test.txt`, 17,593 Area-5a/5b samples, SHA-256 `b9de196c6c1aa8f9ac37926910af0806ce59b91eb068998711ddbb78eb24423a`.
- Train/test intersection: zero.
- RGB, Depth16, RawDepth, HHA, Label, and Pose each contain 70,496 matching samples.
- RGB, label, HHA, and training inputs are 480x480. `Depth16` and K retain the original 1080x1080 calibration.
- Classes: 13; label 0 is transformed to ignore 255, labels 1..13 map to class indices 0..12.

Per-sample files share the same relative identifier:

```text
RGB/<area>/<sample>.png
Depth16/<area>/<sample>.png
Label/<area>/<sample>.png
Pose/<area>/<sample>.json
```

REL+ is generated at calibrated 1080x1080 resolution, then bilinearly resized to 480x480 with a nearest-resized validity mask. It is cached only below the new run directory.

## Depth, K, pose, and coordinate convention

The official 2D-3D-S documentation and the existing HHA preparation agree that perspective depth is camera-z depth. It is not ray distance. Valid metres are `(uint16 + 1) / 512`; raw `65535` is invalid and is explicitly removed before floating-point conversion.

Each Pose JSON supplies:

- `camera_k_matrix`: 3x3 K for 1080x1080;
- `camera_rt_matrix`: row-major 3x4 world-to-camera `[R_wc | t_wc]`;
- `camera_location`: camera center `C` in world coordinates.

Across 32 audited real samples, rotations were orthonormal and `t_wc = -R_wc C` held with maximum residual `3.363e-6`. Therefore the implemented row-vector transforms are:

```text
p_world = p_camera @ R_wc + C
p_rel = p_camera @ R_wc = p_world - C
n_world = n_camera @ R_wc
p_camera = (p_world - C) @ R_wc.T
```

Camera axes are x right, y down, z forward. World +Z is up and gravity is `G=(0,0,-1)`. There is no identity-pose fallback and inconsistent metadata raises an error.

The local perspective HHA/REL point-cloud function indexes pixels from 1 through W/H, so this implementation uses one-based pixel coordinates. K resize/crop/pad/flip functions and tests freeze that decision.

## Frozen REL+ mathematics

Output is exactly uint8 `[ReD, EGVIA, LOA]`, range 0..255. Invalid pixels are `[255,255,255]`.

For camera-centred points expressed in gravity-aligned world axes and world normals:

```text
ReD = sqrt(px^2 + py^2)
H = pz - min_valid(pz)
A = acos(N dot G)
theta = atan2(py, px)
LOA = acos(Nx cos(theta) + Ny sin(theta))
```

ReD and H use valid-image min-max scaling. A and LOA use their fixed physical 0..pi range. With `alpha=45 degrees` and `lambda=0.5`:

```text
EGVIA = 0.5*A_hat + 0.5*H_hat, when A < 45 or A > 135 degrees
EGVIA = A_hat,                         otherwise
```

Exactly 45 and 135 degrees use pure angle. The plus sign means only calibrated perspective adaptation; it adds no learnable module.

REL-default keeps the native REL camera-centred cylinder: `p_rel=p_camera@R_wc=p_world-C`. K performs perspective backprojection and the real pose rotation aligns points and normals with gravity. Translation `t/C` is still parsed and validated through `t_wc=-R_wc C`, but it is provenance-only and does not enter ReD, EGVIA, or LOA. This corrects the earlier absolute-world interpretation without changing CMX or the three REL channels.

Normals use the local perspective square-support algebraic plane estimator at radius 3, not the historical REL+ gradient/cross approximation. Real pose supplies gravity, so no per-image gravity fitting is performed. This retains the REL plane normal estimator while avoiding an unnecessary and potentially conflicting gravity estimate.

## Conflicts resolved

The old `/data/bxh_copy/Pano_MA_Seg/getHHA.py::getRLE` contains several implementation/definition conflicts: it stacks `[EGVIA,LOA,ReD]`, applies the height blend to the opposite mask, leaves LOA at 0..180, uses a sign-different LOA expression, and uses percentile height normalization. The later thread formula and current prompt take priority.

The old `/home/zhuzhaoziao/rel_exp/rel_plus/utils/relplus.py` also cannot be reused: it defaults to `[EGVIA,LOA,ReD]`, all-image fixed blending at a 30-degree threshold, 1/99-percentile normalization, gradient normals, and an incompatible pose default. These differences are covered by regression tests.

## CMX and initialization

The model is unchanged CMX MiT-B2 plus MLPDecoder. Both RGB and REL+ encoders receive the same ImageNet MiT-B2 tensors from:

`/data/zhuzhaoziao/cmx/raw/pretrained/segformer/mit_b2.pth`

SHA-256: `ced22617efb7bae3c34ad0a80f20a9b8afb4d27368cb0835a23456baa9d0e092`.

No HHA or task checkpoint initializes REL+. Decoder remains Kaiming-initialized and CM-FRM/FFM constructors are unchanged. `initialization_report.json` records all loaded tensors/parameters, missing keys, unexpected keys, unmapped checkpoint keys, shape mismatches, and the loaded parameter ratio. Any mapped backbone shape mismatch is fatal.

## Frozen 2D training protocol

- 480x480, MiT-B2, MLPDecoder, 13 classes, CE with ignore 255.
- 32 epochs, eight physical GPUs implementing four logical DDP ranks, global batch 12, and 4,409 optimizer steps per epoch. Each former three-sample logical-rank batch is split across paired physical ranks `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)` as `2+1`; the sampler reconstructs the original four-rank `DistributedSampler(seed=0)` batches exactly.
- The paired loss uses local cross-entropy sums and the scale `2 / paired_valid_pixels` before the eight-rank DDP average. This exactly preserves each logical rank's `CrossEntropyLoss(mean, ignore_index=255)` despite unequal physical batches and unequal valid-pixel counts.
- AdamW, LR 6e-5, weight decay 0.01, polynomial schedule, 10-epoch warmup.
- Synchronized RGB/REL+/label mirror, random scale `[0.5,0.75,1,1.25,1.5,1.75]`, crop and padding. Continuous modalities use bilinear interpolation; labels use nearest.
- Single-scale `[1]`, no-flip Area-5 evaluation at 480x480.
- Periodic checkpoints every four epochs. Epoch 32 is the only preregistered checkpoint eligible for the primary conclusion; `best.pth` and `last.pth` both point to epoch 32. Periodic checkpoints are not cherry-picked on Area 5.
- Configured seed is 12345. The eight physical ranks use independent execution seeds `[0,1,2,3,4,5,6,7]`; logical-rank base seeds remain `[0,1,2,3]`. Reusing each base seed on its paired rank was rejected because it would duplicate DropPath/Dropout masks and augmentation RNG streams across the `2+1` split. Consequently, sampling and loss algebra match the four-rank protocol, but the stochastic training trajectory is explicitly not bitwise equivalent to the historical four-rank runs.

The user explicitly requested the switch from four to eight physical GPUs on 2026-07-22. The first four-GPU attempt was stopped without a completed epoch checkpoint at epoch 1, iteration 1,575 and is preserved as an aborted run at `/data/zhuzhaoziao/cmx/outputs/cmx_relplus_2d_stanford2d3d_mitb2_seed12345_20260722_141229`. The completed absolute-world eight-GPU run is also retained as historical evidence only. Neither its cache nor its checkpoints may be reused by this REL-default rerun.

## Comparable baselines

- HHA: `/data/zhuzhaoziao/cmx/outputs/stanford2d3d_b2_hha_formal_seed12345_20260711_004244/checkpoints/epoch-32.pth`, SHA-256 `37df767cb312981e86e8266ff6e552263ebf7b5efc276a1d121d526c3bea0e3e`, original result 61.623 mIoU / 82.526 pixel accuracy / 70.150 mean accuracy; train/eval exit codes 0.
- RawDepth: `/data/zhuzhaoziao/cmx/outputs/stanford2d3d_b2_rawdepth_formal_seed12345_20260711_163209/checkpoints/epoch-32.pth`, SHA-256 `1f535608ec16b2d585cdd64f432d3dcd2a34cc20ddca4504019fc6acdb2b295b`, original result 58.015 / 80.017 / 66.486; train/eval exit codes 0.

Both checkpoints are re-evaluated by this target code, and deltas are emitted only if the machine-readable fairness gate confirms every frozen field except X modality/root/encoding matches.

Panoramic Fold-1 CMX-REL results are explicitly excluded from the 2D delta table.

## Environment at audit

- 8x RTX 3090, all idle; no process is killed or preempted.
- Driver 595.71.05; system CUDA 13.2.
- Experiment Python `/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python`: Python 3.8.16, PyTorch 1.8.2, CUDA 11.1, cuDNN 8005.
- Root filesystem approximately 3.2 TB free; `/data` approximately 4.4 TB free.

The formal run records a fresh environment snapshot and all code/input hashes.
