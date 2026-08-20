# CMX-REL integration report

## 1. Task identity

```text
Representation: Original panoramic REL executable path
Architecture: Original CMX
Backbone: MiT-B2
Decoder: MLPDecoder
Classes: 13
X modality: three-channel REL
Gate: disabled
SMMF: disabled
DyMM: disabled
REL+: disabled
Perspective REL: disabled
```

This was an integration smoke test, not a performance reproduction. No mIoU
was computed or inferred.

## 2. Code sources

- Official CMX baseline: `/home/zhuzhaoziao/RELPlus/CMX`
- CMX remote/branch: `https://github.com/huaaaliu/RGBX_Semantic_Segmentation.git`, `main`
- CMX latest date/title: `2024-09-02 22:15:42 +0800`, `Update download link for ZJU-RGB-P`
- Already verified local REL: `/home/zhuzhaoziao/RELPlus/REL`
- Public REL reference: `/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference`
- Reference remote/branch: `https://github.com/SrtaEstrella/REL-SF4PASS.git`, `main`
- Reference latest date/title: `2026-06-01 09:58:29 +0800`, `Remove unused _gitignore template`
- Integrated REL source: `third_party/rel_original/`

The public reference's missing `utils/hha_util.py` is supplied by the same
compatibility file used by the already verified local reproduction. This
provenance limitation is documented in `REL_SOURCE_ALIGNMENT.md`.

## 3. REL semantic audit

### Input and geometry

- Input: complete `2048 x 4096` ERP depth PNG, not a crop.
- Raw type: `uint16`.
- Actual public read path: OpenCV flag value 6, in-place `D += 1`, conversion
  to `float32`, division by 512.
- Invalid behavior: raw `65535` wraps to zero during the increment and is then
  marked missing.
- ERP point cloud: longitude/latitude rays from the full panorama.
- Normal: `computeNormalsSquareSupport_ERP`, radius 2.
- Gravity: initial `[0,0,-1]`, thresholds `[15,5]`, iterations `[5,5]`.
- Rotation: `getRMatrix` then `rotatePC` for normals and points.

### Channels

- ReD: planar radius of the rotated point cloud, per-image min/max encoded.
- EGVIA: vertical angle plus code-defined height blend.
- LOA: ERP longitude and rotated horizontal normal relation.
- `alpha=45`, `lambda=0.5`.
- Core output: `uint8`, `[0,255]`, channel order `[EGVIA, LOA, ReD]`.
- Missing pixels: all three channels set to 255.

The supplied paper summary says EGVIA blends near horizontal surfaces, while
the public/local code blends `~is_horizontal`; the supplied height definition
also differs from the public code's percentile encoding. This implementation
preserves the verified executable source and records the conflict rather than
silently changing it.

By project decision on 2026-08-16, this verified executable behavior is the REL
implementation standard for future work. The paper/code difference remains a
provenance fact, but is no longer an unresolved project-definition blocker. See
`REL_IMPLEMENTATION_STANDARD.md`.

### OpenCV round trip

The file's RGB components visually correspond to `R=ReD`, `G=LOA`,
`B=EGVIA`. `IMREAD_UNCHANGED` returns array indices `[EGVIA, LOA, ReD]`.
CMX-REL does not apply `BGR2RGB` to the REL array, so the final PyTorch tensor
remains `[EGVIA, LOA, ReD]`.

## 4. Isolation from REL+

The called implementation contains no perspective intrinsic, perspective
extrinsic, FoV, pitch/roll/yaw, crop-aware intrinsic, S2D XYZ, pose gravity or
REL+ floor reference. It also contains no Gate, SMMF, DyMM, region slicing,
soft/hard gate training or temperature scheduler.

## 5. CMX integration

- REL is generated offline from full ERP depth by
  `tools/generate_stanford2d3d_rel.py`.
- `RGBXDataset` reads the saved PNG as an unchanged three-channel array.
- `x_is_single_channel=False`, `in_chans=3`, `in_chans_x=3`.
- RGB, REL and label share one mirror decision, scale, crop position and
  padding path. RGB/REL use linear interpolation; label uses nearest-neighbor.
- RGB and REL both retain official CMX ImageNet normalization.
- REL enters `extra_patch_embed1`, the original X encoder, then the original
  CM-FRM/FFM and decoder.
- `models/` is identical to the official CMX baseline.

## 6. Test result

- Real S3D samples: 3.
- Integrated core versus public executable path: exact equality for all 3.
- Generated PNG versus in-memory integrated output: exact equality for all 3.
- Each channel: non-constant, finite, range 0–255.
- Real data batch: RGB `[1,3,256,256]`, REL `[1,3,256,256]`, label
  `[1,256,256]`.
- Valid label values observed: `3,4,5,6,8,11,255` after conversion.
- Logits: `[1,13,256,256]`, finite.
- Loss: `2.615638494491577`, finite.
- Backward: passed.
- One optimizer step: passed; an in-memory RGB encoder parameter changed.
- RGB encoder gradient: finite and nonzero.
- REL encoder gradient: finite and nonzero.
- CM-FRM/FFM gradient: finite and nonzero.
- Decoder gradient: finite and nonzero.
- Checkpoint saved or replaced: no.

Parameter counts observed in the unchanged model:

| Group | Parameters |
|---|---:|
| Total | 66,567,573 |
| RGB encoder | 24,196,288 |
| REL encoder | 24,196,288 |
| CM-FRM/FFM | 16,591,880 |
| Decoder | 1,583,117 |

Evidence:

```text
/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/tests/smoke.log
/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/tests/smoke_results.json
/data/zhuzhaoziao/RELPlus/outputs/cmx_rel_integration/tests/channel_visualization/
```

## 7. Not performed

- Full-dataset REL generation
- Any complete epoch
- 200-epoch training
- Fold 1 formal evaluation
- mIoU reproduction
- Fold 2 or Fold 3
- SMMF, Gate, DyMM or REL+

## 8. Next-step judgment

```text
PROJECT_REL_IMPLEMENTATION_STANDARD: ACCEPTED
PAPER_CODE_DIFFERENCE: DOCUMENTED
FORMULA_SCOPE_BLOCKER: RESOLVED_BY_PROJECT_DECISION
```

The CMX wiring and executable-source equivalence have passed. The project now
accepts the public/local executable EGVIA condition and percentile height
encoding as its REL definition. Formal Fold 1 is therefore no longer blocked by
formula ambiguity, but it remains unperformed and requires a separate execution
decision. No performance claim follows from this scope decision.
