# Original ERP-REL reproduction

This directory contains only the Stanford2D3DPano ERP depth path needed to
reproduce the original three-channel REL array from
`SrtaEstrella/REL-SF4PASS`. It does not contain CMX integration, S2D
adaptation, REL+, training, segmentation evaluation, or full-dataset
generation.

## Provenance

- `rel_original/rel.py`: `getImage` and `getREL` numerical statements extracted
  from the public `getREL.py`; only the package import was changed.
- `rel_original/rgbd_util.py`: the called ERP geometry, normal, gravity setup,
  and rotation path extracted from public `utils/rgbd_util.py`; only the
  package import was changed.
- `rel_original/hha_util.py`: compatibility helper copied verbatim from
  `/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py` because the public repository
  imports this file but does not contain it in its inspected history. It
  supplies the called matrix/filter/gravity helpers and is not claimed to be
  an author-tracked source file.
- `reproduce_rel.py`: small manifest-driven command-line wrapper written for
  this reproduction.
- `tests/compare_with_reference.py`: observation and exact-comparison runner;
  it does not edit the reference checkout.

The copied public source is covered by the included `LICENSE`.

## Environment

Required Python packages:

```text
numpy
opencv-python
```

The verified server interpreter is:

```text
/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python
Python 3.8.16
NumPy 1.21.6
OpenCV 4.5.5
```

## Generate REL for the frozen manifest

The wrapper processes only paths explicitly listed in the manifest:

```bash
/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python reproduce_rel.py \
  --manifest /data/zhuzhaoziao/RELPlus/outputs/rel_original_reproduction_20260815_175130/sample_manifest.csv \
  --output-dir /data/zhuzhaoziao/RELPlus/outputs/rel_original_reproduction_20260815_175130/wrapper_output \
  --alpha 45 \
  --lam 0.5
```

This command does not discover or process the full dataset.

## Compare with the public reference path

```bash
/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python tests/compare_with_reference.py \
  --reference-root /home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference \
  --compatibility-hha /data/bxh_copy/Pano_MA_Seg/utils/hha_util.py \
  --manifest /data/zhuzhaoziao/RELPlus/outputs/rel_original_reproduction_20260815_175130/sample_manifest.csv \
  --output-root /data/zhuzhaoziao/RELPlus/outputs/rel_original_reproduction_20260815_175130
```

The final criteria are exact `np.array_equal` results for both the HxWx3
`uint8` REL arrays and the PNGs reloaded with `cv2.IMREAD_UNCHANGED`. Floating
intermediates are also compared exactly while treating matching NaN positions
as the same observed state; no tolerance or re-normalization is used.

## Original channel order

```text
channel 0 = angle / EGVIA
channel 1 = HA / LOA
channel 2 = RD / ReD
```

The actual source statement is `np.stack([angle, HA, RD], axis=2)`.
