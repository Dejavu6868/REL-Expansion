# REL+ v2.1 for Stanford2D3D S2D

This independent tree implements protocol
`RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`. It preserves every v2
`[EGVIA, LOA, ReD]` byte and hardens only camera provenance, validation,
invalid diagnostics, transforms, dtype and real CMX integration.

The formal output is canonical `480x480x3 uint8`. Native
`1080x1080 Depth16` is z-depth `raw / 512`; `0` and `65535` are
depth-invalid. The stored order is `[EGVIA, LOA, ReD]` and an invalid pixel is
`[255,255,255]`.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

Live-source regression requires the audited source paths documented in
`SOURCE_AUDIT.md`.

## Single image

```bash
python tools/generate_rel_plus.py \
  --depth /path/to/native_depth16.png \
  --camera-json /path/to/camera.json \
  --output /path/to/rel_plus_v2_1.png
```

The CLI always uses `STANFORD_S2D_PROFILE`; it cannot infer the K reference
shape from the input depth.

## Full preflight and pilot

```bash
python tools/build_full_manifest.py --output /path/to/full_manifest.csv
python tools/preflight_dataset.py \
  --manifest /path/to/full_manifest.csv \
  --output /path/to/full_preflight.csv \
  --class-mapping /data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480/class_mapping.json \
  --workers 24 --resume
python tools/generate_pilot_cache.py \
  --preflight /path/to/full_preflight.csv \
  --output-root /path/to/pilot_cache
```

The pilot tool refuses failed preflight rows and writes exactly 36 samples,
six per Area 1-6 (Area 5 combines 5a/5b). It never generates the full cache.

## CMX integration

`cmx_integration/` is the isolated real-CMX copy. Mode
`x_mode="rel_plus_v2_1"` reads stored bytes with `IMREAD_UNCHANGED`,
propagates a nearest diagnostic valid mask, disables flips and calls the shared
tested transform. The model still receives only RGB plus three-channel REL+.

The provided single-batch tool executes forward, loss and backward only. It
does not construct an optimizer, call `optimizer.step()`, start an epoch loop
or write a checkpoint.

See `REL_PLUS_V2_1_SPEC.md`, `INVALID_INPUT_CONTRACT.md`,
`AUGMENTATION_CONTRACT.md`, `CMX_PREPROCESS_AUDIT.md`,
`V2_TO_V2_1_DIFF.md`, `FULL_PREFLIGHT_REPORT.md`, `VISUAL_REVIEW_CN.md` and
`IMPLEMENTATION_REPORT_CN.md`.
