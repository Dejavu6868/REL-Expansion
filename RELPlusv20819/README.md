# REL+ v2 for Stanford2D3D S2D

This is the independent source-aligned v2 implementation derived from the reviewed `RELPlus0818` baseline. It fixes only the frozen v2 differences and does not modify v1 or production CMX.

The formal byte contract is `HxWx3 uint8`, stored as `[EGVIA, LOA, ReD]`; a depth-invalid pixel is `[255,255,255]`. Stanford S2D depth is z-depth in metres (`raw / 512`), with raw `0` and `65535` invalid. A camera K is explicitly bound to its reference `(height, width)` and generation fails on a mismatch.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Single image:

```bash
python tools/generate_rel_plus.py \
  --depth /path/to/native_depth16.png \
  --camera-json /path/to/camera.json \
  --output /path/to/rel_plus_v2.png \
  --debug-dir /path/to/debug
```

Review workflow:

```bash
python tools/generate_review_samples.py --output-root /path/to/review --select-only
python tools/preflight_dataset.py --manifest /path/to/review/selected_manifest.csv --output /path/to/preflight.csv
python tools/generate_review_samples.py --manifest /path/to/review/selected_manifest.csv --output-root /path/to/review --limit 12
python tools/validate_real_geometry.py --manifest /path/to/review/real_samples_manifest.csv --output-root /path/to/review
```

`rel_plus_display_only.png` is for human inspection only. A standard viewer displays stored OpenCV bytes as RGB `[ReD, LOA, EGVIA]`; model loading must use `cv2.IMREAD_UNCHANGED` without colour conversion.

The CMX compatibility adapter first consumes a complete canonical REL+ image, then applies one shared scale/crop/pad transform to RGB, REL+ and label. It never regenerates geometry after random augmentation and rejects every flip, arbitrary rotation and perspective warp.

See `REL_PLUS_V2_SPEC.md`, `V1_TO_V2_DIFF.md`, `AUGMENTATION_CONTRACT.md`, `SOURCE_AUDIT.md`, and `IMPLEMENTATION_REPORT_CN.md`.

This repository does not start training, perform backpropagation or optimizer updates, create checkpoints, generate a full cache, or calculate mIoU.
