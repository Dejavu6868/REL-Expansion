# CMX-REL+ v2.1 Evaluation Protocol

## Canonical entry

The sole formal REL+ evaluator is `tools/eval_rel_plus_v2_1.py`, backed by
`engine/relplus_evaluator.py`. `eval.py` and `tools/eval_fold1.py` are legacy
entries and explicitly reject `x_mode=rel_plus_v2_1`.

The canonical entry obtains its dataset contract from the same
`build_data_setting(config, split="val")` function used by training and uses
`ValPre(x_mode="rel_plus_v2_1")`. A missing valid-mask file or field fails
loudly; there is no standard-mode fallback.

## S2D inference

- Input and prediction size: full-image 480x480.
- Scale list: `[1]`.
- Flip: false.
- Network upsampling: Original CMX bilinear interpolation with
  `align_corners=False`.
- Classes: 13.
- Ignore label: 255.
- Accumulation: 13x13 confusion matrix over valid semantic labels.
- The mask remains diagnostic and is never passed to `network(rgb, modal_x)`.

An optional checkpoint may contain either a top-level state dict or a `model`
field. Missing or unexpected checkpoint keys fail the canonical entry. The
delivered plumbing smoke intentionally used random model initialization; its
purpose was interface validation only.

## Prediction saving

Raw class-index PNG and palette-color PNG use explicit OpenCV and Pillow
imports and preserve nested sample IDs. The pilot save smoke wrote and reopened
both output types without a NameError.

## Interpretation boundary

The pilot report states only `evaluator plumbing smoke PASS`. It records
prediction and confusion shapes but computes and reports no mIoU or other
scientific accuracy claim.
