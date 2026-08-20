# REL+ v2.1 invalid input contract

Protocol: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`.

## Five distinct layers

1. Storage invalid: a canonical depth-invalid pixel is saved as
   `[255,255,255]` in stored `[EGVIA, LOA, ReD]` order.
2. Geometry invalid: only decoded `depth_valid` controls the formal encoding
   mask. Normal finite/nonzero/support masks remain diagnostics.
3. Augmentation validity: the canonical boolean depth-valid mask is resized by
   nearest neighbour and follows the same crop/pad transform as the data.
4. Formal model input: REL+ is resized by CMX `INTER_LINEAR`, normalized with
   the common three-channel mean/std, and is not masked or zeroed.
5. Diagnostic reference: a mask-aware bilinear reference may be computed only
   to quantify boundary contamination. It never replaces formal `modal_x`.

## Why storage uses 255

This is the frozen source-compatible baseline inherited from v2. It preserves
the executable byte contract and makes no claim that 255 is physically optimal.
Linear scaling can mix this value into neighbouring valid pixels. The
`analyze_invalid_interpolation()` diagnostic reports source/nearest-invalid
ratios, boundary and affected ratios, channel deviation, and normalized-value
quantiles.

The nearest valid mask is returned as `modal_x_valid_mask` for audit and
visualization. It is not a fourth channel, is not passed to CMX forward, and
does not alter the loss. Mask-zero, masked bilinear and mask-as-input variants
would require a new method name and are outside v2.1.
