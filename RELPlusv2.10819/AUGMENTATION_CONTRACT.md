# REL+ v2.1 augmentation contract

Protocol ID: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`.

1. Decode native depth, bind native K and parse confirmed W2C pose.
2. Deterministically canonicalize depth and K to `480x480`.
3. Generate one complete REL+ v2 frame using full-frame normal support and statistics.
4. Sample one `SpatialTransform` exactly once.
5. Apply it to RGB, stored REL+, label and the diagnostic valid mask together.
6. Use `INTER_LINEAR` for RGB and REL+, `INTER_NEAREST` for label and mask.
7. Normalize RGB/REL+, crop, centred-pad (`0` normalized RGB/REL+, `255` label), then HWC-to-CHW, matching the audited CMX-REL compatibility order.

Crop/scale/pad never update K or regenerate REL+. This keeps every physical pixel's representation independent of a random crop. RGB-only colour augmentation is permitted and cannot be applied to REL+, label or depth.

`SpatialTransform` stores source and scaled dimensions and application fails
if any of the four arrays or the actual resize shape differs. RGB photometric
callbacks receive RGB only and must return HxWx3 uint8 in [0,255].

`horizontal_flip`, `vertical_flip`, `arbitrary_rotation` and
`perspective_warp` are rejected through
`validate_rel_plus_augmentation_policy()`. The real v2.1 `TrainPre`
constructs this gate at startup. Future three-arm comparisons must use the same
no-horizontal-flip setting.

The frozen configuration values are `train_horizontal_flip=False` and `eval_flip=False`.
