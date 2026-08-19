# REL+ v2 augmentation contract

1. Decode native depth, bind native K and parse confirmed W2C pose.
2. Deterministically canonicalize depth and K to `480x480`.
3. Generate one complete REL+ v2 frame using full-frame normal support and statistics.
4. Sample one `SpatialTransform` exactly once.
5. Apply it to RGB, stored REL+ and label together.
6. Use `INTER_LINEAR` for RGB and REL+, `INTER_NEAREST` for label.
7. Normalize RGB/REL+, crop, centred-pad (`0` normalized RGB/REL+, `255` label), then HWC-to-CHW, matching the audited CMX-REL compatibility order.

Crop/scale/pad never update K or regenerate REL+. This keeps every physical pixel's representation independent of a random crop. RGB-only colour augmentation is permitted and cannot be applied to REL+, label or depth.

`horizontal_flip`, `vertical_flip`, `arbitrary_rotation` and `perspective_warp` are all rejected by the adapter through `validate_rel_plus_augmentation_policy()`. Future three-arm CMX-RGBD/CMX-HHA/CMX-REL+ fairness work must use the same no-horizontal-flip setting. This document defines an interface only; no training was executed.

The frozen configuration values are `train_horizontal_flip=False` and `eval_flip=False`.
