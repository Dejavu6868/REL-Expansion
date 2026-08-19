# REL+ v2.1 frozen specification

Protocol ID: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`.

## Representation invariants

REL+ v2.1 is byte-identical to v2. It decodes Stanford2D3D S2D `Depth16` as
perspective z-depth `raw / 512`, treats `0` and `65535` as depth-invalid,
uses JSON half-pixel intrinsics and explicit 3x4 world-to-camera pose, and
generates one complete canonical 480x480 frame. The stored `uint8` channel
order is `[EGVIA, LOA, ReD]`; depth-invalid pixels are `[255,255,255]`.

ReD, EGVIA, LOA, alpha 45 degrees, lambda 0.5, radius 2, centimetre encoding,
full-image source statistics, uint8 truncation, gravity alignment and
depth-only production validity are unchanged from v2.

## Trusted camera profile

`STANFORD_S2D_PROFILE` binds native shape `(1080,1080)`, canonical shape
`(480,480)`, `json_half_pixel` K and `world_to_camera_3x4` pose. Production
entry points require this profile; native K shape is never inferred from the
depth array. Camera construction rejects non-finite values, non-positive focal
lengths, non-zero skew, an implausible principal point, non-orthonormal R,
`det(R) != +1`, or a camera centre inconsistent with metadata.

## Pose evidence states

- `PASS_STRONG`: global-XYZ/camera-XYZ or reprojection evidence passes.
- `PASS_WEAK`: quality-masked floor, ceiling or wall normals pass only the
  gross convention check.
- `REVIEW_REQUIRED`: semantic/normal evidence exists but is insufficient.
- `FAIL`: a supplied strong oracle or sufficient semantic convention check
  fails.
- `NOT_APPLICABLE`: no physical evidence was supplied; it is never counted as
  a pass.

The semantic mask is
`depth_valid & finite & nonzero & normal_quality`; it affects validation only.

## Training input

The formal invalid policy is `SOURCE_COMPAT_STORAGE_255`. CMX reads REL+ with
`cv2.IMREAD_UNCHANGED`, applies `INTER_LINEAR` scale, standard normalization,
crop/pad and HWC-to-CHW, and does not zero invalid pixels. A nearest-neighbour
valid mask is propagated for diagnostics but never passed into the model.
Model inputs are explicit `float32 CHW`.

Spatial transforms record source and scaled shapes and fail on a mismatch.
RGB, REL+, label and diagnostic valid mask share one transform. RGB/REL+ use
linear interpolation; label/mask use nearest interpolation. Horizontal and
vertical flips, arbitrary rotation and perspective warp are disabled.

## Scope

This protocol permits full-manifest read-only preflight, exactly 36 pilot cache
items and one CMX forward/loss/backward wiring check. It does not permit a full
cache, optimizer or scheduler steps, checkpoints, epoch loops, mIoU, or formal
training.
