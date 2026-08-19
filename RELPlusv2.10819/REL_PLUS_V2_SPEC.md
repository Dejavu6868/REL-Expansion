# REL+ v2 frozen specification

## Identity and output

REL+ v2 is the executable REL encoding adapted to Stanford2D3D S2D perspective geometry. Output is `HxWx3 uint8` in stored order `[EGVIA, LOA, ReD]`. Only depth invalidity sets the whole pixel to `[255,255,255]`.

## Depth and camera

- Depth16 is perspective z-depth: `depth_m = raw_uint16 / 512`.
- Raw `0` and `65535` are invalid and decoded to zero metres.
- Native depth is deterministically resized to `480x480` with OpenCV `INTER_NEAREST`.
- `K_json` follows Stanford JSON half-pixel coordinates: pixel centres are `(u+0.5, v+0.5)`.
- Backprojection is `X=(u+0.5-cx)z/fx`, `Y=(v+0.5-cy)z/fy`, `Z=z`.
- The source one-based helper receives `cx+0.5, cy+0.5` exactly once.
- Every K carries `intrinsics_shape=(height,width)`. Generation rejects a different depth shape. Resizing K uses only that bound source shape and rebinds the destination.

## Pose, gravity and normals

- Pose is explicit `X_camera = R_world_to_camera X_world + t_world_to_camera` from a confirmed `3x4` JSON matrix.
- Camera centre is `C_world = -R.T @ t` and must match `camera_location`.
- World down is `[0,0,-1]`; camera gravity is `R_world_to_camera @ world_down`.
- Alignment preserves source `getRMatrix(target, source)` and `rotatePC(..., R.T)` behavior.
- The original `getRMatrix` can produce undefined values for anti-parallel axes. REL+ v2 intentionally deviates only there: gravity within the frozen anti-parallel threshold of 180 degrees raises `GravityAlignmentSingularity`; no arbitrary axis fallback exists.
- Normals preserve the perspective square-support source helper at canonical radius 2.
- Finite, nonzero and support-count masks are diagnostics only. The formal encoding mask remains `depth_valid`.

## Encoding

- Geometry/backprojection/alignment arrays remain metres.
- Only the encoding input is converted: `points_for_encoding_cm = points_aligned_m * 100`.
- ReD is horizontal radius followed by full-image min/max, including invalid-as-zero source values.
- EGVIA uses full-image P1/P99 height, `alpha=45`, `lambda=0.5`, and blends only `~is_horizontal`.
- LOA uses the perspective horizontal tangent `[ry,-rx,0]`, remains degrees in `[0,180]`, and is truncated directly to uint8.
- NaN normal at valid depth yields channel-local source behavior: EGVIA 255, LOA 90, ReD valid.
- Zero normal remains finite source data; it is not converted to NaN or made depth-invalid.
- Public generator parameters are frozen; alpha, lambda and radius are not exposed.

## Storage and augmentation

- Save and load with OpenCV and no BGR/RGB conversion.
- Generate one complete canonical frame before random training transforms.
- Apply the same sampled scale/crop/pad to RGB, REL+ and label.
- Compatibility interpolation is linear for RGB/REL+ and nearest for labels.
- Normalization, crop, centred pad and HWC-to-CHW follow the audited CMX-REL order.
- RGB-only photometric augmentation cannot touch REL+, label or depth.
- Horizontal/vertical flips, arbitrary rotation and perspective warp are rejected.

## Validation gate

Native and canonical global-XYZ checks report component, Euclidean and reprojection statistics. Canonical reprojection P95 is a required PASS condition. Pose physical checks are independent of generation and return `PASS`, `FAIL` or `NOT_APPLICABLE`. A full cache remains prohibited until these gates pass.
