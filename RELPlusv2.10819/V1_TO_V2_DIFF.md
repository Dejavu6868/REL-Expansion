# Controlled v1 to v2 differences

| Change | v1 | v2 | Reason | Expected numeric effect |
|---|---|---|---|---|
| Encoding point unit | metres | centimetres | executable source identity | normally scale-invariant; changes degenerate raw-value branches |
| Formal valid mask | depth and finite normal | depth only | source `missingMask` derives from depth | changes depth-valid abnormal-normal pixels only |
| Normal return | normals plus validity mask, with rewriting | untouched source normals plus diagnostics | preserve NaN/zero behavior | diagnostics do not overwrite bytes |
| K resolution | caller-provided source shape | K-bound `intrinsics_shape` | prevent silent mismatch | no normal-data byte change |
| Canonical validation | native XYZ plus K arithmetic | native and nearest-mapped canonical XYZ | validate executed 480 path | evidence only |
| Pose validation | algebra and sampled XYZ | independent strong/weak physical layer and transpose counterexample | catch self-consistent wrong pose | evidence only |
| Gravity singularity | generic failure | typed exception, preflight and per-sample batch record | avoid arbitrary 180-degree yaw | normal data unchanged |
| Parameters | public alpha/lambda/radius options | frozen public identity | prevent accidental variants | defaults unchanged |
| CMX policy | declaration only | executed compatibility adapter | unambiguous next-step interface | no training in this task |

The executed six-case comparison reported dense-valid and depth-invalid byte equality. One NaN-normal pixel changed under `INTENTIONAL_NORMAL_MASK_FIX`; one single-pixel degenerate case changed under `INTENTIONAL_SOURCE_UNIT_FIX`; unexpected differences were zero. The exact tables are emitted by `tools/compare_v1_v2.py`.
