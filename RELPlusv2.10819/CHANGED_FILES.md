# REL+ v2.1 change inventory

This directory is a standalone implementation. No v2 or historical CMX tree is changed.

- Core: v2 plus `profiles.py`, hardened `camera.py`/`stanford_s2d.py`,
  byte-identical `generator.py`, normal diagnostics and CMX preprocessing.
- Validation: `validation/geometry_oracle.py`, `canonical_geometry.py`, `pose_physics.py`, `v1_v2_diff.py`.
- Integration: `integration/cmx_preprocess.py`.
- Tools: full manifest, resume-capable full preflight, 36-sample pilot,
  v2-v2.1 byte regression, single generation and review.
- CMX: isolated `cmx_integration/` with explicit v2.1 mode, no-flip
  `TrainPre`, diagnostic valid mask, config, loader sentinel and one-batch
  wiring check.
- Tests: K profile/cross-resolution/skew/principal checks, strong/weak/review
  pose evidence, invalid contamination, transform/dtype, frozen unit golden,
  live source, channel/RGB loader sentinels and pilot selection.
- Documents: the required v2.1 spec, diff, invalid, augmentation, CMX audit,
  full preflight, pilot visual review and Chinese implementation report.
