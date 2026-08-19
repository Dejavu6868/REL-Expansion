# REL+ v2 delivered files

This directory is a standalone implementation. No v1 or production CMX file was changed.

- Core: `constants.py`, `camera.py`, `depth.py`, `source_helpers.py`, `normal_diagnostics.py`, `geometry.py`, `encoding.py`, `generator.py`, `storage.py`, `policy.py`, `stanford_s2d.py`.
- Validation: `validation/geometry_oracle.py`, `canonical_geometry.py`, `pose_physics.py`, `v1_v2_diff.py`.
- Integration: `integration/cmx_preprocess.py`.
- Tools: single generation, review selection/generation, dataset preflight, real geometry validation, v1-v2 comparison, source fixture generation and visualization.
- Tests: camera/K, units, normal policy, source offline/live regression, synthetic planes, canonical geometry, pose counterexample, gravity singularity/preflight/batch behavior, augmentation and channel sentinel.
- Documents: `README.md`, `REL_PLUS_V2_SPEC.md`, `V1_TO_V2_DIFF.md`, `AUGMENTATION_CONTRACT.md`, `SOURCE_AUDIT.md`, `IMPLEMENTATION_REPORT_CN.md`, this inventory and notices.
