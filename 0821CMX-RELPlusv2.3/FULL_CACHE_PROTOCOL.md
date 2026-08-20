# Full REL+ cache protocol

## Generation

`tools/generate_full_relplus_cache.py` reads the frozen manifest and calls only
the frozen `load_canonical_frame()` and `generate_rel_plus_v2_1()` paths. It
does not reimplement depth decoding, intrinsic resizing, pose, gravity,
normals or REL+ encoding.

Each Python worker calls `cv2.setNumThreads(1)`. Every PNG is written to a
same-directory temporary file, decoded and checked, then committed with
`os.replace()`. Resume skips a pair only when REL+ and mask both decode with
the expected shapes, dtypes, channel counts and binary-mask contract. A
missing, corrupt or wrong-shape member is regenerated atomically.

Repository authorization stays false. Full execution additionally requires
the explicit `--authorize-full-cache` command flag. `full_cache_generated` is
true only for a non-dry full-manifest run with zero failures and exactly one
generated-or-verified result per manifest row.

## Audit

`tools/audit_full_relplus_cache.py` checks all 70,496 rows and both cache trees:

- O(N) Counter duplicate detection;
- exact train/test counts, ordered list identity, uniqueness and disjointness;
- absolute cache, manifest, resolved-manifest and split-list identity;
- missing/extra files, decode, shape, dtype, channels and binary masks;
- invalid-mask pixels stored as `[255,255,255]`;
- train/test RGB, label, depth and metadata paths remain disjoint.

It joins risk fields by exact sample ID to the existing full preflight CSV and
selects ten rows per area across low/median/P90/high invalid ratio, low/median/
high normal quality, large gravity tilt, a distinct room/camera and a fixed
random sample. All 70 are regenerated from source depth/camera data and must
have zero changed pixels, channels and maximum difference.

## Full CMX preflight

`tools/preflight_cmx_training_data_v2_3.py` then decodes RGB, label, REL+ and
valid mask for all rows. Label IDs are accepted only through the real dataset
`class_mapping.json`. Formal execution does not use preflight resume, so the
summary must state that every sample was decoded in the current run.

Actual generation, audit and preflight outcomes are reported separately in
`FULL_CACHE_GENERATION_REPORT.md`, `FULL_CACHE_AUDIT_REPORT.md` and
`FULL_TRAINING_DATA_PREFLIGHT_REPORT.md`.
