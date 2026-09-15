# Full cache audit report

## Final status

`PASS` (attempt2, exit code 0)

## Preserved failed attempt

Attempt1 ran from approximately 00:09:34 to 00:18:34 and exited 1 before
writing a summary. The traceback is preserved in
`logs/full_cache_audit_attempt1.log`. It exposed an auditor implementation
error: the real manifest uses `area=area_5a/area_5b` while deliberately
collapsing `area_group=area_5`; the selector incorrectly preferred
`area_group` and reported fewer than ten rows.

The real field structure was added to a regression test. The focused test was
RED before the one-line grouping fix and GREEN after it. No cache file was
changed. Attempt2 then reran the complete audit rather than reusing attempt1.

## Successful full audit

Attempt2 ran from approximately 00:21:02 to 00:30:11 and exited 0.

- Integration protocol: `CMX_RELPLUS_V2_3`
- Representation protocol: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`
- Manifest/unique samples: 70,496/70,496
- Train/test: 52,903/17,593
- REL+ files: 70,496
- Valid-mask files: 70,496
- Structural/numeric/identity failures: 0
- Failure CSV: header only
- Audited resolved manifest: 70,496 data rows plus header

All recorded cache, source-manifest, resolved-manifest and split-list paths are
normalized absolute paths. Ordered train/test IDs are bound to the manifest.

## Risk-stratified regeneration

- Total: 70
- Areas: area_1, area_2, area_3, area_4, area_5a, area_5b and area_6
- Per area: 10
- Each of the ten reasons appears exactly seven times: invalid low, median,
  P90 and high; normal quality low, median and high; large gravity tilt;
  distinct room/camera; fixed random.
- PASS rows: 70
- Rows with any nonzero changed pixel/channel/maximum difference: 0
- Regeneration failures: 0

## Invalid interpolation diagnostic

The same 70 selected samples were diagnosed without changing formal input:

- source invalid ratio mean: 0.0172781;
- bilinear-affected ratio mean: 0.00130295;
- affected pixels: 21,014;
- affected mean/max channel deviation: 35.0039 / 115.826;
- affected label ignore ratio: 0.515323;
- affected valid-semantic ratio: 0.484677;
- per-class affected counts are retained in the JSON report.

This is a diagnostic of the accepted `SOURCE_COMPAT_STORAGE_255` baseline, not
an input correction or a new ablation.

## Evidence

- Summary: `formal_cache/audit/cache_audit_summary.json`
- Failures: `formal_cache/audit/cache_audit_failures.csv`
- Regeneration: `formal_cache/audit/cache_audit_sample_regeneration.csv`
- Resolved manifest: `formal_cache/audit/cache_manifest_resolved.csv`
- Failed log/exit: `logs/full_cache_audit_attempt1.log`, attempt1 exitcode
- Passing log/exit: `logs/full_cache_audit_attempt2.log`, attempt2 exitcode
- TDD evidence: `tests/tdd_area5_group_red.log`, `tests/tdd_area5_group_green.log`
- Invalid diagnostic: `reports/invalid_interpolation_diagnostic_70.json`

No file hash was generated or written.
