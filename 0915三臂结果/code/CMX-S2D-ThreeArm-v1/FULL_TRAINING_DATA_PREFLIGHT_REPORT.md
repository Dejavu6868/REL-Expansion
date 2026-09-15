# Full CMX training-data preflight report

## Final status

`PASS` (exit code 0)

The V2.3 preflight ran from approximately 00:31:20 to 00:51:28 and decoded
every formal-training sample. It was a read-only validation: it performed no
backpropagation, optimizer update, checkpoint write or cache modification.

## Full-dataset result

- Integration protocol: `CMX_RELPLUS_V2_3`
- Representation protocol: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`
- Manifest/unique samples: 70,496/70,496
- Train/test: 52,903/17,593
- RGB files decoded: 70,496
- Label files decoded and class-mapped: 70,496
- REL+ files decoded: 70,496
- Valid-mask files decoded: 70,496
- Missing, decode, shape, dtype, range or finite-value failures: 0
- Ordered split-identity failures: 0
- Reused rows: 0
- Rescanned rows: 70,496
- All samples decoded during this run: true
- Failure CSV: header only
- Resolved preflight CSV: 70,496 data rows plus header

The label mapping was loaded from the configured Stanford2D3D mapping file.
Stored labels are converted by the declared `stored_id - 1` transform, with
stored ID 0 mapped to ignore label 255; no class IDs were guessed or
hard-coded as a fallback.

The preflight also bound its normalized absolute manifest, cache, audit and
ordered train/test paths to the formal configuration. A passing cache audit
was required before any sample scan began.

## Evidence

- Summary: `formal_cache/preflight/cmx_training_data_preflight_summary.json`
- Per-sample rows: `formal_cache/preflight/cmx_training_data_preflight.csv`
- Failures: `formal_cache/preflight/cmx_training_data_preflight_failures.csv`
- Log: `logs/full_training_data_preflight_attempt1.log`
- Exit code: `logs/full_training_data_preflight_attempt1.exitcode`

No file hash was generated or written.
