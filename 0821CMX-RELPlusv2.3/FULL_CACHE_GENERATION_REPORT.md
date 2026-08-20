# Full cache generation report

## Status

`PASS`

The formal generator completed with exit code 0. This stage generated data
only; it did not perform backpropagation, optimizer updates or checkpoint
writes.

## Throughput estimate

Two fresh 500-image runs used the same frozen generator and one OpenCV thread
per Python worker.

| Workers | Elapsed | Images/s | Aggregate CPU | Peak single-process RSS | Encoded write | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 15.04 s | 33.24 | 794.25% of one core | 286.25 MiB | 6.43 MiB/s | 0 |
| 16 | 10.63 s | 47.03 | 1569.63% of one core | 286.28 MiB | 9.10 MiB/s | 0 |

The benchmark estimated 14,305,674,465 bytes and 140,992 PNG inodes for the
full cache. Sixteen workers were selected from the measured result; the server
had 112 CPUs, 491 GiB available memory, 2.8 TiB available storage and ample
inodes before generation.

Evidence:

- `reports/cache_throughput_500_w8_attempt1.json`
- `reports/cache_throughput_500_w16_attempt1.json`

## Full execution

- Started: 2026-08-20 23:47:35 +08:00
- Summary written: 2026-08-21 00:08:38 +08:00
- Approximate wall clock: 21 min 03 s
- Workers: 16
- Manifest rows: 70,496
- Generated or verified: 70,496
- Train list: 52,903
- Test list: 17,593
- Generation failures: 0
- `full_cache_generated`: true
- REL+ PNG files: 70,496
- Valid-mask PNG files: 70,496
- Temporary PNG files after exit: 0
- Actual tree bytes after generation: 14,548,117,919
- Actual regular files after generation: 140,997
- Actual directories after generation: 17

The resolved manifest has 70,496 data rows plus one header. The failure CSV
contains only its header. No file hash was generated or written.

## Evidence paths

- Cache: `/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/formal_cache`
- Summary: `formal_cache/cache_generation_summary.json`
- Resolved manifest: `formal_cache/cache_manifest_resolved.csv`
- Failures: `formal_cache/cache_generation_failures.csv`
- Log: `logs/full_cache_generation_attempt1.log`
- Exit code: `logs/full_cache_generation_attempt1.exitcode`
