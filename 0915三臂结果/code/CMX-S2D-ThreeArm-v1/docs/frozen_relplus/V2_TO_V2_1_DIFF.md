# REL+ v2 to v2.1 change map

| Item | v2 | v2.1 | Changes three channels |
|---|---|---|---|
| K profile | caller supplied shape | trusted dataset profile | no |
| principal point | general shape binding | general bound plus Stanford centre sanity | no |
| skew | implicit helper limitation | explicit zero-skew failure | no |
| pose status | PASS/FAIL/N/A | strong/weak/review/fail/N/A | no |
| semantic pose mask | finite normals | depth + finite + nonzero + quality | no |
| invalid model contract | incomplete | source-compatible storage-255 | no |
| valid mask | not propagated to CMX | nearest diagnostic field only | no |
| transform shape | caller responsibility | source and scaled shape checked | no |
| dtype | implicit downstream cast | explicit float32 CHW | no |
| CMX adapter | independent arrays | real RGBXDataset/TrainPre branch | no |
| normal scan | 12 review samples | complete 70,496-row manifest | no |
| unit test | encoder self-comparison risk | generator-boundary capture and golden regression | no |

The v2 generator remains available as an independent regression entry. The
v2.1 generator delegates to that frozen byte path; all new behavior is at
loading, validation, diagnostics and CMX integration boundaries.
