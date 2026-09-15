# Code change summary

| File or directory | Change | Reason | Changes model mathematics |
|---|---|---|---|
| `third_party/rel_original/` | Added the minimum executable ERP REL source and its source notice/license | Make CMX-REL self-contained without copying REL-SF4PASS models | No |
| `tools/generate_stanford2d3d_rel.py` | Added offline full-ERP REL generation with area, limit, worker and overwrite arguments | Generate REL before training augmentation | No |
| `tools/prepare_cmx_rel_smoke_data.py` | Added three-real-sample CMX layout and official semantic-label mapping | Build a bounded smoke dataset | No |
| `tools/run_with_config.py` | Added explicit config-module selection for unchanged CMX entry points | Use the new configuration without replacing official entry points | No |
| `configs/stanford2d3dpano/cmx_mit_b2_rel.py` | Added 13-class MiT-B2/MLPDecoder REL configuration | Define the S3D REL experiment | No |
| `configs/stanford2d3dpano/cmx_mit_b2_rel_smoke.py` | Added bounded 256-square smoke settings | Limit the integration test to one in-memory step | No |
| `dataloader/RGBXDataset.py` | Added `rel_original` mode using `IMREAD_UNCHANGED`, three-channel validation and a clear missing-file error | Preserve the actual REL array order | No |
| `dataloader/dataloader.py` | Passed `x_mode` to the existing Dataset | Select the REL loader while retaining the original synchronized transforms | No |
| `tests/run_rel_integration_smoke.py` | Added source-body, exact-array, channel, data, gradient and optimizer-step checks | Verify the requested integration | No |
| `docs/rel_integration/` | Added provenance, alignment, channel, command, model-diff and final reports | Make the result auditable | No |
| `README.md` | Added a short CMX-REL entry point | Identify this working copy | No |

The following were not changed:

- `models/encoders/dual_segformer.py`
- CM-FRM and FFM
- `models/decoders/`
- `models/builder.py`
- `train.py`
- `eval.py`
- losses and engine code

No Gate, SMMF, DyMM, region slicing, temperature schedule, soft/hard gate,
perspective camera model or REL+ implementation was added.
