# CMX-REL+ v2.1 Completion Checklist

Evidence root: `/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_1_integration_fix`.

1. PASS — frozen REL+ generator core files are unchanged.
2. PASS — `audit/generator_byte_invariant.json`: 0 changed pixels/channels/max difference.
3. PASS — independent code directory created at `/home/zhuzhaoziao/RELPlus/CMX-RELPlusv2.1`.
4. PASS — REL+ TrainPre runs through a real pilot DataLoader.
5. PASS — REL+ ValPre runs through a real pilot DataLoader without TypeError.
6. PASS — train/eval use `dataloader.data_setting.build_data_setting`.
7. PASS — canonical evaluator requires REL+ mode; legacy entries reject it.
8. PASS — `eval/evaluator_smoke.json` records evaluator plumbing PASS.
9. PASS — the eval smoke wrote both raw and palette-color PNGs.
10. PASS — Focal map/scalar/gradients match the live author source.
11. PASS — gamma 1 and gamma 2 differ; forward uses `self.gamma`.
12. PASS — author seed formula, sampler location and cuDNN flags are tested.
13. PASS — two author parameter groups and AdamW defaults are tested.
14. PASS — both optimizer-group LR traces match for the first 300 iterations.
15. PASS — `audit/mit_b2_initialization.json` proves both encoders and decoder initialization.
16. PASS — source-compatible RGB sentinel and public behavior remain unchanged.
17. PASS — `[EGVIA, LOA, ReD]` sentinel passes disk-to-DataLoader.
18. PASS — no-flip is enforced in the real shared TrainPre profile.
19. PASS — `audit/transform_trace_first50.json` proves identical three-arm traces.
20. PASS — formal invalid=255 source-compatible behavior remains unchanged.
21. PASS — `pilot/invalid_label_diagnostic.json` contains label-joint diagnostics for 36 samples.
22. PASS — `pilot/single_batch_focal_backward.json` records finite/nonzero four-group gradients.
23. PASS — full formal configuration exists and resolves against 52,903/17,593 lists.
24. PASS — formal `training_authorized=False`.
25. PASS — formal `data_ready=False`.
26. PASS — no full REL+ cache was generated.
27. PASS — no optimizer or scheduler step was executed.
28. PASS — no checkpoint was produced.
29. PASS — no formal training process or epoch loop was started.
30. PASS — no complete-test scientific mIoU was computed or reported.

Final pytest evidence: `tests/pytest_final.log` and `tests/pytest_final.exitcode`
record 93 passed and exit code 0.
