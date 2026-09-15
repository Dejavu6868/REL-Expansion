# CMX-REL+ V2.3 specification

## Scope

V2.3 changes infrastructure around the frozen REL+ representation. It does
not change REL+ geometry, channel semantics, invalid-value policy, model
architecture, dataset split or training hyperparameters.

- V2.2 baseline: `/home/zhuzhaoziao/RELPlus/RELPlusv2.2`
- V2.3 code: `/home/zhuzhaoziao/RELPlus/RELPlusv2.3`
- Runtime evidence: `/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3`
- Full manifest: `/data/zhuzhaoziao/RELPlus/outputs/REL_plus_v2_1_implementation/full_manifest.csv`
- Formal cache: `/data/zhuzhaoziao/RELPlus/outputs/CMX_RELPlus_v2_3/formal_cache`

The server copies are delivery trees, not Git worktrees. Git status or branch
claims must therefore not be inferred from those directories.

## Frozen representation

- Representation protocol: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`
- Integration protocol: `CMX_RELPLUS_V2_3`
- Tensor: uint8, 480 x 480 x 3
- Channel order: `[EGVIA, LOA, ReD]`
- Invalid storage: `SOURCE_COMPAT_STORAGE_255`
- Valid mask: binary diagnostic artifact; never passed to the model
- Camera intrinsics, pose and gravity: read only through the frozen Stanford
  S2D loader; there is no identity or invented fallback

`rel_plus/` must remain source-identical to V2.2. The executable byte
regression additionally requires zero changed pixels, zero changed channels
and zero maximum difference.

## Data contract

The frozen manifest contains 70,496 ordered samples: 52,903 train and 17,593
test. Formal execution requires all of the following identities to agree by
normalized absolute path and ordered sample ID:

- cache root, REL+ root and valid-mask root;
- source manifest and audited resolved manifest;
- train and test lists;
- representation and integration protocol IDs;
- sample and split counts.

The CMX preflight decodes RGB, label, REL+ and valid mask for every sample.
Label legality comes from the dataset's real `class_mapping.json`, including
the loader transform from stored IDs to model IDs.

## Infrastructure changes

- Counter-based O(N) duplicate detection;
- atomic PNG generation, one OpenCV thread per worker and strict resume;
- correct `full_cache_generated` semantics;
- risk-stratified, 10-per-area regeneration of 70 cache entries;
- cache plus full CMX preflight training gate;
- checkpoint filename/payload epoch binding;
- fraction and percent metric fields;
- one sequential 8-rank evaluation launch per checkpoint;
- real 8-rank optimizer/save/restore/resume smoke;
- one-run resolved config and fail-closed formal launcher;
- read-only RawDepth/HHA contract audit for a possible future comparison.

## Formal training boundary

Only one CMX-REL+ V2.3 run is authorized: seed 12345, Original CMX, MiT-B2,
MLPDecoder, 200 epochs, global batch 8, eight GPUs, AMP off and SyncBN on.
RGBD, HHA, original REL, extra seeds, folds, tuning and additional formal runs
are outside this task.
