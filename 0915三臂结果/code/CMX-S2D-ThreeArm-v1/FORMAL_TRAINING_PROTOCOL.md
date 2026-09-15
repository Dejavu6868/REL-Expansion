# Formal training protocol

## Experiment identity

- Mainline stage: V2.3 full-data infrastructure closed, then one formal
  CMX-REL+ training run.
- Scientific question: under the frozen Stanford2D3D S2D and author-aligned
  CMX protocol, can the frozen REL+ input train stably to the declared formal
  endpoints?
- Changed factor in the future three-arm comparison: X modality only. This
  task launches only the REL+ arm.
- Representation: `RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT`.
- Integration: `CMX_RELPLUS_V2_3`.
- Seed: 12345.

This is retraining. The DDP smoke and formal run execute backpropagation and
optimizer updates. The smoke writes only a disposable checkpoint outside the
formal sweep directory; the formal run creates new V2.3 checkpoints and never
replaces V2.2 checkpoints.

## Frozen controls and data

- Dataset: Stanford2D3D S2D ordered manifest, 52,903 train and 17,593 test;
- Original CMX, MiT-B2 RGB encoder, MiT-B2 X encoder, MLPDecoder;
- Gate, SMMF, DyMM and SGA off;
- `[EGVIA, LOA, ReD]`, valid mask diagnostic only;
- 200 epochs, global batch 8, eight ranks, SyncBN on, AMP off;
- FocalLoss2d, gamma 2, `reduction=none`, then spatial `mean()`;
- AdamW, learning rate 6e-5, weight decay 0.01;
- WarmUpPolyLR, ten warm-up epochs, iteration-wise update;
- no horizontal/vertical flip, arbitrary rotation or perspective warp;
- seed policy `base_seed + epoch + local_rank * 1000` in DDP;
- cuDNN benchmark off and deterministic off, matching the frozen author path.

The sampler uses 52,904 logical samples per epoch, one padding sample, and
6,613 iterations per epoch at global batch 8.

## Hard gates

The repository config stays fail-closed. The explicit launcher creates a
one-run resolved config only after all of these pass:

1. frozen-generator byte invariant;
2. complete 70,496-pair cache with zero generation failures;
3. cache audit bound to absolute cache/manifest/resolved-manifest/split paths,
   ordered IDs and both protocol IDs;
4. 70 risk-stratified regenerations with zero byte difference;
5. full CMX RGB/label/REL+/mask preflight with every sample decoded;
6. real eight-rank DDP smoke with finite logits/loss/gradients, optimizer
   updates, all four parameter groups changing, disposable checkpoint save,
   exact parameter restore, LR continuity and 2--5 resumed updates;
7. MiT-B2 pretrained file exists and was loaded in the smoke;
8. exactly eight visible GPUs.

Missing, mismatched or stale evidence is `BLOCKED_BEFORE_FORMAL_TRAINING`.

## Runtime and stopping rules

The formal loop preserves the author NaN policy: NaN elements in the loss map
are replaced with zero and counted. A non-finite reduced loss or any non-finite
gradient fails loudly and stops the run. Infrastructure identity failure,
process failure or a GPU/NCCL failure also stops the run. There is no metric
early stopping and no hyperparameter tuning.

Primary endpoint: epoch 200. Secondary descriptive endpoint:
`test_selected_best` over epochs 100, 105, ..., 200, disclosed as test-set
selection. A PASS at the launch stage unlocks only continued training and the
frozen checkpoint-evaluation schedule; it does not authorize another arm or
seed.

Runtime status is written atomically every ten iterations to
`formal_training/CMX_RELPlus_v2_3_seed12345/runtime_status.json`.
