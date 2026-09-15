# CMX Training Protocol Alignment

The executable author reference audited for this delivery is
`/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference`.

| Item | Old CMX fork | Author REL executable behavior | Formal CMX-REL+ |
|---|---|---|---|
| Focal reduction | `mean` in train | `none`, then training-loop `mean()` | `none`, then `mean()` |
| Focal gamma | default 0 and exponent hard-coded 2 | default/live gamma 2 | live `self.gamma`, configured 2 |
| Ignore | 255 | 255 | 255 |
| Seed | base plus rank | base plus epoch plus distributed rank times 1000 | author formula |
| Worker seed | no explicit worker init | no explicit worker init | no explicit worker init |
| cuDNN | benchmark true | benchmark false; deterministic left false | benchmark false; deterministic false |
| Optimizer groups | CMX `group_weight` | decay plus no-decay groups | same author grouping |
| AdamW | 6e-5, betas 0.9/0.999, wd 0.01 | same | same |
| Scheduler | WarmUpPolyLR per iteration | WarmUpPolyLR per iteration | same; first 300 values directly compared |
| Sampler | `set_epoch` present | `set_epoch(epoch)` before epoch iteration | same location |
| Initialization | Original CMX dual-path load | MiT backbone initialization | both CMX encoders copy mapped MiT tensors; decoder uses Original CMX default |

## V2.3 authorization gates

`train.py` calls `assert_training_ready(config)` before Engine, model,
optimizer, TensorBoard or loader construction. The repository formal config
remains fail-closed. The explicit V2.3 launcher produces a one-run resolved
overlay only after all of these are verified:

1. explicit formal-training and source-compatible-invalid acceptance flags;
2. full generation summary with `full_cache_generated=true`;
3. cache audit PASS with 70/70 byte-identical risk regenerations;
4. full RGB/label/REL+/mask preflight PASS;
5. real eight-rank optimizer/checkpoint/resume smoke PASS;
6. pretrained MiT-B2 load and exactly eight visible GPUs.

No `data_ready=True` boolean can substitute for these bound reports.

## Initialization evidence

The initialization audit loaded the actual MiT-B2 file and compared source and
target state-dict keys, shapes and tensor values. RGB and X encoders each
matched all 332 mapped tensors; missing, unexpected, shape-mismatch and
value-mismatch lists were empty. The classification-only `head.weight` and
`head.bias` were explicitly recorded as expected ignored source keys. The
decoder `init_weight` call was traced, changed two tensors without changing
shapes, and left all decoder tensors finite/nonzero. No file hash was written.

## Historical V2.2 pilot and V2.3 DDP smoke

The V2.2 single-batch tool built no optimizer or scheduler. It performed one
real DataLoader -> CMX -> formal Focal map -> mean -> backward chain, checks
finite logits/loss and finite nonzero gradients for RGB encoder, X encoder,
fusion and decoder, then exits without an epoch loop or checkpoint.

V2.3 separately requires a real eight-rank NCCL/DDP/SyncBN run of 50--100
optimizer steps, a disposable checkpoint, exact parameter and LR restoration,
and another 2--5 optimizer steps. This is training activity with
backpropagation and optimizer updates, but its checkpoint is excluded from the
formal evaluation sweep.
