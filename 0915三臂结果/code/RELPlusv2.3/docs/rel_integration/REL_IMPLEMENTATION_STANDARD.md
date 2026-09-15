# REL implementation standard

## Status

Accepted on 2026-08-16 by project decision.

## Scope

This decision applies to REL generation, CMX-REL integration, training and
evaluation in this project unless a later explicit decision supersedes it.

## Decision

The verified executable REL implementation is the project standard.  For this
checkout, the canonical implementation entry is
`third_party/rel_original/rel.py` together with its called ERP geometry helpers.
Future REL artifacts must preserve the code-defined behavior, including:

- height encoded from the 1st and 99th percentiles of rotated `z`;
- EGVIA height blending on `~is_horizontal`;
- `alpha=45` and `lambda=0.5` unless an experiment explicitly changes them;
- output array and tensor order `[EGVIA, LOA, ReD]`;
- missing pixels encoded as 255 in all three channels.

The equivalent `getRLE()` body in
`/data/bxh_copy/Pano_MA_Seg/getHHA.py` is supporting provenance, not the
canonical entry point.  Its mixed HHA/RLE command-line wrapper is not adopted
as the REL standard.

## Consequences

- Differences between the supplied paper prose and executable code remain
  documented as provenance, but no longer block this project's REL definition.
- Paper prose must not silently override the executable behavior.
- A future paper-faithful alternative must be separately named and treated as
  a different representation in any controlled comparison.
- This decision resolves only the representation-definition blocker.  It does
  not itself authorize training or establish an mIoU result.
