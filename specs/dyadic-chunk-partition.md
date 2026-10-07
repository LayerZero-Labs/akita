# Spec: Canonical Dyadic Chunk Partition

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-06 |
| Status | implemented |
| PR | [#372](https://github.com/LayerZero-Labs/akita/pull/372) |
| Supersedes | The residual chunk rules in the distributed-prover design records and `archive/2026-Q3/digit-innermost-layout.md` |
| Superseded-by | Recursive ownership: [#175](https://github.com/LayerZero-Labs/akita/pull/175); [Book: chunks and fold challenges](../book/src/how/proving/opening-points-layout.md#chunks-and-fold-challenges) |
| Book-chapter | book/src/how/proving/opening-points-layout.md |

## Summary

This specification governs proportional partitions of root sources and frozen
commitment groups. It replaced the earlier remainder-first rule with nested
proportional boundaries for supported power-of-two chunk counts.

It no longer governs inherited recursive witness ownership. After a multi-chunk
producer, each Z/E/T body is padded to the consumer's source-block coefficient
width. The consumer inherits those bodies, merges adjacent owners when chunking
contracts, and assigns the complete shared tail and final padding to the last
owner. This also applies when the consumer contracts to one chunk. Single-chunk
producers retain their contiguous bodies without body alignment padding.

Inherited partitions need not be balanced, and alignment can change witness
lengths, proof/setup costs, and selected schedules. The current complete contract
is [Chunks and fold challenges](../book/src/how/proving/opening-points-layout.md#chunks-and-fold-challenges)
and [Multi-chunk folding](../book/src/how/proving/advanced-relation-layouts.md).
The acceptance results below describe the original proportional-partition
cutover, not guarantees for inherited recursive layouts.

## Intent

### Goal

Use one checked function to derive canonical nested proportional ranges over
the exact live block prefix of roots and frozen commitment groups.

### Invariants

For `B` live blocks, `P` parts, and part index `i`, the canonical range is

```text
[floor(i * B / P), floor((i + 1) * B / P))
```

The implementation evaluates this formula with quotient and remainder
arithmetic so verifier supplied `usize` values cannot overflow during endpoint
calculation.

For proportional root/frozen-group partitions, the following properties hold:

- `P` is one or a power of two no greater than 64.
- `B` is positive. `P` may exceed `B`. The partition still contains exactly
  `P` ranges, and consecutive equal boundaries represent empty machine slots.
- Ranges are ordered, contiguous, and cover `[0, B)` exactly once.
- Range lengths differ by at most one block.
- If `P` divides `Q`, every boundary for `P` is also a boundary for `Q`.
- `WitnessLayout` remains the owner of all physical witness coefficient ranges.
  Its callers use the schedule-owned block ranges; recursive witnesses may
  inherit their ranges instead of using the proportional formula.
- Planner sizing, prover folding, setup evaluation, relation evaluation, and
  verification consume the same ranges.
- The original proportional-partition cutover changed only uneven block
  ownership. Its unchanged-length/cost/geometry claims do not apply to inherited
  recursive ownership introduced by #175.
- Invalid geometry returns `AkitaError::InvalidSetup` before allocation.
- The Akita instance descriptor version changes because the same public chunk
  count now has different derived protocol meaning for uneven block counts.

For example, ten blocks have these partitions:

```text
P = 2: [0, 5), [5, 10)
P = 4: [0, 2), [2, 5), [5, 7), [7, 10)
```

The boundary at five appears in both partitions.

When there are more parts than blocks, the same formula places empty ranges at
the repeated floor boundaries. Four blocks split into eight parts give

```text
[0, 0), [0, 1), [1, 1), [1, 2), [2, 2), [2, 3), [3, 3), [3, 4)
```

Each empty slot keeps its replicated Z segment. Its E and T segments are empty,
and the honest prover writes zero into Z because its challenge window is empty.

### Non-Goals of the original cutover

- This PR does not add B commitment slicing.
- This PR does not change the supported chunk counts or activation depths.
- This PR does not change planner objectives or schedule geometry.
- This PR does not change the fold challenge or transcript order.
- This PR does not add a compatibility path for the old partition rule.

## Evaluation of the original proportional-partition cutover

### Acceptance Criteria

- [x] One exported function derives every proportional witness chunk block range.
- [x] The old remainder-first range implementation is removed.
- [x] Exhaustive tests cover balance, exact coverage, and nesting for live block
      counts from 1 through 512 and supported part counts through 64.
- [x] A ten-block regression distinguishes the new nested four-part partition
      from the old crossing partition.
- [x] Invalid zero, non-power-of-two, and over-cap counts return
      `AkitaError::InvalidSetup`.
- [x] Over-partitioned positive block counts preserve all slots with canonical
      empty ranges.
- [x] Endpoint calculation succeeds for `usize::MAX` live blocks.
- [x] Existing multi-chunk prover and verifier tests pass.
- [x] Generated schedule parameters stay unchanged apart from catalog identity.
- [x] The instance descriptor version and generated catalog identities change.
- [x] Repository format, documentation, dependency, and lint checks pass.

### Testing Strategy

The unit tests in `akita-params` check the partition laws directly, including
`B < P`. Layout tests check that E and T still tile the live blocks, that empty
slots have empty E and T ranges, and that Z remains replicated once per chunk.

The planner tests check that challenge work reads the same canonical ranges.
The existing multi-group and multi-chunk proof tests provide end to end prover
and verifier coverage. The original cutover produced no schedule geometry changes.

### Performance

The helper performs constant work per chunk. Supported layouts use at most 64
chunks. The proportional formula itself does not change total witness length
or proof size; recursive body alignment can change both.

Some blocks move between chunks when `B` is not divisible by `P`. Every chunk
still receives either `floor(B/P)` or `ceil(B/P)` blocks, so maximum per-chunk
work does not increase.

## Design

### Architecture

`akita-params::dyadic_block_ranges` is the proportional partition authority.
`CommittedGroupParams::witness_block_ranges` selects proportional ranges for
roots/frozen groups and schedule-owned inherited endpoints for recursive
witnesses. `WitnessChunkShape::align` derives those inherited endpoints and
lengths. Relation, setup, trace, and verifier code consume the checked ranges
stored in `WitnessLayout`.

The public function replaces
`WitnessLayout::resolve_chunk_block_ranges`. There is no forwarding wrapper or
second formula.

During the original cutover, `AKITA_INSTANCE_DESCRIPTOR_VERSION` reset from 3
to 1. Generated catalog identity included this value as its protocol epoch, so
schedule table regeneration updated every affected catalog identity without
adding a second partition policy field. This is a historical cutover decision;
current descriptor binding follows the current protocol epoch and ownership
layout.

### Alternatives Considered

The first alternative kept the old remainder-first split and required future B
slicing to check boundary alignment. It was rejected because powers of two
would not guarantee refinement.

The second alternative padded the live block count to a power of two. It was
rejected because Akita commits and proves only the exact live block prefix.

The third alternative serialized every boundary. It was rejected because the
counts and exact live block geometry already determine all boundaries.

## Documentation

This spec records the cutover and updates the chunk rule in
`book/src/how/proving/opening-points-layout.md`. The distributed verifier chapter
uses the same exact range formula. No `AGENTS.md` change is needed because the
verifier error contract and development commands do not change.

PR #372 merged with all acceptance criteria complete. The Book owns the
durable rule, and this record remains in the root as the current load-bearing
partition contract until the next archive pass.

## Execution

1. Add `dyadic_block_ranges` in `akita-types`.
2. Replace every call to the old chunk-specific method.
3. Add direct partition law tests and update residual range fixtures.
4. Reset the unreleased instance descriptor epoch to 1.
5. Regenerate schedule catalogs.
6. Run the affected prover and verifier tests, then repository preflight.

## References

- [`archive/2026-Q3/digit-innermost-layout.md`](archive/2026-Q3/digit-innermost-layout.md)
- [`commitment-compression-cutover.md`](archive/2026-Q3/commitment-compression-cutover.md)
- [`book/src/how/proving/opening-points-layout.md`](../book/src/how/proving/opening-points-layout.md)
