# Spec: Canonical Dyadic Chunk Partition

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-06 |
| Status | active |
| PR | |
| Supersedes | The residual chunk rules in `distributed-prover.md`, `distributed-planner.md`, and `digit-innermost-layout.md` |
| Superseded-by | |
| Book-chapter | book/src/how/proving/opening-points-layout.md |

## Summary

Akita currently puts every residual live block into the earliest witness
chunks. That gives balanced ranges, but partitions with two and four chunks do
not always nest. This change uses proportional boundaries for every witness
chunk partition. All supported chunk counts are powers of two, so every finer
partition then refines every coarser partition.

This PR changes chunk ownership only. It does not add commitment slicing.

## Intent

### Goal

Use one checked function to derive canonical nested ranges over the exact live
block prefix.

### Invariants

For `B` live blocks, `P` parts, and part index `i`, the canonical range is

```text
[floor(i * B / P), floor((i + 1) * B / P))
```

The implementation evaluates this formula with quotient and remainder
arithmetic so verifier supplied `usize` values cannot overflow during endpoint
calculation.

The following properties must hold:

- `P` is one or a power of two no greater than 64.
- `P` is no greater than `B`, so every range is nonempty.
- Ranges are ordered, contiguous, and cover `[0, B)` exactly once.
- Range lengths differ by at most one block.
- If `P` divides `Q`, every boundary for `P` is also a boundary for `Q`.
- `WitnessLayout` remains the owner of all physical witness coefficient ranges.
  Its callers use the same canonical block ranges.
- Planner sizing, prover folding, setup evaluation, relation evaluation, and
  verification consume the same ranges.
- The total witness length, proof size, setup size, and selected schedule
  geometry do not change. Only uneven block ownership changes.
- Invalid geometry returns `AkitaError::InvalidSetup` before allocation.
- The Akita instance descriptor version changes because the same public chunk
  count now has different derived protocol meaning for uneven block counts.

For example, ten blocks have these partitions:

```text
P = 2: [0, 5), [5, 10)
P = 4: [0, 2), [2, 5), [5, 7), [7, 10)
```

The boundary at five appears in both partitions.

### Non-Goals

- This PR does not add B commitment slicing.
- This PR does not change the supported chunk counts or activation depths.
- This PR does not change planner objectives or schedule geometry.
- This PR does not change the fold challenge or transcript order.
- This PR does not add a compatibility path for the old partition rule.

## Evaluation

### Acceptance Criteria

- [x] One exported function derives every witness chunk block range.
- [x] The old remainder-first range implementation is removed.
- [x] Exhaustive tests cover balance, exact coverage, and nesting for live block
      counts from 1 through 512 and supported part counts through 64.
- [x] A ten-block regression distinguishes the new nested four-part partition
      from the old crossing partition.
- [x] Invalid zero, non-power-of-two, over-cap, and over-partitioned counts
      return `AkitaError::InvalidSetup`.
- [x] Endpoint calculation succeeds for `usize::MAX` live blocks.
- [x] Existing multi-chunk prover and verifier tests pass.
- [x] Generated schedule parameters stay unchanged apart from catalog identity.
- [x] The instance descriptor version and generated catalog identities change.
- [x] Repository format, documentation, dependency, and lint checks pass.

### Testing Strategy

The unit tests in `akita-types` check the partition laws directly. Existing
layout tests then check that E and T ranges still tile the exact live blocks and
that Z remains replicated once per chunk.

The planner tests check that challenge work reads the same canonical ranges.
The existing multi-group and multi-chunk proof tests provide end to end prover
and verifier coverage. Schedule generation must produce no geometry changes.

### Performance

The helper performs constant work per chunk. Supported layouts use at most 64
chunks. The formula does not change total witness length or proof size.

Some blocks move between chunks when `B` is not divisible by `P`. Every chunk
still receives either `floor(B/P)` or `ceil(B/P)` blocks, so maximum per-chunk
work does not increase.

## Design

### Architecture

`akita-types::dyadic_block_ranges` is the single block partition authority.
`WitnessLayout`, planner challenge pricing, and prover fold construction call it
directly. Relation, setup, trace, and verifier code continue to consume the
checked ranges stored in `WitnessLayout`.

The public function replaces
`WitnessLayout::resolve_chunk_block_ranges`. There is no forwarding wrapper or
second formula.

`AKITA_INSTANCE_DESCRIPTOR_VERSION` resets from 3 to 1. Akita is still in
development and has no compatibility promise. Generated catalog identity
already includes this value as its protocol epoch, so schedule table
regeneration updates every affected catalog identity without adding a second
partition policy field or carrying development epoch churn.

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

After the PR merges, set the status to `implemented` and record its number. The
book owns the durable rule, so the spec can then be archived during the normal
spec pruning pass.

## Execution

1. Add `dyadic_block_ranges` in `akita-types`.
2. Replace every call to the old chunk-specific method.
3. Add direct partition law tests and update residual range fixtures.
4. Reset the unreleased instance descriptor epoch to 1.
5. Regenerate schedule catalogs.
6. Run the affected prover and verifier tests, then repository preflight.

## References

- [`distributed-prover.md`](distributed-prover.md)
- [`distributed-planner.md`](distributed-planner.md)
- [`digit-innermost-layout.md`](digit-innermost-layout.md)
- [`commitment-compression-cutover.md`](commitment-compression-cutover.md)
- [`book/src/how/proving/opening-points-layout.md`](../book/src/how/proving/opening-points-layout.md)
