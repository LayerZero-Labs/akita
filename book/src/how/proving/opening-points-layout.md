# Opening points and digit-innermost layout

Akita uses one physical source order at the root and at every recursive level.
For one commitment group, let

```text
N = exact live source ring elements per claim (num_live_ring_elements_per_claim)
L = num_positions_per_block, positions in one block, a power of two
F = num_live_blocks = ceil(N / L), the exact number of live blocks
```

The physical source index is

```text
source_index = block_idx * num_positions_per_block + position
```

so position is the low-order coordinate and block_idx is the high-order coordinate.
The final block may be partial; it is not padded in the stored source.

## Opening point split

An opening point first contains `log2(num_positions_per_block)` position coordinates. The remaining
coordinates address `next_power_of_two(num_live_blocks)` block slots. Akita constructs all `L`
position weights but retains only the exact live prefix of `F` block weights.
There is no virtual compact-to-padded address map and no root-versus-recursive
block-order mode.

`RingOpeningPoint` exposes the resulting factors directly:

```text
position_weights: length num_positions_per_block
live_block_weights:    length num_live_blocks
```

Both the Lagrange and monomial bases use this same physical order.

## Witness order

Decomposition digits are innermost. For each group and chunk, the canonical
physical unit is

```text
[z_hat | e_hat | t_hat]
```

The logical orders are

```text
z_hat[position][commit_digit][fold_digit][A_coefficient]
e_hat[claim][block_idx][D_subcolumn][opening_digit][D_coefficient]
t_hat[claim][block_idx][A_row][B_subcolumn][outer_digit][B_coefficient]
r_hat[relation_row][quotient_digit][native_row_coefficient]
```

The T digit is the outer commitment digit. The active
[`role-native-projected-digit-layout`](../../../../specs/role-native-projected-digit-layout.md)
spec defines the mixed-dimension cutover. Under that contract, E and T split
each A-ring value into native D or B subrings before decomposition. Their target
coefficient orders are:

```text
e_hat[claim][block_idx][D_subcolumn][opening_digit][D_coefficient]
t_hat[claim][block_idx][A_row][B_subcolumn][outer_digit][B_coefficient]
```

Only live subcolumns are stored. When every role dimension equals A, the
subcolumn axis has length one and the byte order is the uniform layout.

[`WitnessLayout`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-types/src/witness.rs)
is the range authority shared by planning, proving, setup, relation evaluation,
recursive handoff, and verification. Units are ordered by chunk and then
authenticated relation group. Each unit records its exact
`global_block_start`, `num_live_blocks`, and coefficient ranges.

The complete physical order depends on the realization:

```text
raw:
  [chunk-major, relation-group-ordered Z | E | T units]
  [ordinary consistency, A, B, and D quotient rows]

compressed:
  [chunk-major, relation-group-ordered Z | E | T units]
  [ordinary consistency, A, B, and D quotient rows]
  [alignment to the common relation coefficient block]
  for compression layer l = 1, 2:
    [alignment to the largest native F_l or H_l ring dimension]
    [F_l digit spans, one per group in relation order]
    [the shared H_l digit span]
    [F_l quotient rows, in the same group order]
    [the H_l quotient row]
  [suffix alignment to the common relation coefficient block]
```

Thus `r_hat` is logically one quotient family in relation-row order, but its
compression rows are physically interleaved with the corresponding digit
layer rather than stored in one contiguous quotient tail. Every alignment
range is zero. Boolean padding, if needed by a later flat-table sumcheck, is a
separate zero suffix after the complete live witness.

In compressed mode, the negative-binary constraint is active exactly on one
interval per layer, from that layer's first `F` digit through the end of its
shared `H` digit. These intervals include the padded `F` and `H` digit spans,
but exclude all ordinary and compression quotient rows and all alignment
ranges. Raw mode has no compression layers or such support intervals.

## Chunks and fold challenges

Chunks own contiguous ranges of the exact `F` live blocks. For chunk `i` of
`P`, the canonical range is

```text
[floor(i * F / P), floor((i + 1) * F / P))
```

The ranges stay exact and contain no padding. Their lengths differ by at most
one block. If `P > F`, repeated boundaries give empty ranges while preserving
all `P` machine slots. An empty slot has no E or T coefficients, but it keeps
the full replicated Z segment and the honest prover fills that segment with
zero. All supported chunk counts are powers of two. Therefore, every finer
chunk partition refines every coarser partition.

Each commitment group owns a fold challenge with `F` independent sparse
coefficients, one for every live block.

## Validation boundary

Malformed dimensions, overflowing sizes, invalid powers of two, and block or
chunk indices outside the exact live ranges are rejected with `AkitaError`.
Verifier-reachable code does not recover an obsolete block-order interpretation.
