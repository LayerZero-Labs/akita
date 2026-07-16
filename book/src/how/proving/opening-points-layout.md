# Opening points and digit-innermost layout

Akita uses one physical source order at the root and at every recursive level.
For one commitment group, let

```text
N = exact live source ring elements per claim (live_ring_elements_per_claim)
L = positions_per_block, positions in one block, a power of two
F = live_block_count = ceil(N / L), the exact number of live blocks
```

The physical source index is

```text
source_index = block_idx * positions_per_block + position
```

so position is the low-order coordinate and block_idx is the high-order coordinate.
The final block may be partial; it is not padded in the stored source.

## Opening point split

An opening point first contains `log2(positions_per_block)` position coordinates. The remaining
coordinates address `next_power_of_two(live_block_count)` block slots. Akita constructs all `L`
position weights but retains only the exact live prefix of `F` block weights.
There is no virtual compact-to-padded address map and no root-versus-recursive
block-order mode.

`RingOpeningPoint` exposes the resulting factors directly:

```text
position_weights: length positions_per_block
live_block_weights:    length live_block_count
```

Both the Lagrange and monomial bases use this same physical order.

## Witness order

Decomposition digits are innermost. For each group and chunk, the canonical
physical unit is

```text
[z_hat | e_hat | t_hat]
```

and one shared `r_hat` quotient tail follows every unit. The logical orders are

```text
z_hat[position][commit_digit][fold_digit]
e_hat[claim][block_idx][opening_digit]
t_hat[claim][block_idx][A_row][opening_digit]
r_hat[relation_row][quotient_digit]
```

`WitnessLayout` is the range authority shared by planning, proving, setup,
relation evaluation, recursive handoff, and verification. Units are ordered by
relation group and then chunk. Each unit records its exact `global_block_start` and
`live_block_count`.

## Chunks and tensor challenges

Chunks own contiguous ranges of the exact `F` live blocks. Internal allocation
uses a power-of-two `blocks_per_chunk_granule`; any residual blocks remain tight in the final
chunk. The planner chooses the granule independently of `positions_per_block` and of the challenge shape.

A flat fold challenge has `F` independent coefficients. A tensor challenge
chooses a power-of-two low-factor width `Q` and derives

```text
H = ceil(F / Q)
coefficient(b) = fold_high[b / Q] * fold_low[b % Q]
```

Only the first `F` products are live, so the last high-factor row may be
partial. Each commitment group owns its own flat-or-tensor shape.

## Validation boundary

Malformed dimensions, overflowing sizes, invalid powers of two, and block or
chunk indices outside the exact live ranges are rejected with `AkitaError`.
Verifier-reachable code does not recover an obsolete block-order interpretation.
