# Opening points and digit-innermost layout

Akita uses one physical source order at the root and at every recursive level.
For one commitment group, let

```text
N = exact live source ring elements per claim
L = positions in one fold slice, a power of two
F = ceil(N / L), the exact number of live fold slices
```

The physical source index is

```text
source_index = fold_index * L + position
```

so position is the low-order coordinate and fold is the high-order coordinate.
The final fold slice may be partial; it is not padded in the stored source.

## Opening point split

An opening point first contains `log2(L)` position coordinates. The remaining
coordinates address `next_power_of_two(F)` fold slots. Akita constructs all `L`
position weights but retains only the exact live prefix of `F` fold weights.
There is no virtual compact-to-padded address map and no root-versus-recursive
block-order mode.

`RingOpeningPoint` exposes the resulting factors directly:

```text
position_weights: length L
fold_weights:     length F
```

Both the Lagrange and monomial bases use this same physical order.

## Witness order

Decomposition digits are innermost. For each group and shard, the canonical
physical unit is

```text
[z_hat | e_hat | t_hat]
```

and one shared `r_hat` quotient tail follows every unit. The logical orders are

```text
z_hat[position][commit_digit][fold_digit]
e_hat[claim][live_fold][opening_digit]
t_hat[claim][live_fold][A_row][opening_digit]
r_hat[relation_row][quotient_digit]
```

`WitnessLayout` is the range authority shared by planning, proving, setup,
relation evaluation, recursive handoff, and verification. Units are ordered by
relation group and then shard. Each unit records its exact global fold start and
live fold count.

## Shards and tensor challenges

Shards own contiguous ranges of the exact `F` live folds. Internal allocation
uses a power-of-two granule `S`; any residual folds remain tight in the final
shard. The planner chooses `S` independently of `L` and of the challenge shape.

A flat fold challenge has `F` independent coefficients. A tensor challenge
chooses a power-of-two low-factor width `Q` and derives

```text
H = ceil(F / Q)
coefficient(f) = fold_high[f / Q] * fold_low[f % Q]
```

Only the first `F` products are live, so the last high-factor row may be
partial. Each commitment group owns its own flat-or-tensor shape.

## Validation boundary

Malformed dimensions, overflowing sizes, invalid powers of two, and fold or
shard indices outside the exact live ranges are rejected with `AkitaError`.
Verifier-reachable code does not recover an obsolete block-order interpretation.
