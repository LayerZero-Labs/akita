# Matrix evaluation at a point

The verifier evaluates the relation matrix multilinear extension without
materializing the matrix. Its column geometry comes from the same
`WitnessLayout` that emitted the witness.

## Canonical walk

`WitnessLayout` orders group-and-shard units as

```text
group 0, shard 0: [z_hat | e_hat | t_hat]
group 0, shard 1: [z_hat | e_hat | t_hat]
...
group g, shard s: [z_hat | e_hat | t_hat]
shared tail:       [r_hat]
```

Each unit carries an exact global fold range. Relation, setup, and trace
evaluators consume these checked ranges; they do not reconstruct offsets from a
second chunk-layout description. Multi-group and multi-shard layouts are the
ordinary product of the same two indices.

`z_hat` is replicated per shard because it participates in every shard-local
relation. `e_hat` and `t_hat` are partitioned by the shard's exact fold range.
The quotient `r_hat` is shared once after all units.

## Exact fold weights

For a group with exact live fold count `F`, a flat challenge supplies `F`
coefficients. A tensor challenge supplies factors of lengths

```text
Q = fold_low_len
H = ceil(F / Q)
```

and fold `f` has weight `high[f / Q] * low[f % Q]`. The final high row may be
partial. Group-local challenge shape and exact `F` are transcript-bound and are
validated before any indexing.

## Structured evaluation

Tensor coefficients stay factored. The verifier's affine digit-interval kernel
combines:

```text
exact live prefix
base column offset
digit stride and interval
fold-high and fold-low factors
partial final row
```

This avoids allocating or scanning the Cartesian `H * Q` table and avoids
materializing one coefficient per padded fold slot. Sparse challenge values use
the ring add, subtract, and double fast paths where applicable.

Flat challenges necessarily cost linear work in `F`, because their entries are
independent. Tensor work is priced from the compact factors (`H + Q`) and shard
subwindows, with a checked work cap at the verifier boundary.

## Setup roles and mixed rings

The A, B, and D setup contributions use the same group and shard ranges. D group
offsets follow checked relation-group prefix sums. `SetupProjectionGeometry`
owns mixed-ring projection, so verifier evaluation does not maintain a parallel
layout carrier for setup columns.

## Safety contract

Before evaluation, the verifier checks the opening dimensions, group-local
layout, tensor shape, unit ranges, setup geometry, and work bounds. Malformed
proof data returns `AkitaError`; verifier-reachable evaluation does not panic or
allocate from an unchecked proof-controlled dimension.
