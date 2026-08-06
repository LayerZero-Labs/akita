# Matrix evaluation at a point

The verifier evaluates the relation matrix multilinear extension without
materializing the matrix. Its column geometry comes from the same
`WitnessLayout` that emitted the witness.

## Canonical walk

`WitnessLayout` orders group-and-chunk units as

```text
group 0, chunk 0: [z_hat | e_hat | t_hat]
group 0, chunk 1: [z_hat | e_hat | t_hat]
...
group g, chunk c: [z_hat | e_hat | t_hat]
shared tail:       [r_hat]
```

Each unit carries an exact global block range. Relation, setup, and trace
evaluators consume these checked ranges; they do not reconstruct offsets from a
second chunk-layout description. Multi-group and multi-chunk layouts are the
ordinary product of the same two indices.

`z_hat` is replicated per chunk because it participates in every chunk-local
relation. `e_hat` and `t_hat` are partitioned by the chunk's exact block range.
The quotient `r_hat` is shared once after all units.

## Exact block weights

For a group with exact live block count `F`, the fold challenge supplies `F`
independent sparse coefficients. The exact count is transcript-bound and is
validated before any indexing. Sparse challenge values use the ring add,
subtract, and double fast paths where applicable.

## Setup roles and mixed rings

The A, B, and D setup contributions use the same group and chunk ranges. D group
offsets follow checked relation-group prefix sums. `SetupProjectionGeometry`
owns mixed-ring projection, so verifier evaluation does not maintain a parallel
setup-column layout.

The active
[`role-native-projected-digit-layout`](../../../specs/role-native-projected-digit-layout.md)
spec defines the E and T verifier cutover. Its target physical order is:

```text
[semantic value][role subcolumn][role digit][native coefficient]
```

The setup matrix and relation witness use the same subcolumn and digit axes.
At the ring evaluation point `alpha`, a role subcolumn `s` of dimension `r`
has weight `alpha^(s * r)`. The verifier includes this power in the projected
equality tensor and applies the role gadget power on the digit axis.

When the projection ratio is one, the verifier does not allocate projection
powers or multiply by one. It uses the unprojected contiguous equality window
directly. Mixed groups use exact coefficient ranges and the shared minimum
relation block; there are no carrier spans or per-role padding to scan.

## Safety contract

Before evaluation, the verifier checks the opening dimensions, group-local
layout, unit ranges, setup geometry, and work bounds. Malformed
proof data returns `AkitaError`; verifier-reachable evaluation does not panic or
allocate from an unchecked proof-controlled dimension.
