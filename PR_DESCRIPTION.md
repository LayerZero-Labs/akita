# Adaptive ring dimensions for fp128 one-hot

## Summary

This PR makes adaptive ring dimensions the default configuration for direct fp128 one-hot and dense commitments.

Instead of selecting a fixed ring dimension for the entire protocol, the offline planner searches the leading fold levels and selects ring dimensions independently for commitment matrices A, B, and D. Later folds use a uniform D64 suffix.

The selected schedules are generated ahead of time and stored in the schedule catalog. Proving and verification only load and validate the generated row; they do not run the planner.

## What changed

### Declarative ring-dimension policy

`CommitmentConfig` now describes its ring-dimension behavior using:

```rust
RingDimensionScheduleMode
```

with two variants:

```rust
UniformDimension {
    ring_dimension,
}

AdaptiveDimension {
    num_search_levels,
    uniform_suffix_dimension,
    potential_a_dimensions,
    potential_b_dimensions,
    potential_d_dimensions,
}
```

Uniform configurations retain the existing single-dimension planner path.

Adaptive configurations use a bounded offline search over the configured dimensions and fold levels.

### Adaptive planner

For the fp128 one-hot configuration, the planner:

- Searches ring dimensions at L0 and L1.
- Uses D64 uniformly from L2 through the terminal fold.
- Searches A dimensions from `[64, 128, 256]`.
- Derives B and D from `[64, 128]`.
- Requires A dimensions to be non-increasing across searched levels.
- Allows adjacent A dimensions to be equal.
- Computes the exact secure rank for each matrix candidate.

A is the branching search dimension. B and D do not introduce additional planner branches. For each selected A geometry, the planner scans their allowed dimensions and selects the dimension with the smallest secure rank:

- Stop increasing the dimension once rank 1 is reached.
- If rank 1 is unavailable, retain the dimension producing the smallest rank.
- Prefer the smaller dimension when two dimensions produce the same rank.

Complete schedules are selected using the following ordering:

1. A rank and A dimension at each searched level.
2. Physical preprocessing-matrix footprint.
3. Estimated proof size.
4. Canonical schedule descriptor bytes as the deterministic tie-break.

This allows larger dimensions when they materially reduce rank without requiring rank 1 in every case.

### Default fp128 one-hot configuration

`fp128::OneHot` is now the canonical direct fp128 one-hot preset.

It uses:

```text
Setup-generation dimension: D256
Adaptive levels: L0 and L1
A candidates: D64, D128, D256
B candidates: D64, D128
D candidates: D64, D128
Uniform suffix: D64
```

For the generated `nv = 36` row, the resulting dimensions are:

```text
L0:  A/B/D = 256/128/128
L1:  A/B/D = 256/64/64
L2+: A/B/D = 64/64/64
Terminal: D64
```

The generated table currently contains direct one-hot rows for `nv = 32` and `nv = 36`.

The former standalone uniform D64 one-hot catalog has been removed.

### Default fp128 dense configuration

`fp128::Dense` is the canonical adaptive dense preset. It uses the same D256
setup-generation envelope, two searched leading levels, candidate dimensions,
and uniform D64 suffix as `fp128::OneHot`, while retaining the dense coefficient
bounds and honest-fold policy.

The generated `fp128_dense` catalog covers the existing fp128 dense scalar and
batched key set. `fp128_dense_precommitted` stores the corresponding standalone
precommitted descriptors. The former standalone uniform D64 dense catalog has
been removed; existing multi-chunk paths retain their dedicated D64 catalogs.

### Generated catalog identity

Generated schedule identity now includes the complete `RingDimensionScheduleMode`, including:

- Number of adaptive levels.
- Uniform suffix dimension.
- Ordered A candidate dimensions.
- Ordered B candidate dimensions.
- Ordered D candidate dimensions.

Catalog validation checks that:

- Every generated fold uses dimensions admitted by the policy.
- A dimensions are non-increasing across adaptive levels.
- The uniform suffix begins at the configured level.
- The terminal uses the configured suffix dimension.
- Every potential A dimension is covered by the challenge-configuration identity.
- Policy changes invalidate stale generated tables.

Runtime proving and verification remain catalog-only. A missing row is rejected rather than falling back to planner execution.

### Configuration and catalog cleanup

Because adaptive one-hot is now the default fp128 configuration, this PR removes redundant fp128 catalogs and presets:

- D128 one-hot
- D128 one-hot precommitted
- D256 one-hot generated catalog
- D256 one-hot precommitted catalog
- D128 dense
- D128 dense precommitted

The tableless `D256OneHot` marker remains available for experiments that need a fixed D256 configuration, but it no longer has a shipped runtime catalog.

The previous `fp128_mixed_dim_onehot` naming has also been removed. The canonical generated family is now simply:

```text
fp128_onehot
```

and the corresponding profiling mode remains:

```text
onehot_fp128
```

### Profiling and CI

Profiling documentation, feature selection, benchmark dispatch, and CI coverage have been updated for the canonical adaptive fp128 one-hot family.

The profiler reports the actual A/B/D dimensions and matrix ranks selected at each fold, making adaptive schedule choices directly observable.

## Scope and limitations

This PR supports adaptive dimensions for direct fp128 one-hot and dense schedules.

The following paths remain on their existing uniform-D configurations:

- fp32 and fp64 presets
- recursive setup offloading
- distributed/multi-chunk proving
- heterogeneous multi-group roots

In particular, adaptive dimensions and recursive setup offloading are not yet combined. Recursive and distributed fp128 one-hot configurations continue to use their existing D64 catalogs.

## Compatibility

This is an intentional breaking configuration and generated-catalog change.

Users of the old fp128 D128/D256 one-hot generated families or the `fp128_mixed_dim_onehot` names should migrate to:

```rust
akita_config::proof_optimized::fp128::OneHot
```

Standalone direct fp128 use should select `fp128::OneHot` or `fp128::Dense`.

## Validation

The implementation includes coverage for:

- Adaptive policy validation.
- A-dimension monotonicity.
- Independent B/D rank minimization.
- Rank greater than one when no candidate reaches rank one.
- Uniform suffix enforcement.
- Generated-row expansion and replay.
- Catalog identity drift.
- Planner/generated schedule parity.
- Deterministic candidate selection.
- Honest proving and verification.
- Wrong openings and commitment/proof tampering.
- Malformed or unsupported dimensions.
- Setup-capacity validation.
- Preservation of existing uniform planner paths.

Documentation guardrails and generated-catalog consistency checks pass.
