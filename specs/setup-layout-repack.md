# Spec: Packed Setup Layout Repack

| Field     | Value |
|-----------|-------|
| Author(s) | Quang Dao |
| Created   | 2026-05-27 |
| Status    | proposed |
| Branch    | `quang/setup-layout-repack` |
| PR         | #112 |

## Summary

Akita currently stores one shared public setup vector in `FlatMatrix`, but most
call sites view that vector through a single rectangular envelope:

```text
ring_view(role_rows, setup.seed.max_stride)
```

where `max_stride` is the maximum column width needed by any A, B, or D role
over all supported shapes. This makes every role row pay for the same stride,
even when its natural width is smaller. The extra row holes are harmless for
correctness, but they inflate setup capacity, complicate setup-claim
offloading, and force the weight evaluator to reason about
`row * max_stride + col` pullbacks.

This spec proposes removing the global setup stride contract. Each role should
view the same raw `FlatMatrix` prefix using its natural width:

```text
A: ring_view(n_a, a_setup_width)
B: ring_view(n_b, b_setup_width)
D: ring_view(n_d, d_setup_width)
```

The setup still consists of one shared flat random vector. The roles are still
prefix views of that vector; they are not disjoint stored matrices. The only
change is that row layout is packed per role instead of padded to one global
stride.

## Motivation

### Setup Size

Today setup generation allocates:

```text
max_rows * max_stride
```

ring elements at the setup generation dimension.

The packed layout only needs:

```text
max_setup_len = max over supported levels/shapes of {
  n_a * a_setup_width,
  n_b * b_setup_width,
  n_d * d_setup_width
}
```

This can be substantially smaller whenever the largest row count and largest
column width occur in different roles or different schedule shapes.

### Setup Claim Offloading

The offloaded setup matrix claim should be over the raw setup vector:

```text
M_raw(shared_idx, coeff_idx)
```

With the current stride layout, role coordinates map to raw indices as:

```text
shared_idx = role_row * max_stride + role_col
```

At a random sumcheck point, evaluating the corresponding weight requires a
mixed-radix equality pullback with carry behavior from `max_stride`.

With packed role views, the role maps become:

```text
A: shared_idx = a_row * a_setup_width + a_col
B: shared_idx = b_row * b_setup_width + b_col
D: shared_idx = d_row * d_setup_width + d_col
```

where each width is the actual role width. This is still a pullback, but it is
the intended packed setup layout rather than an artifact of an envelope stride.

## Current Layout

The relevant storage type is:

```text
crates/akita-types/src/layout/flat_matrix.rs
```

`FlatMatrix` stores raw field data plus `gen_ring_dim`. At verifier ring
dimension `D`, it can be viewed as any prefix of D-sized ring elements:

```text
ring_view::<D>(num_rows, num_cols)
```

The storage layer is already flexible enough. The global stride is imposed by
metadata and call sites:

```text
AkitaSetupSeed {
    max_stride,
    ...
}

CommitmentConfig::max_setup_matrix_size(...) -> (max_rows, max_stride)

AkitaProverSetup::generate_with_capacity(...)
    derives max_rows * max_stride ring elements
```

Then prover and verifier paths pass `setup.seed.max_stride` to role views.

Examples:

```text
commit_inner_witness(..., setup.expanded.seed.max_stride)
mat_vec_mul_ntt_single_i8(..., setup.expanded.seed.max_stride, ...)
setup.shared_matrix.ring_view::<D>(params.a_key.row_len(), setup.expanded.seed.max_stride)
setup.shared_matrix.ring_view::<D>(r_max, setup.seed.max_stride)
```

## Target Layout

### Role Widths

For a level/proof shape, define the setup role widths:

```text
a_setup_width = lp.inner_width()
              = lp.block_len * lp.num_digits_commit

d_setup_width = prepared.depth_open * prepared.num_blocks * prepared.num_claims
              = n_cols_w

b_setup_width = max(num_polys_per_point) * n_a * prepared.depth_open * prepared.num_blocks
              = n_cols_t
```

The B width is intentionally based on `max(num_polys_per_point)`, not the total
number of T vectors, because the current verifier/prover grouped layout uses a
group-local B matrix width and maps each point's `poly_idx` through
`group_offsets[point_idx]` only when forming the `r_x` equality target.

Under `feature = "zk"`, the natural B/D widths must include their role-local
blinding tails:

```text
b_setup_width_zk = b_setup_width + b_blinding_digit_planes_per_point * max_group_count_policy
d_setup_width_zk = d_setup_width + d_blinding_segment_len
```

The implementation should pin the exact formulas at the ZK call sites; this
spec's non-ZK formulas are the canonical starting point.

### Capacity

The config/setup capacity function should return a single packed length:

```text
max_setup_len
```

computed as the maximum role footprint across all generated setup levels and
supported batch shapes:

```text
max_setup_len = max(
  n_a * a_setup_width,
  n_b * b_setup_width,
  n_d * d_setup_width
)
```

with feature-gated ZK width extensions included when built with
`feature = "zk"`.

### Role Views

Introduce role-view helpers instead of scattering shape math through prover and
verifier code:

```text
setup_a_view(setup, lp) -> ring_view(n_a, a_setup_width)
setup_b_view(setup, prepared_or_shape) -> ring_view(n_b, b_setup_width)
setup_d_view(setup, prepared_or_shape) -> ring_view(n_d, d_setup_width)
```

The helper names and module location are implementation choices. The important
property is that callers stop spelling `setup.seed.max_stride`.

## Protocol Impact

This is a protocol-visible setup layout change.

- The setup seed serialization changes.
- Setup descriptor digests change.
- Old expanded setups should be rejected rather than adapted.
- Existing proof bytes do not need to remain valid.
- No backward-compatibility shim is required.

The public setup remains transparent and deterministic. A vector prefix of
uniform random ring elements is still uniform random, and every role matrix is
still obtained from a prefix of the same shared vector. The security argument
does not rely on the artificial row padding created by `max_stride`.

## Implementation Plan

### 1. Add Packed Setup Envelope Types

Replace the two-number envelope:

```text
(max_rows, max_stride)
```

with an explicit packed-capacity value. Prefer a named type to prevent
regressing into row/stride thinking:

```text
SetupMatrixEnvelope {
    max_setup_len: usize,
}
```

If a named type creates too much churn in the first patch, a direct
`max_setup_len: usize` return is acceptable, but the spec recommends a type.

Affected areas:

```text
crates/akita-config/src/lib.rs
crates/akita-config/src/proof_optimized.rs
crates/akita-planner/src/test_utils.rs
crates/akita-setup/src/lib.rs
crates/akita-prover/src/api/setup.rs
```

### 2. Change Setup Seed Metadata

Change:

```text
AkitaSetupSeed {
    max_stride,
    ...
}
```

to:

```text
AkitaSetupSeed {
    max_setup_len,
    ...
}
```

Update serialization, validation, descriptor digests, and tests.

Old setups should fail descriptor/cache validation naturally because both the
seed bytes and shared matrix length expectations change.

Affected areas:

```text
crates/akita-types/src/proof/setup.rs
crates/akita-types/src/instance_descriptor.rs
crates/akita-types/src/proof/batch.rs
crates/akita-verifier/src/proof/claims.rs
```

### 3. Generate Packed Setup Capacity

Change setup generation from:

```text
derive_public_matrix_flat(max_rows * max_stride, seed)
```

to:

```text
derive_public_matrix_flat(max_setup_len, seed)
```

The NTT cache can continue to build over:

```text
ring_view(1, total_ring_elements_at_D)
```

because it is already over the full flat vector and does not depend on role
stride.

### 4. Cut Over A/B/D Role Reads

Replace every role read that uses `setup.seed.max_stride` with a natural-width
view. Important paths include:

```text
crates/akita-prover/src/api/commitment.rs
crates/akita-prover/src/backend/{dense,onehot,sparse_ring,recursive_witness,field_reduction,multilinear_polynomial}.rs
crates/akita-prover/src/protocol/ring_switch.rs
crates/akita-verifier/src/protocol/batched.rs
crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs
crates/akita-verifier/src/protocol/slice_mle/zk_blinding.rs
```

Where the backend trait currently accepts `matrix_stride`, rename or replace it
with the natural role width. This is an API cleanup, not a compatibility layer.

### 5. Rewrite Fused Setup Contribution

Today `compute_setup_contribution` fuses A/B/D through one temporary view:

```text
ring_view(r_max, setup.seed.max_stride)
```

After repacking, it should compute the same scalar as the sum of packed role
prefix contributions:

```text
D contribution over d_row * d_setup_width + d_col
B contribution over b_row * b_setup_width + b_col
A contribution over a_row * a_setup_width + a_col
```

This can still share precomputed column weights, but it should no longer need
the stride-based row/column envelope.

This step is the riskiest part of PR 01. It should land with equivalence tests
that compare old-style fixture expectations to the new packed calculation on
small shapes.

### 6. Update Validation

Verifier-reachable validation must reject undersized setups with `AkitaError`,
not panic.

New checks should prove:

```text
setup.shared_matrix.total_ring_elements_at::<D>() >= required_role_len
```

for every role view requested. Avoid unchecked indexing in hot verifier paths;
validate once at setup/prepared-layout boundaries where possible.

## Tests

Minimum tests for PR 01:

- `FlatMatrix` can view the same raw vector through multiple packed shapes.
- Setup generation creates exactly `max_setup_len` ring elements.
- Cache validation rejects old/smaller setup artifacts.
- A/B/D role-view helpers reject insufficient setup length.
- `compute_setup_contribution` matches the existing test fixture after changing
  the fixture from `r_max * max_stride` to packed role widths.
- Direct witness recomputation still verifies root direct commitments.
- ZK blinding tests continue to pass under `feature = "zk"` or are explicitly
  adjusted if the repository does not currently run them in default CI.

Required commands before making the implementation PR ready:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

## Non-Goals

- No setup claim offloading proof in this PR.
- No new matrix-claim sumcheck.
- No `f`/meta commitment work; that belongs to the three-tier commitment PR.
- No Jolt-style path-generated stack automation.
- No backward-compatibility support for old setup artifacts.

## Open Questions

1. Should `SetupMatrixEnvelope` record only `max_setup_len`, or also debug-only
   maxima for A/B/D role footprints to improve diagnostics?
2. Should role-view helpers live in `akita-types` next to `FlatMatrix`, or in
   prover/verifier-facing modules that already know `LevelParams` and prepared
   batch shape?
3. Under `feature = "zk"`, should B blinding capacity be sized by
   `max(num_polys_per_point)` or by an explicit per-point/group policy? The
   implementation should follow the existing group-local commitment semantics.
4. Do generated schedule tables need to encode any new setup-length objective,
   or is the packed capacity purely a setup-envelope computation over existing
   `LevelParams`?

## Acceptance Criteria

- `max_stride` is removed from `AkitaSetupSeed`.
- Setup capacity is expressed as packed `max_setup_len`.
- Setup generation no longer allocates `max_rows * max_stride`.
- Prover and verifier role matrix views use natural widths.
- `compute_setup_contribution` no longer depends on a global setup stride.
- Existing proofs generated in-tree verify under the new setup layout.
- Documentation in `STACK.md` remains accurate for the rest of the stack.
