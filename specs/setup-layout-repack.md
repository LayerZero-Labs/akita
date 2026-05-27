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

ZK blinding tails should not be folded into this packed base matrix. They have
different, group-local semantics, they are much smaller than the base W/T/Z
segments, and they do not need to participate in the main setup-claim
offloading path. The target design is:

```text
base setup seed/matrix: A, B, D base matrices only
ZK setup seed/domain:   small B-blinding and D-blinding matrices
```

The verifier can evaluate the ZK blinding terms directly from the ZK setup
seed, while the offloaded matrix claim remains over the raw base setup matrix
`M_setup`.

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

These are setup-view widths, not necessarily the serialized key column widths
stored in `LevelParams`. The existing verifier checks that
`lp.a_key.col_len()`, `lp.b_key.col_len()`, and `lp.d_key.col_len()` are large
enough for the runtime shape. PR 01 should preserve that distinction: compute
the packed role width from the prepared proof shape, validate the corresponding
key width is at least that large, and only then take the packed setup view. In
particular, do not blindly substitute `lp.outer_width()` for the grouped B
setup width unless the shape really is the same.

These widths are the base setup widths. They intentionally exclude ZK blinding
tail columns.

### ZK Blinding Split

Under `feature = "zk"`, B/D blinding terms should be generated from a separate
ZK setup seed or domain, not by extending the base B/D setup widths.

Use one seed field with role-separated PRG labels, or two explicit seed fields:

```text
zk_blinding_seed

ZK_B_BLINDING(point_idx, b_row, local, coeff_idx)
ZK_D_BLINDING(d_row, local, coeff_idx)
```

The B and D ZK shapes remain:

```text
b_blinding_digit_planes_per_point =
  blinding_digit_plane_count(n_b, D, log_basis)

b_blinding_segment_len =
  num_points * b_blinding_digit_planes_per_point

d_blinding_segment_len =
  blinding_digit_plane_count(n_d, D, log_basis)   // intermediate M-row layout
  0                                               // terminal M-row layout
```

B differs from D because B blinding is point/group-local. The witness segment
contains `num_points` consecutive B blinding groups, but the setup randomness
needed for each group is only:

```text
ZK_B_BLINDING(point_idx, b_row, local)
where local < b_blinding_digit_planes_per_point
```

D has one global tail:

```text
ZK_D_BLINDING(d_row, local)
where local < d_blinding_segment_len
```

The verifier should evaluate these terms directly:

```text
b_blinding_part =
  sum_{idx = 0}^{b_blinding_segment_len - 1}
    eq_full_vec(b_blinding_offset + idx)
    * sum_{b_row = 0}^{n_b - 1}
        eq_tau1[b_start + point_idx * n_b + b_row]
        * eval_ring_at_pows(
            ZK_B_BLINDING(point_idx, b_row, local),
            alpha_pows
          )

where point_idx = idx / b_blinding_digit_planes_per_point
      local     = idx % b_blinding_digit_planes_per_point

d_blinding_part =
  sum_{local = 0}^{d_blinding_segment_len - 1}
    eq_full_vec(d_blinding_offset + local)
    * sum_{d_row = 0}^{n_d_active - 1}
        eq_tau1[d_start + d_row]
        * eval_ring_at_pows(
            ZK_D_BLINDING(d_row, local),
            alpha_pows
          )
```

This local verifier work is intentionally outside the base `M_setup` matrix
claim. If we later want to offload ZK blinding too, that should be a separate
small claim over the ZK setup seed/domain, not an expansion of the base setup
matrix.

The verifier cost is proportional to the number of ZK blinding digit planes:

```text
O(num_points * n_b * b_blinding_digit_planes_per_point
  + n_d_active * d_blinding_segment_len)
```

That is the small, irregular term we intentionally keep local. The base
offloading path should not pay for these columns in `M_setup`, and the ZK PRG
indices should not include `group_message_planes` or `w_len`; those are witness
segment offsets, not physical columns in the base setup matrix.

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

Do not add ZK blinding tail widths to `max_setup_len`. Those tail matrices are
owned by the separate ZK setup seed/domain described above.

This is a maximum, not a sum. A/B/D are all prefix views over one shared
`FlatMatrix`; they are not three disjoint matrices placed back-to-back. The
same physical `shared_idx` may be reached by more than one role view. That
aliasing is intentional and the setup-claim weight for a physical index is the
sum of all role contributions that land there. PR 01 changes the physical
aliasing pattern from `role_row * max_stride + role_col` to natural-width
prefix maps, so the setup descriptor and cache identity must treat it as a new
layout.

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

- The base setup seed serialization or setup layout domain tag changes.
- Under `feature = "zk"`, the ZK blinding seed/domain is distinct from the base
  setup seed/domain.
- Setup descriptor digests and disk cache keys change.
- Old expanded setups and old disk-cache artifacts are unsupported in the new
  format. They should not be decoded, adapted, or migrated.
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
The type should also expose non-protocol diagnostics for the maximum A, B, and
D role footprints, or an equivalent "forcing role" field, so cache/setup
rejections can report which role shaped the packed length. Those diagnostics
must not become additional proof metadata.

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
    public_matrix_seed,      // base A/B/D setup only
    zk_blinding_seed,        // feature = "zk"; or equivalent domain separation
    ...
}
```

Update serialization, validation, descriptor digests, and tests.

This is a full cutover. Do not add legacy decoding, migration, or compatibility
shims. Do add a semantic layout boundary so the packed format is unambiguous.
The preferred option is a serialized setup-layout domain tag in
`AkitaSetupSeed`. An acceptable alternative is a fixed packed-setup domain tag
in every setup artifact digest, instance descriptor, and disk cache file name.
Do not introduce numbered setup/cache names. Under `feature = "zk"`, the
ZK blinding seed/domain must also be bound in the descriptor.

The setup load/validation path must also check the physical matrix length at
the setup generation dimension against the packed envelope. For the first
implementation, use the simple exact rule:

```text
setup.shared_matrix.total_ring_elements() == setup.seed.max_setup_len
```

Role views at smaller ring dimensions can still use
`total_ring_elements_at::<D>()`; that is a view-capacity check, not the seed's
physical-length identity check.

Do not support larger cached supersets in this PR. Cache reuse is exact for the
requested setup metadata; incompatible artifacts are rejected or regenerated.

Affected areas:

```text
crates/akita-types/src/proof/setup.rs
crates/akita-types/src/instance_descriptor.rs
crates/akita-types/src/proof/batch.rs
crates/akita-setup/src/lib.rs
crates/akita-prover/src/api/setup.rs
crates/akita-verifier/src/proof/claims.rs
```

### 3. Add ZK Blinding Setup Domain

When `feature = "zk"` is enabled, stop reading B/D blinding columns from
`setup.shared_matrix`. Introduce a small setup evaluator such as:

```text
zk_b_blinding_ring(setup, point_idx, b_row, local) -> CyclotomicRing<F, D>
zk_d_blinding_ring(setup, d_row, local) -> CyclotomicRing<F, D>
```

Both functions derive ring coefficients from `zk_blinding_seed` with
role-separated labels. They should be used by both prover witness construction
and verifier replay. They should not build a large `FlatMatrix` view and should
not extend `b_setup_width` or `d_setup_width`.

This is part of the same setup-layout branch. Do not create a separate layout
sub-stack for ZK blinding. PR 01 should route the blinding tails through this
separate domain rather than preserving the old shared-stride columns as a
temporary shim.

### 4. Generate Packed Setup Capacity

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

This statement only applies to the cache material itself. Any kernel that
turns the flat cache back into logical role rows must be cut over to packed
role widths.

### 5. Cut Over A/B/D Role Reads

Replace every role read that uses `setup.seed.max_stride` with a natural-width
view. Important paths include:

```text
crates/akita-prover/src/api/commitment.rs
crates/akita-prover/src/backend/{dense,onehot,sparse_ring,recursive_witness,field_reduction,multilinear_polynomial}.rs
crates/akita-prover/src/kernels/linear.rs
crates/akita-prover/src/protocol/ring_switch.rs
crates/akita-verifier/src/protocol/batched.rs
crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs
crates/akita-verifier/src/protocol/slice_mle/zk_blinding.rs
```

Where the backend trait currently accepts `matrix_stride`, rename or replace it
with the natural role width. This is an API cleanup, not a compatibility layer.
For `zk_blinding.rs`, the cutover is stronger: remove its dependency on the
base shared-matrix B/D views for blinding tails and use the ZK setup evaluator.

### 6. Cut Over Fused NTT Quotient Kernel

`fused_split_eq_quotients` currently accepts one `stride` and slices cached NTT
rows as:

```text
&cache[i * stride .. (i + 1) * stride]
```

That is a same-stride invariant, not just an implementation detail. PR 01 must
replace this API with one that can address the packed A/B/D role rows
independently. A reasonable shape is:

```text
D-cyclic rows: d_row * d_setup_width + d_col
B-cyclic rows: b_row * b_setup_width + b_col
A-cyclic rows: a_row * a_setup_width + a_col
A-neg rows:    a_row * a_setup_width + a_col
```

The helper below the dispatch layer should receive separate row slices for D,
B, and A rather than one shared `cyc_rows` array. This keeps the one-pass/tiled
cache reuse optimization, but removes the false assumption that D, B, and A
have the same physical row width.

### 7. Rewrite Fused Setup Contribution

Today `compute_setup_contribution` fuses A/B/D through one temporary view:

```text
ring_view(r_max, setup.seed.max_stride)
```

After repacking, it should compute the same scalar as the sum of packed role
prefix contributions for the base setup matrix:

```text
D contribution over d_row * d_setup_width + d_col
B contribution over b_row * b_setup_width + b_col
A contribution over a_row * a_setup_width + a_col
```

This can still share precomputed column weights, but it should no longer need
the stride-based row/column envelope. ZK blinding parts are not included in
this base setup contribution; they are evaluated by the separate ZK blinding
path.

This step is the riskiest part of PR 01. It should land with equivalence tests
that compare old-style fixture expectations to the new packed calculation on
small shapes.

### 8. Update Validation

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
- Current-format setup validation checks physical matrix length equals
  `seed.max_setup_len`.
- Cache validation rejects smaller or physically mismatched current-format setup
  artifacts.
- A/B/D role-view helpers reject insufficient setup length.
- `fused_split_eq_quotients` has a test where A, B, and D use different role
  widths, so the old one-stride row slicing would fail.
- `compute_setup_contribution` matches the existing test fixture after changing
  the fixture from `r_max * max_stride` to packed role widths.
- Direct witness recomputation still verifies root direct commitments.
- B/D blinding tests derive their entries from `zk_blinding_seed` and no longer
  allocate tail columns in the base setup matrix.

Required commands before making the implementation PR ready:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

## Non-Goals

- No setup claim offloading proof in this PR.
- No new matrix-claim sumcheck.
- No ZK blinding offload. ZK blinding remains a small local verifier term from
  a separate setup seed/domain.
- No `f`/meta commitment work; that belongs to the three-tier commitment PR.
- No Jolt-style path-generated stack automation.
- No backward-compatibility support for old setup artifacts.

## Open Questions

1. Should role-view helpers live in `akita-types` next to `FlatMatrix`, or in
   prover/verifier-facing modules that already know `LevelParams` and prepared
   batch shape?
2. Do generated schedule tables need to encode any new setup-length objective,
   or is the packed capacity purely a setup-envelope computation over existing
   `LevelParams`?

## Acceptance Criteria

- `max_stride` is removed from `AkitaSetupSeed`.
- Setup layout domain separation makes the packed format explicit
  without legacy decoding or migration.
- Setup capacity is expressed as packed `max_setup_len`.
- Setup generation no longer allocates `max_rows * max_stride`.
- Prover and verifier role matrix views use natural widths.
- ZK B/D blinding tails are not counted in `max_setup_len` and are derived from
  a separate ZK setup seed/domain.
- The fused NTT quotient path no longer has a same-stride row slicing
  invariant.
- `compute_setup_contribution` no longer depends on a global setup stride.
- Existing proofs generated in-tree verify under the new setup layout.
- Documentation in `STACK.md` remains accurate for the rest of the stack.
