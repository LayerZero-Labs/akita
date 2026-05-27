# Spec: Packed Setup Layout Repack

| Field     | Value |
|-----------|-------|
| Author(s) | Quang Dao |
| Created   | 2026-05-27 |
| Status    | proposed |
| Branch    | `quang/setup-layout-repack` |
| PR         | #112 |

## Scope

This PR is a spec-only cleanup PR. It records the target packed setup layout
and names the downstream setup-claim-offloading constraints that the layout
must not obstruct. It does not change Rust code, proof bytes, setup
serialization, generated tables, or benchmarks.

The next implementation branch should re-implement from current `main` using
this spec, not continue the older prototype commits that used to live on this
branch.

The next implementation branch is a pure layout branch. It must not introduce
setup-prefix commitments, setup-claim delegation, a setup product sumcheck, or
new proof objects. It may touch existing prover/verifier paths only where those
paths already consume A/B/D setup rows and therefore must be taught the packed
role views.

## Summary

Akita currently stores one shared public setup vector in `FlatMatrix`, but most
call sites view that vector through one rectangular envelope:

```text
ring_view(role_rows, setup.seed.max_stride)
```

Here `max_stride` is the maximum column width needed by any A, B, or D role
over all supported shapes. This padding is harmless for correctness, but it
does three bad things:

- it inflates setup capacity;
- it makes the physical layout look like a single global row stride even
  though A, B, and D have different natural widths;
- it complicates setup-claim offloading by forcing the weight evaluator to
  reason about `row * max_stride + col` instead of the actual role map.

The target layout removes the global setup-stride contract. There is still one
shared random setup object, denoted `S`, but A, B, and D are packed prefix views
of it:

```text
A: ring_view(n_a, a_setup_width)
B: ring_view(n_b, b_setup_width)
D: ring_view(n_d, d_setup_width)
```

These role matrices are not stored disjointly as `A || B || D`. They overlap as
prefix views of the same raw setup vector. If a raw setup index is reached by
more than one role view, later setup-claim offloading adds the corresponding
role weights at that shared coordinate.

ZK blinding tails are not part of this base setup object. They have separate,
smaller, point-local semantics and should use a dedicated ZK setup seed/domain.

## Notation

`S` is the base shared setup object. For the layout branch, `S[lambda]` is a
ring element. Later setup-claim offloading will expose its coefficient axis:

```text
S(lambda, y) = coefficient y of S[lambda]
```

The flattened coefficient vector is:

```text
S^flat[lambda * D_setup + y]
```

If later setup-offloading code or prose uses `M_setup`, it should mean the
selected committed prefix of this same object, not `S_full` and not an
alpha-evaluated matrix. The layout branch itself should not introduce that
commitment.

The alpha-evaluated matrices are:

```text
A_alpha, B_alpha, D_alpha
```

They are verifier-local evaluations of ring entries at the Stage-2 point
`alpha`. They are not preprocessing commitment targets.

## Current Layout

`FlatMatrix` is already flexible enough for the target representation. It
stores raw field data plus a generation ring dimension, and it can view a
prefix as a matrix of ring elements:

```text
ring_view::<D>(num_rows, num_cols)
```

The global stride is imposed by setup metadata and call sites:

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
Representative uses include inner witness setup multiplication, outer B setup
multiplication, root-direct recomputation, ring-switch quotient kernels, and
`compute_setup_contribution`.

## Target Base Setup Layout

### Role Dimensions

For one concrete level/proof shape, define:

```text
W_A = a_setup_width
W_B = b_setup_width
W_D = d_setup_width
```

The base active widths are:

```text
W_A = block_len * depth_commit
```

```text
W_D = num_claims * num_blocks * depth_open
```

```text
W_B = max(num_polys_per_point) * n_a * num_blocks * depth_open
```

The maximum over `num_polys_per_point` is deliberate. A B row is
point/group-local: at point `p`, the row sees only the `n_p` polynomial slots
opened at that point. A single packed B width is still used; slots beyond the
local `n_p` are zero for that point.

The base packed setup footprint for the active shape is:

```text
N_active^R = max(n_a * W_A, n_b * W_B, n_d * W_D)
```

This counts ring slots. For coefficient-level setup-claim delegation:

```text
N_active^F = D_setup * N_active^R
```

A/B/D dimensions are not generally powers of two. Only protocol axes such as
`num_blocks = 2^r` are power-of-two aligned.

### Capacity

The implementation branch should replace the two-number envelope:

```text
(max_rows, max_stride)
```

with an explicit packed capacity:

```text
SetupMatrixEnvelope {
    max_setup_len: usize,   // ring slots at setup generation dimension
}
```

where:

```text
max_setup_len = max over supported levels/shapes of {
  n_a * W_A,
  n_b * W_B,
  n_d * W_D
}
```

This is a maximum, not a sum. A/B/D remain overlapping prefix views of one
shared setup vector.

Do not add ZK blinding tail widths to `max_setup_len`.

### Setup Seed Metadata

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
    zk_blinding_seed,        // or equivalent separated ZK domain
    ...
}
```

This is a protocol-visible setup layout change:

- setup seed serialization or a setup-layout domain tag changes;
- setup descriptor digests and disk cache keys change;
- old expanded setups and old cache artifacts are unsupported;
- no backward-compatibility shim is required.

Use exact cache identity initially:

```text
setup.shared_matrix.total_ring_elements() == setup.seed.max_setup_len
```

Role views at smaller ring dimensions may still use
`total_ring_elements_at::<D>()`; that is a view-capacity check, not the seed's
physical-length identity check.

### Role View Helpers

The implementation should centralize role-view construction rather than
scattering shape math:

```text
setup_a_view(setup, dimensions) -> ring_view(n_a, W_A)
setup_b_view(setup, dimensions) -> ring_view(n_b, W_B)
setup_d_view(setup, dimensions) -> ring_view(n_d, W_D)
```

The exact module is an implementation choice. The invariant is that prover and
verifier callers stop spelling `setup.seed.max_stride`.

## Role Column Order

The setup role column order is a view used by the current folding step. It is
not a separate committed setup object.

Root witnesses are digit-fast today, and root one-hot commitment only occurs at
the root. Therefore root setup views must stay digit-fast, including the A
view.

Recursive folded witnesses are block-fast. For recursive levels where we use
setup-claim delegation, D/B should use block-fast views, and recursive A should
also use a block-fast view if recursive setup offloading is enabled.

Let `B = 2^r`, `delta = depth_open`, `delta_c = depth_commit`,
`L = block_len`, `K = max_p n_p`, and `C = num_claims`.

Root digit-fast D/B/A views:

```text
j_D_root(c, b, d) = (c * B + b) * delta + d
```

```text
j_B_root(s_p, b, a, d)
  = s_p * (n_a * delta * B) + b * (n_a * delta) + a * delta + d
```

```text
j_A_root(b_z, d_c) = b_z * delta_c + d_c
```

Recursive block-fast D/B views:

```text
j_D_rec(c, b, d) = b + B * (c + C * d)
```

```text
j_B_rec(s_p, b, a, d)
  = b + B * (s_p + K * (a * delta + d))
```

Optional recursive block-fast A view:

```text
j_A_rec(b_z, d_c) = b_z + L_bar * d_c
```

where `L_bar` is the power-of-two block width used for the recursive A view.
This may pad the recursive A column view, so it should be enabled only if the
recursive setup-offload evaluator benefits from it.

The root A constraint is not optional: root A must remain digit-fast for
efficient one-hot folding.

## NTT Cache and Kernels

The NTT cache material can still be built over the flat setup prefix:

```text
ring_view(1, total_ring_elements_at_D)
```

The cache itself does not need a global row stride. The important hot-path
invariant is that kernels consume contiguous row slices for the role view they
are multiplying.

Any kernel that turns the flat cache back into logical rows must accept
role-specific widths and, where relevant, a role-column-order view:

```text
D-cyclic rows: d_row * W_D + d_col
B-cyclic rows: b_row * W_B + b_col
A-cyclic rows: a_row * W_A + a_col
A-neg rows:    a_row * W_A + a_col
```

`fused_split_eq_quotients` must not keep the same-stride invariant:

```text
&cache[i * stride .. (i + 1) * stride]
```

for all roles. It should receive separate D/B/A row slices or separate
role-width parameters. The goal is to preserve row-contiguous cache access
without forcing a fake global stride.

## ZK Blinding Split

Under `feature = "zk"`, B/D blinding terms should not be columns of the base
setup matrix. They should be derived from a separate ZK setup seed/domain:

```text
ZK_B_BLINDING(point_idx, b_row, local, coeff_idx)
ZK_D_BLINDING(d_row, local, coeff_idx)
```

B is point-local:

```text
b_blinding_digit_planes_per_point =
  blinding_digit_plane_count(n_b, D, log_basis)

b_blinding_segment_len =
  num_points * b_blinding_digit_planes_per_point
```

D is global and is absent in terminal M-row layout:

```text
d_blinding_segment_len =
  blinding_digit_plane_count(n_d, D, log_basis)   // intermediate
  0                                               // terminal
```

The verifier can evaluate ZK blinding directly from the ZK domain. This work is
intentionally outside the base `S` claim. If ZK blinding is ever offloaded, it
should be a separate small claim over the ZK domain, not an expansion of the
base A/B/D setup matrix.

## Setup Contribution After Repack

Today the direct verifier can fuse A/B/D through one temporary view:

```text
ring_view(r_max, setup.seed.max_stride)
```

After repacking, the direct verifier should compute the same scalar as the sum
of packed role-prefix contributions:

```text
D contribution over d_row * W_D + d_col
B contribution over b_row * W_B + b_col
A contribution over a_row * W_A + a_col
```

ZK blinding parts are not included in this base setup contribution.

This direct computation is still local verification. It is the correctness
baseline for the later setup-claim-offloading work.

## Deferred Setup-Claim Offloading Context

This section is context for later branches. It is intentionally not part of the
layout implementation branch. The layout branch should only make the current
direct verifier/prover use packed A/B/D setup views.

### Committed Object

Later setup-claim offloading should commit to the flat coefficient vector of the
setup prefix:

```text
S^flat[lambda * D_setup + y]
```

not to `S(alpha)` and not to alpha-evaluated A/B/D matrices.

The alpha powers live in the structured weight:

```text
omega_S(lambda, y) = omega_bar_S(lambda) * alpha^y
```

This is important: preprocessing can commit to `S`, but it cannot commit to
`S_alpha`, because `alpha` is transcript-dependent.

### Prefix Ladder

Do not force the verifier to pay for `S_full` if the active shape is smaller.

Later preprocessing should commit to one setup commitment for each power-of-two
flat coefficient prefix in a ladder:

```text
N_min <= N <= N_max
```

At runtime:

```text
N_prefix = 2^ceil(log2(N_active^F))
```

Delegate only if:

```text
N_prefix >= N_min
```

Do not round a smaller active claim up to `N_min`; below the threshold, the
verifier should use the direct setup computation.

Initial choices:

```text
D_setup = 32
N_min = 2^23 field coefficients
```

If a proof level has `D != D_setup`, setup delegation is rejected at that level
and direct setup verification is used.

The first offloading implementation should make only the root and, at most, the
first recursive level eligible; the prefix gate decides whether delegation
actually fires.

### Inner Product Shape

The delegated setup value is:

```text
sigma_S = <S_{<= N_setup}, omega_S>
```

where:

```text
omega_bar_S(lambda)
  = sum of D/B/A role weights that pull back to raw setup slot lambda
```

and:

```text
omega_S(lambda, y) = omega_bar_S(lambda) * alpha^y
```

Overlapping A/B/D prefix coordinates are handled by addition in
`omega_bar_S`.

The product-sumcheck terminal point is:

```text
rho = (rho_lambda, rho_y)
```

and the verifier-side weight evaluator should use:

```text
omega_tilde_S(rho_lambda, rho_y)
  = omega_bar_tilde_S(rho_lambda)
    * MLE(1, alpha, ..., alpha^(D_setup - 1))(rho_y)
```

The terminal setup-side value is:

```text
s_rho = S_tilde_{<= N_setup}(rho_lambda, rho_y)
```

The intended recursive route is to carry this selected-prefix setup opening
claim into the next recursive fold and batch it with the folded-witness
opening, rather than verify a nested setup PCS inside the same level.

### A/J Weight

Do not materialize `A J`. Akita already represents the paper's `A J z_hat`
term by moving `J` to the weight side.

At the root, the compact A column is:

```text
j_A_root(b_z, d_c) = b_z * delta_c + d_c
```

The folded z coordinate is block-fast:

```text
x_Z(p, d_f, d_c, b_z)
  = b_z + L * (p + P * (d_f + delta_f * d_c))
```

The useful adjoint vector is:

```text
eta_Z(b_z, d_c)
  = - sum_p sum_{d_f} g_f[d_f]
      eq(r_x, offset_z + x_Z(p, d_f, d_c, b_z))
```

The A contribution to the coefficient-weight tensor is:

```text
omega_A(iota_A(a, j_A_root(b_z, d_c)), y)
  = alpha^y * eq(tau_1, A_a) * eta_Z(b_z, d_c)
```

The root setup-offload evaluator should evaluate the row-aware root A slice
directly. Because root A is digit-fast while the folded z side is block-fast,
the exact evaluator needs a carry-DP style contraction. This is the right
long-term choice because root one-hot requires digit-fast A.

If setup offloading is enabled at a recursive folded level, recursive A may use
the block-fast view so the large `b_z` axis factors as an equality inner
product instead of entering the carry transducer.

### Transcript Binding

The later setup-offload implementation must bind:

- the setup seed/digest and packed layout tag;
- `D_setup`;
- selected `N_setup`;
- the selected prefix commitment;
- the batching/incidence shape;
- `sigma_S`;
- `r_x`, `tau_1`, and `alpha`;
- the role-column view choices used to define `omega_S`.

## Implementation Plan for the Repack Branch

The implementation branch should land as a fresh branch from current `main`.
Suggested commit boundaries:

1. Add packed setup envelope/types and remove `max_stride` from setup metadata.
2. Update setup generation/cache identity to use exact `max_setup_len`.
3. Add role-view helpers and cut over direct A/B/D reads.
4. Cut over existing setup-matrix consumers to pass natural role widths. This
   includes current prover commitment/recommitment routines only because they
   already multiply by A/B setup rows; it does not add new commitments.
5. Cut over fused NTT quotient kernels to role-width row slices.
6. Rewrite direct `compute_setup_contribution` as explicit packed D/B/A sums.
7. Split ZK B/D blinding into the separate ZK setup seed/domain.
8. Add focused equivalence tests and then broader end-to-end tests.

The direct verifier should continue to work before setup-claim offloading is
introduced.

## Tests

Minimum tests for the implementation branch:

- `FlatMatrix` can view the same raw vector through multiple packed shapes.
- Setup generation creates exactly `max_setup_len` ring elements.
- Setup validation checks physical matrix length equals `seed.max_setup_len`.
- Cache validation rejects smaller or physically mismatched setup artifacts.
- A/B/D role-view helpers reject insufficient setup length.
- `fused_split_eq_quotients` covers different D/B/A role widths.
- Direct `compute_setup_contribution` matches the old logical formula on small
  batched fixtures after converting fixtures from `r_max * max_stride` to
  packed role widths.
- Root direct witness recomputation still verifies commitments.
- ZK B/D blinding tests derive entries from `zk_blinding_seed` and no longer
  allocate blinding tail columns in the base setup matrix.

For the later offloading branches:

- materialized `<S_{<= N_setup}, omega_S>` equals direct setup contribution;
- `omega_S` has alpha on the weight side;
- root A/J evaluator matches materialized `eta_Z` and the row-aware A slice;
- recursive block-fast D/B and optional recursive block-fast A evaluators match
  materialized weights;
- selected-prefix opening claims are transcript-bound and batched with the next
  recursive folded-witness opening.

## Non-Goals

- No Rust implementation in this spec-only PR.
- No setup-prefix commitments in the layout repack implementation branch.
- No setup-claim offloading proof in the layout repack implementation branch.
- No new matrix-claim sumcheck in the layout repack implementation branch.
- No ZK blinding offload in the base setup claim.
- No full-`S_full` runtime opening when a shorter committed prefix covers the
  active proof.
- No legacy `max_stride` compatibility layer.
- No materialization of `A J`.
- No change to the physical `w_hat || t_hat || z_hat || r_hat` witness segment
  order beyond selecting role-column views that match the current folding step.

## Open Questions

1. Where should role-view helpers live: `akita-types` near `FlatMatrix`, or
   prover/verifier-facing modules that know the prepared runtime shape?
2. Should generated setup tables eventually advertise selectable prefix sizes,
   or is the prefix ladder purely derived from `max_setup_len`?
3. For recursive setup offloading, is the recursive block-fast A view worth the
   possible `L_bar` padding, or should recursive offload initially restrict to
   D/B if A padding is awkward?
4. What is the exact NTT cache API that keeps contiguous row slices while
   supporting root digit-fast and recursive block-fast views?
5. Should the offload gate remain one global `N_min = 2^23`, or should it later
   depend on the pair `(base field, extension field)` after matrix-MLE
   benchmarks?
6. What is the exact closure rule for the last eligible delegated setup claim
   if there is no subsequent recursive fold available to batch the setup-prefix
   opening?

## Acceptance Criteria for This Spec PR

- The PR diff contains only durable planning/spec documents.
- The spec states that later committed setup offloading uses `S`, not
  `S_alpha`.
- The spec records the prefix-ladder plan and the initial `D_setup = 32`,
  `N_min = 2^23` decisions.
- The spec records root digit-fast and recursive block-fast role-view policy,
  including the root A/one-hot constraint.
- The spec records that ZK blinding is outside the base setup matrix.
- The implementation plan is explicit enough to restart the code work from
  current `main`.
