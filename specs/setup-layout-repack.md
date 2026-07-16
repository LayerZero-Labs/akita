# Spec: Packed Setup Layout Repack

> **Pre-zk-strip historical.** This spec predates the zk-strip
> ([`akita-zk-strip-for-audit.md`](akita-zk-strip-for-audit.md)). References to
> `feature = "zk"` or `zkB`/`zkD` matrices describe removed code preserved on
> `zk-wip`.

| Field     | Value |
|-----------|-------|
| Author(s) | Quang Dao |
| Created   | 2026-05-27 |
| Status    | proposed |
| Suggested branch | `setup-layout-repack` |
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
shared random setup object, labeled `shared` in setup derivation and formulas,
but A, B, and D are packed prefix views of it:

```text
A: ring_view(n_a, a_setup_width)
B: ring_view(n_b, b_setup_width)
D: ring_view(n_d, d_setup_width)
```

These role matrices are not stored disjointly as `A || B || D`. They overlap as
prefix views of the same raw setup vector. If a raw setup index is reached by
more than one role view, later setup-claim offloading adds the corresponding
role weights at that shared coordinate.

ZK blinding tails are not part of this base setup object. Under
`feature = "zk"`, setup materializes separate smaller `zkB` and `zkD`
matrices from the same `public_matrix_seed` under short, fixed labels.

## Notation

`S` is the base shared setup object. Its setup derivation label is `shared`.
For the layout branch, `S[lambda]` is a ring element. Later setup-claim
offloading will expose its coefficient axis:

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
W_A = positions_per_block * depth_commit
```

```text
W_D = num_claims * live_block_count * depth_open
```

```text
W_B = max(num_polys_per_point) * n_a * live_block_count * depth_open
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

A/B/D dimensions and the exact `live_block_count = B` are not generally powers
of two. The Boolean block-index domain `2^r_blk` is power-of-two aligned.

### Capacity

The implementation branch should replace the two-number envelope:

```text
(max_rows, max_stride)
```

with an explicit packed capacity:

```text
SetupMatrixEnvelope {
    max_setup_len: usize,   // base shared ring slots at setup generation dimension
    max_zk_b_len: usize,    // feature = "zk": B-blinding ring slots
    max_zk_d_len: usize,    // feature = "zk": D-blinding ring slots
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

Do not add ZK blinding tail widths to `max_setup_len`. Under
`feature = "zk"`, size them separately as `max_zk_b_len` and `max_zk_d_len`.

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
    max_zk_b_len,            // feature = "zk"
    max_zk_d_len,            // feature = "zk"
    public_matrix_seed,      // labels: shared, zkB, zkD
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

Let `B = live_block_count`, `delta = depth_open`, `delta_c = depth_commit`,
`M = positions_per_block`, `K = max_p n_p`, and `C = num_claims`. The Boolean
block-index domain has size `B_dom = next_power_of_two(B)`; setup role widths
below use exact live `B`, not `B_dom`.

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
setup matrix. They should not add another seed either. Instead, setup
materializes three labeled public matrices from `public_matrix_seed`:

```text
shared[lambda]
  = PRF(public_matrix_seed, "shared", lambda)

zkB[lambda_B]
  = PRF(public_matrix_seed, "zkB", lambda_B)

zkD[lambda_D]
  = PRF(public_matrix_seed, "zkD", lambda_D)
```

The PRF/XOF input tuple is encoded as seed, label, and the listed flat ring
slot index as an unsigned 64-bit little-endian integer under the setup
derivation's length-prefixed field encoding. Coefficients are output
coordinates in `[0, D_setup)`, not additional seed inputs.

These labels are protocol labels, not version strings. If a future incompatible
derivation is needed, change the setup-layout domain around the seed/descriptor
rather than appending a label version.

B blinding witnesses are point-local:

```text
b_blinding_digit_planes_per_point =
  blinding_digit_plane_count(n_b, D, log_basis)

b_blinding_segment_len =
  num_points * b_blinding_digit_planes_per_point
```

The materialized B-blinding setup view is per commitment and reused for each
point with fresh blinding digits:

```text
setup.zk_b_matrix().ring_view(n_b, b_blinding_digit_planes_per_point)

lambda_B =
  b_row * b_blinding_digit_planes_per_point
  + local
```

D is global and is absent in terminal M-row layout:

```text
d_blinding_segment_len =
  blinding_digit_plane_count(n_d, D, log_basis)   // intermediate
  0                                               // terminal
```

The materialized D-blinding view is:

```text
setup.zk_d_matrix().ring_view(n_d, d_blinding_segment_len)

lambda_D =
  d_row * d_blinding_segment_len
  + local
```

Prover and verifier read these stored matrices during protocol replay. They do
not derive ZK blinding rows on demand. Setup validation checks the materialized
`shared`, `zkB`, and `zkD` matrices against `public_matrix_seed` and the seed's
declared lengths.

This work is intentionally outside the base `S` claim. If ZK blinding is ever
offloaded, it should be a separate small claim over the matching ZK matrix, not
an expansion of the base A/B/D setup matrix.

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
setup_weight_S(lambda, y) = setup_index_weight_S(lambda) * alpha^y
```

This is important: preprocessing can commit to `S`, but it cannot commit to
`S_alpha`, because `alpha` is transcript-dependent.

### Prefix Commitments and Slot Policies

Do not force the verifier to pay for `S_full` if the active shape is smaller.

Later setup offloading commits to power-of-two flat coefficient prefixes of
`S`. A full ladder is one useful policy:

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

The prefix object is not just the public commitment. A prover-ready prefix slot
must contain enough witness material to batch the selected setup opening claim
in the next recursive fold:

```text
SetupPrefixSlot {
  id: (setup digest/layout tag, D_setup, N_prefix, commitment params),
  commitment: RingCommitment,
  hint: AkitaCommitmentHint,
  natural_len,
  padded_len = N_prefix,
}
```

The `AkitaCommitmentHint` carries the decomposed inner rows, i.e. the `t_hat`
material produced while committing to the prefix, plus the recomposed inner
rows when available and any ZK blinding digit streams required by the active
feature set. If this hint is not cached with the prefix slot, the prover would
need to recompute it precisely when the setup claim is being batched, which
defeats much of the purpose of preprocessing.

Verifier setup metadata only needs the public half:

```text
SetupPrefixVerifierSlot {
  id,
  commitment,
  natural_len,
  padded_len = N_prefix,
}
```

The preprocessing policy should be explicit, because the right tradeoff differs
between production artifacts and CI benches:

```text
FullLadder(min, max)
  generate every power-of-two prefix in [min, max]

SelectedSlots({N_0, N_1, ...})
  generate only the listed prefix sizes

Disabled
  never delegate setup claims
```

`FullLadder` is the general deployment policy: one setup artifact can serve
many future batching shapes. `SelectedSlots` is the right policy for CI,
benchmark fixtures, and single-config deployments. If a benchmark knows that
only one root shape will ever be exercised, it can generate exactly that root
prefix slot, and perhaps one first-recursive slot later if the benchmark starts
covering L1 delegation.

Missing prefix slots should be handled by configured behavior:

```text
StrictError
  selected slot must already exist, otherwise return a setup/policy error

GenerateAndPersist
  prover-side local/bench convenience: create the missing slot and save it

DirectFallback
  skip delegation for this shape and use the direct setup computation
```

`StrictError` is the clean production mode once preprocessing is meant to be
complete. `GenerateAndPersist` is ergonomic for CI cache warmup and local
experimentation, but the newly generated commitment still has to be transcript
bound and surfaced to the verifier through ordinary metadata. `DirectFallback`
is useful while setup offloading remains an optimization. Protocol code should
not panic on a missing slot; a benchmark harness may choose to panic, but the
library boundary should return an `AkitaError`.

### Inner Product Shape

The delegated setup value is:

```text
sigma_S = <S_{<= N_setup}, setup_index_weight_S>
```

where:

```text
setup_index_weight_bar_S(lambda)
  = sum of D/B/A role weights that pull back to raw setup slot lambda
```

and:

```text
setup_index_weight_S(lambda, y) = setup_index_weight_bar_S(lambda) * alpha^y
```

Overlapping A/B/D prefix coordinates are handled by addition in
`setup_index_weight_S`.

The product-sumcheck terminal point is:

```text
rho = (rho_lambda, rho_y)
```

and the verifier-side weight evaluator should use:

```text
setup_weight_tilde_S(rho_lambda, rho_y)
  = setup_index_weight_tilde_S(rho_lambda)
    * MLE(1, alpha, ..., alpha^(D_setup - 1))(rho_y)
```

The terminal setup-side value is:

```text
s_rho = S_tilde_{<= N_setup}(rho_lambda, rho_y)
```

The intended recursive route is to carry this selected-prefix setup opening
claim into the next recursive fold and batch it with the folded-witness
opening, rather than verify a nested setup PCS inside the same level.

### Product Sumcheck Placement

There are two protocol placements worth distinguishing.

The cleaner first implementation is a post-Stage-2 placement. Current Stage 2
already fuses the Stage-1 norm claim and the relation claim:

```text
gamma * s_claim + relation_claim
  = sum_{x,y} [
      gamma * eq(r_stage1, (x,y)) * W(x,y) * (W(x,y) + 1)
      + W(x,y) * alpha(y) * m_tau1(x)
    ].
```

At the sampled Stage-2 point `r_stage2 = (r_y, r_x)`, the verifier checks:

```text
gamma * eq(r_stage1, r_stage2) * W(r_stage2) * (W(r_stage2) + 1)
  + W(r_stage2) * alpha(r_y) * m_tau1(r_x).
```

Setup offloading changes only the row evaluation:

```text
m_tau1(r_x) = m_local(r_x) + sigma_S.
```

After Stage 2 fixes `r_x`, a new setup product sumcheck proves:

```text
sigma_S = <S_{<=N_setup}, setup_index_weight_S(tau_1, alpha, r_x)>.
```

This adds a Stage 3, but it is conceptually clean: Stage 2 continues to reduce
the witness side to one folded-witness opening claim, and Stage 3 leaves only
the setup-prefix opening claim `s_rho`.

The more compact no-new-stage optimization, and the optimized target for this
protocol, shifts the relation-matrix work back before the setup product
sumcheck and uses Stage 2 for the setup product. The level then runs two
stages, the same as the baseline scheme, instead of three. The setup product
sumcheck depends on the relation point `r_x` through the `G_fold` weighted
A-column vector `eta_Z`, so it cannot start until `r_x` is fixed; the only
reason it lands in a third stage above is that the baseline fuses the relation
into Stage 2, fixing `r_x` only at the very end.

Shifted schedule:

- Stage 1 batches the ring-switched relation directly into the range/norm
  `Q(S)` sumcheck: with a batching scalar `zeta`, one sumcheck proves
  `zeta * 0 + V_alpha = sum_{x,y} [ zeta * eq(tau_0,(x,y)) * Q(W(W+1)) + W(x,y) * alpha(y) * m_tau1(x) ]`,
  fixing the shared point `r_1 = (r_x, r_y)`. Batching drops the
  `eq(tau_0, .)`-factored compact stage-1 message (it no longer composes with
  the tree-based range prover), which we accept because it is cheaper than
  running the relation as a second stage-1 sumcheck. Stage 1 leaves the relation
  witness claim `W(r_1)`, the deferred row value `m_tau1(r_x)`, and the carried
  norm claim `s = W(r_1)(W(r_1)+1)`, whose binding to the witness is deferred to
  the Stage-2 refold.
- Stage 2 runs the setup product sumcheck for
  `m_tau1(r_x) = m_local(r_x) + sigma_S`, reducing `sigma_S` to the carried
  setup opening `s_rho`, exactly as in the Stage-3 slice.

Batching the relation into Stage 1 splits the witness side across the stages:
the relation opens `W` at `r_1`, while the norm claim `s` is bound to the
witness only by the Stage-2 refold (the `s = W(W+1)` virtualization), which
lands at a fresh point. Close the two with one witness claim-reduction
sumcheck, folded into Stage 2 next to the setup product. With `lambda` sampled
from the transcript:

```text
lambda * W(r_1) + s
  = sum_{x,y} eq(r_1, (x,y)) * [
      lambda * W(x,y)
      + W(x,y) * (W(x,y) + 1)
    ]
```

reduces both claims to one opening `W(r_star)` at a fresh point. This is the
same range-refold-plus-relation fusion the baseline already runs in Stage 2,
with the relation term replaced by the `eq(r_1, .)` reduction of the stage-1
relation claim, and it uses the same degree-2 primitive as the extension
opening reduction (EOR). We standardize on the reduction (rather than opening
`W(r_1)` directly) because it keeps the recursive carry at a single witness
opening at the Stage-2 refold point.

The recursive boundary is unchanged: the level emits one folded-witness opening
`(u', r_star, W(r_star))` plus the carried setup-prefix opening
`(C_S, (rho_lambda, rho_y), s_rho)`. No second witness opening is carried.

Cost. Relative to the Stage-3 slice, batching the relation into Stage 1 fixes
`r_x` early enough to run the setup product in Stage 2, removing the third
stage; the relation rides the range sumcheck's `mu'` rounds at no extra degree,
so the witness-domain cost is `deg Q + 2` rounds in both forms. The trade is
that Stage 1 forgoes the `eq(tau_0, .)`-factored compact range prover. The setup
product sumcheck contributes the same
`ceil(log2 N_setup^R) + log2 D_setup` rounds in either placement, and the
claim-reduction term rides the stage-2 refold for free. The net transcript
delta is on the order of a few hundred bytes at level-0 round counts.

Alternative. One can instead keep the range check in its `eq(tau_0, .)`-factored
compact form and run the relation as a second, parallel stage-1 sumcheck. This
preserves the tree-based range prover but spends an extra witness-domain
sumcheck and produces the relation and norm claims at two unrelated points; the
Stage-2 claim reduction keeps the same degree-2 shape either way. We batch the
relation into the range sumcheck instead, trading the compact stage-1 message
for one fewer stage-1 sumcheck.

The Stage-3 post-Stage-2 placement remains the first implementation step: it is
correct, conceptually clean, and reduces the witness side to one opening
without the extra stage-1 relation sumcheck. The optimized relation-shift form
is the target once recursive carried-opening batching exists and the witness
claim-reduction sumcheck is wired.

### Recursive Carried-Opening Boundary

The current recursive protocol boundary is singleton-shaped: it carries one
current witness commitment, one opening point, and one claimed opening value.
Setup offloading needs the recursive boundary to become genuinely batched.

The target recursive carry object is a list of claims:

```text
(commitment, point, value, basis, natural_len, padded_len, kind)
```

The first claim is the ordinary folded-witness opening. When setup offloading
fires, the setup product sumcheck appends the selected-prefix setup claim:

```text
(S_{<=N_setup} commitment, (rho_lambda, rho_y), s_rho, ...)
```

The clean first implementation should use root-style incidence at the recursive
boundary and batch all carried claims in one common padded power-of-two field
domain. Claims whose natural MLE domain is smaller are embedded by zero-padding
the table and fixing the extra point coordinates. This avoids heterogeneous
MLE arity in the first cut and lets the recursive verifier consume one
incidence shape.

If a level has no subsequent recursive fold, setup offloading should be
disabled at that level in the first implementation. A terminal closure rule can
be added later only if benchmarks justify it.

### A and G_fold Weight

Do not materialize `A * G_fold`. Akita keeps `A` as the digit-domain setup
prefix. It applies `G_fold` on the weight side when it builds the setup weights.

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
  = - sum_p sum_{d_f} G_fold[d_f]
      eq(r_x, offset_z + x_Z(p, d_f, d_c, b_z))
```

The A contribution to the setup-weight tensor is:

```text
setup_index_weight_A(iota_A(a, j_A_root(b_z, d_c)), y)
  = alpha^y * eq(tau_1, A_a) * eta_Z(b_z, d_c)
```

Equivalently, the A setup rows represent:

```text
A * z, where z[b_z, d_c]
  = sum_{d_f} G_fold[d_f] * z_hat[b_z, d_c, d_f].
```

This is `A * G_fold * z_hat`. It is not
`A * G_commit * G_fold * z_hat`.

The root setup-offload evaluator should evaluate the row-aware root A slice
directly. Because root A is digit-fast while the folded z side is block-fast,
the evaluator must account for the carry that appears when `offset_z` is added
to the block index. This is the right long-term choice because root one-hot
requires digit-fast A.

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
- the role-column view choices used to define `setup_index_weight_S`.

## Downstream Parallel Work Slices

The setup-layout repack is the shared foundation. After it lands, the rest of
setup offloading should be developed as parallel lanes, then integrated.

### Layout Foundation

This is the implementation branch for this spec. It removes `max_stride`,
introduces `max_setup_len`, centralizes packed A/B/D role views, cuts over all
existing setup-matrix consumers, and moves ZK B/D blinding out of base `S`
into stored `zkB` and `zkD` matrices derived from `public_matrix_seed`. It must
not add setup-prefix commitments, setup product sumchecks, recursive
carried-claim batching, or new proof objects for setup offloading.

### Materialized Inner-Product Oracle

This lane rewrites the direct setup contribution as:

```text
<S_{<=N_setup}, setup_index_weight_S>
```

using a materialized `setup_index_weight_S`. Its purpose is to pin the exact role pullbacks
and give every later branch a correctness oracle. It should test D, B, and
`A * G_fold * z_hat` weights, including the root digit-fast A slice and
recursive block-fast role views.

### Succinct Weight Evaluator

This lane replaces materialized `setup_index_weight_S` on the verifier side. It should
evaluate:

```text
setup_index_weight_tilde_S(rho_lambda)
```

without scanning `S` and without building a dense setup-index equality table.
The hard part is the root A slice with the `G_fold`
weight: root A remains digit-fast for one-hot, while the folded z side is
block-fast. The evaluator must account for the carry that appears when
`offset_z` is added to the block index.

### Prefix Commitment Artifacts

This lane is preprocessing and metadata only. It commits to the power-of-two
prefixes of the flat coefficient vector of `S` and exposes runtime prefix
selection:

```text
N_prefix = 2^ceil(log2(N_active^F))
delegate iff N_prefix >= N_min
N_setup = N_prefix
```

It should implement both full-ladder and selected-slot policies. A selected
slot must be keyed by setup identity, `D_setup`, `N_prefix`, and commitment
parameters. Prover-ready slots store `RingCommitment + AkitaCommitmentHint`
so the later product-sumcheck integration can batch the setup-prefix opening
without recomputing `t_hat`; verifier slots store only the public commitment
and shape metadata.

It does not verify a setup product sumcheck by itself.

### Recursive Carried-Opening Batching

This lane is independent of the setup weight algebra. It generalizes the
recursive proof state from one carried opening to a root-style incidence batch
of carried openings. The singleton recursive protocol should become the
size-one incidence case.

The concrete planner rollout, including per-level direct/recursive selection,
prefix-size gating, and the setup-prefix-plus-witness suffix shape, is specified
in [`setup-offloading-planner.md`](setup-offloading-planner.md).

### Product Sumcheck and Integration

This lane wires the delegated setup claim. It should start as a post-Stage-2
Stage 3 against a materialized setup-opening oracle, then integrate the
succinct `setup_index_weight_S` evaluator, selected prefix commitments, gating policy, and
recursive carried-opening batching.

The no-new-stage relation-shift placement is the optimized target (see Product
Sumcheck Placement). The Stage-3 post-Stage-2 shape is the first implementation
step toward it: it is the correctness baseline and the simplest delegated-claim
wiring. Move to the relation-shift form, with the explicit witness
claim-reduction sumcheck folded into Stage 2, once recursive carried-opening
batching exists.

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
7. Split ZK B/D blinding into stored `zkB`/`zkD` matrices derived from
   `public_matrix_seed`.
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
- ZK B/D blinding tests read entries from stored `zkB`/`zkD` matrices derived
  from `public_matrix_seed` with labels `zkB`/`zkD`, and no longer allocate
  blinding tail columns in the base setup matrix.

For the later offloading branches:

- materialized `<S_{<= N_setup}, setup_index_weight_S>` equals direct setup contribution;
- `setup_index_weight_S` has alpha on the weight side;
- root A evaluator with `G_fold` weights matches materialized `eta_Z` and the
  row-aware A slice;
- recursive block-fast D/B and optional recursive block-fast A evaluators match
  materialized weights;
- selected-slot preprocessing can generate only the known CI/root prefix and
  still exposes a prover-ready `RingCommitment + AkitaCommitmentHint` bundle;
- missing prefix slots obey the configured behavior: strict error,
  generate-and-persist, or direct fallback;
- selected-prefix opening claims are transcript-bound and batched with the next
  recursive folded-witness opening.

## Deferred: Multi-Point B Commitment Kernel

The layout branch commits multi-point B with one cyclic NTT pass per opening
point.
`repeated_b_commitment_rows` loops over the `num_polys_per_point` groups and
calls the single-RHS cyclic mat-vec once per group, zero-padding each group to
the packed `W_B = max(num_polys_per_point) * n_a * live_block_count * depth_open`
width.
Single-point B avoids the extra passes by fusing into the
`fused_split_eq_quotients` pass that already computes D and A.

This single-vs-multi split is a consequence of a missing kernel, not of the
packed B layout itself.
There is no cyclic-domain mat-vec that multiplies one B setup view by several
right-hand-side digit columns in a single cache pass, even though the
negacyclic side already batches multiple RHS over shared column tiles.
Because that kernel does not exist, multi-point B pays `num_points` full B
cache traversals, and the prover forks single-point vs multi-point B in
`compute_r_split_eq` via `use_relation_b_rows`.

The clean follow-up extends `fused_split_eq_quotients` so its B branch loops
over the point groups inside the existing column-tile pass, reusing each loaded
B row across groups.
D and A stay point-independent and unchanged.
This recovers cache amortization for multi-point B and lets single-point be the
`num_points == 1` case, which removes the prover fork and `repeated_b` entirely.
ZK per-group B blinding folds in as a per-group add after the B accumulation.

This is a prover-performance and code-simplification follow-up.
It is out of scope for the layout cutover, which only needs the direct path to
stay correct.
It is independent of setup-claim offloading and can land before or after the
offloading lanes.

## Implemented: Setup Derivation Performance

The packed layout does not change how the shared setup vector is derived, but
the repack branch also tightened the derivation hot path (`setup_expand`),
which dominates setup time.

The derivation keeps its per-element domain separation
(`domain || seed || matrix || index`), so the absorbed byte stream and every
derived ring element are bit-for-bit unchanged.
The implementation now absorbs the fixed `domain || seed || matrix` prefix once
and clones the SHAKE state per element before absorbing only the element index,
fills the flat coefficient buffer in place instead of collecting an
intermediate `Vec<CyclotomicRing>` and copying it, and holds the XOF reader
inline instead of behind a per-element boxed trait object.
The matrix-validation paths reuse the same pre-absorbed prefix.
These are equivalence-preserving and are pinned by the existing determinism,
prefix-stability, and ring-random-stream tests.
Local `setup_expand` drops roughly 14 to 20 percent across the fp16/fp32/fp64
D32 profile cases.

The profile benchmark workflow compiles with a fixed `x86-64-v3` ISA instead of
`-C target-cpu=native`.
GitHub-hosted `ubuntu-latest` is a heterogeneous CPU fleet, so `native` built a
different binary per run and compared a PR run against a main-baseline run that
targeted and executed on different silicon, which produced phantom setup-time
regressions even when the derivation code was unchanged.
The workflow also records an `lscpu` fingerprint so any residual cross-run
variance is visible in the logs.

## Non-Goals

- No Rust implementation in this spec-only PR.
- No setup-prefix commitments in the layout repack implementation branch.
- No setup-claim offloading proof in the layout repack implementation branch.
- No new matrix-claim sumcheck in the layout repack implementation branch.
- No ZK blinding offload in the base setup claim.
- No full-`S_full` runtime opening when a shorter committed prefix covers the
  active proof.
- No legacy `max_stride` compatibility layer.
- No materialization of `A * G_fold`.
- No change to the physical `w_hat || t_hat || z_hat || r_hat` witness segment
  order beyond selecting role-column views that match the current folding step.

## Open Questions

1. Where should role-view helpers live: `akita-types` near `FlatMatrix`, or
   prover/verifier-facing modules that know the prepared runtime shape?
2. Which prefix-slot policy should each preset use: full ladder, selected
   root/L1 slots, or direct fallback until the active size exceeds the gate?
3. What is the exact cache serialization and validation boundary for
   prover-ready prefix slots, especially `AkitaCommitmentHint` material and
   ZK blinding digit streams?
4. For recursive setup offloading, is the recursive block-fast A view worth the
   possible `L_bar` padding, or should recursive offload initially restrict to
   D/B if A padding is awkward?
5. What is the exact NTT cache API that keeps contiguous row slices while
   supporting root digit-fast and recursive block-fast views?
6. Should the offload gate remain one global `N_min = 2^23`, or should it later
   depend on the pair `(base field, extension field)` after matrix-MLE
   benchmarks?
7. After the common padded recursive carry works, is heterogeneous-domain
   carried-opening batching worth the extra verifier and scheduler complexity?

## Acceptance Criteria for This Spec PR

- The PR diff contains only durable planning/spec documents.
- The spec states that later committed setup offloading uses `S`, not
  `S_alpha`.
- The spec records the prefix-commitment policy plan, including full-ladder and
  selected-slot modes, plus the initial `D_setup = 32`, `N_min = 2^23`
  decisions.
- The spec records root digit-fast and recursive block-fast role-view policy,
  including the root A/one-hot constraint.
- The spec records that ZK blinding is outside the base setup matrix.
- The spec records that recursive setup openings are batched through carried
  opening claims, with a common padded domain in the first implementation.
- The implementation plan is explicit enough to restart the code work from
  current `main`.
