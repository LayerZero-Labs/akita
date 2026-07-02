# Spec: Tensor-Structured Folding Challenges

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @sumchecker                   |
| Created     | 2026-05-26                     |
| Status      | Implemented                    |
| PR          | #106                              |

## Summary

This branch implements the tensor-structured folding-challenge optimization:
stage-1 folding challenges can be sampled as a tensor product of two sparse
challenge vectors instead of as one independent sparse challenge per logical
block. For a level with `B = 2^r` witness blocks, the tensor path samples
`left_len + right_len ~= 2 * sqrt(B)` sparse challenges per claim and interprets
the logical block challenge as `c_{p,q} = left_p * right_q` in
`Z[X] / (X^D + 1)`. Relative to `main`, this branch adds the shared
flat-vs-tensor challenge representation, transcript binding for the two tensor
halves, prover kernels that fold with widened integer tensor products, verifier
row-replay code that can contract factored tensor aggregates without expanding
every logical challenge, and an fp128 `D=64` one-hot preset that enables the
tensor shape at the root level only.

## Intent

### Goal

Add a protocol-selected tensor shape for stage-1 fold challenges so verifier
challenge evaluation can use Kronecker structure while the rest of the folding
protocol continues to consume a single logical challenge per `(claim, block)`.

The key implementation surfaces are:

- `akita-challenges`: introduces `ChallengeShape`, `Challenges`,
  `TensorChallenges`, `sample_folding_challenges`, `tensor_split`,
  `tensor_left_digest`, and exact tensor-product evaluation helpers.
- `akita-transcript`: adds stage-1 tensor labels
  `CHALLENGE_TENSOR_FOLD_LEFT`, `ABSORB_TENSOR_FOLD_LEFT`, and
  `CHALLENGE_TENSOR_FOLD_RIGHT`.
- `akita-types::LevelParams`: carries `fold_challenge_shape` and computes
  `challenge_l1_mass()` from the active challenge shape.
- `akita-prover`: samples `Challenges` in the quadratic-equation builder and
  routes tensor-shaped folds through backend-owned
  `decompose_fold_tensor_batched` kernels.
- `akita-verifier`: preserves the flat verifier path and adds
  `PreparedChallengeEvals::Tensor` for factored deferred ring-switch row replay.
- `akita-config::tensor_verifier::fp128::D64OneHotTensor`: enables tensor
  challenges for the root fold of the fp128 `D=64` one-hot preset and keeps
  recursive levels flat.
- `akita-types::generated`: adds schedule tables for the tensor one-hot preset,
  including the ZK variant.
- `akita-pcs`: adds the `onehot_fp128_d64_tensor` profile mode and end-to-end tensor
  tests for one-hot and dense polynomials.

The tensor profile mode follows this branch's generated D64 tensor preset. It is
a direct local comparison mode and is not part of PR #107's D32 profile
benchmark matrix.

### Invariants

1. Flat challenge behavior remains the default. Existing presets keep
   `fold_challenge_shape = Flat`, use the original `CHALLENGE_STAGE1_FOLD`
   label, and interpret challenges in claim-major flat order.
2. Tensor sampling is transcript-bound in two stages. The left vector is sampled
   first, a canonical SHA3-256 digest of the left vector and shape is absorbed,
   and the right vector is sampled from the updated transcript.
   `tensor_sampling_absorbs_left_digest_before_right` protects this invariant.
3. Sampled tensor dimensions come from `tensor_split(num_blocks)`, which splits
   `2^r` into balanced dimensions `2^{floor(r/2)}` and `2^{ceil(r/2)}`. The
   lower-level `TensorChallenges` container stores explicit left/right lengths
   and validates power-of-two dimensions, vector lengths, and product size, but
   it does not require manually constructed tensors to use the balanced split.
4. The logical block order is unchanged. For claim `c` and local block
   `b = p * right_len + q`, the logical challenge is the negacyclic ring product
   `left[c, p] * right[c, q]`.
5. Tensor products are logical ring products, not `SparseChallenge` values,
   because multiplying two sparse challenges can create coefficients outside the
   `i8` sampler envelope. Tests use dense negacyclic ring multiplication as the
   reference model.
6. Evaluation at a ring-switch point is exact in `Z[X] / (X^D + 1)`. The
   verifier cannot use only `eval(left, alpha) * eval(right, alpha)` at a
   generic `alpha`; it must subtract the wrap quotient multiplied by
   `alpha^D + 1`. This is protected by
   `tensor_product_only_formula_is_not_exact_for_generic_alpha`,
   `tensor_exact_aggregate_collapses_to_product_at_negacyclic_root`, and
   `tensor_evals_at_pows_match_ring_product_reference`.
7. Prover and verifier see the same logical challenge stream. Both keep tensor
   factors and must match the direct ring-product reference.
8. Multipoint batching preserves claim routing. `Challenges::select_claims`
   returns a tensor-shaped subset for point-local folds, preserving left/right
   factor grouping by selected claim.
9. Tensor verifier summaries must be equivalent to expanded flat summaries for
   every offset/carry case used by structured row replay. This is protected by
   `factored_carry_summary_matches_flat_for_tensor_challenges`.
10. `LevelParams::challenge_l1_mass()` returns `cfg.l1_norm()` for flat and
    `cfg.l1_norm()^2` for tensor. Generated tensor schedule entries stamp the
    root fold shape before singleton layout derivation, so the table-backed
    singleton root layout is sized with the tensor mass. Both the runtime DP
    planner's root-candidate fold-digit sizing and
    `akita_types::scale_batched_root_layout` derive the batched fold digit
    count from `challenge_l1_mass · num_claims`, so tensor batched roots are
    sized for the tight `omega² · num_claims` bound rather than the previous
    `max(omega², omega · num_claims)` floor.
11. Tensor presets require backend tensor kernels. The tensor path fails with
    `AkitaError` if a backend does not implement the tensor-shaped fold kernel;
    it does not silently expand through an unoptimized fallback.
12. Tensor challenges do not add proof object fields. Challenges remain
    Fiat-Shamir-derived; proof shape changes only through the schedule selected
    for a tensor preset, not through serialized challenge payloads.
13. Verifier-reachable tensor validation rejects malformed encodings with
    `AkitaError`, not by panicking. Today this covers malformed sparse factors,
    non-power-of-two dimensions, length mismatches, and product mismatches
    against the expected block count; it does not reject an otherwise valid
    unbalanced explicit factorization.
14. The implemented preset applies only the tensor-challenge optimization.
    Claim-reduction sumcheck for setup-side verifier cost is not implemented in
    this branch.

### Non-Goals

1. No implementation of claim-reduction sumcheck or shared-matrix commitment.
2. No default migration of existing production presets to tensor challenges.
   This branch introduces an explicit tensor-verifier preset.
3. No generic tensor support for arbitrary non-power-of-two block counts.
4. No verifier compatibility with historical tensor transcript experiments.
   Tensor labels and digest bytes are canonical for this branch.
5. No proof-object serialization change for challenge material. Challenges are
   still transcript-derived.
6. No support for a 3-level tensor challenge decomposition.
7. No small-field tensor preset. The implemented public preset is fp128
   `D=64` one-hot.
8. No fallback evaluator on the tensor prover path. Unsupported polynomial
   backends must report that they cannot satisfy the tensor shape.

## Evaluation

### Acceptance Criteria

- [x] `akita-challenges` exposes a flat-vs-tensor challenge container shared by
      prover and verifier code.
- [x] Tensor sampling uses two sparse challenge vectors per claim and absorbs a
      canonical digest of the left vector before deriving the right vector.
- [x] Tensor products stay implicit as factor pairs and match dense negacyclic
      multiplication in `Z[X] / (X^D + 1)`.
- [x] Tensor evaluations at `alpha` include the negacyclic wrap correction and
      match direct ring-product references.
- [x] The verifier can evaluate factored tensor aggregates for carry summaries
      without materializing every logical block challenge.
- [x] The prover's batched root fold can route tensor challenges through
      homogeneous dense, one-hot, sparse-ring, and root-projection backends.
- [x] Multipoint/same-point batching can select point-local tensor claim subsets
      without losing the factorized shape.
- [x] `LevelParams` exposes the tensor effective L1 mass, and generated tensor
      schedule entries use it for singleton root fold digit sizing.
- [x] A dedicated fp128 `D=64` one-hot tensor-verifier preset exists and sets the
      root fold to tensor while keeping recursive folds flat.
- [x] Generated schedule tables exist for the tensor preset and ZK tensor preset.
- [x] The profile example exposes `AKITA_MODE=onehot_fp128_d64_tensor`.
- [x] End-to-end tensor tests prove and verify dense and one-hot singleton
      openings through serialization/deserialization.
- [x] Existing flat challenge tests continue to cover the legacy sparse sampler
      and flat challenge path.

### Testing Strategy

Focused tensor tests added by this branch:

- `cargo test --release -p akita-challenges dense_negacyclic_product_reference_handles_wrap_and_cancellation`
- `cargo test --release -p akita-challenges tensor_sampling_uses_two_vectors`
- `cargo test --release -p akita-challenges tensor_sampling_absorbs_left_digest_before_right`
- `cargo test --release -p akita-challenges tensor_left_digest_rejects_duplicate_positions`
- `cargo test --release -p akita-challenges tensor_lazy_evals_match_ring_product_reference`
- `cargo test --release -p akita-challenges tensor_factored_aggregate_matches_ring_product_reference`
- `cargo test --release -p akita-challenges tensor_evals_at_pows_match_ring_product_reference`
- `cargo test --release -p akita-challenges tensor_product_only_formula_is_not_exact_for_generic_alpha`
- `cargo test --release -p akita-challenges tensor_exact_aggregate_collapses_to_product_at_negacyclic_root`
- `cargo test --release -p akita-verifier factored_carry_summary_matches_flat_for_tensor_challenges`
- `cargo test --release -p akita-pcs --test single_poly_tensor_e2e`

Broader validation before merge should include:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test --release -p akita-challenges`
- `cargo test --release -p akita-verifier`
- `cargo test --release -p akita-pcs --test single_poly_tensor_e2e`
- `AKITA_MODE=onehot_fp128_d64_tensor AKITA_NUM_VARS=22 cargo run --release -p akita-pcs --example profile`

The e2e tests intentionally cover both `OneHotPoly` and `DensePoly` under the
tensor preset at `nv = 15, 20, 22`. They assert that the selected layout is
tensor-shaped at the root, produce a proof, serialize/deserialize it, and verify
against the same transcript domain.

### Performance

The optimization target is to reduce the verifier's challenge-dependent work
from `O(B)` to `O(sqrt(B))` per claim, where `B = 2^r` is the number of block
challenges at the level. In the concrete implementation, the savings apply
where the verifier can keep the tensor factors and contract separable row
weights directly:

```text
flat:   sample B sparse challenges per claim and evaluate B logical challenges
tensor: sample left_len + right_len sparse factors per claim and contract
        weighted aggregates through the factored API
```

For an even split, `left_len = right_len = sqrt(B)`. For an odd `r`, the split
is balanced as `2^{floor(r/2)} x 2^{ceil(r/2)}`.
The factored verifier path is still an exact negacyclic evaluation: each
aggregate includes a `D`-coefficient product and wrap correction. Since the
implemented preset fixes `D = 64`, this is a constant-size correction at the
level where the tensor shape is enabled, not a loop over all `B` logical block
challenges.

The current branch does not implement claim-reduction sumcheck, so
setup-dependent verifier work still exists. The expected near-term profile
comparison is therefore `onehot_fp128_d64` versus `onehot_fp128_d64_tensor`, with the
largest improvement in the challenge-evaluation portion of ring-switch row
replay rather than in total verification time. The `onehot_fp128_d64_tensor` mode is
the canonical local profile lens for the D64 tensor preset:

```text
AKITA_MODE=onehot_fp128_d64_tensor AKITA_NUM_VARS=<nv> cargo run --release -p akita-pcs --example profile
```

Tensor products increase the effective per-logical-block L1 envelope from
`omega` to `omega^2`. This branch sizes the table-backed singleton tensor root
through `LevelParams::challenge_l1_mass()` and uses dedicated generated schedule
tables for the tensor preset, so performance and proof-size comparisons must be
made against that preset's generated schedule, not by only changing the sampler
at runtime. Batched-root scaling derives the fold digit count from
`challenge_l1_mass · num_claims` directly via `LevelParams::challenge_l1_mass()`,
so tensor batched roots are sized for `omega^2 · num_claims`. Generated tensor
table entries with `num_t_vectors > 1` reflect this tight sizing.

## Design

### Architecture

The implemented shape uses two tensor-challenge sampling rounds and packages the
result behind the same challenge object consumed by the existing fold.

```text
QuadraticEquation::new_prover
        |
        v
sample_folding_challenges(shape = Flat | Tensor)
        |
        +-- Flat:
        |     sample num_claims * num_blocks SparseChallenge values
        |
        +-- Tensor:
              split num_blocks = left_len * right_len
              sample left[claim, p]
              absorb tensor_left_digest(left, D, num_claims, left_len)
              sample right[claim, q]
              logical c[claim, p, q] = left[claim, p] * right[claim, q]
        |
        v
Challenges enum
        |
        +-- Prover: select point-local claims and call backend tensor fold
        |
        +-- Verifier: store flat evals or tensor factors for deferred row replay
```

`ChallengeShape` is a selector, not sampled state. `Challenges` is the runtime
state. The flat variant stores `Vec<SparseChallenge>` with explicit
`num_blocks_per_claim` and `num_claims`. The tensor variant stores
`TensorChallenges { left, right, left_len, right_len, num_claims }`. Sampling
uses the balanced `tensor_split(num_blocks)` shape; explicit tensor containers
may carry any power-of-two factorization whose product matches the expected
block count.

The tensor sampler uses the same sparse challenge families as the flat sampler.
The crucial difference is interpretation: a sampled tensor factor is a normal
`SparseChallenge`, but the logical block challenge is a ring product of two
factors and therefore may have larger integer coefficients. The implementation
keeps that product implicit and contracts/evaluates it from the factors.

#### Transcript Flow

Flat sampling:

```text
sample_sparse_challenges(CHALLENGE_STAGE1_FOLD, num_claims * num_blocks)
```

Tensor sampling:

```text
left = sample_sparse_challenges(CHALLENGE_TENSOR_FOLD_LEFT,
                                num_claims * left_len)
digest = SHA3-256("akita/tensor-left-digest/v1"
                  || D
                  || num_claims
                  || left_len
                  || left.len()
                  || canonical sorted sparse left terms)
transcript.append_bytes(ABSORB_TENSOR_FOLD_LEFT, digest)
right = sample_sparse_challenges(CHALLENGE_TENSOR_FOLD_RIGHT,
                                 num_claims * right_len)
```

The digest includes the ring degree, claim count, left length, total left count,
and every left sparse challenge's sorted nonzero terms. Sorting makes the digest
canonical even if an equivalent sparse representation arrives in a different
term order, while `SparseChallenge::validate` rejects duplicate positions.

#### Logical Challenge Semantics

For claim `c`, left coordinate `p`, and right coordinate `q`:

```text
local_block = p * right_len + q
logical_index = c * (left_len * right_len) + local_block
c_logical = left[c * left_len + p] * right[c * right_len + q]
```

The product is computed in `Z[X] / (X^D + 1)`:

```text
X^i * X^j =  X^{i+j}       if i + j < D
X^i * X^j = -X^{i+j-D}     if i + j >= D
```

The prover uses this integer product when folding witness blocks. The verifier
has two exact evaluation APIs:

- `TensorChallenges::evals_at_pows` expands one field evaluation per logical
  block, matching the flat interface.
- `TensorChallenges::eval_factored_aggregate_at_pows` computes
  `sum_{p,q} u[p] * v[q] * eval(left[p] * right[q], alpha)` for one claim,
  where `u` and `v` are separable row weights.

The second API is the verifier optimization boundary. It builds weighted dense
left/right factors, evaluates their product, and subtracts the negacyclic wrap
quotient:

```text
eval(reduce(L * R), alpha)
  = eval(L, alpha) * eval(R, alpha)
    - (alpha^D + 1) * eval(quotient_wrap(L, R), alpha)
```

This is why the implementation remains exact at arbitrary ring-switch
challenges, not only at roots satisfying `alpha^D + 1 = 0`.

#### Prover Path

`QuadraticEquation::new_prover` samples `Challenges` after the prover's `v`
message is absorbed. It then groups claims by opening point. For a flat challenge
set, the point-local code preserves the existing path: select the corresponding
flat challenge slice and call `decompose_fold_batched` or individual
`decompose_fold`.

For a tensor challenge set, the code calls `Challenges::select_claims` to produce
the point-local tensor factors and requires the backend to implement
`decompose_fold_tensor_batched`. This branch adds tensor kernels for:

- `DensePoly`: contracts one tensor factor at a time over cached digit planes
  when available, otherwise decomposes ring coefficients in partitioned chunks
  and applies the same factor contraction.
- `OneHotPoly`: reuses the sparse one-hot block representation, rotates only
  the right tensor factor per block, and applies the left factor to the
  accumulated temporary rows.
- `SparseRingPoly`: flattens sparse ring blocks across the batched claims and
  streams right-factor rotated rows into a temporary accumulator before the
  left-factor pass.
- `RootTensorProjectionPoly`: dispatches homogeneous dense projections to the
  dense tensor kernel and homogeneous sparse projections to the sparse-ring
  tensor kernel. Mixed projection batches report unsupported.
- `MultilinearPolynomial`: dispatches homogeneous dense or homogeneous one-hot
  batches to the matching backend.

The output remains a normal `DecomposeFoldWitness`: `z_pre`, centered
coefficients, and the centered infinity norm used for the fold-bound check.

The ring-relation quotient uses the same tensor payload without allocating the
logical product vector: the high-half quotient term streams the left/right
sparse factors directly and tests against a direct ring-product oracle.

#### Verifier Path

Verifier ring-switch replay prepares either:

```text
PreparedChallengeEvals::Flat(Vec<E>)
PreparedChallengeEvals::Tensor { challenges: TensorChallenges, alpha_pows: Vec<E> }
```

The flat path stores `c_i(alpha)` for every logical challenge. The tensor path
stores the factored challenges and alpha powers, validates that
`left_len * right_len == lp.num_blocks`, and defers contraction until row
evaluation.

Structured W/T row replay needs block-carry summaries of the form:

```text
sum_block eq_low[offset + block] * c_block(alpha)
```

For tensor challenges, the low block bits split into right bits first and then
left bits. The verifier builds separable right weights `v[q]` and left weights
`u[p]` for each carry case, then calls
`eval_factored_aggregate_at_pows` for each claim. This exactly matches the
logical flat summary while avoiding a logical block loop in the cases where row
weights factor through the tensor split.

Non-challenge row data, setup contribution logic, ZK blinding segment layout,
sumcheck transcript flow, and final proof objects remain owned by the existing
ring-switch verifier machinery.

#### Schedule and Config

`LevelParams` now carries:

```text
fold_challenge_shape: TensorChallengeShape
```

The default is `Flat`. Tensor-aware schedule sizing calls:

```text
challenge_l1_mass =
  Flat   => stage1_config.l1_norm()
  Tensor => stage1_config.l1_norm() * stage1_config.l1_norm()
```

This feeds the existing fold decomposition formulas:

```text
beta = challenge_l1_mass * num_claims * 2^(r_vars + log_basis - 1)
```

The implemented tensor preset relies on generated schedule tables. When
materializing a generated fold entry, `akita-derive` stamps the configured
fold shape onto the generated level params before deriving the singleton root
layout, so that singleton fold digits and the `(m_vars, r_vars)` split observe
`omega^2` for tensor roots.

One current limitation is intentionally reflected in this spec:

- The offline DP fallback carries a `fold_challenge_shape` hook and now
  sizes root-candidate fold digits with `effective_l1_mass · num_t_vectors`
  (tight `omega^2 · num_claims` for tensor). The from-scratch root search
  still derives root candidates through the configured singleton default
  params, so production tensor schedule selection is expected to use the
  generated table — DP search is a fallback rather than a primary path.

`scale_batched_root_layout` reads the per-claim effective L1 mass directly
from `LevelParams::challenge_l1_mass()`, so batched-root scaling sizes
tensor batched roots for the tight `omega^2 · num_claims` bound. The
historical `root_stage1_l1_mass` argument was structurally redundant
with the layout's stage-1 config and has been removed.

The fp128 tensor-verifier preset is:

```text
akita_config::tensor_verifier::fp128::D64OneHotTensor
```

It uses the fp128 `D=64` exact-shell sparse challenge family:

```text
count_mag1 = 30
count_mag2 = 12
l1_norm = 54
```

and applies:

```text
level 0 fold_challenge_shape = Tensor
recursive levels             = Flat
```

The root-only policy matches the branch's near-term scope: validate and profile
the tensor-challenge path at the high-cost root fold without changing every
recursive level's schedule and security envelope.

### Alternatives Considered

1. **Materialize all tensor products in the verifier.** This is simpler and is
   still available through `evals_at_pows`, but it gives up the verifier-side
   `O(sqrt(B))` target for challenge-dependent structured rows.
2. **Use `SparseChallenge` for tensor products.** This would assume product
   coefficients fit the `i8` sampler envelope. The assumption is false in
   general, so the branch keeps products implicit instead of narrowing them.
3. **Sample right challenges without absorbing a left digest.** This would lose
   the intended two-stage Fiat-Shamir binding. The digest makes the right
   challenge depend on the exact sampled left vector.
4. **Enable tensor challenges through the existing fp128 one-hot preset.** The
   tensor effective L1 mass changes schedule sizing, so this branch uses a new
   explicit preset and dedicated generated schedule tables.
5. **Add a generic prover fallback that expands tensor products and calls the
   flat fold path.** That would hide missing backend support and risk surprising
   prover regressions. Tensor presets require tensor kernels instead.
6. **Implement claim-reduction sumcheck in the same branch.** That path touches
   setup commitments, proof scheduling, and next-level witness semantics. This
   branch keeps tensor challenges isolated.

## Documentation

This spec is the PR-facing design record for the implemented branch delta versus
`main`. It should be linked from the PR description. The existing profile
documentation can mention `AKITA_MODE=onehot_fp128_d64_tensor` once the preset is ready
to advertise as a standard profiling target.

One implementation detail is especially important for reviewers: because Akita
evaluates tensor products at arbitrary ring-switch challenges, the verifier's
factored evaluation includes the negacyclic wrap correction instead of using a
bare product of factor evaluations.

## Execution

The branch implementation is complete for the root-level fp128 D64 one-hot
tensor preset. Remaining merge-readiness work should focus on verification:

1. Run the release-mode focused tests listed above.
2. Compare `onehot_fp128_d64` and `onehot_fp128_d64_tensor` profile output for the PR's
   target `AKITA_NUM_VARS` values.
3. Confirm generated tensor schedule tables are accepted by normal config lookup
   and do not require planner fallback in production paths; runtime DP fallback
   is not the tensor-aware scheduling source for this branch.
4. Confirm ZK builds still compile with the generated ZK tensor table, even
   though the non-ZK e2e test is the primary behavior test in this branch.
5. Keep claim-reduction sumcheck out of scope unless the PR explicitly expands
   to that design.

## References

- `crates/akita-challenges/src/tensor.rs`
- `crates/akita-challenges/src/challenge.rs`
- `crates/akita-prover/src/protocol/quadratic_equation.rs`
- `crates/akita-verifier/src/protocol/ring_switch/tensor_challenges.rs`
- `crates/akita-config/src/tensor_verifier.rs`
- `crates/akita-pcs/tests/single_poly_tensor_e2e.rs`
