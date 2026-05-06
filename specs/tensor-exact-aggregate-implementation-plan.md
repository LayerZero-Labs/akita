# Tensor Exact Aggregate Implementation Plan

| Field | Value |
|---|---|
| Status | in progress |
| Branch | `feat/fourth-root-verifier-optimizations` |
| Base | `feat/l1-bound-challenges` |
| Source spec | `specs/tensor-exact-aggregate-evaluator.md` |
| Scope | Verifier tensor challenge aggregation, M-eval integration, tests, security gating |

## Objective

Implement tensor-structured stage-1 challenge verification without materializing
all logical `c_{p,q}(alpha)` values in verifier hot paths, while preserving the
exact expanded semantics:

```text
c_{p,q}(alpha) = eval(reduce(L_p * R_q), alpha)
```

The implementation must not use the product-only shortcut
`L_p(alpha) * R_q(alpha)` except in tests or diagnostics that explicitly show
why it is incomplete for generic ring-switch `alpha`.

## Current Baseline

The plumbing commit `d346d38` established the protocol shape needed for this
work:

- `Stage1ChallengeShape::Tensor` samples left/right sparse challenge vectors.
- Prover and verifier derive tensor challenges from separated transcript labels.
- Prover folding expands tensor products exactly in `Z[X] / (X^D + 1)`.
- Verifier preparation still calls `Stage1Challenges::evals_at_pows`, which
  expands all logical tensor products and evaluates each reduced product.
- Tensor mode is opt-in; default generated/runtime schedules remain flat.

This is correct but still uses the conservative `O(B * omega^2)` verifier path.

## Implementation Items

### 1. Add Exact Factored Tensor Aggregate Evaluator

Add an exact verifier-side helper for one factored weighted aggregate:

```text
S = sum_{p,q} u[p] * v[q] * eval(reduce(L_p * R_q), alpha)
```

Implementation shape:

- Accumulate dense coefficient vectors:
  - `Lbar[i] = sum_p u[p] * L_p[i]`
  - `Rbar[j] = sum_q v[q] * R_q[j]`
- Compute `product_eval = eval(Lbar, alpha) * eval(Rbar, alpha)`.
- Compute the high-half quotient evaluation:
  - `quotient_eval = sum_{i+j>=D} Lbar[i] * Rbar[j] * alpha^(i+j-D)`.
- Return `product_eval - (alpha^D + 1) * quotient_eval`.

Placement:

- Prefer `akita-challenges/src/stage1.rs` because the evaluator only depends on
  tensor challenge structure and scalar weights.
- Export the helper through `akita-challenges` only if verifier integration
  needs public access across crates.

Tests:

- Exact aggregate equals explicit tensor expansion for random/fixed sparse
  challenges and factored weights.
- The product-only formula differs from exact expansion for a generic `alpha`
  counterexample.
- The `alpha^D + 1 = 0` local algebra case collapses to the product term.

Completion criteria:

- Unit tests cover the algebra in `akita-challenges`.
- No verifier call sites are changed in this item.

### 2. Introduce Prepared Challenge Evaluation Storage

Replace `PreparedMEval`'s unconditional dense `c_alphas: Vec<F>` with an enum:

```text
PreparedChallengeEvals:
  Flat(Vec<F>)
  Tensor {
    tensor challenges,
    alpha_pows,
    alpha_pow_d_plus_one,
  }
```

Implementation shape:

- `prepare_m_eval` keeps dense flat behavior for `Stage1Challenges::Flat`.
- `prepare_m_eval` stores compact tensor challenges for
  `Stage1Challenges::Tensor`.
- Preserve `logical_len` validation for both modes.
- Keep an expanded debug/reference method for tests.

Tests:

- Existing flat `prepared_m_eval_matches_materialized` still passes.
- Tensor storage preserves challenge shape and avoids dense `num_claims *
  num_blocks` allocation for challenge scalars.

Completion criteria:

- Flat behavior is unchanged.
- Tensor mode can prepare without calling tensor `evals_at_pows`.

### 3. Add Tensor Carry Summary Decomposition

Implement tensor-aware replacement for:

```text
summarize_pow2_block_carries(eq_low, offset_low, c_alphas_for_claim)
```

For `block = p * right_len + q` and
`offset = offset_left * right_len + offset_right`, decompose each carry summary
into a small constant number of factored terms:

- Split the low equality point into left/right pieces.
- Build `eq_left` and `eq_right` tables.
- Partition by `carry_q = floor((offset_right + q) / right_len)`.
- Partition by final carry:
  `floor((offset_left + p + carry_q) / left_len)`.
- For each non-empty partition, call the exact aggregate evaluator with
  `u[p]` and `v[q]`, then add it to `out[final_carry]`.

Tests:

- For many offsets, splits, and random verifier points, tensor summaries equal
  `summarize_pow2_block_carries` applied to expanded `c_alphas`.
- Include odd `r` where `right_len` has the extra bit.
- Include non-zero `offset_low` cases that exercise both carry layers.

Completion criteria:

- Produces bit-identical `[F; 2]` summaries to dense expansion.
- No downstream M-eval algebra changes are required.

### 4. Integrate Tensor Summaries Into `PreparedMEval::eval_at_point`

Update the challenge-summary construction in
`crates/akita-verifier/src/protocol/ring_switch.rs`:

- Flat mode keeps the current dense slice plus `summarize_pow2_block_carries`.
- Tensor mode calls the tensor carry summary helper per claim.
- Existing `w_carry_terms`, `t_carry_terms`, and
  `eval_offset_eq_peeled_carry_terms` code remains unchanged.

Tests:

- Add a tensor variant of `prepared_m_eval_matches_materialized` in
  `crates/akita-pcs/tests/ring_switch.rs`.
- Compare prepared tensor M-eval against materialized prover M-table evaluation
  on small random layouts.

Completion criteria:

- Flat and tensor `prepare_m_eval` paths both match materialized evaluation.
- Tensor verifier path avoids full `c_alphas` expansion.

### 5. E2E Tensor Verification Coverage

Add small end-to-end prove/verify tests with tensor mode enabled:

- One dense or full polynomial case.
- One one-hot case.
- At least one batched or multi-claim root case if existing helpers make it
  practical without excessive runtime.

Also keep explicit flat E2E coverage to confirm defaults are unchanged.

Completion criteria:

- E2E tensor proof verifies.
- Simple transcript/challenge tampering causes rejection.

### 6. Security and Schedule Gating

Before enabling tensor in production presets:

- Keep `Stage1ChallengeShape::Flat` as the default.
- Ensure batched root scaling uses `root_lp.challenge_l1_mass()` rather than
  bare `stage1_config.l1_norm()`.
- Audit generated schedule metadata so tensor-enabled entries pin a shape and an
  effective challenge mass consistently.
- Review SIS derivation paths that use `infinity_norm` for A-role collision
  bounds and document whether tensor products require a wider proxy.
- Do not enable tensor mode in generated production schedules until the
  planner/SIS audit passes.

Completion criteria:

- Tensor mode remains explicitly opt-in.
- Runtime assertions/tests reject inconsistent challenge mass metadata.

### 7. Benchmarks and Diagnostics

Add benchmarks or measurement hooks for:

- Flat baseline.
- Tensor expanded exact path.
- Tensor exact aggregate path.
- Product-only diagnostic path, test-only and clearly not production-safe.

Completion criteria:

- Benchmark output separates challenge aggregation from full verifier replay.
- The exact aggregate path shows the intended `O(2^(r/2) + D^2)` trend for the
  challenge-dependent summaries.

## Progress Log

### 2026-05-06

- Rebased `feat/fourth-root-verifier-optimizations` onto
  `feat/l1-bound-challenges`.
- Dropped the report-only extra commit that added
  `specs/bugfix-review-comment.md`.
- Committed current tensor plumbing as `d346d38`
  (`feat(protocol): add tensor challenge plumbing`).
- Added this implementation/progress plan.
- Implemented item 1's exact factored tensor aggregate evaluator in
  `akita-challenges`.
- Implemented item 2's compact `PreparedMEval` challenge storage:
  flat challenges still store dense scalar evaluations, while tensor challenges
  retain compact tensor data plus `alpha` powers and expose a debug expansion
  bridge for later integration work.
- Implemented item 3's tensor carry-summary decomposition helper and reference
  test against expanded `c_alphas`.
- Implemented item 4's `PreparedMEval::eval_at_point` integration so tensor
  mode uses exact aggregate summaries instead of expanding `c_alphas` in the
  verifier summary hot path.
- Added item 5's first end-to-end tensor coverage: a singleton one-hot proof
  uses a test-only config wrapper that tensorizes the root fold schedule while
  leaving production configs flat.
- Dense/full and batched tensor E2E coverage still need a schedule-retiming
  helper that can safely update multi-fold schedules after tensor digit-depth
  changes.
- Implemented item 6's first security/schedule gate: batched root scaling now
  derives fold digit depth from `root_lp.challenge_l1_mass()`, so tensor-shaped
  roots use the effective tensor challenge mass instead of the flat sampler
  mass.
- Implemented item 7's first benchmark diagnostics in the existing
  `akita-challenges` Criterion benchmark:
  expanded exact tensor aggregation, exact aggregate evaluation, and
  product-only diagnostic evaluation.
- Implemented the first production-hardening slice:
  tensor right challenges are now sampled after absorbing a canonical digest of
  the sampled tensor-left vector, generated schedule tables carry explicit
  stage-1 challenge-shape metadata, runtime reconstruction rejects shape/mass
  mismatches, and tensor prover paths reject schedules that exceed conservative
  `i32` accumulator headroom.
- Implemented shape-aware SIS extraction plumbing: `LevelParams` now exposes
  the honest fold mass separately from the tensor CWSS extraction degradation,
  and root/recursive SIS derivation sizes A-role collisions with the tensor
  `4 * omega` relative MSIS factor while preserving flat-mode behavior.

## Validation Log

- `cargo test -p akita-challenges` passed after item 1.
- First attempt used `Fp64<5>` for the `alpha^D + 1 = 0` edge case, but that
  modulus is too small for this field implementation's reduction constants. The
  test now uses the existing 32-bit test field with root `983270775` of
  `X^2 + 1`.
- `cargo clippy -p akita-challenges --all-targets --message-format=short -q
  -- -D warnings` passed after item 1.
- `cargo test -p akita-verifier` passed after item 2.
- `cargo clippy -p akita-verifier --all-targets --message-format=short -q --
  -D warnings` passed after item 2.
- `cargo test -p akita-verifier` passed after item 3.
- `cargo clippy -p akita-verifier --all-targets --message-format=short -q --
  -D warnings` passed after item 3.
- `cargo test -p akita-pcs --test ring_switch prepared_m_eval -- --nocapture`
  passed after item 4, including the tensor materialized-M-table comparison.
- `cargo clippy -p akita-verifier -p akita-pcs --tests
  --message-format=short -q -- -D warnings` passed after item 4.
- `cargo test -p akita-pcs --test single_poly_e2e
  onehot_tensor_stage1_prove_verify -- --nocapture` passed after item 5's first
  E2E tensor test.
- `cargo clippy -p akita-pcs --test single_poly_e2e --message-format=short -q
  -- -D warnings` passed after item 5's first E2E tensor test.
- `cargo test -p akita-types
  batched_root_scaling_uses_shape_aware_challenge_mass` passed after item 6.
- `cargo test -p akita-config
  batched_commitment_direct_fallback_scales_root_layout` passed after item 6.
- `cargo test -p akita-scheme --lib
  same_point_batched_root_preserves_opening_geometry` compiled the scheme test
  target after item 6; the filtered test is intentionally ignored without
  planner fallback.
- `cargo clippy -p akita-types -p akita-config -p akita-planner --tests
  --message-format=short -q -- -D warnings` passed after item 6.
- A broad `cargo test -p akita-config batched_commitment -- --nocapture` also
  selected two pre-existing planner-fallback tests that are expected to fail
  without the planner feature; the directly relevant fallback scaling test
  passed.
- `cargo test -p akita-challenges --benches --no-run` passed after item 7.
- `cargo clippy -p akita-challenges --benches --message-format=short -q --
  -D warnings` passed after item 7.
- Full workspace `cargo test` passed after item 7 and the schedule scaling
  fixes.
- `cargo test -p akita-challenges tensor_stage1` passed after the transcript
  binding hardening.
- `cargo test -p akita-types accumulator` and
  `cargo test -p akita-types generated_level_params` passed after the schedule
  metadata/headroom hardening.
- `cargo clippy --all --message-format=short -q -- -D warnings` passed after
  the hardening slice.
- Full workspace `cargo test` passed after the hardening slice.
- `cargo test -p akita-types tensor_extraction_bound_is_separate_from_honest_mass`
  and `cargo test -p akita-types
  sis_root_derivation_uses_tensor_extraction_collision_bucket` passed after the
  SIS extraction hardening.
- `cargo test -p akita-planner` passed after making planner schedule assertions
  shape-aware.
