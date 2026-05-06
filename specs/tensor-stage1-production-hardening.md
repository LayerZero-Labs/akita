# Spec: Tensor Stage-1 Production Hardening

| Field | Value |
|---|---|
| Author(s) | TBD |
| Created | 2026-05-06 |
| Status | proposed |
| Branch | `feat/fourth-root-verifier-optimizations` |
| Scope | tensor schedule/SIS gating, transcript binding, tensor test coverage, accumulator headroom |

## Summary

Tensor stage-1 challenges are implemented and algebraically corrected for Hachi's generic ring-switch point, but tensor mode is not production-ready. This spec defines the next steps required to make tensor mode safe to enable in audited schedules: pin tensor shape in schedule metadata, align the Fiat-Shamir transcript with the chosen proof model, extend tests to the security surface, and enforce accumulator bounds or widen hot-path accumulators.

The goal is not just to make current tests pass. The goal is to make it hard to accidentally deploy tensor challenges with flat schedule assumptions, unaudited Module-SIS bounds, ambiguous transcript semantics, or integer wraparound in prover accumulation.

## Context

The current branch implements the challenge-dependent half of the fourth-root verifier optimization:

- `Stage1ChallengeShape::Tensor` samples left/right challenge vectors.
- Prover folding expands logical challenges as reduced tensor products in `R_q`.
- Verifier `PreparedMEval` stores tensor challenge data compactly.
- The exact aggregate evaluator computes the missing quotient correction:

  ```text
  eval(reduce(L_p * R_q), alpha)
    = L_p(alpha) * R_q(alpha)
      - (alpha^D + 1) * Q_{p,q}(alpha)
  ```

This fixes the residual-term issue created by the simplified product-only assumption in the tex draft. It does not, by itself, finish production hardening.

The remaining risks are:

1. Tensor schedule/SIS gating is incomplete.
2. Right-challenge transcript binding differs from the tex proof model.
3. Tensor tests cover only a thin slice of the proof surface.
4. Future large tensor schedules can exceed current integer accumulator headroom.

## Intent

### Goal

Make tensor stage-1 challenges production-enableable only when the schedule, transcript, tests, and arithmetic bounds are all consistent with the tensor security proof and implementation semantics.

### Invariants

- The prover and verifier must derive identical stage-1 challenge shapes from the same public schedule and transcript prefix.
- A proof generated under `Flat` must never verify under `Tensor`, and vice versa.
- Every schedule table entry must bind the challenge shape used to derive `challenge_l1_mass`, fold digit depth, witness size, and SIS/security estimates.
- Tensor schedules must account for both:
  - honest folded-witness mass, roughly `omega^2` per logical challenge;
  - extraction norm degradation, modeled by the two-level CWSS proof.
- The Fiat-Shamir implementation must match the documented proof model exactly, or the proof model must be updated to match the implementation exactly.
- Tensor challenge products must remain reduced products in `R_q`; verifier shortcuts must include the `(alpha^D + 1)` correction unless the protocol is explicitly redesigned.
- Prover accumulator arithmetic must not wrap in debug or release builds for any tensor-enabled schedule.
- Tensor mode must remain opt-in until all acceptance criteria below are satisfied.

### Non-Goals

- This spec does not implement Section 5 claim-reduction sumcheck or tiered setup commitments.
- This spec does not change the ring-switch challenge distribution.
- This spec does not attempt to remove the tensor quotient correction.
- This spec does not enable tensor mode in generated production schedules by default.

## Design

### Workstream 1: Schedule And SIS Gating

#### Problem

`GeneratedFoldStep` currently pins `challenge_l1_mass`, but the generated schedule metadata does not encode `Stage1ChallengeShape`. Runtime generated-level reconstruction starts from `LevelParams::params_only`, whose default shape is `Flat`, then validates the pinned mass against flat `challenge_l1_mass()`.

That is correct for today's generated tables, because they are flat. It is unsafe as a long-term tensor production path, because a tensor-generated entry needs to bind both the tensor shape and the mass/security calculations derived from that shape.

#### Rationale

Tensor changes more than the number of sampled challenges. It changes:

- logical challenge distribution;
- effective honest fold mass;
- fold digit depth;
- next-witness length;
- root and recursive capacity planning;
- extraction denominators in the CWSS proof;
- concrete Module-SIS margins.

If any part of the schedule path silently assumes flat mass, the prover can create witnesses sized for flat challenges while the transcript samples tensor products. That may cause honest failures, invalid soundness claims, or accidental deployment below the intended security floor.

#### Required Changes

1. Add a generated/runtime representation of stage-1 challenge shape.

   Suggested shape:

   ```rust
   pub enum GeneratedStage1ChallengeShape {
       Flat,
       Tensor,
   }
   ```

   Add it to `GeneratedFoldStep` or an equivalent schedule metadata object.

2. Make `generated_level_params` reconstruct `LevelParams` with the pinned shape before checking `challenge_l1_mass`.

   Required invariant:

   ```text
   step.challenge_shape == params.stage1_challenge_shape
   step.challenge_l1_mass == params.challenge_l1_mass()
   ```

3. Split honest fold mass from extraction security mass if needed.

   The honest folded witness bound uses the effective logical challenge L1 mass. The security audit may need a different quantity for A-role and challenge-dependent rows, because tensor extraction uses a product of challenge differences and the tex draft models a `4 * omega` relative MSIS degradation. Do not conflate these without a documented derivation.

4. Extend planner/SIS APIs to consume shape-aware parameters.

   Review these code paths:

   - `crates/akita-types/src/layout/params.rs`
   - `crates/akita-types/src/layout/digit_math.rs`
   - `crates/akita-types/src/layout/sis_derivation.rs`
   - `crates/akita-types/src/schedule.rs`
   - `crates/akita-planner/src/schedule_params.rs`
   - `crates/akita-planner/src/sis_security.rs`
   - `crates/akita-config/src/schedule_policy.rs`

5. Add production gating policy.

   Tensor mode can be enabled only when:

   - generated shape is tensor;
   - schedule mass is tensor-effective;
   - `num_digits_fold` matches tensor mass;
   - next witness length matches tensor `num_digits_fold`;
   - SIS floor remains above the configured target after tensor extraction degradation;
   - accumulator headroom checks pass.

#### Acceptance Criteria

- [ ] Generated schedule entries encode stage-1 challenge shape.
- [ ] Runtime schedule reconstruction rejects shape/mass mismatches.
- [ ] Unit tests show tensor-generated entries fail if loaded as flat and flat entries fail if loaded as tensor.
- [ ] SIS/planner report includes tensor extraction margins for every tensor-enabled preset.
- [ ] Production configs keep tensor disabled unless an audited tensor schedule is present.

### Workstream 2: Fiat-Shamir Transcript Alignment

#### Problem

The tex proof model describes the tensor protocol as two verifier challenge rounds:

```text
left = H(prefix, "tensor/left")
right = H(prefix, left, "tensor/right")
```

The implementation samples left and right with distinct labels, but it does not absorb the sampled left vector before deriving the right vector. The right vector is bound to the transcript state after the left sampling context, not to the left challenge output itself.

This is probably not an exploitable issue when there is no prover message between left and right, but it is a proof-model mismatch.

#### Rationale

Fiat-Shamir proofs should not rely on informal "morally equivalent" transcript ordering. The implementation and proof should agree exactly. Otherwise future changes, such as inserting an empty prover message marker, proof metadata, or challenge compression, can invalidate the intended challenge tree.

There are two acceptable designs:

1. Sequential binding design:

   ```text
   left = H(prefix, label_left)
   absorb(canonical_digest(left))
   right = H(prefix, left_digest, label_right)
   ```

2. Joint independent design:

   ```text
   (left, right) are independent challenge vectors sampled from the post-v prefix
   under domain-separated labels.
   ```

The code currently resembles design 2. The tex report currently describes design 1.

Decision: use design 1, sequential binding. This matches the tex proof model,
avoids an unnecessary proof-model fork, and costs only one canonical digest
absorb between tensor-left and tensor-right sampling. Design 2 remains a
possible future simplification only if accompanied by an explicit no-downgrade
soundness argument.

#### Required Changes

Choose exactly one design.

If choosing sequential binding:

- Add a canonical digest/serialization for `TensorStage1Challenges.left`.
- Absorb the left digest before sampling right.
- Add a dedicated transcript label such as:

  ```text
  ABSORB_STAGE1_TENSOR_LEFT
  ```

- Update prover and verifier sampling through the same helper only.

If choosing joint independent sampling:

- Update the spec and soundness write-up to state that left and right are domain-separated independent random oracle outputs from the same post-`v` transcript state.
- Explain why the two-level CWSS extraction remains valid without an intermediate prover message or left-output absorption.
- Keep the current code structure but add tests that lock the exact transcript behavior.

#### Acceptance Criteria

- [ ] The implementation and written proof model use the same transcript definition.
- [ ] Transcript fixture tests pin flat, tensor-left, and tensor-right outputs for fixed transcript prefixes.
- [ ] A test proves that changing the shape, labels, order, count, ring dimension, or sparse challenge config changes derived challenges.
- [ ] A tensor proof generated with one transcript policy is rejected by the other.

### Workstream 3: Tensor Test Coverage

#### Problem

Current tensor tests cover the exact aggregate algebra and one one-hot E2E case. That is enough for a prototype, not for production.

Tensor soundness changes the root challenge extractor and touches batching, multipoint routing, schedule sizing, and recursive handoff. Tests must cover these surfaces.

#### Rationale

The highest-risk bugs are not local algebra bugs in `eval_factored_aggregate_at_pows`; those are already well tested. The highest-risk bugs are mismatched shapes, ordering assumptions, layout retiming, and prover/verifier disagreement under non-singleton proof shapes.

#### Required Test Matrix

Add tests in phases.

Phase A: deterministic algebra and transcript tests.

- Tensor product reduced multiplication equals dense negacyclic multiplication.
- Product-only formula fails for generic `alpha`.
- Exact aggregate equals expanded reduced products.
- Carry summaries equal expanded `c_alpha` summaries for:
  - random offsets;
  - odd and even `r`;
  - multiple claims;
  - left/right splits with right side carrying the extra bit.
- Transcript fixture tests for left/right challenge derivation.
- Toy 2-level CWSS algebra test:

  ```text
  (z(L', R') - z(L', R)) - (z(L, R') - z(L, R))
    = (L'_p - L_p) * (R'_q - R_q) * s_{p,q}
  ```

Phase B: E2E tests.

- Dense/full tensor root-to-direct E2E.
- One-hot tensor root-to-direct E2E for more than one schedule bucket.
- Same-point batched tensor E2E.
- Multipoint tensor E2E.
- Mixed commitment groups if supported by current helpers.
- Recursive multi-fold tensor E2E once schedule retiming supports it.

Phase C: tampering tests.

- Change tensor shape metadata.
- Swap left/right labels.
- Change tensor split.
- Change `num_blocks` or `num_claims`.
- Tamper `stage1.s_claim`.
- Tamper proof messages after tensor challenges.
- Verify tensor proof under flat schedule and flat proof under tensor schedule.

Phase D: negative schedule tests.

- Tensor schedule with flat mass is rejected.
- Tensor schedule with insufficient accumulator headroom is rejected.
- Tensor schedule with SIS margin below target is rejected.

#### Acceptance Criteria

- [ ] Tensor test matrix covers dense, one-hot, batched, multipoint, and recursive paths or explicitly documents why a path is unsupported.
- [ ] Every tensor-enabled schedule has at least one E2E test.
- [ ] Tensor-specific tamper tests reject invalid proofs.
- [ ] Materialized-reference comparisons remain available for small layouts.

### Workstream 4: Accumulator Headroom

#### Problem

Several prover fold paths accumulate centered coefficients in `i32`. Tensor products can increase logical challenge mass from `omega` to roughly `omega^2`. Future high-block or batched tensor schedules can exceed `i32` even if current tests do not.

#### Rationale

Release-mode integer overflow in prover arithmetic is unacceptable. Even if the verifier later catches an invalid relation, wraparound can cause honest proof failures, nondeterministic behavior, or misleading benchmark/security results. The system should reject unsafe schedules before proving, or use accumulator types wide enough for all approved schedules.

#### Required Bound

For each tensor-enabled level, compute a conservative centered coefficient bound:

```text
accum_bound =
  num_blocks_for_fold
  * num_claims_folded_together
  * effective_challenge_l1_mass
  * max_digit_abs
```

where:

```text
effective_challenge_l1_mass = stage1_config.l1_norm()^2
max_digit_abs = 2^(log_basis - 1)
```

The exact expression may need per-backend refinement, but the schedule gate must be conservative.

If using `i32`:

```text
accum_bound <= i32::MAX
```

If this bound is too restrictive, widen tensor integer accumulation paths to `i64` or `i128`.

#### Required Changes

1. Add a schedule-level headroom function, likely in `akita-types`:

   ```rust
   pub fn stage1_accumulator_bound(lp: &LevelParams, num_claims: usize) -> Result<u128, AkitaError>
   ```

2. Call the check from:

   - schedule construction;
   - tensor test schedule wrapper;
   - prover entry points before accumulation.

3. Decide whether to reject or widen.

   Recommended path:

   - short term: reject unsafe tensor schedules;
   - medium term: widen tensor-specific integer accumulators where benchmarks justify larger schedules.

4. Add tests that construct a deliberately unsafe tensor schedule and verify rejection.

#### Acceptance Criteria

- [ ] Every tensor-enabled schedule proves an explicit accumulator headroom inequality.
- [ ] Unsafe tensor schedules fail before prover accumulation.
- [ ] Release builds cannot silently wrap tensor fold accumulators for approved schedules.
- [ ] If accumulator widening is implemented, tests compare widened and narrow paths on safe schedules.

## Performance Plan

Production hardening should not erase the verifier win. Track these measurements before enabling tensor mode:

- challenge sampling time, flat vs tensor;
- tensor exact aggregate time, expanded vs compact;
- `PreparedMEval::eval_at_point`, flat vs tensor;
- full verifier replay, flat vs tensor;
- full prover time, flat vs tensor;
- proof size and next-witness size;
- allocation counts in `summarize_tensor_block_carries`.

Expected near-term optimizations:

- reuse `alpha_pows` stored in prepared tensor state;
- avoid repeated `EqPolynomial::evals` for the same block prefix;
- reuse or stack-allocate `u_weights`, `v_weights`, `left_bar`, and `right_bar`;
- avoid cloning the full tensor challenge object where shared ownership suffices;
- keep product-only evaluation only as a diagnostic benchmark, never as verifier logic.

## Execution Plan

### Phase 1: Proof-Model Lockdown

1. Choose transcript design: sequential binding or joint independent sampling.
2. Update `specs/tensor-exact-aggregate-evaluator.md` and this spec if needed.
3. Add transcript fixture tests.
4. Add toy 2-level CWSS algebra tests.

### Phase 2: Shape-Aware Schedule Metadata

1. Add generated challenge-shape metadata.
2. Make generated schedule materialization shape-aware.
3. Add shape/mass mismatch rejection tests.
4. Extend planner/SIS report to include tensor extraction margins.

### Phase 3: Accumulator Headroom

1. Add conservative bound helpers.
2. Gate tensor schedules and prover entry points.
3. Add unsafe-schedule rejection tests.
4. Decide whether any path needs widened accumulators.

### Phase 4: Test Matrix And Benchmarks

1. Add dense tensor E2E.
2. Add batched and multipoint tensor E2E.
3. Add tensor-specific tamper tests.
4. Add verifier-focused benchmarks.
5. Run full workspace tests and clippy with tensor tests enabled.

### Phase 5: Production Enablement

Tensor mode can be enabled for a generated preset only after:

- all acceptance criteria above pass;
- the preset has a documented SIS margin after tensor extraction costs;
- accumulator headroom is checked;
- verifier/prover benchmarks show the intended improvement;
- the exact aggregate correction remains in the verifier path;
- the config explicitly marks the preset as tensor-audited.

## Documentation

Update or add:

- `specs/tensor-exact-aggregate-evaluator.md`: transcript proof-model decision and production gating link.
- `specs/tensor-exact-aggregate-implementation-plan.md`: progress log for hardening phases.
- `specs/fourth-root-verifier-audit.md`: mark hardening spec as the follow-up plan.
- Generated schedule docs or comments explaining shape-aware schedule metadata.
- Planner/SIS report output documenting tensor extraction margins.

## References

- `specs/fourth-root-verifier-audit.md`
- `specs/tensor-exact-aggregate-evaluator.md`
- `specs/tensor-exact-aggregate-implementation-plan.md`
- `specs/fourth-root-verifier-optimizations.md`
- `crates/akita-challenges/src/stage1.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-prover/src/protocol/quadratic_equation.rs`
- `crates/akita-types/src/layout/params.rs`
- `crates/akita-types/src/schedule.rs`
- Hachi paper `2026-156.pdf`, ring-switch lift and CWSS sections.
- Lattice Jolt draft `sections/5_fourth_root_verifier.tex`, tensor challenges and claim reduction.
