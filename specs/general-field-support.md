# Spec: General Field Support Scaffolding

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-02 |
| Status | implementation |
| PR | #60 (`quang/general-fields`) |
| Follow-up | `specs/extension-claim-incidence-cutover.md` (#69), then `specs/extension-field-trace-cutover.md` (#71), then `specs/extension-field-opening-batching.md` (next) |

## Summary

Akita should support base fields beyond the current fp128 production field,
starting with 32-bit and 64-bit prime-field profiles. This PR implements the
scaffolding needed for that generalization while deliberately stopping before
the native extension-opening and batching cutover. It splits field roles,
threads extension-aware transcript helpers, adds Hachi field-reduction
reference utilities, makes planner/proof-size accounting field-width-aware, adds
static fp32/fp64 configs and E2E coverage, and preserves the existing fp128
verifier behavior.

The next PR owns native small-field commitments opened at extension-field
points, including the batching generalization needed for one committed group
opened at many Frobenius-conjugate points.

As of PR #71, several items that were deferred from this scaffolding PR have
landed in follow-up commits: proof payloads are generic over `F, L`, root
folded extension openings run through the packed-inner path when supported,
unsupported extension shapes fall back to root-direct, and SIS sizing is keyed
by `Q32`/`Q64`/`Q128` families with larger small-field D candidates. Generated
production fp32/fp64 schedule tables remain deferred until the measured profile
data selects defaults.

## Intent

### Goal

Prepare Akita's crate-decomposed protocol stack for general base fields by
introducing typed field roles, extension transcript plumbing, reference
field-reduction math, field-width-aware planning, and small-field integration
coverage, without changing the public base-field opening API in this PR.

### Scope Boundary

This PR does everything up to the point where extension-field openings require
a better public claim model. The current public prover/verifier inputs still
take base-field opening points and base-field claimed evaluations. That is
intentional. `ClaimField` and `ChallengeField` are introduced as typed staging
points so the follow-up can migrate existing surfaces in place.

This PR must not add parallel public APIs for each batching special case. The
follow-up spec defines the point/group/claim incidence model for the native
extension-opening cutover.

### Invariants

- Ring commitments, setup matrices, digit decomposition, CRT/NTT work, and SIS
  bounds remain over `Cfg::Field`.
- SIS bounds are still served by the existing fp128-calibrated registry in this
  scaffolding PR. Field-family-specific SIS floor tables are deliberately
  deferred to `specs/extension-field-opening-batching.md` before any generated
  fp32/fp64 schedule tables are baked.
- Existing fp128 presets continue to use
  `Field = ClaimField = ChallengeField`.
- Extension absorption is coordinate-order sensitive and deterministic.
- Extension challenge sampling draws all base-field limbs and must not silently
  project challenges back into the base field.
- The current verifier shortcut remains the `k = 1` specialization of the
  subgroup trace relation and stays constant time in the hot path.
- Planner digit counts and proof-size estimates respect the configured field
  bit width instead of assuming 128-bit elements.
- Static fp32/fp64 configs are scaffolding profiles, not generated production
  schedules.
- Local scratch notes and generated planning scripts remain untracked and are
  not part of this PR.

### Non-Goals

- This does not complete extension-valued public opening claims in the
  production prover/verifier API.
- This does not implement the real `k > 1` Hachi embedding in the proof path.
- This does not implement the Frobenius-conjugate optimized base/ext opening
  path.
- This does not generalize the public batched-claim input model.
- This does not regenerate production fp32/fp64 schedule tables.
- This does not make fp32/fp64 SIS sizing security-calibrated. The static
  small-field profiles are E2E correctness scaffolds until the SIS floor
  registry is keyed by the active modulus family.
- This does not replace every protocol challenge with `Cfg::ChallengeField`.
- This does not benchmark or tune extension-field arithmetic.
- This does not change the default fp128 production preset or its security
  parameters.

## Evaluation

### Acceptance Criteria

- [x] `CommitmentConfig` exposes separate base, claim, and challenge field
  roles.
- [x] Existing fp128 proof-optimized presets set
  `Field = ClaimField = ChallengeField`.
- [x] `WCommitmentConfig` forwards the wrapped config's field roles.
- [x] Config-level helpers append `ClaimField` values and sample
  `ChallengeField` values through extension-aware transcript code.
- [x] `akita_transcript::append_ext_field` absorbs extension elements in
  deterministic base-coordinate order.
- [x] `akita_transcript::sample_ext_challenge` samples all extension limbs and
  reconstructs the extension element without base-field projection.
- [x] `akita_field::ExtField` and `LiftBase` cover the base field, `Fp2`, and
  `Fp4`.
- [x] `akita_types::field_reduction` exposes subgroup exponent enumeration,
  `trace_h`, and `psi_pack`.
- [x] `SubfieldParams` validates nonzero `D`, power-of-two `D`, nonzero `k`,
  `k | D/2`, and invertibility of `4k + 1` modulo `2D`.
- [x] `field_reduction` tests cover production-size subgroup cardinalities for
  `D = 64` and `D = 128`.
- [x] The verifier-side `k = 1` relation is tested against the general trace
  helper while preserving the constant-time shortcut.
- [x] Planner digit and proof-size helpers can model non-128-bit field widths.
- [x] Runtime schedule/layout scaling receives the configured field bit width
  instead of assuming fp128.
- [x] Static fp32/fp64 proof-optimized profiles compile.
- [x] Static fp32/fp64 dense commit/prove/verify E2E tests pass.
- [x] Transcript tests cover extension absorption, limb-order divergence, and
  extension challenge replay.
- [x] Bugbot findings on extension labels and subgroup validation are addressed.
- [ ] Final PR head has green GitHub CI after the latest spec-only commits.
- [ ] Human review accepts the scoped boundary: batching and native extension
  openings are deferred to the follow-up spec.

### Testing Strategy

Existing tests that must continue passing:

- `cargo fmt -q`
- `cargo clippy --all --all-targets --all-features -- -D warnings`
- `cargo test`
- fp128 E2E tests in `crates/akita-pcs/tests/akita_e2e.rs`
- batched and multipoint E2E tests in `crates/akita-pcs/tests/`
- transcript tests in `crates/akita-pcs/tests/transcript.rs`

Targeted tests added or exercised by this PR:

- [x] `field_reduction` unit tests for subgroup validation, exponent
  cardinality, `trace_h`, `psi_pack`, and the `k = 1` shortcut relation.
- [x] Extension transcript replay tests for `Fp2` and `Fp4`.
- [x] Config helper tests proving `CommitmentConfig` helper output matches raw
  transcript helper output.
- [x] fp32 static dense E2E round trip.
- [x] fp64 static dense E2E round trip.
- [ ] Final CI run on the PR head after documentation-only updates.

### Performance

No runtime performance regression is expected for the default fp128 path. The
hot verifier relation remains the existing constant-term shortcut for `k = 1`;
the general `trace_h` implementation is a reference/test utility in this PR.

Planner and proof-size output changes are expected only where field byte widths
are explicitly different from fp128. Static fp32/fp64 profiles are integration
scaffolds, not tuned performance presets.

## Design

### Architecture

The PR touches five layers:

1. **Field layer**: `akita-field` owns the `ExtField` and `LiftBase` traits,
   plus fp32/fp64-wide arithmetic support needed by static small-field tests.
2. **Transcript layer**: `akita-transcript` owns coordinate-wise extension
   absorption and challenge sampling over a base-field transcript.
3. **Config layer**: `akita-config` owns role selection:
   `Field`, `ClaimField`, and `ChallengeField`.
4. **Types/layout layer**: `akita-types` owns reference field-reduction helpers
   and field-bit-width-aware digit math inputs.
5. **Planner/test layer**: `akita-planner`, `akita-config`, and `akita-pcs`
   exercise non-fp128 byte accounting and static fp32/fp64 E2E flows.

The field-role split is intentionally ahead of the public API. Today:

```text
commit/prove/verify public inputs:
  opening point, claimed eval  : Cfg::Field
  commitments, rings, setup    : Cfg::Field
```

After the follow-up cutover:

```text
commit/prove/verify public inputs:
  opening point, claimed eval  : Cfg::ClaimField
  commitments, rings, setup    : Cfg::Field
```

This PR stops before that second diagram becomes true.

### Field Roles

`CommitmentConfig` exposes:

- `type Field`: the base field used by rings, commitments, setup matrices,
  digit decomposition, and SIS bounds.
- `type ClaimField`: the intended field for public opening points and claimed
  evaluations.
- `type ChallengeField`: the intended field for Fiat-Shamir scalar challenges
  in sumcheck-style steps.

Config helper methods centralize transcript behavior:

```rust
fn append_claim_field<T: Transcript<Self::Field>>(
    transcript: &mut T,
    label: &[u8],
    x: &Self::ClaimField,
);

fn sample_challenge_field<T: Transcript<Self::Field>>(
    transcript: &mut T,
    label: &[u8],
) -> Self::ChallengeField;
```

### Extension Transcript Semantics

For `EXT_DEGREE = 1`, extension helpers preserve the existing transcript label
behavior. For `EXT_DEGREE > 1`, each base-field coordinate is absorbed or
sampled under a derived limb label:

```text
original label || 0xff || limb_index_le_u64 || "ext"
```

This makes coordinate order binding explicit and avoids accidental transcript
collisions between extension limbs.

### Field Reduction Reference Utilities

`SubfieldParams<D>::new(k)` models the Hachi subgroup
`H = <sigma_-1, sigma_(4k+1)>` modulo `2D`.

It rejects malformed parameters before exponent enumeration:

- `D = 0`;
- non-power-of-two `D`;
- odd `D`;
- `k = 0`;
- `k` not dividing `D/2`;
- `gcd(4k + 1, 2D) != 1`.

`h_exponents()` enumerates distinct odd exponents in `H`.
`trace_h(params, x)` computes `sum_{sigma in H} sigma(x)`.
`psi_pack(params, values)` implements the coefficient-placement part of
Hachi's `psi` map:

```text
values[0 .. D/(2k))       -> coeffs[0 .. D/(2k))
values[D/(2k) .. D/k)     -> coeffs[D/2 .. D/2 + D/(2k))
```

This is not yet the complete `k > 1` proof-path embedding. It is a reference
utility with tests that anchor the algebra before the follow-up implementation.

### Field-Width Accounting

Proof-size and layout helpers take `field_bits` or derive it from
`Cfg::decomposition().field_bits()`. This affects:

- serialized field element byte counts;
- ring-vector byte estimates;
- sumcheck byte estimates;
- stage-1 proof-size estimates;
- batched root-layout scaling.

The important behavior is not that fp32/fp64 schedules are optimized or
security-calibrated, but that they no longer pay fp128 byte accounting by
construction. SIS rank floors and ring-dimension ladders remain a separate
modulus-family registry issue: fp32/fp64 must eventually use Q32/Q64 SIS tables
and larger candidate ring dimensions, while fp128 should use a Q128
representative such as `2^128 - (2^32 - 22537)` rather than silently reusing a
larger `2^128 - 275` table. A rough sizing intuition is
`fp128 D=32 ~ fp64 D=64 ~ fp32 D=128`, with D=256/D=512 candidates reserved for
small-field planning and profiling.

### Static Small-Field Profiles

The PR adds static fp32/fp64 config scaffolds under `akita-config` proof
optimized presets. They are intentionally small and planner-backed. Their job
is to make the existing end-to-end commit/prove/verify path run over smaller
base fields without changing the public API yet.

### Alternatives Considered

**Complete extension openings in this PR.**
Rejected because the moment `Cfg::ClaimField` becomes public input, the current
nested batch shape becomes the wrong abstraction for the optimized Frobenius
route. That belongs in the follow-up incidence-model PR.

**Add a one-poly-many-points API now.**
Rejected because it would create another public special case beside existing
same-point batching. The follow-up will use one general point/group/claim model.

**Replace the fp128 verifier shortcut with generic trace enumeration.**
Rejected because the shortcut is the hot production path and is exactly the
`k = 1` specialization. This PR tests the equivalence instead of slowing the
runtime path.

**Keep planner proof-size constants at 128 bits.**
Rejected because it would make fp32/fp64 configs pass functionally while giving
misleading proof-size and layout estimates.

## Documentation

This spec is the main documentation artifact for PR #60. The follow-up design
is documented in `specs/extension-field-opening-batching.md`.

No user-facing README/API documentation is required yet because extension-field
openings are not exposed by the public prover/verifier API in this PR.

## Execution

### Progress Tracker

Field traits and concrete fields:

- [x] Add `LiftBase<F>` for constant-term base-to-extension embedding.
- [x] Add `ExtField<F>` with `EXT_DEGREE`, `from_base_slice`, and
  `to_base_vec`.
- [x] Implement `ExtField` for the base field.
- [x] Implement `ExtField` for `Fp2`.
- [x] Implement `ExtField` for `Fp4`.
- [x] Add fp32/fp64 wide accumulation support needed by small-field E2E tests.

Transcript plumbing:

- [x] Move extension challenge sampling to `akita-transcript`.
- [x] Add `append_ext_field`.
- [x] Add `sample_ext_challenge`.
- [x] Use derived limb labels for higher-degree extensions.
- [x] Preserve exact label behavior for degree-one fields.
- [x] Re-export the transcript sampling helper from `akita-challenges`.

Config role split:

- [x] Add `CommitmentConfig::ClaimField`.
- [x] Add `CommitmentConfig::ChallengeField`.
- [x] Add `CommitmentConfig::append_claim_field`.
- [x] Add `CommitmentConfig::sample_challenge_field`.
- [x] Update fp128 presets to set all roles to the base field.
- [x] Update small-field static presets to set all roles to the base field.
- [x] Forward roles through `WCommitmentConfig`.

Field-reduction reference math:

- [x] Add `SubfieldParams<D>`.
- [x] Validate power-of-two ring dimensions.
- [x] Validate `k | D/2`.
- [x] Validate `4k + 1` is invertible modulo `2D`.
- [x] Bound subgroup exponent enumeration to avoid non-terminating loops.
- [x] Add `trace_h`.
- [x] Add `psi_pack`.
- [x] Test `D = 64` and `D = 128` subgroup sizes.
- [x] Test `k = 1` trace collapse to scaled constant coefficient.
- [x] Test current verifier shortcut against `trace_h`.

Planner and proof-size accounting:

- [x] Add field-bit-width-aware byte helpers.
- [x] Thread field bit width into ring-vector byte estimates.
- [x] Thread field bit width into sumcheck/stage byte estimates.
- [x] Thread field bit width into batched root-layout scaling.
- [x] Update planner search/proof-size code to use configured field width.
- [x] Keep fp128 behavior unchanged when `field_bits = 128`.

Small-field integration coverage:

- [x] Add fp32 static config.
- [x] Add fp64 static config.
- [x] Add fp32 dense E2E round trip.
- [x] Add fp64 dense E2E round trip.
- [x] Keep existing fp128 E2E coverage passing.

Bugbot and review fixes:

- [x] Avoid extension limb label collisions.
- [x] Avoid base-field projection in extension challenge replay.
- [x] Validate subgroup generator invertibility.
- [x] Validate power-of-two `D` for Hachi subgroup cardinality.

Remaining for this PR:

- [ ] Wait for CI on the latest PR #71 head.
- [ ] Address any new review or Bugbot comments that land on the final PR head.
- [ ] Confirm reviewers agree that batching generalization is deferred to
  `specs/extension-field-opening-batching.md`.

Explicitly deferred to follow-up:

- [ ] Generalize public batched claims into a point/group/claim incidence graph.
- [ ] Migrate public opening points and claimed evaluations to
  `Cfg::ClaimField`.
- [ ] Implement the full `k > 1` Hachi embedding in the proof path.
- [ ] Implement Frobenius-conjugate optimized base/ext openings.
- [ ] Add extension-point dense and one-hot E2E tests.
- [ ] Add redistribution-attack regression tests.
- [ ] Teach the planner the base/ext split-parameter tradeoff.
- [x] Replace the single fp128 SIS floor registry with modulus-family-specific
  Q32/Q64/Q128 SIS floor tables before generating fp32/fp64 schedule tables.
- [x] Add larger small-field ring-dimension candidates, including D=256 and
  D=512 where appropriate, and select defaults from profile data rather than
  assuming the larger rings are faster or slower.
- [x] Add stack-backed sparse challenge sampler tiers through D=512 so larger
  small-field candidates do not use a heap-backed fallback.
- [x] Fix all-features zk setup sizing for extension-root-fold tests by
  reserving B-blinding columns in hand-rolled test configs.

### Files Modified In This PR

- `crates/akita-field/src/fields/lift.rs`
- `crates/akita-field/src/fields/wide.rs`
- `crates/akita-transcript/src/lib.rs`
- `crates/akita-challenges/src/lib.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-config/src/schedule_policy.rs`
- `crates/akita-types/src/field_reduction.rs`
- `crates/akita-types/src/layout/digit_math.rs`
- `crates/akita-types/src/layout/sis_derivation.rs`
- `crates/akita-types/src/schedule.rs`
- `crates/akita-planner/src/proof_size.rs`
- `crates/akita-planner/src/search.rs`
- `crates/akita-pcs/tests/transcript.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
- `specs/general-field-support.md`
- `specs/extension-field-opening-batching.md`

## References

- Hachi field-reduction implementation:
  `crates/akita-types/src/field_reduction.rs`
- Config field roles:
  `crates/akita-config/src/lib.rs`
- Extension transcript helpers:
  `crates/akita-transcript/src/lib.rs`
- Extension field traits:
  `crates/akita-field/src/fields/lift.rs`
- Field-width digit math:
  `crates/akita-types/src/layout/digit_math.rs`
- Proof-size accounting:
  `crates/akita-planner/src/proof_size.rs`
- Transcript coverage:
  `crates/akita-pcs/tests/transcript.rs`
- Small-field E2E coverage:
  `crates/akita-pcs/tests/akita_e2e.rs`
- Follow-up batching and extension-opening design:
  `specs/extension-field-opening-batching.md`
