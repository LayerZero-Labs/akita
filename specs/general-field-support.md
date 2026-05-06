# Spec: General Field Support Scaffolding

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-02 |
| Status | proposed |
| PR | quang/general-fields |
| Follow-up | extension-field-opening-batching.md |

## Summary

Akita should support base fields beyond the current fp128 production field,
starting with 32-bit and 64-bit prime-field profiles. This PR intentionally
implements the scaffolding needed for general fields while stopping before the
public batching and extension-opening cutover. The scope is: split field roles,
add extension-aware transcript helpers, add Hachi field-reduction reference
utilities, make planner/proof-size accounting field-width-aware, add static
fp32/fp64 configs and E2E coverage, and preserve the existing fp128 verifier
behavior.

The follow-up spec owns native small-field commitments opened at extension-field
points and the batching generalization needed to represent one committed group
opened at many points without duplicating public inputs.

## Intent

Introduce the protocol scaffolding required for general base-field and
extension-field support while preserving the existing fp128 behavior as the
default production path.

The key surfaces in this PR are:

- `CommitmentConfig::{Field, ClaimField, ChallengeField}` for separating the
  ring/setup base field from public claim fields and Fiat-Shamir challenge
  fields.
- Config-level helpers:
  - `append_claim_field(transcript, label, &ClaimField)`;
  - `sample_challenge_field(transcript, label) -> ChallengeField`.
- `akita_transcript::{append_ext_field, sample_ext_challenge}` for deterministic
  coordinate-wise extension-field transcript handling over a base-field
  transcript.
- `akita_types::field_reduction` for subgroup trace and fixed-subfield packing
  reference utilities.
- Field-bit-width-aware planner and proof-size accounting for fp32/fp64
  profiles.
- Static fp32/fp64 proof-optimized config modules and dense E2E tests.

## Scope Boundary

This PR does everything up to the point where native extension openings require
a better claim model. In particular, it is acceptable and intentional that the
current public prover/verifier API still takes base-field opening points and
base-field claimed evaluations. The new `ClaimField` and `ChallengeField` roles
are introduced now so later work can migrate the existing surfaces in place
without adding compatibility wrappers.

This PR must not introduce a separate public API for each batching special case.
The follow-up PR will generalize the claim input model once extension openings
need one committed polynomial group to be opened at several Frobenius-conjugate
points.

## Invariants

- Ring commitments, setup matrices, digit decomposition, CRT/NTT work, and SIS
  bounds remain over `Cfg::Field`.
- Existing fp128 presets continue to use
  `Field = ClaimField = ChallengeField`.
- Extension absorption is coordinate-order sensitive and deterministic.
- Extension challenge sampling draws all base-field limbs and must not silently
  project challenges back into the base field.
- The current verifier shortcut remains the `k = 1` specialization of the
  subgroup trace relation and stays constant time in the hot path.
- Planner digit counts and proof-size estimates must respect the configured
  field bit width rather than assuming 128-bit elements everywhere.
- Static fp32/fp64 configs are scaffolding profiles. They are not generated
  production schedules.

## Implementation Details

### Field Roles

`CommitmentConfig` exposes three associated field roles:

- `Field`: the base field used by rings, commitments, setup matrices, digit
  decomposition, and SIS bounds.
- `ClaimField`: the field intended for public opening points and claimed
  evaluations.
- `ChallengeField`: the field intended for Fiat-Shamir scalar challenges in
  sumcheck-style steps.

For this PR, existing public prove/verify surfaces still use `Field`. The role
split is a typed staging point for the follow-up cutover, not a completed
extension-opening API.

### Extension Transcript Helpers

Extension elements are absorbed by serializing their base-field coordinates in
canonical basis order. For degree-one extensions, helpers preserve the existing
label behavior. For higher-degree extensions, each limb uses a derived label so
coordinate swaps or nested-limb changes transcript-diverge.

Challenge sampling draws `EXT_DEGREE` base-field challenges under the same limb
label convention and reconstructs the extension element with
`ExtField::from_base_slice`.

### Field Reduction Reference Utilities

`SubfieldParams<D>` models the Hachi subgroup
`H = <sigma_-1, sigma_(4k+1)>` for power-of-two ring degree `D` and extension
degree `k | D/2`. It validates:

- nonzero ring dimension and extension degree;
- power-of-two `D`;
- `k | D/2`;
- invertibility of `4k + 1` modulo `2D`.

`trace_h(params, x)` computes `sum_{sigma in H} sigma(x)`.
`psi_pack(params, values)` implements the coefficient-placement part of Hachi's
packing map. This is a reference utility; it does not yet implement the full
production `k > 1` extension-opening transform.

The current verifier relation is tested as the `k = 1` specialization, where
the trace collapses to scaled constant-term extraction.

### Field-Width Accounting

Planner and proof-size helpers take explicit field-bit widths where byte sizes
depend on serialized field elements. This lets fp32/fp64 scaffolding profiles
exercise the same schedule/proof-size code without inheriting fp128 byte costs.

### Static Small-Field Profiles

The fp32/fp64 profiles are static integration scaffolds. They demonstrate that
setup, commit, prove, and verify can run over smaller base fields with the
current base-field opening API. They do not claim to be tuned production
parameters and do not exercise extension-field opening points.

## Non-Goals

- This does not complete extension-valued public opening claims in the
  production prover/verifier API.
- This does not implement the real `k > 1` Hachi embedding in the proof path.
- This does not implement the Frobenius-conjugate optimized base/ext opening
  path.
- This does not generalize the public batched-claim input model.
- This does not regenerate production fp32/fp64 schedule tables.
- This does not replace every protocol challenge with `Cfg::ChallengeField`.
- This does not benchmark or tune extension-field arithmetic.
- This does not change the default fp128 production preset or its security
  parameters.

## Acceptance Criteria

- `CommitmentConfig` exposes separate base, claim, and challenge field roles.
- Config-level helpers append `ClaimField` values and sample `ChallengeField`
  values through extension-aware transcript code.
- `field_reduction` exposes subgroup exponent enumeration, `trace_h`, and
  `psi_pack` with production-size coverage for `D = 64` and `D = 128`.
- `SubfieldParams` rejects invalid ring dimensions and subgroup generators that
  would break trace cardinality or exponent enumeration.
- The verifier-side `k = 1` relation is covered against the general trace
  helper while preserving the constant-time specialization.
- Planner digit and proof-size helpers can model non-128-bit field widths.
- Static fp32/fp64 configs compile and pass dense commit/prove/verify E2E
  tests.
- Transcript tests cover extension challenge replay without base-field
  projection.
- Existing fp128 protocol tests, clippy, docs, and CI pass.

## Implementation Order

1. Add field-reduction reference utilities and tests.
2. Split config field roles and add config-level transcript helpers.
3. Add extension-aware transcript and replay tests.
4. Parameterize planner digit/proof-size accounting by field width.
5. Add fp32/fp64 static profiles and E2E tests.
6. Preserve fp128 verifier behavior and prove the `k = 1` trace specialization.
7. Stop before public claim-shape or extension-opening migration; continue in
   the follow-up spec.

## References

- `crates/akita-types/src/field_reduction.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-transcript/src/lib.rs`
- `crates/akita-field/src/fields/lift.rs`
- `crates/akita-types/src/layout/digit_math.rs`
- `crates/akita-planner/src/proof_size.rs`
- `crates/akita-pcs/tests/transcript.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
