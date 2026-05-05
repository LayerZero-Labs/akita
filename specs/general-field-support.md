# Spec: General Field Support

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-02 |
| Status | proposed |
| PR | quang/general-fields |

## Summary

Akita should support commitment, opening, transcript, and planning paths for
base fields beyond the current fp128 production field, starting with 32-bit and
64-bit prime-field profiles. This spec establishes the field-role split,
extension-field transcript plumbing, field-reduction reference utilities,
field-width-aware planner accounting, and static fp32/fp64 E2E coverage needed
to incrementally generalize the PCS without weakening the existing fp128
verifier path.

## Intent

Introduce the protocol scaffolding required for general base-field and
extension-field support while preserving the existing fp128 behavior as the
default production path.

The key surfaces are:

- `CommitmentConfig::{Field, ClaimField, ChallengeField}` for separating the
  ring/setup base field from public claim fields and Fiat-Shamir challenge
  fields.
- `akita_types::field_reduction` for subgroup trace and fixed-subfield packing
  reference utilities.
- Extension-aware transcript helpers for appending claim-field elements and
  sampling challenge-field elements over a base-field transcript.
- Field-bit-width-aware planner and proof-size accounting for fp32/fp64
  profiles.
- Static fp32/fp64 proof-optimized config modules and dense E2E tests.

## Invariants

- Ring commitments, setup matrices, digit decomposition, CRT/NTT work, and SIS
  bounds remain over `Cfg::Field`.
- Existing fp128 presets continue to use
  `Field = ClaimField = ChallengeField`.
- Transcript absorption of extension fields is coordinate-order sensitive and
  deterministic.
- `ChallengeField` sampling must not silently project extension challenges
  back into the base field.
- The current verifier shortcut remains the `k = 1` specialization of the
  subgroup trace relation and stays constant time in the hot path.
- Planner digit counts and proof-size estimates must respect the configured
  field bit width rather than assuming 128-bit elements everywhere.
- Small-field static configs must pass commit/prove/verify without changing
  the existing public prover API.

## Non-Goals

- This does not complete full extension-valued public opening claims in the
  production prover/verifier API.
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
- The verifier-side `k = 1` relation is covered against the general trace
  helper while preserving the constant-time specialization.
- Planner digit and proof-size helpers can model non-128-bit field widths.
- Static fp32/fp64 configs compile and pass dense commit/prove/verify E2E
  tests.
- Transcript tests cover extension challenge replay without base-field
  projection.
- Existing fp128 protocol tests, clippy, and doc tests pass.

## Implementation Order

1. Add field-reduction reference utilities and tests.
2. Split config field roles and add config-level transcript helpers.
3. Add extension-aware transcript and replay tests.
4. Parameterize planner digit/proof-size accounting by field width.
5. Add fp32/fp64 static profiles and E2E tests.
6. Preserve fp128 verifier behavior and prove the `k = 1` trace specialization.

## References

- `crates/akita-types/src/field_reduction.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-transcript/src/lib.rs`
- `crates/akita-types/src/layout/digit_math.rs`
- `crates/akita-planner/src/proof_size.rs`
- `crates/akita-pcs/tests/transcript.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
