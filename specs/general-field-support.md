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
64-bit prime-field profiles and culminating in native small-field commitments
opened at extension-field points. This spec establishes the field-role split,
extension-field transcript plumbing, field-reduction reference utilities,
field-width-aware planner accounting, static fp32/fp64 E2E coverage, and the
claim-shape model needed for optimized base-field-polynomial /
extension-field-point openings without weakening the existing fp128 verifier
path.

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
- A canonical batched-opening incidence model that separates distinct opening
  points, distinct committed polynomial groups, and individual opening claims.
- A native optimized path for base-field-valued polynomials evaluated at
  extension-field points using Frobenius-conjugate same-commitment openings.

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
- The first scaffolding phase keeps the existing public prover API, but the
  native extension-opening phase must migrate the existing API surfaces in
  place to use `Cfg::ClaimField` for opening points and claimed evaluations.
- Optimized base-field-polynomial / extension-field-point openings must not use
  the literal Hachi optimization that asks the prover for independent
  extension-valued partial evaluations and checks only one fixed linear
  relation among them.
- Same-commitment multipoint openings, same-point multipolynomial openings, and
  arbitrary mixtures should be represented by one general claim incidence
  model. Special cases may receive execution sharing, but should not create
  separate public protocol variants.

## Native Extension Opening Strategy

For a polynomial with coefficients in `Cfg::Field` and an evaluation point in
`Cfg::ClaimField = F_{q^k}`, Akita should avoid blindly paying the generic
extension-valued transform whenever the coefficient table is base-field-valued.
The intended optimized route is:

1. Choose a split parameter `t <= log_2(k)` and write the base polynomial as
   slices
   `f(X_head, X_tail) = sum_h lambda_h(X_head) f_h(X_tail)`.
2. Choose `F_q`-linearly independent coefficients `theta_h` in
   `Cfg::ClaimField` and form the extension-valued tail polynomial
   `g = sum_h theta_h f_h`.
3. Commit to the Hachi ring transform of `g`.
4. Open the same committed transformed polynomial at the Frobenius-conjugate
   tail points `x_tail^{q^j}`.
5. Use the Moore system induced by the conjugates of the `theta_h` to bind the
   slice evaluations `f_h(x_tail)`, then check the original claimed value
   `f(x_head, x_tail)`.

This trades a wider same-commitment multipoint opening for fewer transformed
variables. It should not add an extra ring-switching sumcheck.

## Batched Claim Incidence Model

The current nested shape groups claims by opening point and then by committed
group. That is enough for same-point batching, but one polynomial opened at
many points is naturally represented by repeating the same committed group
under many point groups. The native extension-opening path would hit exactly
this case: the transformed polynomial `g` is one committed object opened at
several Frobenius-conjugate points.

The principled model is a small bipartite incidence graph:

- `points[p]`: distinct opening points.
- `groups[g]`: distinct committed polynomial groups, with one commitment and
  one prover hint per group.
- `claims[c]`: individual openings, each carrying `(point_idx, group_idx,
  poly_idx_within_group, claimed_eval)`.

This single model covers:

- one point, many polynomials;
- one polynomial or one committed group, many points;
- many points and many groups with arbitrary matching; and
- the Frobenius-conjugate openings required by the base/ext optimization.

Protocol code should derive the existing flattened schedule quantities from
this incidence graph: total claim count, point count, group sizes, and
claim-to-point routing. Execution optimizations should be local sharing
decisions keyed by `point_idx`, `group_idx`, or `(point_idx, group_idx)`, not
new public protocol variants. A first implementation may preserve the current
flattened behavior internally, but the public/input normalization layer should
avoid forcing callers to duplicate commitments, hints, or polynomial slices
when the same group is opened at multiple points.

## Non-Goals

- This does not regenerate production fp32/fp64 schedule tables.
- This does not replace every protocol challenge with `Cfg::ChallengeField`.
- This does not benchmark or tune extension-field arithmetic.
- This does not change the default fp128 production preset or its security
  parameters.
- This does not introduce parallel public APIs for each batching special case.
  The existing commit/prove/verify surfaces should be migrated in place.

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
- The next phase migrates opening points and claimed evaluations to
  `Cfg::ClaimField` while preserving base-field ring commitments over
  `Cfg::Field`.
- Base-field-polynomial / extension-field-point E2E tests cover dense and
  one-hot polynomials, including a wrong-claim rejection test.
- A Frobenius-conjugate same-commitment opening test demonstrates that one
  committed transformed polynomial can be opened at multiple conjugate points
  without duplicating the commitment group in the public claim input.

## Implementation Order

1. Add field-reduction reference utilities and tests.
2. Split config field roles and add config-level transcript helpers.
3. Add extension-aware transcript and replay tests.
4. Parameterize planner digit/proof-size accounting by field width.
5. Add fp32/fp64 static profiles and E2E tests.
6. Preserve fp128 verifier behavior and prove the `k = 1` trace specialization.
7. Generalize the batched claim input into a point/group/claim incidence graph.
8. Migrate opening points and claimed evaluations to `Cfg::ClaimField`.
9. Implement the real `k > 1` Hachi embedding and trace opening relation.
10. Add the Frobenius-conjugate optimized base/ext opening path.
11. Teach the planner/proof-size model the split-parameter tradeoff:
    fewer transformed variables versus more same-commitment conjugate openings.

## References

- `crates/akita-types/src/field_reduction.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-transcript/src/lib.rs`
- `crates/akita-types/src/layout/digit_math.rs`
- `crates/akita-planner/src/proof_size.rs`
- `crates/akita-pcs/tests/transcript.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
