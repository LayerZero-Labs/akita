# Spec: Extension-Field Openings and Batched Claim Incidence

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-06 |
| Status | proposed |
| Depends on | general-field-support.md |
| Target PR | follow-up to quang/general-fields |

## Summary

Akita should support native small-field commitments opened at extension-field
points. The core use case is a polynomial with coefficients in `Cfg::Field`
evaluated at a point in `Cfg::ClaimField = F_{q^k}` for sumcheck soundness.
This requires two changes beyond the scaffolding PR:

- migrating public opening points and claimed evaluations to `Cfg::ClaimField`;
- replacing the current nested batched-claim input with one incidence model that
  can represent one committed group opened at many points without duplicating
  commitments, hints, or polynomial slices.

## Motivation

The current public claim shape is grouped by opening point and then by
committed polynomial group. That is ergonomic for same-point batching, but it
does not naturally express one committed object opened at many points. The
optimized base/ext strategy needs exactly that shape: one transformed
polynomial is opened at several Frobenius-conjugate tail points.

Trying to add separate APIs for one-point-many-polys, one-poly-many-points, and
arbitrary matchings would create protocol churn and duplicate optimization
logic. Instead, there should be one canonical claim representation, with
special cases handled as execution-sharing choices beneath that representation.

## Claim Incidence Model

The public input should normalize to a small bipartite incidence graph:

- `points[p]`: distinct opening points in `Cfg::ClaimField`.
- `groups[g]`: distinct committed polynomial groups, each with one commitment
  and, on the prover side, one hint.
- `claims[c]`: individual openings, each carrying:
  - `point_idx`;
  - `group_idx`;
  - `poly_idx_within_group`;
  - `claimed_eval`.

This model covers:

- one point, many polynomials;
- one polynomial or one committed group, many points;
- many points and many committed groups with arbitrary matching;
- Frobenius-conjugate openings for the optimized base/ext path.

The protocol should derive existing flattened scheduling quantities from this
incidence graph:

- number of distinct points;
- number of distinct groups;
- total claim count;
- claim-to-point routing;
- claim-to-group routing;
- per-group polynomial counts;
- per-point claim counts.

Execution optimizations should be keyed by `point_idx`, `group_idx`, or
`(point_idx, group_idx)`:

- share commitment and hint material per `group_idx`;
- share point-derived ring reductions per `point_idx`;
- compute point/group-specific rows per edge only when both sides are needed;
- preserve the existing same-point batching and multipoint schedule machinery
  as derived views where possible.

## Extension-Field API Cutover

The existing commit/prove/verify surfaces should be migrated in place rather
than duplicated. Conceptually:

- commitment and setup types stay over `Cfg::Field`;
- polynomial coefficients stay over `Cfg::Field` for base-field commitments;
- public opening points become slices of `Cfg::ClaimField`;
- claimed evaluations become `Cfg::ClaimField`;
- transcript absorption of points/evaluations uses the config-level
  `append_claim_field` helper;
- Fiat-Shamir scalar challenges that require extension-field soundness use
  `Cfg::ChallengeField`.

The `Field = ClaimField = ChallengeField` fp128 path should remain the
degree-one specialization.

## Generic Extension-Valued Transform

For an extension-valued polynomial and an extension-valued evaluation point,
Akita should implement the real Hachi `k > 1` embedding:

- embed each extension element into the fixed multiplicative subfield
  `R_q^H ~= F_{q^k}`;
- pack `(R_q^H)^{D/k}` into `R_q` using the Hachi `psi` map;
- recover the scalar inner product through the subgroup trace relation;
- preserve the current `k = 1` coefficient embedding as the degeneration of the
  same construction.

This is the sound fallback for arbitrary extension-valued claims.

## Optimized Base-Coefficient / Extension-Point Path

For `f` with coefficients in `Cfg::Field` and opening point
`x in Cfg::ClaimField^ell`, Akita should avoid blindly paying the generic
extension-valued transform when the coefficient table is base-field-valued.

The intended optimization is:

1. Choose a split parameter `t <= log_2(k)` and write
   `f(X_head, X_tail) = sum_h lambda_h(X_head) f_h(X_tail)`.
2. Choose `F_q`-linearly independent coefficients `theta_h` in
   `Cfg::ClaimField`.
3. Form `g = sum_h theta_h f_h`, an extension-valued tail polynomial.
4. Commit to the Hachi ring transform of `g`.
5. Open the same committed transformed polynomial at Frobenius-conjugate tail
   points `x_tail^{q^j}`.
6. Use the Moore system induced by `theta_h^{q^{-j}}` to bind the slice
   evaluations `f_h(x_tail)`.
7. Verify the original claim
   `f(x_head, x_tail) = sum_h lambda_h(x_head) f_h(x_tail)`.

This trades a wider same-commitment multipoint opening for fewer transformed
variables. It should not add an extra ring-switching sumcheck.

Akita should not implement the literal Hachi optimization that asks the prover
for independent extension-valued partial evaluations and checks only one fixed
linear relation among them. Those partial evaluations are not uniquely bound by
that relation over the extension field.

## Planner Model

The planner should model a split-parameter tradeoff:

- smaller `t`: fewer same-commitment openings, more transformed variables;
- larger `t`: more Frobenius-conjugate openings, fewer transformed variables;
- `t = log_2(k)`: full base/ext optimization, opening width `k`, transformed
  variables reduced by `log_2(k)`.

Proof-size accounting must distinguish:

- base-field bytes for ring commitments, setup rows, digits, and SIS objects;
- extension-field bytes for public scalar claims, opening points, and
  extension-field sumcheck messages;
- shared per-group material versus per-point material in the incidence graph.

## Acceptance Criteria

- Public prove/verify inputs accept extension-field opening points and claimed
  evaluations via `Cfg::ClaimField`.
- The fp128 path remains the degree-one specialization with no public
  compatibility wrappers.
- The claim incidence model represents one committed group opened at multiple
  points without duplicating commitments, hints, or polynomial slices.
- Generic extension-valued openings use the real Hachi fixed-subfield embedding
  and trace relation for `k > 1`.
- Base-field-polynomial / extension-field-point openings support the
  Frobenius-conjugate optimized route.
- E2E tests cover fp32 or fp64 commitments opened at extension-field points.
- Dense and one-hot polynomial backends are covered.
- Wrong-claim rejection and redistribution-attack regressions are covered.
- Planner/proof-size tests cover the split-parameter tradeoff.

## Non-Goals

- This does not introduce separate public APIs for each batching special case.
- This does not keep old base-field-only aliases once the public cutover is
  complete.
- This does not require every optimization to land in the first implementation;
  the incidence model should support them without further public API churn.

## References

- `specs/general-field-support.md`
- `crates/akita-prover/src/lib.rs`
- `crates/akita-types/src/proof/scheme.rs`
- `crates/akita-types/src/proof/batch.rs`
- `crates/akita-prover/src/protocol/flow.rs`
- `crates/akita-verifier/src/proof/claims.rs`
- `crates/akita-types/src/field_reduction.rs`
