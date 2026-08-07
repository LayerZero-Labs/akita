# Security model

One canonical security narrative: the hardness assumption, how Ajtai ranks
connect to security bits, the weak-binding fold price, and the current SIS table
model. Keep the marketing claim separate from audited reality. See
[Introduction → Security status](../intro.md#security-status-honest).

## SIS / MSIS and Ajtai sizing

Production Ajtai key sizing uses generated Module-SIS width tables. The
generator certifies scalar cutoffs `(B, n) -> max m` under
`Quantum128BitADPS16`, and the checked-in runtime artifact stores the
Module-SIS projection:

```text
(sis_security_policy, modulus_profile, d, coeff_linf_bound)
    -> max secure ring widths by module rank
```

where `width[r - 1] = cutoff_m(B, n = r * d) / d`.

The shipped policy is `Quantum128BitADPS16`. It accepts a row only when the
exhaustive ADPS16 quantum certificate reports a finite score or a classified
above-target lower bound of at least 128 bits. A lookup for an unsupported
policy, exact modulus profile, role, or scalar cell fails closed.

The checked-in policy table may use `local-minimum` only to discover a candidate
boundary. Every emitted boundary and its immediate rejected successor are
certified by exhaustive search over the configured beta and zeta domain.
Parallel generation parallelizes independent rows and does not change the
certificate domain or output ordering.

CSV table-generation artifacts include the certified accepted and rejected
successor witnesses, cutoff kind, cap provenance, and role provenance. These
are audit inputs, not verifier-visible state, and are committed separately from
the runtime table digest.

The planner has two production tables for the committed A role. The default
table uses a coefficient `L∞` bound. A separate Euclidean table is available
only when the selected fold proves a complete physical squared `L2` norm. Both
tables use the 128 bit quantum ADPS16 policy and have separate digests.

For the Euclidean table, the scalar SIS dimensions are `n = rank * D` and
`m = width * D`. The length bound is the square root of the complete collision
norm. The complete norm already includes every scalar coordinate, so the
planner does not multiply it by the matrix width again.

The production lookup is table-only. Verifier-reachable code must reject a
missing table row or unsupported floor with `AkitaError`; it must not run the
estimator at verification time.

### Quantum policy

The production rule is the ADPS16 quantum LGSA model with a 128-bit target. It
is an attack-cost model, not a physical resource estimate or an unqualified
post-quantum security proof.

The complete decision, assumptions, claim language, certificates, and
implementation acceptance criteria live in
[`specs/sis-quantum128-scalar-n-table.md`](../../../specs/sis-quantum128-scalar-n-table.md).

**Implementation map**

- `crates/akita-types/src/sis/mod.rs`, `ajtai_key.rs`, `l2_table.rs`,
  `physical_l2.rs`, `generated_sis_table/`, and `norm_bound.rs`.
- Paper §2.2 `def:msis`, §3.12 `sec:batched-soundness` ("MSIS targets", "Two norm models").
- `docs/security-posture.md`, `specs/sis-quantum128-scalar-n-table.md`.
- `scripts/sis_golden/infinity_width_table.csv` (generation provenance for the
  infinity-width golden grid).

## Norm bounds and weak binding

Every committed level records one A role security route. The coefficient
`L∞` route is always available. A later nonterminal fold may also have an `L2`
candidate when its preset supplies a measured cap for that exact fold level,
incoming witness length, response length, digit basis, and digit count. The
root, early folds, and terminal response do not use the `L2` route.

Let `kappa_1` be the maximum physical coefficient `L1` norm of the fold
challenge. Let `Z_inf` be the accepted physical coefficient bound on the
response, and let `S` be the accepted squared norm of the complete physical
response. The two collision bounds are

```text
C_inf  = 8 * kappa_1 * Z_inf
C_2_sq = 64 * kappa_1^2 * S.
```

These formulas use the physical ring coefficients that enter the A role
Module SIS kernel. The small field extension embedding has already produced
those coefficients. Applying the Hachi logical to physical conversion at this
point would count that conversion twice.

An `L∞` schedule carries no norm proof. An `L2` schedule binds its cap and
integer proof shape into the schedule descriptor. The verifier proves the norm
of the same physical Z coefficients used by the security calculation and then
checks the public cap. The existing digit range proof remains mandatory.

**Implementation map**

- `crates/akita-types/src/sis/norm_bound.rs` owns the two physical collision
  formulas. `crates/akita-types/src/proof/relation_range_image.rs` owns the
  physical response map. `crates/akita-prover/src/protocol/sumcheck/physical_l2_norm.rs`
  and `crates/akita-verifier/src/stages/physical_l2_norm.rs` own proof and replay.
- Paper §3.12 `sec:batched-soundness` (`def:batched-weak-opening`, `lem:batched-weak-binding`, `prop:committed-fold-price`).
- `specs/archive/2026-Q3/weak-binding-norm-fix.md` records the earlier fold reprice.
- `specs/fold-linf-rejection.md` (fold digit-count tightening).
- `specs/selective-l2-fold-security-sizing.md` (implemented physical norm correction
  and optional L2 route).
