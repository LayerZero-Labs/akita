# Security model

> **Status:** stub. Part of the initial Akita Book scaffold.

One canonical security narrative: the hardness assumption, how Ajtai ranks
connect to security bits, the weak-binding fold price, and the current SIS table
model. Keep the marketing claim separate from audited reality. See
[Introduction → Security status](../intro.md#security-status-honest).

## SIS / MSIS and Ajtai sizing

Production Ajtai key sizing uses generated scalar Module-SIS width tables keyed
by the SIS policy, exact modulus profile, coefficient-`L∞` bound, and
scalar dimension:

```text
(sis_security_policy, modulus_profile, coeff_linf_bound, n = rank * d)
    -> certified scalar cutoff
```

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

The planner derives role bounds as coefficient-`L∞` values because those are the
values enforced by the protocol. It does not convert production role bounds
through a Euclidean `d * B^2` key. The Euclidean estimator code remains an
offline comparison path.

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

**Sources to fold in**

- `crates/akita-types/src/sis/mod.rs`, `ajtai_key.rs`, `generated_sis_table/`, `norm_bound.rs`.
- Paper §2.2 `def:msis`, §3.12 `sec:batched-soundness` ("MSIS targets", "Two norm models").
- `docs/security-posture.md`, `specs/sis-quantum128-scalar-n-table.md`.
- `scripts/sis_golden/infinity_width_table.csv` (generation provenance for the
  infinity-width golden grid).

## Norm bounds and weak binding

The fold-response bounds, the committed-fold price as relaxed binding, the
batched weak-opening definition, and why range checks do not lower the binding
norm. Keep the fold-reprice correction explicit.

**Sources to fold in**

- `crates/akita-types/src/sis/norm_bound.rs`, `layout/digit_math.rs` (`optimal_m_r_split`).
- Paper §3.12 `sec:batched-soundness` (`def:batched-weak-opening`, `lem:batched-weak-binding`, `prop:committed-fold-price`).
- `specs/weak-binding-norm-fix.md` (fold reprice — keep the correction section).
- `specs/fold-linf-rejection.md` (fold digit-count tightening).
