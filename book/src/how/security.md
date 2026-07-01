# Security model

> **Status:** stub. Part of the initial Akita Book scaffold.

One canonical security narrative: the hardness assumption, how Ajtai ranks
connect to security bits, the weak-binding fold price, and the current SIS table
model. Keep the marketing claim separate from audited reality. See
[Introduction → Security status](../intro.md#security-status-honest).

## SIS / MSIS and Ajtai sizing

Production Ajtai key sizing uses generated Module-SIS width tables keyed by the
minimum security floor, modulus family, ring dimension, and coefficient-`L∞`
bound:

```text
(min_security_bits, family, ring_dimension, coeff_linf_bound)
    -> max secure width by module rank
```

The first shipped production floor is 138 bits. A lookup for any other floor
returns `None` until a matching table is generated and checked in.

The checked-in 138-bit table was generated with the `local-minimum` optimizer
profile. That profile uses Python-compatible local beta and zeta search inside
each table row. Parallel generation parallelizes rows; it does not make one
row's optimizer exhaustive.

CSV table-generation artifacts include `max_security_margin_bits` and
`next_failure_margin_bits` so review can see how close each binary-searched
width boundary is to the target floor. Narrow margins are not verifier-visible
state, but they are sensitive provenance and should be audited before treating a
new table as a durable security floor.

The planner derives role bounds as coefficient-`L∞` values because those are the
values enforced by the protocol. It does not convert production role bounds
through a Euclidean `d * B^2` key. The Euclidean estimator code remains an
offline comparison path.

The production lookup is table-only. Verifier-reachable code must reject a
missing table row or unsupported floor with `AkitaError`; it must not run the
estimator at verification time.

**Sources to fold in**

- `crates/akita-types/src/sis/mod.rs`, `ajtai_key.rs`, `generated_sis_table.rs`, `norm_bound.rs`.
- Paper §2.2 `def:msis`, §3.12 `sec:batched-soundness` ("MSIS targets", "Two norm models").
- `docs/security-posture.md`, `specs/sis-linf-table-cutover.md`.

## Norm bounds and weak binding

The fold-response bounds, the committed-fold price as relaxed binding, the
batched weak-opening definition, and why range checks do not lower the binding
norm. Keep the fold-reprice correction explicit.

**Sources to fold in**

- `crates/akita-types/src/sis/norm_bound.rs`, `layout/digit_math.rs` (`optimal_m_r_split`).
- Paper §3.12 `sec:batched-soundness` (`def:batched-weak-opening`, `lem:batched-weak-binding`, `prop:committed-fold-price`).
- `specs/weak-binding-norm-fix.md` (fold reprice — keep the correction section).
- `specs/fold-linf-rejection.md` (fold digit-count tightening).
