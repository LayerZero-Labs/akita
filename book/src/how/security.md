# Security model

> **Status:** stub. Part of the initial Akita Book scaffold.

One canonical security narrative: the hardness assumption, how Ajtai ranks
connect to security bits, the weak-binding fold price, and the current SIS table
model. Keep the marketing claim separate from audited reality. See
[Introduction → Security status](../intro.md#security-status-honest).

## SIS / MSIS and Ajtai sizing

Production Ajtai key sizing uses generated Module-SIS width tables keyed by the
versioned SIS policy, modulus family, ring dimension, and coefficient-`L∞`
bound:

```text
(sis_security_policy, family, ring_dimension, coeff_linf_bound)
    -> max secure width by module rank
```

The shipped policy is
`Classical138Quantum128WithIdealizedBcssV1`. It accepts a row only when the
full optimizer finds both at least 138 bits under the classical ADPS16 model
and at least 128 bits under the conventional ADPS16 quantum model. A lookup
for an unsupported policy returns `None` until a matching table is generated
and checked in.

The checked-in policy table was generated with the `local-minimum` optimizer
profile. Each hard model and the idealized BCSS diagnostic runs an independent
Python-compatible local beta and zeta search inside each table row. Parallel
generation parallelizes rows; it does not make one row's optimizer exhaustive.

CSV table-generation artifacts include separate classical and conventional-
quantum target margins, optimizer witnesses, and BCSS review flags so review can
see how close each binary-searched width boundary is to every policy line.
Narrow margins are not verifier-visible state, but they are sensitive provenance
and should be audited before treating a new table as a durable security floor.

The planner derives role bounds as coefficient-`L∞` values because those are the
values enforced by the protocol. It does not convert production role bounds
through a Euclidean `d * B^2` key. The Euclidean estimator code remains an
offline comparison path.

The production lookup is table-only. Verifier-reachable code must reject a
missing table row or unsupported floor with `AkitaError`; it must not run the
estimator at verification time.

### Classical and quantum policy

The checked-in table enforces the joint policy: 138 classical bits and 128 bits
under the conventional ADPS16 quantum Core-SVP model. It separately reports one
idealized BCSS diagnostic using the `0.2563 * beta` exponent, with a 124-bit
manual-review line. BCSS is not a production rank constraint in this policy.
The BCSS model assumes heuristic asymptotics, exponential reusable-sieve
storage, and writable QRAQM with coherent access; it is not a concrete
fault-tolerant resource estimate or an unqualified post-quantum security proof.

The complete decision, assumptions, review line, claim language, and
implementation acceptance criteria live in
[`specs/sis-classical138-quantum128-bcss-policy.md`](../../../specs/sis-classical138-quantum128-bcss-policy.md).

**Sources to fold in**

- `crates/akita-types/src/sis/mod.rs`, `ajtai_key.rs`, `generated_sis_table/`, `norm_bound.rs`.
- Paper §2.2 `def:msis`, §3.12 `sec:batched-soundness` ("MSIS targets", "Two norm models").
- `docs/security-posture.md`, `specs/sis-linf-table-cutover.md`.
- `specs/sis-classical138-quantum128-bcss-policy.md` (implemented policy and
  generation provenance).

## Norm bounds and weak binding

The fold-response bounds, the committed-fold price as relaxed binding, the
batched weak-opening definition, and why range checks do not lower the binding
norm. Keep the fold-reprice correction explicit.

**Sources to fold in**

- `crates/akita-types/src/sis/norm_bound.rs`, `layout/digit_math.rs` (`optimal_m_r_split`).
- Paper §3.12 `sec:batched-soundness` (`def:batched-weak-opening`, `lem:batched-weak-binding`, `prop:committed-fold-price`).
- `specs/weak-binding-norm-fix.md` (fold reprice — keep the correction section).
- `specs/fold-linf-rejection.md` (fold digit-count tightening).
