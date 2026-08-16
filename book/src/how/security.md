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
complete ADPS16 quantum certificate reports a finite score or a classified
above-target lower bound of at least 128 bits. The beta search checks values
from 40 through the capped Euclidean baseline and stops once the monotone
ADPS16 lower bound exceeds the best complete candidate. For each visited beta,
the LGSA profile transition proves that the checked zeta endpoints cover the
full zeta domain. A lookup for an unsupported policy, exact modulus profile,
role, or scalar cell fails closed.

The checked-in policy table may use `local-minimum` only to discover a candidate
boundary. Every emitted boundary and its immediate rejected successor are
certified under the proven-pruned beta and zeta domain. Parallel generation
parallelizes independent rows and does not change the certificate domain or
output ordering.

CSV table-generation artifacts include the certified accepted and rejected
successor witnesses, cutoff kind, cap provenance, and role provenance. These
are audit inputs, not verifier-visible state. The shared table digest commits
to the compact runtime table and its policy audit files together.

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
- `crates/akita-types/src/sis/generated_sis_table/policy_audit.csv` (canonical
  production table certificate).

## Norm bounds and weak binding

Every committed level records one A role security route. The coefficient
`L∞` route is always available. Every production profile also enables the typed
`L2` response model from level 3 onward. The root and early folds do not use the
`L2` route. A clear terminal response may use the route because the verifier
computes its complete integer norm directly.

Let `kappa_1` be the maximum physical coefficient `L1` norm of the fold
challenge. Let `gamma` be the bound used for challenge multiplication. This is
either `kappa_1`, or a verifier enforced operator norm threshold. Let `Z_inf`
be the accepted physical coefficient bound on the response, and let `S` be the
accepted squared norm of the complete physical response. The two collision
bounds are

```text
C_inf  = 8 * kappa_1 * Z_inf
C_2_sq = 64 * gamma^2 * S.
```

These formulas use the physical ring coefficients that enter the A role
Module SIS kernel. The small field extension embedding has already produced
those coefficients. Applying the Hachi logical to physical conversion at this
point would count that conversion twice.

An `L∞` schedule carries no norm proof. An `L2` schedule binds its cap and
integer proof shape into the schedule descriptor. The verifier proves the norm
of the same physical Z coefficients used by the security calculation and then
checks the public cap. The existing digit range proof remains mandatory. For a
clear terminal response, the verifier decodes every coefficient, computes the
integer square sum, and checks the same cap without a sumcheck.

The D64 and D128 L2 routes use transcript replayed operator norm rejection.
D64 uses the `(31, 11)` signed shell and runtime threshold 18. D128 uses the
production `(31, 0)` shell and runtime threshold 13. The fixed point checker
uses 48 fractional bits and accepts only a certified subset below the stated
mathematical threshold. Its rounding margins are 600 units for D64 and 351
units for D128. Exact support certificates show that each accepted family
retains at least 128 bits.

The response model is an honest prover model, not a security assumption. An
eligible source carries a modeled squared norm through the typed Z, E, T, R,
compression, and extension packing operations. The planner rounds that source
estimate upward, adds a 3 percent model envelope, and permits a response up to
`40/39` times the resulting conditional mean. Markov's inequality gives a
distribution-free grinding bound if the 3 percent envelope covers the source
model error. The planner freezes the resulting cap into the schedule. The
verifier enforces that exact cap. A model error can make proving fail more
often, but it cannot make the verifier accept a response above the cap.

The fold nonce does not incur a fixed 12-bit soundness loss. Every nonce trial
is another random-oracle query, so the Fiat-Shamir reduction charges it through
the adversary's total query budget. See
[Polynomial commitments and binding](../foundations/pcs-and-binding.md#fiat-shamir-queries-and-fold-nonces).

## Subring coefficient packing

Before sampling a packing challenge, the prover binds every coordinate of the
partial opening through the D payload or its compressed H payload. The
transcript also binds the method, challenge subring dimension, challenge
family, group order, claim count, and block count. After challenge folding, the
prover binds `Q_pack` and the next witness before sampling `alpha`.

The production primes satisfy the fixed LS18 congruence and shortness
condition used for unit pairwise challenge differences. This fact belongs to
the field and challenge security review. It is not planner metadata and does
not require a per-schedule certificate.

Extraction forks one claim and block position at a time while holding the
other transcript inputs fixed. The complete packed consistency equation is one
polynomial identity in `E[Y]`. After including the `(Y^s + 1)Q_pack` term, its
degree is at most `2s-1`, so its root bound is `(2s-1)/|E|`. The proof does not
project to one base field coordinate and does not add a factor of `k` or a
`1/|K|` error term.

This error composes with the existing A, B, D, F, and H Module SIS binding
events, the range and sumcheck errors, and the random oracle forking loss. See
the [subring coefficient packing design record](../../../specs/archive/2026-Q3/subring-coefficient-packing.md)
for the full derivation and acceptance checklist.

The challenge response identity is exact when the accepted challenge has
scalar covariance. The fixed point operator norm filter is not assumed to have
perfect symmetry. Five million sampled orbit comparisons at D64 and D128 had no
acceptance mismatches. Full orbit tests also had no mixed outcomes. The measured
covariance defect was about 0.07 percent, and explicit orbit randomization did
not improve it. The protocol therefore keeps the existing challenge sampler.

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
