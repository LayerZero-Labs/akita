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
above-target lower bound of at least 128 bits. The decision threshold is an
explicit estimator configuration value supplied by the policy profile. The
beta search checks values from 40 through the capped Euclidean baseline and
stops once the monotone ADPS16 lower bound exceeds the best complete candidate.
It also returns a classified above-target result once both the best visited
attack and the lower bound for all unvisited beta values exceed 128 bits. For
`B > 1`, the global infinity estimate includes the Euclidean baseline
explicitly because `L2 <= B` implies `L-infinity <= B`; the baseline is
therefore an attack, not only a search cap. The diagnostic compression cells
with `B = 1` are the production instance of the `B <= 1` edge case, where this
model has no defined Euclidean-dimension optimization. They do not substitute
another bound: they omit that attack and sweep through the estimator's full
supported beta range. For each visited beta,
the optimizer exhausts every valid effective dimension before the LGSA profile
stabilizes, checks both endpoints of the stable unit-vector tail, and checks
both sides of any probability-regime transition inside that tail. The full
valid tall-lattice domain is `0 <= zeta < d - n`. Fixed and exhaustive estimates
reject an effective dimension `d - zeta <= n` instead of pricing a non-tall
q-ary embedding. Width generation starts at the first tall ring width
`width = rank + 1`. Narrower widths inherit a cutoff only when that tall
instance is certified; if it already fails, the row emits cutoff zero and
runtime must choose a higher rank. A lookup for an unsupported policy, exact
modulus profile, role, or scalar cell fails closed.

The checked-in policy table may use `local-minimum` only to discover a candidate
boundary. Every emitted boundary and its immediate rejected successor are
certified under the proven-pruned beta and zeta domain. Parallel generation
parallelizes independent rows and does not change the certificate domain or
output ordering.

The estimator hardening described above changes the acceptance model. The
checked-in SIS table retains the unversioned `Quantum128BitADPS16` policy ID
and wire tag `1`; evaluator revision `akita-infinity-width-v3`, the regenerated
table digest, and dependent catalog identities bind the corrected semantics.
The q32 Inner/A profile guard stops at `2^28 - 1`; q64 uses `2^41 - 1`; q128
uses `2^44 - 1`. Within those guards, the audited A cells are the exact
one-response protocol collisions `4 * ||c||_1 * (2^t - 1)` for reachable
opening-basis/digit-depth products `t <= 33` and dimension-compatible challenge
families. A raw collision target selects the smallest covering audited cell for
the same profile and dimension; it does not jump to a generic power-of-two
bucket. B and D retain their exact gadget-anchor rounding.

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

### Inspect the modeled cost of generated schedules

The generated tables establish that each scheduled SIS instance meets the
policy threshold. Maintainers can also run the estimator directly at every
matrix coordinate in an expanded schedule. The report includes A, B, shared D,
terminal A, precommitted and setup-prefix groups, and both maps in each
compressed payload chain.

Run the summary for every checked-in catalog row:

```bash
cargo run --release -p akita-planner \
  --features catalog-security \
  --example catalog_security -- --check
```

The command prints one tab-separated row per schedule. `--check` returns an
error if a row falls below the target named by its SIS policy. Add `--details`
to print the expanded schedule and every SIS occurrence. Restrict the report
with a family name, `--final-group NUM_VARSxNUM_POLYNOMIALS`, or the exact
`--row-digest HEX` identity printed by the summary.

These numbers are derived diagnostics. They are not fields of the generated
schedule, schedule digest, proof, or verifier configuration, and the repository
does not track a perpetually regenerated report. A security- or schedule-review
PR may retain a head-pinned TSV under `specs/evidence/`, but that snapshot is
review evidence rather than runtime authority.

The reported value is the minimum scalarized SIS attack cost under the
policy's ADPS16 quantum/LGSA model. It is not an unqualified concrete-security
claim: the scalar estimator does not price attacks that exploit ring or module
structure, CRT splitting, subfield projection, or role-specific matrix
structure.

The production lookup is table-only. Verifier-reachable code must reject a
missing table row or unsupported floor with `AkitaError`; it must not run the
estimator at verification time.

### Quantum policy

The production rule is the ADPS16 quantum LGSA model with a 128-bit target. It
is an attack-cost model, not a physical resource estimate or an unqualified
post-quantum security proof.

The conventional `0.2650 * beta` quantum Core-SVP cost is deliberate. Akita
previously evaluated the newer idealized BCSS23 `0.2563 * beta` sieve as an
independently optimized diagnostic over 6,240 generated rows. Its reusable
quantum walks assume exponential sieve storage and writable coherent QRAQM;
zero accepted ADPS16 rows fell below the corresponding 124-bit review line.
The idealized model therefore remains documented sensitivity evidence rather
than a production constraint.

LGSA is likewise an explicit attacker strategy: rerandomize the q-ary basis so
BKZ forgets its canonical q-vectors. On representative widened q64 and q128
rows, LGSA is no more expensive than ordinary GSA and is cheaper than the
determinant-preserving Chen-Nguyen profile simulations. The optional symmetric
ZGSA compatibility path caps its paired smoothing steps at the smaller of the
q-vector and identity-vector zones, preserving the lattice determinant even
when q-vectors are the majority.

Infinity-norm probabilities are priced on the coordinates that remain after
the attacker's `zeta` projection. In particular, the small-box condition uses
`sqrt(d - zeta) * B <= q`. Using the original dimension can select the wrong
probability formula and overstate security; this active-dimension rule is
regression tested, and integer production bounds use an exact boundary
comparison. Requests for the unimplemented high-precision backend fail closed.

The complete decision, assumptions, claim language, certificates, and
implementation acceptance criteria live in
[`specs/sis-quantum128-scalar-n-table.md`](../../../specs/sis-quantum128-scalar-n-table.md).

**Implementation map**

- `crates/akita-types/src/sis/mod.rs`, `ajtai_key.rs`, `l2_table.rs`,
  `physical_l2.rs`, `generated_sis_table/`, and `norm_bound.rs`.
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

For a chunked response, the A rows do not bind each chunk independently. They
bind the effective response

```text
z_sum = z^(0) + ... + z^(C-1).
```

Stage 1 nevertheless range-checks every chunk in the same balanced
base-`2^ell` digit interval. If each chunk uses `delta` digits, the exact
difference between two accepted values in one chunk has diameter
`2^(ell * delta) - 1`. Differences add across the `C` chunks, so the exact
coefficient collision target used by the A role is

```text
C_inf_chunked = 4 * kappa_1 * C * (2^(ell * delta) - 1).
```

This is the chunk-aware form of the same weak-binding calculation. Pricing a
chunked row as though `C = 1` would certify only one accepted response even
though the shared relation admits their sum. Akita therefore treats the
response-chunk count as a schedule parameter, re-derives this target during
planner replay and verifier admission, and rejects a row whose stored A bound
is smaller. All chunks use the same digit basis, digit depth, and full ambient
Z width. Their honest norms and their assigned E/T block counts may differ.
The selective `L2` route remains restricted to one response chunk.

The checked-in infinity table does not multiply its coverage by every supported
response-chunk count. Instead, a chunked raw target reuses the smallest audited
A cell that covers it. This is usually tight, but two- and four-chunk schedules
can inherit a conservative collision bound when their target lies in a gap
between existing cells. That choice can make the resulting SIS rank larger
than a table search specialized to that exact chunk geometry. We accept this
possible loss to keep the certified table small. Maintainers add a sparse
refinement cell only when catalog measurements show a material schedule
improvement; lookup fails closed when no existing cell covers the target. The
current table adds eight fp128 refinement cells for measured two- and
four-chunk operating points, contributing 160 certified rank rows. It adds no
eight-chunk coverage axis or dedicated eight-chunk cell.

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

### The accepted committed-source space

A committed level stores `num_digits_inner` balanced base-`2^log_basis_inner`
digits per source coefficient. `CommittedSourceContract::accepted_bounds`
computes the accepted centered interval for a balanced signed digit source. It
intersects what those digits can represent with the declared
`DecompositionParams::log_commit_bound` that the schedule was priced for.

The source class and numeric bound are independent. The class selects either the
unit one-hot structural contract or the balanced signed digit contract. The bound
selects the digit depth for either class. A balanced signed digit source may use
any valid bound, including `1` and the field width. A unit one-hot source remains
unit one-hot at any valid bound.

The intersection matters because the depth rounds up, so the representable
envelope is strictly wider than the declaration — by 256x at some shipped
geometries. The declaration is the binding side, because the planner prices a
bounded source's final digit plane at only the range its bound leaves.

A smaller bound is a smaller accepted witness space, not a weaker commitment. The
A-role collision bounds above are computed from the same digit envelope the
verifier admits, so a bounded family is priced for exactly what it accepts. The
unit one-hot class has a separate structural admission check. The declared bound
is inside `DecompositionParams`, which is hashed into the external artifact's
policy identity and serialized into the instance descriptor, so a proof cannot be
replayed against a family with a different bound.

The obligation the smaller space creates is on the *producer*, and it has two
halves. Committing above the representable envelope would bind a truncation,
because the decomposition keeps only the scheduled digits. Committing above the
declared bound would instead inflate the level-1 witness past the L2 response caps
frozen into the recursion suffix, because those caps were priced from the
declaration. `commit` rejects both. See
[Bounded committed sources](./configuration.md#bounded-committed-sources).

The fold nonce does not incur a fixed 12-bit soundness loss. Every nonce trial
is another random-oracle query, so the Fiat-Shamir reduction charges it through
the adversary's total query budget. See
[Polynomial commitments and binding](../foundations/pcs-and-binding.md#fiat-shamir-queries-and-fold-nonces).

## Reduced ring-relation soundness

Quotient lifting and reduced evaluation enforce the same native-ring
relations through different witness geometries. For one physical row of native
dimension $d$, reduced evaluation forms the residual

$$
Z(X)
=
\left(\sum_c A_c(X)W_c(X)-Y(X)\right)
\bmod (X^d+1).
$$

The protocol checks $Z(\alpha)$ at the existing ring-switch challenge
$\alpha$. If the row relation is false, $Z$ is a nonzero polynomial of degree
less than $d$, so random evaluation over the extension challenge field has the
usual Schwartz--Zippel bound. The existing $\tau_1$ challenge batches the
canonically ordered physical rows after each has been reduced in its own native
ring dimension.

The schedule, witness layout, relation mode, public statement, commitments,
and outgoing witness are bound before $\alpha$ is sampled. Consequently the
prover cannot choose the relation realization or its witness after seeing the
evaluation point. `RingRelationMode` is part of the instance descriptor and
effective schedule digest, not a proof field.

Reduced evaluation never divides by $\alpha^d+1$. An evaluation point that is
a root of the cyclotomic modulus therefore needs no special rejection: the
signed-wrap residue recurrence and terminal verifier kernel remain defined.
Removing quotient spans also does not change the A-role SIS binding argument,
the digit-range proof, or the scheduled `L∞`/`L2` response cap. It only removes
coordinates that existed to witness polynomial divisibility.

## Subring coefficient packing

For the coefficient layout, the three ring domains, and the exact reason the
partial evaluation is shorter, first read
[Subring coefficient packing](./proving/root-fold-ring-switch.md#subring-coefficient-packing).
This section focuses on transcript binding and soundness.

Before sampling a packing challenge, the prover binds every coordinate of the
partial opening through the D payload or its compressed H payload. The
transcript also binds the method, challenge subring dimension, challenge
family, group order, claim count, and block count. After challenge folding, the
prover binds `Q_pack` and the next witness before sampling `alpha`.

Both opening methods squeeze one dedicated 32-byte root per commitment group.
They then derive coordinate `(claim, block)` from a fresh SHAKE256 stream whose
input is exactly that root followed by the claim-major coordinate index as a
little-endian `u64`. The fixed widths make this encoding unambiguous. The
coordinate streams do not mutate the live transcript. Sequential and parallel
samplers therefore return the same ordered vector, and one coordinate can be
forked without changing any other coordinate.

The production primes satisfy the fixed LS18 congruence and shortness
condition used for unit pairwise challenge differences. This fact belongs to
the field and challenge security review. It is not planner metadata and does
not require a per-schedule certificate.

For CWSS extraction, fix the transcript prefix, group root, shared fold-response
nonce, and all coordinate-oracle answers except one. Reprogramming coordinate
`j` gives two accepting transcripts whose challenge difference is zero outside
`j`. The production LS18 condition makes every nonzero sparse-challenge
difference a unit, so subtracting the accepted relations isolates that opening.
The extractor uses the central accepting vector and one such fork for every
claim-major coordinate. The random-oracle reduction charges all root and
coordinate queries, including the jointly searched fold nonce. It does not
assume that two arbitrary full-vector forks are enough.

The packed consistency equation still gives one polynomial identity in `E[Y]`.
After including the `(Y^s + 1)Q_pack` term, its degree is at most `2s-1`, so the
conditional polynomial-check error is `(2s-1)/|E|`. This term is added to the
existing CWSS, random-oracle forking, sum-check, collision, and MSIS terms. See
the active
[subring coefficient packing design record](../../../specs/subring-coefficient-packing.md)
for the complete accounting.

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
- `specs/archive/2026-Q3/weak-binding-norm-fix.md` records the earlier fold reprice.
- `specs/fold-linf-rejection.md` (fold digit-count tightening).
- `specs/selective-l2-fold-security-sizing.md` (implemented physical norm correction
  and optional L2 route).
- `crates/akita-types/src/config.rs` (`DecompositionParams::log_commit_bound`) and
  `crates/akita-prover/src/api/commitment.rs`
  (`ensure_sources_fit_accepted_interval`) own the declared
  committed-source bound and the producer-side range check.
