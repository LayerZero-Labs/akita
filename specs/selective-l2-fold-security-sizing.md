# Spec: Selective L2 Fold Security Sizing

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-06 |
| Revised | 2026-08-14 |
| Status | implemented |
| PR | [#369](https://github.com/LayerZero-Labs/akita/pull/369) |
| Supersedes | The physical A role embedding factor in `archive/2026-Q3/weak-binding-norm-fix.md` |
| Superseded-by | |
| Book-chapter | book/src/how/security.md |

## Purpose

This specification explains how Akita sizes the A matrix for a folded
response. It covers two security routes:

1. A coefficient bound, written Linf.
2. A bound on the sum of squared physical coefficients, written L2.

The planner treats these as separate candidates. A selected Linf route proves
only the existing digit range statement. A selected L2 route also proves, or
directly checks at the terminal boundary, a public squared norm cap. The
verifier derives the selected route from the schedule. The prover cannot choose
the route after seeing the witness.

The planner runs only while Akita generates a catalog. Runtime code resolves a
frozen row from that catalog. A committed group stores one checked
`GroupCommitPhaseParams`, and those parameters are the only source for its matrix
geometry, decomposition, and slice count. The verifier resolves the public row
digest and never runs planner search.

The specification also explains the planner model that predicts honest
responses. That model affects completeness, proof size, and prover work. It is
not a premise of the binding proof. The distinction is important:

* The verifier-enforced cap and the SIS table determine soundness.
* The response model predicts whether an honest prover will satisfy that cap.
* Measurements test whether the prediction is useful in supported profiles.

This document is both an implementation contract for PR 369 and a design
record for readers who need to understand this part of Akita without reading
the full planner and proof implementation first.

## Result in one page

Akita keeps the Linf route at every committed fold. It may add an L2 candidate
at fold level 3 or later when all of these conditions hold:

* The fold has one opening claim.
* The response has one physical chunk.
* The response basis is at least 8.
* The typed response model produces a finite public cap.
* The physical norm proof shape is valid.
* The Euclidean SIS table contains the required cell.
* The L2 route lowers the A matrix rank.

The terminal response may also use L2. Its coefficients are already public, so
the verifier computes their exact integer squared norm directly. No recursive
norm proof is needed at that boundary.

For a response cap `S_max` and a challenge multiplication bound `gamma`, the L2
route prices the extracted collision with

```text
C_2_sq = 64 * gamma^2 * S_max.
```

For D64 and D128 selective L2 challenges, `gamma` is a verifier-enforced
operator norm threshold. Other supported dimensions use the deterministic
challenge L1 norm. The coefficient route remains

```text
C_inf = 8 * kappa_1 * Z_inf.
```

Here `kappa_1` is the physical challenge L1 bound and `Z_inf` is the maximum
physical response coefficient admitted by the scheduled digit depth.

Both formulas act on the same physical A response. Neither formula applies a
Hachi logical-to-physical embedding factor at this boundary. The Euclidean
estimator also receives the complete collision length once. It does not
multiply that length by the A matrix width a second time.

The planner carries total and typed source statistics through recursive folds:

```text
M             = estimated total source squared L2 norm
M_i           = estimated squared L2 norm of component i
P_i,ring      = estimated coordinate second moment across a complete packed ring
P_i,local     = estimated coordinate second moment for a strict packed subring
D_pack        = ring dimension used to pack the source
```

The component index `i` ranges over Z, E, T, R, and compression. The planner
uses `M` for L2 response caps. It uses the typed values for Linf digit depth.
The model follows the component counts in `WitnessLayout`.

The current calibration uses

```text
L2 cap = ceil(M * q_c * 1.03 * 40 / 39),
```

where `q_c` is the challenge squared L2 norm. The `1.03` factor covers measured
source-model error. The `40/39` factor gives a distribution-free Markov lower
bound of `1/40` on per-nonce acceptance, conditional on the source envelope
being valid. With 4096 fresh nonce attempts, the resulting exhaustion bound is
below `2^-149` for one response.

Fresh end-to-end validation covers 13 production profiles and 51 selected L2
responses. It includes dense and one-hot witnesses over fp32, fp64, and fp128,
plus direct, recursive-setup, multi-group, and W2, W4, and W8 multi-chunk
adapters. Observed L2 cap slack was 0.0847 to 10.4720 percent. Every proof
passed both verifier modes.

## Scope

### Included protocol changes

This specification covers the following changes.

* Per-fold Linf and L2 A matrix security routes.
* Removal of the physical A role Hachi norm double count.
* A separate quantum ADPS16 Euclidean SIS table.
* Certified D64 and D128 operator norm challenge rejection.
* A hidden recursive proof of the complete physical response norm.
* A direct terminal check of the clear physical response norm.
* Typed recursive response models for L2 caps and Linf digit depth.
* Complete-suffix planner selection and generated schedule audit.
* Checked committed profiles and strict runtime catalog resolution.
* Exact proof-size and response-bound reporting.

### Supported families

The typed response model is enabled for every generated scalar family in this
PR. This includes dense and one-hot fp32, fp64, and fp128 profiles, recursive
setup, multi-group, and multi-chunk adapters. A multi-chunk schedule can select
L2 after its recursive state has become a single physical chunk. An individual
L2 candidate never spans multiple chunks.

The independent Euclidean table covers the three production modulus profiles
and ring dimensions D32, D64, D128, D256, and D512. A geometry outside that
table, including a future D1024 L2 geometry, has no L2 table row and therefore
uses Linf until the audited Euclidean table is extended.

### Bundled performance work

The PR also contains stacked implementation work that makes the larger profile
matrix practical. This includes ring-switch output reuse, faster one-hot and
dense decomposition, sparse fold work, verifier selector optimization, and
profile-report cleanup. Reviewers should inspect those changes, but none is a
premise of the security argument in this specification.

### Not included

This change does not do the following.

* It does not replace Linf with L2 globally.
* It does not select L2 before fold level 3.
* It does not remove the digit range proof from an L2 route.
* It does not use a Gaussian response assumption for soundness.
* It does not use an uncertified operator norm threshold.
* It does not add operator norm rejection outside D64 and D128.
* It does not force a particular rank or number of folds.
* It does not add four-square slack, carry witnesses, or a field inequality
  proof.
* It does not reject and resample setup seeds from a planner envelope.

The last item remains a possible future completeness improvement. A verifier
could reproduce exact digit-plane or cyclic peak statistics from a public
setup seed and reject seeds outside a frozen envelope. This PR does not need
that mechanism for soundness or for its measured completeness target.

## Terms and coordinate system

Let

```text
R = Z[X] / (X^D + 1)
```

with centered integer coefficients. The A matrix is a module matrix over this
negacyclic ring.

The following terms have precise meanings in this document.

| Term | Meaning |
|---|---|
| Physical coefficient | One centered base-field ring coefficient after all extension-field packing. |
| Physical response | The complete coefficient vector consumed by the selected A matrix. |
| Logical value | A value before extension-field coordinates are packed into physical ring coefficients. |
| Response basis | The power-of-two basis used to decompose the folded response into Z digit planes. |
| Source witness | The committed witness entering one fold. |
| Folded response | The challenge-weighted sum produced by that fold. |
| Recursive witness | The typed Z, E, T, R, compression, and related segments committed for the next fold. |

For one fold, define

```text
kappa_1 = maximum physical coefficient L1 norm of a challenge
gamma   = accepted L2-to-L2 multiplication-operator bound for a challenge
Z_inf   = accepted maximum physical coefficient magnitude of the response
S       = accepted sum of squared physical response coefficients
```

The complete squared norm is

```text
S = sum_i z_i^2,
```

where the index covers every physical A response coefficient exactly once. It
is not a per-row norm, a per-chunk norm, or a matrix-column average.

The `SparseChallenge` is already a physical polynomial in the base-field ring.
Its `l1_norm()` counts physical coefficients. The prover also stores the folded
response as centered physical coefficients before it decomposes those values
into Z digit planes.

## What is rigorous and what is modeled

The design uses several kinds of reasoning. They should not be conflated.

| Claim | Status | Consequence if wrong |
|---|---|---|
| Extracted collision formulas from accepted bounds | Rigorous, subject to the Akita weak-binding reduction | Security failure if implemented incorrectly |
| Physical response mapping | Exact protocol definition | Security or correctness failure if inconsistent |
| Direct and LimbGram integer reconstruction | Exact after public no-wrap checks | Soundness failure if implemented incorrectly |
| D64 and D128 accepted-support certificates | Exact finite-family certificate with checked artifacts | Challenge entropy failure if incorrect |
| Euclidean and Linf SIS tables | Audited concrete-security estimates under the chosen lattice cost model | Security estimate changes if the model changes |
| One-hot root MGF cap | Analytic tail bound with directed floating-point inflation and a universal dominance guard | Honest rejection rate may increase if incorrect |
| Universal balanced-source Linf cap | Tail bound from public source bounds and random challenge signs | Honest rejection rate may increase if incorrect |
| Typed source moments | Component model plus empirical error envelope | Honest L2 or Linf grinding may take longer or exhaust |
| Rounded-normal Z digit moments | Conditional approximation | Honest grinding or later schedule choices may be worse |
| Typed Linf Gaussian slab model | Gaussian correlation bound applied to modeled coordinate variances, capped by the universal digit depth | Honest Linf grinding may take longer or exhaust if the Gaussian approximation is inaccurate |
| Measured challenge covariance | Empirical validation, not a proof of exact symmetry | Source model may have more error than measured |

Only the first five rows are part of the verifier security contract. The
remaining rows guide offline planning. The verifier never evaluates them.

## Security contract

### Extracted A collision

The weak-binding extractor compares two accepted openings. If each challenge
has physical L1 norm at most `kappa_1`, their difference has L1 norm at most
`2 * kappa_1`. If each response has Linf norm at most `Z_inf`, their difference
has Linf norm at most `2 * Z_inf`.

For an L2 route, let multiplication by each accepted challenge have operator
norm at most `gamma`. The difference of two accepted challenge operators then
has norm at most `2 * gamma`. If each response has squared L2 norm at most `S`,
their difference has L2 norm at most `2 * sqrt(S)`.

For physical negacyclic ring coefficients, Young's inequalities give

```text
||a * b||_inf <= ||a||_1 * ||b||_inf
||a * b||_2   <= ||M_a||_(2 to 2) * ||b||_2,
```

where `M_a` is negacyclic multiplication by `a`. Clearing the two weak-opening
denominators adds two products. The three factors of two come from:

1. Taking the difference of two challenges.
2. Taking the difference of two responses.
3. Adding the two denominator-clearing products.

The complete bounds are therefore

```text
C_inf  = 8 * kappa_1 * Z_inf
C_2_sq = 64 * gamma^2 * S.
```

When no certified operator norm policy exists, Akita sets
`gamma = kappa_1`. This recovers the L1-to-L2 convolution bound. At D64 and
D128, the selective L2 challenge sampler enforces a smaller certified `gamma`.

The selected route determines the table and the proof obligation. The planner
may compare ranks from both routes, but one accepted schedule needs only its
selected sound route.

### Linf response admission

At a nonterminal Linf fold, the schedule stores a response digit depth. Stage 1
proves that every Z digit lies in the scheduled balanced alphabet. The verifier
therefore accepts every recomposed coefficient in the full range represented
by that depth. A role security uses this representable range, not the smaller
honest-prover cap that the planner used to choose the depth.

At the terminal boundary, the response is clear. A Linf route stores and checks
a raw coefficient cap. An L2 route stores no independent Linf cap. It requires
the signed coefficient wire representation, canonical Golomb-Rice encoding,
and the scheduled payload budget, then checks the complete decoded squared
norm directly.

An error in the planner's Linf response prediction can reject an honest proof.
It cannot make the verifier accept a coefficient outside the range priced by
the A matrix.

### L2 response admission

An L2 schedule stores `S_max`. The recursive verifier proves the exact physical
square sum, or the terminal verifier computes it directly, and then checks

```text
S <= S_max.
```

The schedule audit recomputes

```text
C_2_sq = 64 * gamma^2 * S_max
```

and checks the selected A rank against the generated Euclidean table. The
planner response model is absent from this audit.

An error in the planner's L2 prediction can make nonce search fail. It cannot
make the verifier accept a response above `S_max`.

### No Hachi factor at the physical boundary

The Hachi map `psi` packs logical extension-field coordinates into physical
base-field ring coefficients. A logical-to-physical norm factor belongs only
in an argument that starts before this packing step.

The A role weak-binding path starts after packing:

1. The challenge is a physical sparse base-ring polynomial.
2. The response is a centered physical base-ring vector.
3. Stage 1 constrains the physical Z digits.
4. The A role kernel vector uses those same physical coordinates.

Applying `ring_subfield_norm_bound` here counts a conversion that has already
happened. The old fp32 and fp64 path priced

```text
8 * kappa_1 * 2 * Z_inf
```

instead of

```text
8 * kappa_1 * Z_inf.
```

This PR removes that extra factor from physical A role sizing. Other code that
starts from a logical extension-field norm must still apply the appropriate
conversion once.

### No Euclidean width factor in the length

For A matrix rank `r`, ring dimension `D`, and input width `w`, the scalar SIS
dimensions are

```text
n = r * D
m = w * D.
```

`C_2_sq` already sums the collision over all `w * D` scalar input
coefficients. The Euclidean estimator must use

```text
length_bound = sqrt(C_2_sq).
```

Using `sqrt(w * C_2_sq)` counts the input rows twice. The width remains in `m`,
where it belongs.

### Route identity and no-panic rule

Every committed level has one typed security route:

```text
InnerCommitSecurityRoute::Linf(SisTableKey)

InnerCommitSecurityRoute::L2 {
    table_key,
    response_l2_sq_cap,
    norm_proof_shape,
}
```

The route, cap, physical response length, proof shape, and table key are part of
schedule identity and descriptor bytes. The proof stream stays headerless. The
schedule tells the decoder exactly which values to read.

Verifier-reachable code validates lengths, integer ranges, allocation sizes,
and shape consistency before use. Malformed input returns `AkitaError` or
`SerializationError`. It must not panic.

## Certified challenge selection

### Why an operator norm helps

The deterministic convolution inequality uses `||c||_1`. For L2, the tighter
quantity is the spectral norm of negacyclic multiplication by `c`. The
mathematical value is

```text
Gamma(c) = max_j |c(zeta_j)|,
```

over the appropriate complex roots of `X^D + 1`. Rejecting challenges with
large `Gamma(c)` lowers `gamma` in `C_2_sq` without changing the response cap.

The prover and verifier replay the same bounded rejection sampler from the
transcript. The predicate uses a fixed-point upper enclosure. A challenge is
accepted only when that checked enclosure is below the runtime threshold.

### Production certified families

| Ring | Signed sparse shell | Squared L2 mass `q_c` | Runtime threshold `gamma` | Fixed-point containment | Certified accepted support |
|---|---:|---:|---:|---:|---:|
| D64 | 31 coefficients at magnitude 1 and 11 at magnitude 2 | 75 | 18 | `Gamma < 18 - 600 / 2^48` | 128.062439 bits |
| D128 | 31 coefficients at magnitude 1 | 31 | 13 | `Gamma < 13 - 351 / 2^48` | 128.563317 bits |

The runtime checker uses 48 fractional bits. It validates the certified root
coordinate error bound and proves the rounding margin with checked integer
arithmetic. The margin condition is

```text
h^2 > 8 * (max_l1 * eps_root)^2.
```

The D64 margin is 600 units and the D128 margin is 351 units. Security pricing
uses the runtime threshold. The exact certificate replay proves that the
contained accepted subset retains the support stated above.

Each challenge draw has at least 128 bits of accepted support. The schedule and
transcript bind the sparse shell and both thresholds. D64 and D128 have
separate certificate checkers under `scripts/operator_norm/`.

Other ring dimensions use their production sparse shell without operator norm
rejection. For those dimensions, L2 security sets `gamma` to the shell L1 norm.

### Measured challenge covariance

The fixed-point predicate could have broken the ring symmetries that preserve
the mathematical operator norm. Measurements found no such effect. Five
million random orbit pairs at each of D64 and D128 produced no acceptance
mismatch. Thirty-two complete orbits at each dimension produced no mixed
accepted and rejected orbit. The measured covariance defect was about 0.07
percent before and after explicit orbit randomization.

The data does not prove exact scalar covariance, so the planner still treats
the small measured defect as part of its 3 percent source envelope. There is no
observed asymmetry to correct and no deferred symmetrization work. Soundness
does not depend on covariance because the verifier enforces `gamma` for every
accepted challenge.

## Physical L2 proof

### One physical response authority

`PhysicalResponsePlan` derives the norm domain from `WitnessLayout` and the
Stage 2 range-image plan. It maps every physical response coefficient to its Z
digit planes exactly once. Padding addresses map to zero.

The plan also checks

```text
physical_response_len = A input width * A ring dimension.
```

The constructor, schedule validation, prover, verifier, and proof-size code use
the same equality. A layout cannot change the A width while retaining an old
norm-proof length.

The recursive L2 path currently applies to one scalar opening claim, one chunk,
one committed group, and no precommitted group at that fold. These restrictions
match the candidate eligibility rule.

### Direct shape

If the largest square sum allowed by the response digit ranges is strictly
below the base-field modulus, Stage 1 proves

```text
S = sum_x z(x)^2
```

directly in the field. The public digit bounds make the field equality an
integer equality because wraparound is impossible.

The prover cannot justify direct mode merely by reporting a small `S`. The
schedule derives the worst-case no-wrap bound before proof generation.

### LimbGram shape for small fields

When direct integer soundness does not fit in the field, write each centered
response coefficient in balanced base `B`:

```text
z(x) = sum_j B^j * z_hat_j(x).
```

For a public address block, define

```text
I_jk = sum_x z_hat_j(x) * z_hat_k(x).
```

The exact block square sum is

```text
S_block = sum_j B^(2j) * I_jj
        + 2 * sum_(j < k) B^(j+k) * I_jk.
```

The schedule chooses a block length that makes every allowed `I_jk` lie
strictly inside the centered interval `(-q/2, q/2)`. The verifier takes the
unique centered integer lift of each subclaim, reconstructs every block with
checked integer arithmetic, sums the blocks, and compares the result with
`S_max`.

The schedule fixes all of this shape:

* Physical response length.
* Block length.
* Response limb count.
* Upper-triangular limb pairs.
* Number and order of subclaims.

The verifier rejects an invalid block, missing or extra subclaim, bad centered
lift, integer overflow, inconsistent length, or reconstructed value above the
cap.

This construction proves equality to the integer norm. It does not prove a
field inequality and does not require four-square slack.

### Stage 1 fusion

The norm term is fused into the last Stage 1 range leaf.

* Direct mode adds a degree-two square term.
* LimbGram mode adds selector times limb times limb, which has degree three.
* The fused round degree is the maximum degree required by the range and norm
  terms.

A fresh transcript scalar batches the norm relation with the existing digit
range relation. The proof shape accounts for the exact extra coefficients,
claims, and final evaluations. Planner byte estimates call the same proof-shape
code used by serialization.

### Stage 2 binding

Stage 2 binds the final Stage 1 values to the committed Z digit witness. Direct
mode checks balanced recomposition at the sampled physical point:

```text
z(r) = sum_j B^j * z_hat_j(r).
```

LimbGram mode binds the required limb evaluations instead. Group, chunk, row,
coefficient, and padding selectors come from the shared physical plan.

The transcript samples the Stage 2 batching challenge only after it has
absorbed the Stage 1 claims. A prover cannot choose two false relations that
cancel under a challenge known in advance.

### Transcript and serialization

`PhysicalL2NormProof` contains the public integer norm, the shape-derived field
subclaims, virtual evaluations, and the norm sumcheck. Transcript absorption
separates the integer claim, subclaims, batching scalars, sumcheck, and virtual
evaluations.

Serialization remains schedule driven and headerless. Mutation tests cover the
norm, cap, route, subclaims, virtual evaluations, nonce, and Stage 2 values.

## Terminal L2 check

The terminal response has no recursive Stage 1 or Stage 2 proof. It is decoded
in the clear. For a terminal L2 route, the verifier:

1. Validates the terminal response shape and coefficient bounds.
2. Reconstructs every centered physical coefficient.
3. Computes the exact checked integer sum of squares.
4. Rejects unless the result is at most the scheduled `S_max`.
5. Audits the A matrix against `64 * gamma^2 * S_max`.

The terminal schedule uses the same certified challenge family and Euclidean
table as a recursive L2 level. It pays no hidden norm-proof bytes.

## Planner model

### Planner state

The response model stores

```text
M = estimated total squared L2 norm of the source witness
P = estimated largest per-coordinate second moment, in parts per million
```

The suffix memo key contains both values. This matters because two witnesses
can have the same total energy but different high-energy components.

Both values retain seven leading bits and round upward. The rounding creates a
finite reusable dynamic-program state and adds less than `1/64` relative error.

The planner constructs source moments after the root. Selective L2 admission
starts at fold level 3. These are different boundaries. The typed Linf model
can guide the fold immediately after the root even though L2 is not yet
eligible.

### Core response identity

For a fixed source vector `s` and a random negacyclic challenge `c` with scalar
coefficient covariance,

```text
E[||c * s||_2^2 | s] = E[||c||_2^2] * ||s||_2^2.
```

Write

```text
q_c = E[||c||_2^2].
```

The production sparse shells have fixed magnitudes, so `q_c` is also their
exact squared L2 mass. The identity is exact when challenge covariance is
scalar. The accepted operator norm sampler is only measured to be close to
that symmetry. The model envelope covers the observed defect and the other
approximations below.

### Root sources

The root policy depends on the declared witness type.

#### Dense root

A dense committed polynomial is caller controlled. The planner makes no
distributional assumption about it. For every balanced digit plane, it uses
the deterministic maximum magnitude `basis / 2` and sums the corresponding
squares over the exact logical source length. The peak moment is the largest
plane square.

This root bound is conservative by design. Removing the old digit snaps exposed
one unsupported fp128 dense `nv=50` key whose root response exceeded every
audited A table bucket. The generated dense catalog now stops at the largest
key supported without that heuristic.

#### Unit one-hot root

A unit one-hot source permits any hot position inside each policy-owned source
chunk. Root commitment and opening use the canonical coefficient table. For a
ring of dimension `D` and source chunk size `K`, one physical ring therefore
contains at most

```text
max nonzero coefficients = D / K, when K < D,
                           1,     when K >= D.
```

Distinct chunks occupy distinct canonical coefficients, so there are no
coefficient collisions and every nonzero coefficient has magnitude one. The
total root energy is the exact number of source chunks and the peak coefficient
square is one. This is a deterministic maximum over the public unit one-hot
contract and is independent of the extension degree.

The Linf root policy also evaluates the exact one-coordinate moment generating
function for the signed sparse challenge. For coefficient packing, the MGF
population is the challenge subring dimension `s`, while source occupancy is
still derived from the ambient canonical table. Directed floating-point
inflation makes each sampled Chernoff value an upper bound. The final digit
depth is never larger than the universal bound and never relies on the exact
policy outside its declared one-hot geometry.

### Recursive witness components

`next_source_moment` builds the exact `WitnessLayout` for the selected
candidate, then models each typed component. It does not fit one constant to
the final witness length.

| Component | Count and model | Status |
|---|---|---|
| Z | Exact physical response count. The pre-decomposition variance is prior source energy times `q_c`, divided by that count. Balanced digit moments use centered residues of a rounded normal integer. | Conditional approximation |
| E | Exact live count `claims * live blocks * d_a`. Each field digit plane uses the centered uniform second moment. | Exact for uniform power-of-two residues; approximate for protocol values |
| T | Exact live E count times the A output row count. Uses the same field digit moments. | Same as E |
| R | Exact row ranges from `WitnessLayout`. Uses full-width field digit moments. | Same as E |
| Compression | Exact coefficient count. Negative-binary digits contribute expected energy `1/2` per coefficient. | Exact under a balanced bit model |
| Tensor packing | Applies the multiplicity `(2K - 1) / K` to total energy and to the average coordinate moment across a complete packed ring. It also retains the local bound `P` for `K = 1` and `2P` for `K > 1`. A later fold uses the complete-ring value only when its ring dimension matches `D_pack`. | Exact under exchangeable extension coordinates; the local value is a deterministic coefficient bound |
| Padding | Contributes zero. | Exact |

For a power-of-two digit basis `b`, a centered uniform digit in
`[-b/2, b/2)` has second moment

```text
(b^2 + 2) / 12.
```

The top field digit plane uses its actual residual width. It is not treated as
another full plane. The supported pseudo-Mersenne moduli differ negligibly from
the matching power of two for this completeness estimate.

For Z, let `sigma^2` be the modeled pre-decomposition variance per response
coefficient. The model rounds a normal integer, reduces it into each balanced
digit plane, and computes that digit's squared moment. When `sigma` spans at
least one residue period, it uses the uniform digit moment. The first omitted
Fourier coefficient is then below `3e-9`, far below the 3 percent source
envelope.

The normal approximation is not used in the current fold's L2 cap. It predicts
the Z digits that become part of the next recursive witness. Correlations in E,
T, R, and recursive setup values often make the uniform model conservative.

### Setup offloading

A recursively offloaded setup has two sources:

1. A public setup prefix derived from a pseudorandom seed.
2. The recursive witness produced by the preceding fold.

The planner models the setup prefix with the exact balanced-digit moments of a
uniform field element, including the residual top plane. This is a
computational pseudorandomness model, not an information-theoretic statement
about every seed. The recursive source keeps its propagated `M` and `P`.

Adaptive direct profiles price first-direct padded setup capacity first, with
proof bytes and total setup field elements as later tie-breakers. Recursive
setup profiles first compare the power-of-two capacity covering total setup,
then first-direct capacity, proof bytes, and first-direct output-witness length.
Uniform direct profiles minimize estimated proof bytes first.

### Multi-group and multi-chunk states

For multiple groups, the planner keeps one source moment per opening group
until `WitnessLayout` constructs the next witness. It allocates each group's
source energy in proportion to its public share of live blocks, then adds the
typed component energies. The block counts are exact. The assumption that
energy follows those counts is part of the completeness model.

For chunks, the model uses the exact chunk layout. The Linf peak calculation
uses the ceiling of live blocks per chunk. All component classes share that
one column capacity. L2 candidate construction still requires one chunk
because the current physical norm proof and candidate audit are defined for
that boundary. A multi-chunk profile can select L2 at a later single-chunk
state.

## L2 response cap

Let `M` be the upward-bucketed source energy and `q_c` the exact challenge
squared L2 mass. The planner freezes

```text
S_max = ceil(M * q_c * 1.03 * 40 / 39).
```

The factors have different meanings.

* `1.03` is the source-model envelope. It covers observed unfavorable error
  from typed components, pseudo-Mersenne field moments, approximate challenge
  covariance, finite mixing, and rounded-normal Z digits.
* `40/39` is the response allowance for one nonce attempt.

Assume the first factor bounds the true conditional mean:

```text
E[S | source] <= 1.03 * M * q_c.
```

Since `S` is nonnegative, Markov's inequality gives

```text
Pr[S <= (40/39) * E[S | source]] >= 1 - 39 / 40 = 1 / 40.
```

The fold protocol permits 4096 nonce attempts. Under the random-oracle model,
fresh nonces give fresh challenge draws. Even this loose per-attempt lower
bound gives

```text
Pr[all 4096 attempts fail] <= (39/40)^4096 < 2^-149.
```

This argument does not assume a Gaussian response tail. It is rigorous only
conditional on the 3 percent source envelope. The empirical workflow tests
that condition against the exact schedule used by each measured proof.

The 4096 honest attempts are a completeness mechanism. They do not cause a
fixed 12-bit soundness debit at every fold. Each adversarial nonce trial is a
random-oracle query and is charged through the total query budget in the
Fiat-Shamir CWSS theorem. See
[`book/src/foundations/pcs-and-binding.md`](../book/src/foundations/pcs-and-binding.md#fiat-shamir-queries-and-fold-nonces).

## Linf response sizing

### Universal balanced-source candidate

The universal candidate uses public source bounds rather than typed moments.
For one source ring row, let

```text
s_inf = maximum source digit magnitude
s_1   = source ring L1 bound
c_inf = maximum challenge coefficient magnitude
c_1   = challenge L1 norm
B     = number of claims * number of live blocks.
```

Negacyclic convolution gives the deterministic envelope

```text
beta_inf = B * min(c_inf * s_1, c_1 * s_inf).
```

Random challenge signs also give a Rademacher tail proxy

```text
t_star^2 = 2 * B * q_c * s_inf^2 * log_term,
```

where `log_term` is a conservative integer upper bound for
`ln(2N / (1 - 1/8))` and `N` is the response coefficient count for one logical
fold. The implementation computes `log_term` without floating point by
rounding `ln(2)` upward.

The honest-prover cap is

```text
min(beta_inf, ceil(sqrt(t_star^2))).
```

The planner converts that cap to the smallest balanced digit depth that can
represent it. The verifier then prices the full range of that depth. Thus the
tail calculation is a completeness policy, while the range proof and A matrix
remain the soundness boundary.

### Typed Linf candidate

The typed model uses the component statistics. Let

```text
N = physical response coefficient count
L = number of live blocks
C = number of chunks
D = source ring dimension
B = ceil(L / C)
```

One source column contains at most `B * D` coefficients. For component `i`, let
`P_i(D)` be `P_i,ring` when `D = D_pack`. Otherwise it is `P_i,local`.

For each component, write

```text
M_i = k_i * P_i(D) + r_i,
0 <= r_i < P_i(D).
```

This gives `k_i` full slots with value `P_i(D)` and at most one partial slot
with value `r_i`. The planner sorts these ten slot classes by value and takes
the largest `B * D` slots. Their sum is `Q_peak`. This is the largest column
moment allowed by the component totals and coordinate peaks.

The response calculation is

```text
v_avg  = M * q_c / N
v_peak = Q_peak * q_c / D
v       = 1.03 * max(v_avg, v_peak)
p_N     = (1/40)^(1/N)
t       = ceil(sqrt(v) * Phi^-1((1 + p_N) / 2)).
```

Here `Phi` is the standard normal cumulative distribution function. If the
centered response were multivariate Gaussian and every coordinate variance
were at most `v`, then each symmetric coordinate slab `[-t,t]` would have
probability at least `p_N`. The Gaussian correlation inequality applies to
centered Gaussian measures and symmetric convex sets. Applying it to the `N`
coordinate slabs gives

```text
Pr[||z||_inf <= t] >= p_N^N = 1 / 40.
```

This does not assume independent response coordinates. It removes the loss
from the previous coordinatewise union bound while retaining the full modeled
covariance freedom.

The shared capacity is important. Z, E, T, R, and compression occupy disjoint
witness coordinates. Giving every class all `B * D` slots can overcount the
same fp64 column by about four times. The capacity calculation prevents that
error while retaining every component total and peak bound.

The component capacity calculation is a valid upper bound for the supplied
moments. The Gaussian correlation step is rigorous for the modeled Gaussian
law, but the reduction from the actual sparse signed convolution to that law
is heuristic. The component moments are also distribution estimates rather
than bounds on each realized witness. The universal Linf candidate remains
available.

The selected typed digit depth is capped by the universal depth. The suffix
search also retains the relevant universal Linf alternative. No historical
half-tail or three-quarter digit snap remains.

### Terminal norm and encoding sizing

The terminal response is encoded as centered integers. For a Linf route, its
raw cap is the smaller of the typed model cap and the certified universal cap
when the typed model is available. If the typed model is unavailable, it uses
the certified cap alone.

For an L2 route, the same typed response scale selects the Rice remainder width
and payload budget, but it is not emitted or enforced as a coefficient cap.
The prover may retry when the canonical payload exceeds that budget. Every
accepted coefficient must fit the signed wire type, and the complete decoded
response must satisfy `S <= S_max`. There is no later digit decomposition and
no separate Linf security condition at this boundary.

## Candidate generation and selection

### L2 eligibility

`selective_l2_inner_matrix` returns no candidate unless all required public
conditions hold:

```text
fold_level >= 3
num_claims == 1
num_chunks == 1
fold_basis >= 8
response_l2_sq_cap is present
physical response length is valid
norm proof shape is valid
Euclidean table lookup succeeds
secure L2 rank exists
```

The recursive planner also requires the L2 rank to be strictly smaller than the
corresponding Linf A rank. This avoids paying a norm proof without an immediate
A rank reduction.

For each basis and dimension state, the planner first finds the best modeled
Linf split. It evaluates at most one L2 version of that split. It does not run a
second exhaustive split search for L2. This bound keeps catalog generation
tractable, but it can miss an L2 split that is worse under Linf and better after
the Euclidean rank change. The complete-suffix comparison is exhaustive only
over the candidates that this bounded geometry search emits.

The response basis 8 route is supported. Earlier basis-8 failures came from a
missing class-indexed range-image source when no product-stage prefix existed.
The prover now prepares that source for the fused L2 leaf. There is no current
basis-8 exclusion.

### Complete suffix comparison

A smaller A rank at one fold does not imply a smaller proof. Changing A rank
also changes T width, the next witness, later fold geometry, the terminal
response, and possibly the number of folds.

The suffix dynamic program therefore prices complete schedules. Depending on
the profile's declared objective, its comparison includes:

* Exact serialized proof bytes.
* Direct setup capacity or total setup field elements.
* The current A, B, and D matrices.
* Norm-proof bytes.
* T decomposition and the next recursive witness.
* Every later fold.
* The terminal response and its encoding.
* The root output-witness length.
* A canonical descriptor tie-break.

Uniform direct profiles minimize proof bytes, then total setup and root
output-witness length. Adaptive direct profiles minimize first-direct padded
setup capacity, then proof bytes, total setup, and root output-witness length.
Recursive-setup profiles minimize padded total-setup capacity, then first-direct
capacity, proof bytes, and first-direct output-witness length. Numeric ties go
directly to the canonical descriptor. These are product objectives, not
security rules.

The memo key includes `M`, `P`, ring dimensions, setup-prefix state, witness
length, basis, level, and payload phase. Pareto pruning keeps candidates that a
parent can still distinguish.

### Fallback behavior

The planner keeps Linf when any L2 prerequisite is absent. Common reasons are:

* The fold is too early.
* The fold has more than one claim or chunk.
* The response basis is below 8.
* The model cannot produce a finite cap.
* The proof shape cannot certify integer equality.
* The Euclidean table has no cell for the geometry and collision bound.
* The L2 rank does not improve.
* The complete selected objective prefers the Linf suffix.

The generated schedule freezes the result. Runtime proving and verification do
not rerun the response model or lattice estimator.

## Committed profiles and runtime catalog resolution

The offline planner and the runtime resolver have separate jobs.

The planner searches candidate folds and writes compact generated rows. Runtime
code accepts only those rows. `CommitmentConfig::resolve_catalog_row_for_key`
and `resolve_catalog_row_for_profiles` perform a strict catalog lookup and
expand the selected row. They do not call the planner. The verifier uses
`resolve_schedule_selection` with the public `OpeningScheduleSelection` digest.
It performs a bounded digest lookup and does not reconstruct a planner key.

`GroupCommitPhaseParams::try_from_params` is the checked construction boundary
for frozen commitment metadata. It validates the root geometry, digit bases,
digit depths, slice count, A and B widths, modulus profiles, matrix identities,
and audited SIS bounds. Prover, verifier, planner emission, schedule expansion,
and schedule audit use this checked constructor when they assemble a profile.

`GeneratedFrozenGroup` stores the frozen profile and the consuming opening
data. In particular, it stores:

* the checked `GroupCommitPhaseParams` descriptor;
* the fold digit depth used when the grouped root opens that commitment.

The generated row does not store a second copy of the commitment geometry.
Catalog dimension collection and identity hashing read the descriptor directly.
`GroupOpenPhaseParams::admit` derives the shared opening parameters for
the current grouped root, but it cannot replace the descriptor's frozen A or B
matrix, decomposition, or slice count. Precommitted A matrices remain Linf.

This boundary gives runtime one source of truth. The catalog row selects the
fold protocol, the committed profile fixes prior commitment geometry, and the
public row digest selects the exact expanded schedule for verification.

## Euclidean SIS table

The L2 route uses a separate 128-bit quantum ADPS16 table. It does not reuse the
coefficient Linf table and does not use the retired BDGL16 Euclidean profile.

The table domain is:

```text
modulus profiles: q32, q64, q128
ring dimensions:  D32, D64, D128, D256, D512
collision keys:   powers of two from 2^1 through 2^84
```

For each profile, dimension, rank, and collision bucket, generation records the
largest secure A width and rejected successor evidence at the 128-bit
boundary. Generated Rust rows carry a digest. Schedule validation requires the
current digest.

The verifier never runs the estimator. It checks the frozen table key and
width against generated rows.

Regenerate the table with

```sh
cargo run -p akita-sis-estimator --release \
  --example euclidean_width_table -- --format rust-split
```

The full audit CSV is reproducible and intentionally not committed. Golden
tests compare estimator cells with the checked-in table and cover supported
dimensions and boundary behavior.

## Empirical validation

### Questions the data must answer

Measurements validate completeness, not soundness. They answer four separate
questions.

1. Does the typed model predict the exact source energy closely enough?
2. Does `M * q_c` predict the conditional response mean?
3. Does the frozen cap leave enough room for nonce search?
4. Do complete production proofs pass both verifier modes for every adapter
   family?

A total proof size alone cannot answer these questions. The validation records
source energy, modeled response mean, observed response norm, cap, nonce, route,
and exact serialized bytes at each fold.

### Current calibration contract

Empirical values are not copied into planner unit tests. Such literals lose the
profile name, schedule identity, seed, and capture provenance, then become
stale when catalogs change. Instead, a diagnostic production proof emits the
exact source and response measurements beside its planned fold events. The CI
report parser joins both sides by fold level within the same run.

For every successful L2 fold, the parser requires and retains:

* the frozen response cap from that run's selected schedule;
* the measured squared response norm;
* cap utilization;
* the accepted nonce and number of attempts;
* the profile case and run identity carried by the surrounding summary.

A successful run with a missing measurement or a response above its cap makes
report generation fail. Repeated samples are aggregated without discarding the
individual response values or nonces. This validates the current generated
schedule rather than a detached copy of an older model.

This production report checks the frozen cap, realized response, and retry
behavior. It does not by itself validate the three percent source-model
envelope. A binary built with `response-model-diagnostics` also emits the exact
incoming source energy and its challenge-scaled conditional mean. Those values
are retained for separate calibration analysis because multi-group proofs can
carry several source observations into one scheduled fold. Claims about model
error must cite such a diagnostic data set, not the ordinary cap report.

### Historical calibration snapshot

The following measurements were collected before the source-free runtime
cleanup and the kernel-faithful one-hot root correction. They explain how the
3 percent envelope was chosen. They are not a current schedule acceptance test.

That calibration run contained 13 production profiles. It covered:

* Dense and one-hot fp32.
* Dense and one-hot fp64.
* Dense fp128.
* One-hot fp128 direct and recursive setup.
* Direct and recursive multi-group.
* W2, W4, and W8 multi-chunk variants.
* Recursive W8 adapters.

Every proof in that snapshot passed multi-threaded and single-threaded
verification.

This run measures the total-energy formula. The later component-capacity
experiment measures the Linf column formula. A Linf model change can select a
different suffix, so the two data sets answer different questions.

Across 51 selected L2 responses:

| Measurement | Observed range |
|---|---:|
| Frozen cap above observed response | 0.0847% to 10.4720%; mean 6.80% |
| Modeled source energy relative to exact source energy | -0.1750% to +2.0689% |
| Observed response relative to `exact source energy * q_c` | -3.8491% to +6.3244% |
| Attempts | 1 to 6; mean 1.57 |

For the second and third rows, a positive value means the model or observation
is larger than the reference. The third row uses the conditional mean implied
by scalar challenge covariance. The separate orbit measurement tests the small
covariance approximation in that reference. The worst aggregate source
underestimate was 0.1750 percent, below the 3 percent envelope.

Component-level checks covered 88 recursive witnesses. Model error relative to
the measured component was:

| Component | Error range |
|---|---:|
| Z | -1.00% to +16.10% |
| E | -2.24% to +33.47% |
| T | -1.33% to +33.50% |
| R | -1.43% to +4.37% |
| Compression | -1.24% to +2.12% |

Negative values are underestimates. The worst unfavorable component error was
2.24 percent. Large positive Z, E, and T values came from recursive multi-group
W8 setup values that retained correlation instead of behaving as fully mixed
uniform values. Those overestimates cost planner efficiency but do not threaten
honest acceptance.

### Historical schedule remeasurement

The component-capacity correction was measured on every recursive and terminal
state of fp32 and fp64 dense and one-hot profiles. The diagnostic computed the
exact largest cyclic source-column energy before each response. A second final
run then exercised every supported adapter family with the corrected Linf
quantile and L2 grinding allowance.

| Profile | Actual proof bytes | Terminal cap | Observed terminal maximum | Cap above maximum |
|---|---:|---:|---:|---:|
| fp32 dense `nv=26` | 69,416 | 1,146 | 1,137 | 0.8% |
| fp32 one-hot `nv=30` | 69,350 | 1,146 | 1,027 | 11.6% |
| fp64 dense `nv=26` | 72,789 | 1,647 | 1,512 | 8.9% |
| fp64 one-hot `nv=30` | 71,511 | 1,647 | 1,613 | 2.1% |
| fp128 dense `nv=28` | 73,247 | 851 | 812 | 4.8% |
| fp128 one-hot direct `nv=36` | 73,938 | 851 | 840 | 1.3% |
| fp128 one-hot recursive `nv=36` | 78,835 | 851 | 791 | 7.6% |
| fp128 multi-group direct `nv=32`, 4 groups | 72,898 | 851 | 799 | 6.5% |
| fp128 multi-group recursive `nv=32`, 4 groups | 76,199 | 851 | 823 | 3.4% |
| fp128 multi-group recursive W8R2 | 79,259 | 851 | 841 | 1.2% |
| fp128 multi-chunk W2R2 | 73,729 | 851 | 841 | 1.2% |
| fp128 multi-chunk W4R2 | 73,993 | 851 | 836 | 1.8% |
| fp128 multi-chunk W8R2 | 75,583 | 851 | 835 | 1.9% |

Every proof in that snapshot passed multi-threaded and single-threaded
verification. The profile harness used fixed seeds. These maxima are samples,
not estimates of an acceptance quantile and not frozen expectations for the
current catalog.

The old scalar calculation gave every typed component the complete block
column. In the eight-block fp64 terminal experiment, that counted four
disjoint component classes as four separate eight-block columns. The source
proxy was 66,572, while the exact source-column energy was 16,061. The shared
capacity model removes this structural overcount.

The previous coordinatewise union bound made the fp64 maximum too large. It
forced one late response from two digits to three and made the selective L2 A
matrix rank eight instead of seven. The Gaussian correlation inequality gives
a joint slab lower bound without assuming independent coordinates. Combined
with the `40/39` L2 allowance, it restores the compact two-digit and rank-seven
fp64 suffix.

Twenty independent seeds for each fp64 profile produced 160 accepted L2
responses. Dense proofs averaged 1.525 attempts per L2 response and one-hot
proofs averaged 1.775. The largest single attempt count was 12. The inferred
source estimate was 0.03 to 2.49 percent above exact source energy across this
sample. This supports the 3 percent source envelope and is far better than the
theoretical `1/40` per-attempt floor.

These measurements do not prove that all future witness distributions fit the
model. A new field profile, setup distribution, ring dimension, challenge
family, or layout component requires new measurements before it can use the 3
percent envelope.

### Current model limits

The current model remains intentionally conservative or heuristic in several
places.

* A dense root uses a deterministic maximum, so it can be much larger than a
  typical application witness.
* Z digit moments assume an approximately normal folded coefficient before
  balanced reduction.
* E, T, and R use uniform field moments even when protocol correlations remain.
* The typed Linf formula approximates the sparse signed response by a centered
  Gaussian with the modeled coordinate variances. The Gaussian correlation
  step is exact for that model, but the approximation itself is not proved.
* The setup prefix is modeled as pseudorandom rather than checked against a
  deterministic seed-admission envelope.

These limits explain why the universal Linf candidate remains necessary and
why the L2 cap keeps a separate empirical source envelope.

## Reporting contract

All comparisons in the PR report use the current merge base, not an
intermediate PR commit.

The compact report should show only the information needed to identify the
selected route and its cost:

* Fold level and whether it consumes the current or terminal witness.
* A, B, and D ring dimensions and module ranks.
* Response and opening decomposition bases and digit counts.
* Challenge ring dimension, signed sparse shell, and operator norm threshold
  when present.
* Folded-response bound type, written as maximum coefficient magnitude (Linf)
  or sum of squared coefficients (L2).
* Exact proof bytes and merge-base change.
* Observed squared norm and frozen cap for selected L2 responses.
* Terminal Z, E, and T byte totals.

The detailed report retains emitted terminal field-coefficient and ring-element
counts, segment encoding, Golomb budget, Golomb parameters, packed-digit
comparison, setup timing details, and nonce diagnostics. These details belong
in the expandable report rather than the compact main comment.

The report does not repeat values that are directly derivable from a displayed
ratio. It also does not show duplicate verifier-core wrapper timings.

## Invariants and acceptance criteria

### Security invariants

1. Every committed level has exactly one typed A security route.
2. A Linf route serializes no L2 proof values.
3. An L2 route binds its cap, table key, challenge policy, response length, and
   proof shape into schedule identity.
4. The verifier proves or computes the norm of the same physical response that
   the A role SIS reduction uses.
5. The physical A collision formula applies no Hachi embedding factor.
6. The Euclidean estimator receives the complete scalar collision norm once.
7. The existing digit range proof remains on recursive L2 routes.
8. Schedule audit derives its security bound from the verifier-enforced cap,
   never from the planner model.
9. Headerless deserialization validates shape and allocation bounds before
   use.
10. Verifier-reachable malformed input returns a typed error and never panics.
11. A frozen committed profile is checked before it enters catalog, prover, or
    verifier state.
12. Runtime schedule resolution performs catalog lookup and expansion only. It
    never invokes planner search.

### Completed acceptance criteria

* [x] Linf and L2 schedules use separate typed routes and SIS tables.
* [x] Recursive L2 verifies the complete physical integer square sum.
* [x] Terminal L2 recomputes the clear complete physical integer square sum.
* [x] D64 and D128 operator norm rejection has checked accepted-support
      certificates above 128 bits.
* [x] The shared physical plan covers each live response coefficient once and
      padding with zero.
* [x] Direct mode proves deterministic no-wrap before using a field square
      sum.
* [x] LimbGram mode checks every centered lift and reconstruction boundary.
* [x] The Hachi physical double count is removed.
* [x] The Euclidean scalar mapping does not multiply the complete norm by A
      width.
* [x] The planner propagates typed total and peak moments through all generated
      profile families.
* [x] The planner prices complete suffixes and keeps Linf fallback behavior.
* [x] Generated schedules replay against the current planner and table digests.
* [x] Generated precommitted rows store one checked commitment descriptor and
      do not duplicate its geometry.
* [x] Runtime catalog APIs state their resolver role and reject missing rows.
* [x] Proof-size accounting matches actual schedule-driven serialization.
* [x] Transcript, cap, nonce, shape, and Stage 2 mutation tests reject.
* [x] Production proofs cover direct, recursive, multi-group, multi-chunk,
      dense, one-hot, fp32, fp64, and fp128 paths.
* [x] Both verifier modes pass the production profile matrix.

## Architecture and reviewer map

| Concern | Canonical owner |
|---|---|
| Sparse challenge shells and operator norm rejection | `akita-challenges` |
| Collision formulas and SIS table keys | `akita-types::sis` |
| Physical response and witness geometry | `akita-types::layout` |
| L2 proof values and schedule-driven shapes | `akita-types::proof` |
| Physical norm construction and Stage 1 and Stage 2 proving | `akita-prover` |
| Challenge replay, integer reconstruction, and cap checks | `akita-verifier` |
| Typed response moments and complete-suffix selection | `akita-planner` |
| Checked committed profiles and lookup keys | `akita-types::schedule` |
| Frozen routes, catalog resolution, and schedule audit | `akita-schedules` |
| Quantum ADPS16 Euclidean table generation | `akita-sis-estimator` |
| Exact bytes, geometry, norms, and timing presentation | profile tooling |

The main data flow is

```text
source WitnessLayout
        |
        +--> universal or typed honest response sizing
        |
        +--> Linf candidate --> digit range --> Linf SIS table
        |
        +--> L2 candidate ----> physical norm --> Euclidean SIS table
                                      |
                                      +--> Stage 2 binds physical Z digits

selected A rank --> T width --> next WitnessLayout --> next (M, P) --> suffix

offline planner --> generated catalog row --> strict runtime resolver
                                            |
checked committed profiles -----------------+
                                            |
public row digest --> verifier lookup -------+
```

## Alternatives and deferred work

### Global L2 replacement

A global replacement would pay norm-proof bytes where the tighter table gives
no complete-suffix benefit. It would also discard the established Linf route.
Separate candidates avoid both problems.

### Prover-reported norm

A transcript-bound diagnostic norm does not prove that the committed response
has that norm. Production sizing requires the recursive proof or the direct
terminal computation.

### One global cap

Response geometry and source composition change at each fold. One cap would
either reject honest proofs or lose rank reductions. Caps are candidate values
frozen into generated schedules.

### Gaussian security assumption

The rounded-normal model is useful for predicting later Z digits. It is not
needed for binding. The verifier's public cap makes the collision bound hold
for every accepted response, regardless of its distribution.

### Four-square inequality proof

The verifier needs an exact integer norm followed by a public integer
comparison. Direct no-wrap sums and LimbGram reconstruction provide that
statement without slack variables or carry witnesses.

### Deterministic setup-seed admission

A future setup generator could compute exact digit-plane energy or cyclic peak
statistics while expanding the public seed, then resample seeds outside a
frozen planner envelope. The verifier could reproduce the check. This would
turn part of setup-prefix completeness sizing into a deterministic public
condition. It is deferred because current profiles already meet the measured
slack target and the mechanism would change setup generation and transcript
identity.

### Larger Euclidean dimensions

The production challenge ladder includes dimensions above D512, but the
current Euclidean table does not. Adding D1024 or D2048 L2 requires new table
rows, boundary evidence, digest regeneration, planner replay, and empirical
profiles. Linf remains the fallback until that work is complete.

## Testing and validation commands

Unit tests cover collision factors, coordinate mapping, digit bounds, model
moments, operator norm certificates, direct no-wrap checks, LimbGram centered
lifts, table boundaries, shape validation, serialization, and proof-size
arithmetic.

Protocol tests generate valid Linf and L2 proofs, then mutate each new public
value and transcript relation. Small-field tests exercise multiple LimbGram
blocks. Large-field tests exercise recursive and terminal D128 routes.

Planner tests cover local rank improvement versus complete-suffix choice,
Linf fallback, recursive setup, multi-group and multi-chunk propagation, model
memoization, and generated-catalog replay.

Final validation follows `AGENTS.md` and the current CI workflow. Documentation
changes also run

```sh
./scripts/check-doc-guardrails.sh
```

## Documentation lifecycle

Durable user-facing explanations live in:

* `book/src/how/security.md` for the security routes and physical collision
  bounds.
* `book/src/how/proving/sumcheck-stages.md` for the optional Stage 1 norm term
  and Stage 2 binding.
* `book/src/how/configuration.md` for planner models, eligibility, and fallback.
* `book/src/usage/profiling.md` for the compact and detailed reports.

PR 369 is merged and the acceptance criteria are complete, so this
specification is implemented. Retain it in the root as the current load-bearing
security sizing source until the durable text and deferred alternatives are
fully folded into the Book.

## References

* Akita paper, `sections/akita/9_core_security.tex`, for the weak-binding
  extraction and radius-to-collision argument.
* Akita paper, `sections/akita/3_preliminaries.tex`, for physical ring norms and
  the logical-to-physical boundary.
* Hachi, Lemma 7, for denominator clearing in weak binding.
* `crates/akita-types/src/sis/norm_bound.rs`.
* `crates/akita-types/src/sis/physical_l2.rs`.
* `crates/akita-types/src/schedule/profiles.rs`.
* `crates/akita-challenges/src/config.rs`.
* `crates/akita-challenges/src/sampler/mod.rs`.
* `crates/akita-planner/src/response_model.rs`.
* `crates/akita-planner/src/schedule_params/candidate/recursive.rs`.
* `crates/akita-planner/src/schedule_params/suffix_dp/terminal.rs`.
* `crates/akita-schedules/src/resolve.rs`.
* `crates/akita-sis-estimator/src/euclidean.rs`.
* `crates/akita-sis-estimator/src/euclidean_width_table.rs`.
* `scripts/operator_norm/`.
* `specs/fold-linf-rejection.md` for the base universal Linf policy.
* `specs/sis-quantum128-scalar-n-table.md` for the production quantum security
  policy.
