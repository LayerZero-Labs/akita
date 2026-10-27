# Binary source checkpoint and adjoint

| Field | Value |
| --- | --- |
| Status | active |
| Created | 2026-10-27 |
| Scope | Mathematical checkpoint for the proposed binary root; not production admission |

## Blockers and corrections

There is no unresolved algebraic blocker in the checkpoint proved below. Two
points need an explicit correction before implementation admission:

1. **Checkpoint location.** A projection argument about a fixed residual cannot
   start at a checkpoint *after* the projection challenge. The successor-source
   checkpoint must be immediately before the seven-row seed J (order 6), replacing
   the ordinary checkpoint before ring reduction. Its candidate is the source of
   the outgoing commitment, not an adaptively chosen table. The construction in
   Theorem 8 makes this placement explicit and preserves the checkpoint count.
   An implementation that retains only the later checkpoint has not instantiated
   this proof. For example, after seeing a seven-row matrix with more than seven
   columns, one can choose a nonzero vector in its kernel over GF(q); the fixed
   residual bound q^-7 cannot be applied to that adaptive vector. This example
   diagnoses the missing fixation argument, not an attack on binding commitments.
2. **Adjoint convolution length.** Extending a D-entry row through reduction
   gives 2D-1 entries. Its convolution with D reversed weights has length
   3D-2 = **1942**, not 1295. Theorem 7 gives a correct implementation using
   **two** products of length 2D-1 = **1295**. The single padded Karatsuba estimate
   3^10 = 59,049 is consequently not a bound for this particular implementation;
   two such products give 118,098 recursive scalar products before overhead.
   A faster specialized middle product would require its own analysis. For a
   concrete nonzero last coefficient, take rho[322]=1 and v[0]=1, with all
   other entries zero: g[1294]=-1, so g*reverse(v) has coefficient -1 at
   degree 1941.

These are resolved here at the mathematical interface. Production admission must
check the checkpoint placement and the implemented adjoint against that interface.
This record does not certify the new modulus, the ordinary suffix widths, or a
complete implementation. Its tests are arithmetic evidence, not a security
certificate.

## 1. Scope, notation, and inherited hypotheses

This document specifies the binary-root replacement in one ordinary grouped
opening on the caller's transcript. The proposed companion protocol and index-map
specification is `specs/binary-source-groups.md`; it does not yet exist.
New behavior is selected
explicitly under the existing default-off `labinius*` features. Merely enabling
those features must preserve unselected setup, commitment, proof bytes, and
transcript state. No runtime code or application proof helper is introduced here.

Set

```text
P = 2^128 - 275,       F = GF(P),       E = F in the standard profile,
q = 14,253,409,        d = 162,         D = 648,        C = 256,
Phi(Z) = Z^162 + Z^81 + 1,             Psi(Y) = Y^648 - Y^324 + 1,
S = Z[Z]/Phi,         R = Z[Y]/Psi,    B = F2[Z]/Phi,
iota(Z) = -Y^4,       w = 46,          Gamma = 2w = 92.
```

Here Z in `Z[Z]` outside the brackets denotes the integers. Both polynomials
are cyclotomic: Phi = Phi_243 and Psi = Phi_1944. Thus their characteristic-zero
quotients are domains. R is free of rank four over S with basis 1,Y,Y²,Y³,
since Phi(-Y^4)=Psi(Y). Write each ring element uniquely as
`sum_{a=0}^3 Y^a iota(v_a)`. The coefficient at t=4s+a equals
`(-1)^s v_a[s]`. All coefficient norms and integer lifts use these fixed bases.
An infinity norm of an array is the largest absolute coefficient.

N=2^n is the number of physical carriers, M=N/C, and m=M/4. The displayed
profiles use n=22,23,24; arbitrary dimensions require separately checked
padding and bounds. A has rank r and m columns over R/qR, with public centered
integer representatives bounded by B_A=(q-1)/2=7,126,704. For each source column
b the image is T_b. Its eight balanced digits lie in {-4,...,3}; hence its
integer coefficients lie in [-9,586,980,7,190,235]. In particular the malicious
bound is BT=9,586,980, not B_A. The outer commitment binds all image digits using
the ordinary P-field commitment.

A four-digit response z has interval [-2340,1755], magnitude BZ=2340 and
diameter Delta=4095. Let S46 contain exactly one signed polynomial for every
support of size at most 46 in {0,...,161}; signs are a deterministic function
of that support. Uniform support selection is part of admission.

The ordinary suffix is used under these explicit hypotheses:

- Its fixed-source range and relation checks authenticate the decoded outgoing
  table, including every digit, view, and zero-defined virtual slot. Its range
  image proves the balanced alphabet; a claimed honest input range is insufficient.
- Different source projections under the same ordinary commitment yield a
  nonzero short kernel vector for its scheduled P-ring instance. This includes
  image, D, outgoing witness, compression, and setup-prefix commitments, at their
  actual widths and source bounds. Repeated occurrences share a canonical view.
- Each ordinary recursive fold has a backward extraction combiner and a
  successor-source checkpoint. Its terminal authenticates the final small source
  and all pending setup claims. No unconsumed claim is silently discarded.
- Fixed-source checks have their actual per-round degrees, fresh challenges,
  and schedule-specific errors. A schedule supplies the induced SIS instances,
  fold arities, checkpoint count, and extraction-time budget. The underlying
  structured module-SIS assumptions apply at that budget, not just prover time.

For clarity, the generic classical-ROM composition interface used below is an
explicit premise. A checkpoint structure has accepting terminal certificates,
backward combiners, and candidate comparison maps returning binding collisions.
For each fixed candidate, false obligations propagate outside bounded bad sets
and are impossible at a correct terminal row. If its tree size is
`Tree = 2^(number of checkpoints) × product(extraction arities)`, its exact
conditional challenge samplers permit an extractor with expected at most
`(Q+2) Tree` prover runs and error at most
`Pr[not Good] + epsilon_interactive + Q epsilon_max`, apart from setup and
binding reductions. Here epsilon_max bounds one addressed bad set and Q counts
all prover and verification oracle queries, including unsuccessful nonce probes.
Work filtering is admissible only with an independent, transcript-bound,
fixed filter and exact conditional sampling. We prove the new structure below;
we do not derive this general random-oracle extraction result from scratch.

## 2. Unit differences and bounded multiplication

**Theorem 1 (units and entropy).** Distinct members c,c' of S46 have difference
s=c-c' that is nonzero over S, a unit in B and a unit in F[Z]/Phi (and after
extension to E). Multiplication by iota(c) on R has infinity operator norm at
most Gamma=92; multiplication by iota(s) has norm at most 184. Moreover

```text
|S46| = sum_{j=0}^46 binom(162,j),
log2 |S46| = 136.300278309...,
256/|S46| = 2^(-128.300278309...).
```

*Proof.* The order of 2 modulo 243 is 162, so every irreducible factor of
Phi_243 over F2 has degree 162; hence B is a field. Different supports give
different coefficient vectors modulo two of degree less than 162. Because
signs are determined by the support, distinct challenges have distinct supports,
so their difference is nonzero in B. Arbitrary independent signs would fail:
Z^j and -Z^j are distinct but their difference is zero modulo two. The empty
support is allowed; it does not change this argument. A nonzero polynomial of
degree below 162 is nonzero in S and its fraction field.

For the prime-unit condition, let a_i be the coefficients of s. Then
`sum a_i² <= 2(||c||_2²+||c'||_2²) <= 4w`. The average of |s(zeta)|² over the
162 primitive complex 243rd roots is exactly

```text
sum_i a_i² - sum_{i=0}^80 a_i a_{i+81} <= (3/2) sum_i a_i² <= 6w = 276.
```

To verify the equality, expand the square. The normalized sum of zeta^k is
1 for k=0, -1/2 for k=±81, and 0 for other nonzero differences of exponents
between 0 and 161. The inequality follows from 2|xy|<=x²+y².
The arithmetic-geometric mean inequality on the squared absolute embeddings
therefore gives `0 < |Norm(s)| <= (6w)^81`. Phi is irreducible in characteristic
zero, so the norm is a nonzero integer.

P is prime and coprime to 243. Its order modulo 243 is 6: the successive
powers for exponents 1,2,3,6 are 161,163,242,1. Thus Phi modulo P is squarefree
with irreducible factors of degree 6. If s were not a unit, its integer norm
would be divisible by P^6. Indeed the nullity of multiplication by s modulo P
is at least 6; the Smith normal form of the integer multiplication matrix then
has at least six invariant factors divisible by P. But

```text
P^6 > 2^(6×127) = 2^762 >= 2^(81×bit_length(276)) = 2^729 > 276^81.
```

This is impossible. Scalar extension preserves invertibility, as does the
rank-four free module embedding into R. This argument uses the selected P;
it cannot be reused for a different 128-bit prime merely on the basis of size.

Reduction by either trinomial maps each fixed monomial multiplication to rows
with absolute row sum at most two. One direct verification uses, for h=D/2,

```text
Y^k                         (0 <= k < D),
Y^(k-h) - Y^(k-D)            (D <= k < D+h),
-Y^(k-D-h)                  (D+h <= k <= 2D-2).
```

Changing the middle sign to obtain Phi has the same absolute bound.
Each of at most w monomials of iota(c) therefore contributes operator norm at
most two. Triangle inequalities give Gamma and 2Gamma. Finally the support
bijection gives the binomial count; signs supply no entropy. The script checks
both orders, the integer inequalities, and the count. ∎

## 3. A comparison and exact divisibility

**Theorem 2 (tight coefficient kernel and projection).** For
`K[t,u,v]=coeff_t(rem_Psi(Y^(u+v)))`,

```text
max_t sum_{u,v<D} |K[t,u,v]| = 3D/2 = 972.
RA_i = rem_Psi(sum_j A_ij z_j - sum_b iota(c_b) T_(b,i)),
||RA||_infinity <= LR = 972 m B_A BZ + C Gamma BT.
```

For a residual fixed before seven independent uniform rows
`J_l in {0,...,q-1}^(rD)`, set UJ=(q-1)rD LR. If 2UJ<P, authenticating
`y_l = <J_l,RA> mod P`, with signed integers |y_l|<=UJ and y_l=0 mod q,
proves RA=0 mod q except with probability exactly q^-7 for any fixed nonzero
residual modulo q. Seven rows give `7 log2(q)=166.353625728...` bits.

*Proof.* The unreduced convolution coefficient at exponent k contains
`min(k+1,2D-1-k)` input pairs. Apply the three reduction cases above.
For t<h their total absolute multiplicity is `3h-1-t`; for h<=t<D it is
3h. Thus the maximum is 3h=972. The general product bound follows by triangle
inequality and is tight as an absolute kernel bound. Sum it over m columns,
and apply Theorem 1 to each of the C image terms to obtain LR.

The true integer dot product and the claimed y each have magnitude at most UJ.
Their difference is divisible by P and has magnitude at most 2UJ<P; it is
therefore zero in the integers. This proves that the divisibility test concerns
the intended residual, with no probabilistic no-wrap error. If one coordinate
of RA is nonzero modulo the prime q, condition on all other entries of a row.
Exactly one of the q possible entries at that coordinate cancels the dot
product. The seven rows are independent, giving q^-7; no factor rD is needed.
If the residual is zero modulo q all seven divisibility tests hold. ∎

For rank 3 the exact values are:

| n | m | LR | UJ |
| --- | ---: | ---: | ---: |
| 22 | 4096 | 66,394,525,036,769,280 | 1,839,701,006,388,918,996,628,930,560 |
| 23 | 8192 | 132,788,824,280,985,600 | 3,679,395,756,376,625,911,509,811,200 |
| 24 | 16384 | 265,577,422,769,418,240 | 7,358,785,256,352,039,741,271,572,480 |

For rank 4 multiply UJ by 4/3. Even its largest 2UJ is below P; twelve signed
bytes per y suffice (84 bytes total). All values use the full malicious image
window. The earlier coarse multiplier 3D=1944 is replaced by 972; the rounded
166.6-bit projection claim is 166.353625728... bits.

The y values must be fixed before independent output-batching scalars lambda.
For any incorrect vector equality, the inner product with uniform independent
lambda is zero with probability 1/|E|: fix a nonzero coordinate and condition
on the others. This charge is separate from q^-7 and subsequent ring and
sumcheck errors. One power-series batching scalar would instead have a degree
bound. A new J is required for each outgoing commitment/nonce trial.

**Committed-quotient alternative.** A schedule may instead commit a digit table
KA before relation batching, prove `RA=q KA` modulo P, and enforce magnitude
BKA with `LR+q BKA<P`. The same lift proves exact equality and hence divisibility
with zero divisibility error. For canonical honest centered images a sufficient
carry bound is
`ceil((B_A/q)(972m BZ+C Gamma))`. Both positive and negative digit reaches must
cover it. This follows from the product bound and BT=B_A for honest canonical
images, divided by q. The malicious lift still uses LR with BT=9,586,980 and
BKA equal to the full accepted digit magnitude. For n=22,23,24 the honest
carry bounds are respectively 4,658,146,490; 9,316,281,203; 18,632,550,629.
Twelve digits have positive reach 29,451,204,315 and magnitude 39,268,272,420.
The corresponding malicious lift bounds are 626,101,272,562,449,060;
692,495,571,806,665,380; 825,284,170,295,098,020, all less than P.
This is an alternative schedule,
not an extra unchecked quotient in the projection schedule.

## 4. Weak openings and extraction

**Theorem 3 (binary root combiner and source comparison).** Suppose a central
accepting child and one accepting neighbor for each column b share the source
commitment, its authenticated T table, the D-bound U and EP tables, and all
fold coordinates except b, with c_b != c_b^(b). Suppose their outgoing
certificates satisfy the
exact branch relations proved here. Write

```text
a_b = z - z^(b),       s_b = c_b - c_b^(b),
A a_b = iota(s_b) T_b mod q,           ||a_b||_infinity <= Delta=4095,
f*_b = a_b / iota(s_b) in (R tensor Q)^m.
```

These are weak openings of all columns. For two weak openings of a column
under one image, either their rational sources coincide or

```text
h = iota(s'_b) a_b - iota(s_b) a'_b,
A h = 0 mod q,       0 < ||h||_infinity <= 4 Gamma Delta = 1,506,960 < q.
```

Thus disagreement produces a nonzero q-module-SIS vector of that radius.
The extraction debit for I independently addressed coordinates is at most
I/|S46| when conservatively treating the vector as one address.

*Proof.* Subtract the two equations RA=0 modulo q. All other column terms
cancel because T and their challenges agree. Both response coefficients lie in
the same length-4096 interval, giving the diameter bound. Theorem 1 makes s_b
nonzero in the number field, so division there is defined. No division modulo
q occurs or is needed. There is one central child and I neighbors, not 2^I
children, and subtraction never inverts a public equality weight.

For comparison, cross-multiply the two fractions. The images cancel modulo q
because multiplication is commutative. The two product norms are each at most
2Gamma Delta. If the sources differ the resulting integer vector is nonzero;
since its norm is below q it is also nonzero modulo q. If its integer vector
is zero, division in the number field proves equality of the fractions.

If the two image digit tables differ under the same outer commitment, first
use ordinary P binding to output its collision; otherwise they coincide and
the preceding argument applies. The same comparison applies across different
opening points or repeated occurrences of one commitment, provided their
manifest, setup view and source interpretation coincide.

With all other coordinates fixed, two different challenges suffice for the
combiner. An unextractable set of answers contains at most one challenge, of
uniform mass 1/|S46|. The indexed extraction error is the sum over coordinates; the
standard flat indexed combiner uses 1+I children. This is precisely the indexed
stage hypothesis of the composition interface in Section 1. ∎

The denominator norm is odd and prime to P by Theorem 1. The inverse of s in
its number field is its integral adjugate divided by its nonzero integer norm.
Consequently both reductions of f* are well-defined, independently of the
chosen weak opening:

```text
Dec_2(f*) = (s mod 2)^(-1) (a mod 2) on its four B components,
Dec_P(f*) = iota(s mod P)^(-1) (a mod P) on R/PR.
```

Neither R/2R being a field nor s being a unit modulo q is asserted.

## 5. Both parity lifts and terminal rows

Let L_B map the M unpacked scalar response components to their weighted sum
with public binary coefficients Br of the B-valued row equality weights.
Undo signed packing before taking these components. Put V0_r equal to the
component of the response's lowest balanced digit; modulo two it equals the
component of the full response. Let U_b have one digit per scalar coefficient,
so -4<=U_b[t]<=3. Its reduction modulo two exists without booleanity.

**Theorem 4 (response and value lifts).** Authenticate over P the coefficient
identities

```text
rem_Phi(sum_r Br V0_r - sum_b U_b c_b) = 2K,
rem_Phi(sum_b E_b U_b) - v = 2K_val,
```

where E_b is the bit coefficient vector of the column equality weight and v
is the canonical 162-bit terminal value. K uses nine balanced digits for
n=22,23 (ten for the conservative n=24 profile), and K_val uses six.
These identities then hold over the integers, not just modulo P, with the
following sufficient bounds:

| n | Hpar=162M×4+Cw×4 | floor(3Hpar/2) | 3Hpar+2BK, nine digits |
| --- | ---: | ---: | ---: |
| 22 | 10,663,936 | 15,995,904 | 185,383,496 |
| 23 | 21,280,768 | 31,921,152 | 217,233,992 |
| 24 | 42,514,432 | 63,771,648 | 280,934,984 |

The n=24 carry bound exceeds the nine-digit positive reach 57,521,883, so that
row is a failed completeness admission for nine digits, despite valid no-wrap.
Ten digits suffice, with no-wrap bound 1,354,676,808<P. For K_val the malicious no-wrap bound is exactly
`3×162×256×4+1+2×149,796 = 797,257 < P`; an honest carry has magnitude at most
62,208 and fits six digits in [-149,796,112,347].

*Proof.* Before reduction, a coefficient of sum Br V0_r has magnitude at most
162M×4, and that of sum U_b c_b at most Cw×4. Each reduced coefficient collects
at most three unreduced coefficients, giving 3Hpar. The nine-digit magnitude
is BK=4(8^9-1)/7=76,695,844. Subtracting 2K leaves magnitude at most 3Hpar+2BK,
strictly below P, so a zero field residue is an integer zero. The ten-digit
case uses the same formula with its larger BK and also remains below P.
For an even residual the carry magnitude is at most floor(3Hpar/2), proving
the stated sufficient digit reaches.

Similarly the malicious value residual has magnitude at most 3dC×4+1,
including the subtracted bit of v. With the six-digit magnitude 149,796 the
claimed lift is strictly below P. For honest bit U replace four by one: the
residual has magnitude at most 124,417. It is even when the intended B value
identity holds, so the integer carry magnitude is at most floor(124417/2)=62,208.
The verifier does not require every bounded malicious U to admit a carry;
it accepts only those with a range-valid carry satisfying the equation.

Finally reduce both exact integer identities modulo two. All higher response
digits vanish because their weights are multiples of eight. The first gives
`L_B(z mod 2)=sum_b c_b (U_b mod 2)`. In a central/neighbor pair U is fixed
by D, so subtraction gives `L_B(a_b mod 2)=s_b (U_b mod 2)`. Divide by the
B-unit s_b and use S-linearity of the component map to obtain
`U_b mod 2=L_B(Dec_2(f*_b))`. The second identity now authenticates the terminal
binary evaluation. It is essential: the first identity alone does not constrain
the claimed value v. ∎

The analogous prime rows use the **full** response, not its lowest digit.
Put S_E=E[Z]/Phi and unpack each response or source column into M scalar
components v_r in S_E, undoing signed packing. For each prime plane, fix public
row weights e_r in E before the fold challenge and define

```text
L_P : S_E^M -> S_E,       L_P(v) = sum_r e_r v_r.
```

This map is S_E-linear: for every s in S_E,
`L_P(s v)=sum_r e_r s v_r=s L_P(v)`. This linearity, as well as additivity,
is a hypothesis of prime-row extraction and of Theorem 8. The coefficient
functional `<omega,->` is applied only after the plane has been authenticated;
it must not replace L_P in the fold equation. Authenticate

```text
L_P(z) = sum_b c_b EP_b in E[Z]/Phi,
vP = sum_b eq(rho_col,b) <omega,EP_b>.
```

EP is fixed behind D before c. Subtraction gives
`L_P(a_b)=s_b EP_b`. Since `a_b=s_b Dec_P(f*_b)` in scalar components,
S_E-linearity gives `s_b L_P(Dec_P(f*_b))=s_b EP_b`. The P-unit condition
then permits cancellation, proving `EP_b=L_P(Dec_P(f*_b))`. Only now does
the second row apply omega to prove the prime terminal value.

A general E-linear coefficient functional with weights W[r,t] must first be
decomposed as `W[r,t]=sum_alpha e_(alpha,r) omega_(alpha,t)`. Each alpha defines
an S_E-linear plane `L_(P,alpha)(v)=sum_r e_(alpha,r) v_r`; the final value is
`sum_alpha <omega_alpha,L_(P,alpha)(v)>`. Such a decomposition always exists,
for example by taking one plane per row with row-selector weights, but its
plane count must fit the descriptor's cap. All weights are public and fixed
before c, and every EP plane is bound by D. The descriptor caps all planes,
including different polynomials and distinct points, and D admission uses that
actual maximum. No bounded integer lift of a general EP field value is required
for this cancellation.

**Why scalar-algebra linearity is necessary.** Let L_bad(v) be the constant
coefficient of the first unpacked component, embedded as a constant in S_E.
It is E-linear but not S_E-linear. Take the allowed singleton challenge
`c=epsilon Z`, where epsilon in {-1,1} is its deterministic sign, and `c'=0`.
For one source ring element take `z=epsilon iota(Z)`, `z'=0`, `T=A*1 mod q`,
and `EP=0`. Both A relations hold: `A z=iota(c) T mod q` and
`A z'=iota(c') T mod q`. Both erroneous prime fold rows also hold, since
`L_bad(z)=L_bad(z')=0=c EP=c' EP`; the claimed terminal value can be zero.
Nevertheless `s=epsilon Z`, `a=epsilon iota(Z)` extract `f*=1`, and
`L_bad(Dec_P(f*))=1 != EP`. Indeed `L_bad(s*1)=0` whereas
`s L_bad(1)=epsilon Z`. Thus even a unit s cannot justify the inference for
this functional. Using the first component itself as L_P instead forces
`EP=1`, after which a constant-coefficient omega correctly returns one.

## 6. Conditional application agreement

**Theorem 5 (agreement under R_app).** For every committed coefficient of every
extracted column, suppose R_app implies that its prime reading belongs to
`{0,sigma(t)}`, where `sigma(t)=(-1)^floor(t/4)`. Then f* is the corresponding
integral signed-bit source and its binary and prime readings agree exactly.
The deterministic comparison bound is

```text
||iota(s)b-a||_infinity <= 2Gamma+Delta = 184+4095 = 4279 < P.
```

*Proof.* Let b be the unique signed-bit representative of all coefficients
of Dec_P(f*). Multiplying the field equality f*=b by the invertible iota(s)
gives iota(s)b-a=0 modulo P. The operator bound from Theorem 1 and ||b||<=1
prove the displayed inequality. Each coefficient is therefore an integer
multiple of P of magnitude less than P, hence zero. Divide in the number field
to conclude f*=b. Reduction modulo two removes the packing sign and gives
exactly the same unpacked bit. This proof is coefficientwise and has error zero
conditional on R_app. ∎

R_app is a hypothesis on the extracted source, not on the input buffer or the
folded response. It must be proved by the caller through prime terminal claims
consumed by **this same grouped opening**, with the same session, source owner,
polynomial manifest and index map. A boolean flag from a caller is not evidence.
Akita supplies no booleanity, zero-support, or prime-claim fusion helper.

The required domain contains every coefficient that enters A: coordinates
128..161, unused payload positions, padding carriers, partial last components,
and padded ring rows. MLE-only slots 162..255 and 648..1023 are instead zero by
definition and need no application assertion. The source descriptor must specify
the precise bidirectional index map and any explicit kernel permutation.
This map is a hypothesis of the agreement theorem.

**Corollary 6 (payload plus zero support).** Payload booleanity together with
zero on every other committed coefficient implies Theorem 5, with no additional
agreement error. Payload booleanity alone does not imply the theorem.

*Proof.* After undoing signs, the two hypotheses assign every committed
coefficient either a bit or zero, so all hypotheses of Theorem 5 hold. For the
failure of payload booleanity alone, the free rank-four module structure allows
an unexamined component to be any bounded integer polynomial: putting 2 at an
unused committed coordinate gives binary reading 0 and prime reading 2 there,
while all payload positions can be zero. The weak opening can take s=1 and
a=2Y^t at that unused coordinate, within the numerator bound. Thus the
payload-only predicate cannot certify agreement of the full source. ∎

If the application uses a random prime zero evaluation of the high-coordinate
subcube for full 128-bit carriers, its multilinear polynomial has n+7 variables.
A nonzero polynomial vanishes with probability at most (n+7)/|E|: condition one
variable at a time, or apply the total-degree root bound. Virtual high positions
are identically zero, while 128..161 are genuinely checked. Low-coordinate
padding needs its own zero mask in that relation; the high-coordinate subcube
alone does not cover it. The application's proof of booleanity and its reduction
errors belong to its own segment ledger. The deterministic agreement lemma
neither supplies nor removes these errors.

## 7. Typed setup adjoint

**Theorem 7 (Hankel adjoint and rank-one factor).** Over the integers or F, define
`K[t,s,k]=coeff_t(rem_Psi(Y^(s+k)))`. For any rho in F^D let
`g(l)=sum_t rho[t] coeff_t(rem_Psi(Y^l))`, for 0<=l<=2D-2. Then

```text
h(s) = sum_{k=0}^{D-1} v[k] g(s+k),
<rho, rem_Psi(a v)> = sum_s a[s] h(s).
```

For a block matrix A and coefficient vectors u_i, its genuine coefficient
adjoint is

```text
(A^T_adj u)_j[k] = sum_{i,s} A_ij[s] sum_t u_i[t] K[t,s,k],
<u, A v> = <A^T_adj u, v>.
```

When Stage 2 leaves response-index weights `eq(r_j,j) eq(r_k,k)`, the setup
functional on A has coefficient weight

```text
W[i,j,s] = eq(r_j,j) h_i(s),
h_i(s) = sum_k eq(r_k,k) g_i(s+k).
```

Thus for each i the weight has rank one in the axes j and s. It requires no
rank×m×D weight materialization by the verifier.

*Proof.* Expand the left side as the finite sum
`sum_{t,s,k} rho[t] a[s] v[k] K[t,s,k]` and interchange the sums. Since K depends
on s+k, the inner sum is g(s+k), proving the first identity. Adding row and
column indices gives the second. Substitution of the product equality weights
gives the final factorization. Padding the coefficient or row domains adds
zero-defined slots; it does not change the finite sums.

For a constructive convolution, reverse v to v_rev[t]=v[D-1-t]. The coefficient
D-1+s of g*v_rev is h(s), directly by the convolution definition. Split g into
its first D coefficients g0 and remaining D-1 coefficients g1, padding g1 with
a final zero. Then

```text
h(s) = (g0*v_rev)[D-1+s] + (g1*v_rev)[s-1],
```

where the second term is zero at s=0. Both products involve degree less than D
polynomials and have length 2D-1. This proves the corrected algorithm as well as
the dense identity. It is an ordinary exact convolution over P; no unavailable
large power-of-two NTT root of P is assumed. ∎

The A setup stream samples uniformly modulo q by rejection, takes its centered
integer representative, and then embeds that integer into P. It is **not** a
uniform P setup stream. Its domain binds q, ring, dimensions, ordering, profile,
and stream role. Ordinary setup uses separately typed P streams. A prepared
prefix commitment authenticates the exact embedded A coefficients; Stage 3's
product claim uses W above and a prefix claim consumed by the next ordinary
level. Comparison with the canonical setup occurrence must either identify
that source or output an ordinary binding collision. This is the required
setup interface; changing a stream label without authenticating its source is
not an implementation of the theorem. The arithmetic and typed-stream
implementation must satisfy this setup interface.

## 8. Checkpoint construction and composition

**Theorem 8 (mixed-root checkpoint).** Under the Section 1 ordinary-suffix
hypotheses, exact samplers, the S_E-linear prime planes defined in Section 5,
and the fixed-source checks in Theorems 1–7, the
binary root supplies the same weak-opening/candidate-comparison interface as
an ordinary root. Replacing its successor-source checkpoint by the checkpoint
at J preserves the suffix tree size Tree0. One application segment satisfying
the conditions below adds exactly one checkpoint, giving tree size 2Tree0,
expected at most `2(Q+2)Tree0` prover executions, and the bound in Section 9.

The required transcript order, including the reason for each fixation, is:

| Order | Fixed message or stage | What the next challenge may test |
| --- | --- | --- |
| 0 | Caller session, manifests, setup, source commitments, schedules, public claims | One source interpretation across all segments |
| 1 | Caller application reductions, including any multiple-binary-point reduction | Each round polynomial fixed before its challenge |
| 2 | Field-switch partial array, then binary frontend rounds and terminal | Partials fixed before seven row-selector coordinates; terminal awaits opening |
| 3 | D commitment to U, all EP planes, and ordinary groups' partials | Same partial source in central and neighboring fold children |
| 4 | Response nonce and indexed fold coordinates | Distinct supports enable subtraction; retries count in Q |
| 5 | Outgoing witness commitment to Z/T digits, U/EP views, K, K_val, and ordinary segments | Successor source fixed before any relation challenge |
| 6 | **Successor-source checkpoint**, J, then seven signed y | Candidate fixed before J; y may depend on J but not lambda |
| 7 | Output-vector batching lambda, ring reduction and row batching | Authenticate the fixed y vector and all fixed relation rows |
| 8 | Stage 1 range-image sumcheck | Digit alphabet, including U and both carries |
| 9 | Fused Stage 2 | A projection link, B/D views, both parity identities, prime rows, range-image binding |
| 10 | Typed Stage 3 | Setup product and canonical prefix claims |
| 11 | Ordinary recursion and terminal | All witness and prefix claims consumed; no trailing proof data |

*Proof.* Work backward from the ordinary terminal. By hypothesis it yields
incoming weak openings with authenticated scalar rows, or an ordinary binding
collision. At the checkpoint preceding J, a certificate carries a passing
prefix through the next fold and the successor source extracted from the
outgoing commitment. Its candidate is that source, with canonical setup sources
and the fixed decoding/view maps. Its test is the full exact branch relation:
digit membership, view consistency, RA=0 modulo q, the two parity identities,
prime rows, and ordinary B/D/compression/setup identities. This includes all
integer bounds; no separate claimed norm message is trusted.

Two candidates under the same checkpoint input have the same outgoing
commitment and typed manifest. If they differ, the ordinary source comparison
map returns an induced P-ring SIS collision. Thus even though extraction runs
backward, it may analyze one fixed candidate before J. Merely appealing to a
computationally binding commitment without this comparison map would not be a
checkpoint proof.

For completeness of the fixed-candidate defect argument, maintain a list of
false obligations. If a digit is invalid or a view differs, retain that defect
until its ordinary range/view check; candidate coefficients do not change as
messages arrive. Otherwise the bounded integer residual exists. If it is nonzero
modulo q, a J outside the q^-7 bad set has a nondivisible projection. Any passing
y for that row then differs from the projection as an integer; exact lifting
makes its field link false too. If the prover violates the canonical y/range/
divisibility checks instead, its prefix already rejects. Therefore no choice of
y after a good J can erase the defect. The other false branch identities persist
until their relation checks. For false output vectors independent lambda loses
a defect with probability at most 1/|E|. Ring evaluation and row batching have
their polynomial root bounds. Range and relation sumchecks propagate the first
false running claim unless a round challenge is a root of its nonzero difference
polynomial; its degree is the admitted degree of that round. At their terminals,
authenticated candidate evaluations and the canonical setup contribution make a
surviving false claim impossible. Theorem 7 identifies the latter contribution
exactly. Taking the disjunction of pending defects and a union bound gives the
ledger; subsequent prover messages cannot alter the fixed candidate.

Consequently a certificate that passes the checkpoint has the exact branch
relations. Apply Theorem 3 at the preceding binary fold to a central child and
one changed-coordinate neighbor. U/EP are common by D binding, while image views
are common by outer binding; disagreement yields those ordinary collisions.
Theorem 4 transfers the binary partials and terminal, and the prime cancellation
transfers every prime partial and terminal. These are the root terminal rows on
the pair of readings of a single rational source. Formally the root-row predicate
is the conjunction of its B-valued and P-valued linear rows; any compression of
these rows must charge its own field-appropriate batching error. Repeated source occurrences
are compared with Theorem 3. There is no division by an equality weight, so a
zero public multiplier leaves the source terminal authenticated.

All remaining recursive levels use the assumed ordinary combiners. The root
uses one flat indexed extraction stage with arity 1+I and the same number of
successor checkpoints as before: J replaces the later ring-reduction checkpoint.
Moving this checkpoint earlier extends its fixed-source segment; it does not
add a checkpoint factor. This constructs the claimed suffix certificate,
combiner, candidate, comparison, defect, and witness maps.

For the application, let sigma be the family of these source projections and
`Dec(sigma)=(Dec_2(sigma),Dec_P(sigma))` on binary groups, with ordinary decoding
on other groups. Require, for every fixed sigma and every allowed auxiliary
environment in Good, a defect bit and stage bad sets with three properties:

1. If R_app is false on Dec(sigma), the bit is one at the first application
   challenge for every prefix with the same statement and source commitments.
2. Outside a stage's bounded bad set, a one bit remains one for **every** next
   prover message extending that answer.
3. A passing application prefix whose terminal rows hold on sigma has bit zero.

At the first application challenge, take candidate sigma and test R_app. Two
candidates with the same input differ under the same source commitments;
Theorem 3 or ordinary binding returns a collision. At the segment end, the root
weak opening supplies exactly the rows in property 3. The three properties are
therefore the fixed-source checkpoint conditions. All application stages,
including field switching and caller reductions, belong to this one segment;
there is no separate checkpoint for each reduction. The added checkpoint doubles
Tree0. The generic extraction premise from Section 1 now gives the claimed
runs and errors. If R_app includes the full signed-bit hypothesis, Theorem 5
also gives agreement with zero additional error. ∎

The application conditions are universal over candidate sources. An honest
source argument or a proof about a second opening is not sufficient. A separate
opening would require a second extraction/comparison argument and its tree cost.
The field-switch adapter likewise must supply its actual fixed-source defect
argument; its familiar low/high disjoint-coordinate observation does not replace
a proof for an arbitrary application relation. Production admission must validate
each supplied application interface against these three conditions.

## 9. Error ledger and standard profile

For an admitted schedule, the resulting classical-ROM bound is

```text
Pr[accept and extracted relation false]
 <= Delta_setup + Pr[not Good] + epsilon_interactive + Q epsilon_max
    + sum_I Adv_MSIS,I(extractor at its actual budget) + epsilon_hash.
```

The refined address charge is `E[sum_{a in queried addresses} epsilon(a)]`;
`epsilon_interactive+Q epsilon_max` is an envelope for it, not an additional
independent failure event. Bindings include every A, outer B, D, outgoing,
compression, and setup-prefix instance. A q24 scalar attack-cost table alone
does not bound all those advantages or prove structured module-SIS hardness.

| Event | Unfiltered bound or deterministic condition |
| --- | --- |
| Binary fold | I/|S46|, I=256 for one C=256 source |
| A comparison | q-module-SIS radius 1,506,960, rank and width as admitted |
| Other binding | Sum of actual induced ordinary SIS advantages |
| Seven-row divisibility | q^-7 = 2^-166.353625728... |
| Projection output batching | 1/|E| for independent uniform vector scalars |
| Integer lifts | 0 when all bounds in Theorems 2 and 4 hold |
| Agreement | 0 conditional on R_app from Theorem 5 |
| Ring evaluation | Sum of actual degrees/|E|; reduced degree <=647, unreduced <=1294 |
| Row, view and claim batching | Sum of actual degrees/|E|; vector batching degree one |
| Stage 1 | Conservative sum over levels of 5 nW/|E| |
| Stage 2 | Conservative sum over levels of 3 n2/|E| |
| Stage 3 | Sum over levels of 2 nSetup/|E| plus actual output batching |
| Setup sampling | Delta_setup=0 in ideal XOF, exact rejection, fail-closed exhaustion |
| Context labels | Mctx(Mctx-1)/2^257 per independent 256-bit family |
| Other transcript digests | Separate hash reduction; see below |
| Honest response exhaustion | At most (7/8)^4096, conditional on the sign/tail model; completeness only |

For digests substituted for transcript messages, collision-only context
accounting need not cover guessing an unopened digest. A sufficient explicit
ideal-hash charge per type with Qh hash queries and V digest positions is
`(binom(Qh,2)+Qh V)/2^256`, summed over types. The first term covers distinct
preimages colliding; the second covers a fresh hash answer hitting a previously
used digest. Under their complement expansion is unique and precedes the
challenge query. It is not multiplied by Q again. The admission certificate must
enumerate which hash roles are used and avoid counting the same role twice.

Caller-owned stages are separate ledger entries but part of the same Q and
checkpoint: a multiple-point H=GF(2^128) reduction contributes
`(1+2n)/2^128` before filters; a seven-coordinate B row selector contributes
`7/2^162`, and its product sumcheck `2n/2^162`. An application using the stated
payload and zero arguments charges their own degree bounds, such as
`(n+7)/|E|` for the high-coordinate zero claim, plus any low-coordinate masks,
batching and source-reduction sumchecks. These are conditional charges for those
chosen arguments, not a library implementation or a universal R_app bound.

The standard profile is E=F, w46, C256 and seven rows, with the ordinary work
planner enforcing **epsilon_max <= 2^-128**. The binary full-vector address has
**0.300278309... bits of margin**. This is a per-address claim, not total error
below 2^-128 for an arbitrary Q. One additional source at that same address
changes I and invalidates this w46 gate. The minimum weights for I=512 and 1024
are respectively 47 and 48; both require fresh response and SIS admission.

For an eligible degree delta field address an independent g-bit work filter
reduces its weight to `delta/(|E| 2^g)`. Choose g by the exact integer inequality
`delta 2^128 <= P 2^g`, not a rounded log2(P). In particular even delta=1 needs
one bit because P<2^128; delta=2 needs two, delta=647 needs ten, and delta=1294
needs eleven. Over H the degree-two case needs one bit exactly. If an address
bundles several rounds its numerator is the sum of their bad-set sizes. Work
filters do not repair missing fixation, and the response-search nonce supplies
no automatic 12-bit soundness gain.

The optional larger profile w77 has log2|S77|=160.220269844..., fold margin
0.220269844... above 152 bits for I=256, and radius 2,522,520. It requires separate
response, SIS, extension-field, Q and tree-time admission. In particular its
four-digit positive response reach fails at n=26. It is not needed merely to
name the standard per-address profile 128-bit FS.

## 10. Executable checks and admission criteria

`scripts/binary_source_checks.py` and
`scripts/tests/test_binary_source_checks.py` independently compute the constants,
use exhaustive coefficient kernels and small-degree adjoint identities, and
exercise reproducible full-degree random adjoints. Optional full checks are
selected by the script flag. Tests include negative controls for invalid bounds
and transpose conventions. Run the repository Python unittest discovery,
documentation guardrails, and typo checker.

Production admission requires:

- Matching the emitted transcript to the pre-J checkpoint and every fixation in
  Theorem 8, with exactly one caller application checkpoint and fresh J per trial.
- Instantiating the source descriptor's index map, complete signed-bit/zero
  coverage contract, source ownership and prime-plane cap; no unconditional
  agreement output.
- Comparing the trinomial kernels, unit admissions, challenge sampler and typed
  setup streams with these identities, including zero virtual slots and negative
  coefficients; authenticating the Stage-3 prefix at the next level without
  online A scans.
- Certifying q24 rank/width/radius and all ordinary induced instances at their
  maximum plane counts and extractor budgets, with structured-ring assumptions
  stated explicitly; no reuse of q25 evidence as a q24 certificate.
- Emitting the whole schedule's degree/address/work ledger, Tree0, sampling and
  hash budgets, response model and exhaustion qualification. In particular n=24
  needs ten K digits here; C512/w47 has response cap 1862>1755 and is not admitted
  by entropy alone.
- Completing malformed-input, no-panic, runtime selection, unselected-byte and
  caller-transcript tests, plus independent security review of the implemented
  compiler and the concrete application-segment interface.
