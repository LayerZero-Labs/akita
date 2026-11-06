# Spec: exact-prime compatibility and binary-source agreement

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-11-06 |
| Status | proposed |
| PR | |
| Supersedes | |
| Superseded-by | |
| Book-chapter | book/src/how/security.md |

## Summary

Specify what a binary-source root must prove when its prime readings share an
opening with ordinary claims over an exact application prime. The current
binary-source characteristics differ from `Prime128OffsetA7F7`. Moving the
existing challenge family to that prime invalidates its universal
unit-difference premise. A degree-148 scalar algebra repairs that premise and
admits conservative single-owner response and carry bounds. It remains a
research profile: no production modulus, ranks, security certificate or
geometry admission is supplied.

The proposing application has decided to pursue this degree-148 algebra.
Adoption in Akita remains the maintainers' decision; no maintainer approval is
claimed. This document builds on the [multigroup design](binary-source-multigroup.md)
and its shared root, rather than specifying a separate driver.

Source inspection is pinned to `dev`
`ca88498e173276d0d187422f24b8a07a05c407bb`. Relative to the earlier inspected
`10d3496e125e698e58f5b8f284c3169d2abc3d25`,
[#367](https://github.com/LayerZero-Labs/akita/pull/367) has landed the p64
coordinate-wise A adjoint and typed Stage 3 support. It does not instantiate the
algebra or extraction contract below. Statements labelled **proved** are
algebraic implications under their displayed assumptions. **Imported** means an
existing root contract; **open** marks work needed before admission.

## Intent

### Goal

Authenticate binary and exact-prime readings of one commitment-fixed source,
and consume their obligations with the application's ordinary groups through
one shared root and one ordinary successor.

### Invariants

- Bind exact field moduli, scalar/carrier algebras, coefficient maps, owners,
  setups, group order, dimensions and schedule in the public statement.
- Fix the source commitment before caller challenges that test its source.
- Fix every group's D commitment before any column-response challenge. D binds
  the binary and prime planes used in extraction.
- Fix responses, quotient digits, carries and ordinary preimages in one
  outgoing witness before the independent root checks.
- Use one composed typed setup prefix and one successor; retain mandatory
  source terminals even when a public evaluation weight is zero.
- Account for extraction failure, source comparison, integrality, retries and
  all query work in the selected schedule's security statement.
- Reject unsupported or mixed profiles before allocation or indexing. Preserve
  setup, commitment, proof bytes and transcript state when unselected.

### Non-Goals

This proposal does not implement a new field or driver, migrate an application's
ordinary prime, select production parameters, supply a structured-SIS
certificate, or admit any geometry. Caller-specific relations, claim-reduction
protocols and performance estimates are outside its scope. Akita does not
manufacture an agreement token from an honest importer or a matching digest.

## Evaluation

### Acceptance Criteria

- [ ] Independently review the exact-modulus obstruction with its universal-premise scope.
- [ ] Review the source extraction, prime-integrality and integer-agreement contract.
- [ ] Check the cyclotomic identity, irreducibility, deterministic sign condition and norm bounds.
- [ ] Check the single-owner response, comparison, quotient and carry arithmetic.
- [ ] Decide whether the new algebra belongs in the shared driver as a supported profile.
- [ ] Specify exact setup identities, encodings, samplers, source maps and grouped
  response layout.
- [ ] Select and certify the modulus, ranks, complete query/extractor schedule and completeness bounds.
- [ ] Complete the ordinary-group integration and independent security review before admission.

### Testing Strategy

This change requires documentation checks: valid math delimiters, existing local
link targets, documentation guardrails and review against the displayed
hypotheses. It introduces no executable protocol and requires no Cargo build.
A later implementation must test the algebra and its adjoints against exact
reference arithmetic, canonical field encodings, mixed-profile rejection,
mandatory terminals, transcript barriers and unchanged unselected behavior.

### Performance

No implementation or runtime change. Proof-size and prover-cost effects of the
degree-148 profile depend on the modulus and ranks, which this record does not
select; they must be stated, with a planner or estimator command, when those
parameters are proposed.

## Design

### Architecture

This record adds no module, trait or proof object. A supported construction
would add a scalar/carrier algebra profile alongside the existing binary-source
profiles, reusing the multi-group driver of
[binary-source-multigroup](binary-source-multigroup.md); the sections below state
what that profile must prove.

### Exact moduli and the scope of the obstruction

Write the ordinary application prime as

$$p=2^{128}-2^{32}+22537.$$

The current 128-bit binary-root prime and the p64 profile use, respectively,

$$P=2^{128}-275,\qquad p_{64}=2^{64}-59.$$

These are different characteristics. A unital map between distinct prime
fields cannot exist: summing the multiplicative identity a characteristic-many
times must commute with the map. Equal bit width or a type alias cannot bridge
them. The current [prime claim type](../crates/akita-types/src/binary_source/prime.rs)
and [prime weights](../crates/akita-types/src/binary_source/prime_weights.rs)
are specific to the existing characteristic. The single-point API shape from
[#362](https://github.com/LayerZero-Labs/akita/pull/362) is reusable; its modulus
validation and weights require a compatible instantiation.

**Proved obstruction to the universal unit premise.** The old scalar polynomial
is

$$\Phi_0(Z)=Z^{162}+Z^{81}+1=\Phi_{243}(Z).$$

Since $p\equiv1\pmod{243}$, it has 162 distinct linear factors over
$\mathbb F_p$. Let $S_{46}$ contain one deterministically signed polynomial for
each support of size at most 46 in $\{0,\ldots,161\}$. Then

$$M=|S_{46}|=\sum_{j=0}^{46}\binom{162}{j}>2^{136}>p.$$

At any primitive 243rd root $\zeta$, evaluation maps this family into
$\mathbb F_p$. Two distinct challenges therefore have equal evaluations.
Their difference is a nonunit in $\mathbb F_p[Z]/\Phi_0$. Thus the
prime-unit-difference premise of the existing
[checkpoint](binary-source-checkpoint.md), proved over $P$, does not transfer
to $p$ for the unchanged family. It does not disprove extraction
from suitably distributed invertible pairs. The probabilistic alternative
below states the missing obligations.

### Extraction and source agreement contract

**Required backend contract, open as an instantiated protocol.** For every
supported statement and stateful malicious prover, acceptance must yield,
through an expected polynomial-time extractor, one source associated with the
initial commitment. Native binary readings and prime readings must authenticate
the corresponding reductions of that same source. Comparison must make it
consistent across adaptive accepted queries, except on explicitly charged
binding or SIS failures. Evaluation binding alone does not supply this contract.

**Imported weak-opening shape.** Let $S=\mathbb Z[Z]/\Phi$ and
$R=\mathbb Z[Y]/\Phi(-Y^4)$, with $\iota(Z)=-Y^4$. Let $A$ be the source setup
matrix and $T_b$ the authenticated image of source column $b$. A central
response $z$ and a fork $z^{(b)}$ differing only at that column give

$$a_b=z-z^{(b)},\qquad d_b=c_b-c_b^{(b)},\qquad
A a_b=\iota(d_b)T_b\pmod q,\qquad f_b^*=a_b/\iota(d_b).$$

Division is in the carrier number field, not modulo $q$. Write $\Gamma$ for
the infinity operator bound of one challenge and $\Delta$ for the diameter of
an accepted response coefficient interval. Thus
$\|a_b\|_\infty\le\Delta$ and
$\|\iota(d_b)\|_{\infty\to\infty}\le2\Gamma$.
Two different rational openings of the same image give

$$A\bigl(\iota(d'_b)a_b-\iota(d_b)a'_b\bigr)=0\pmod q.$$

The comparison vector has norm at most $4\Gamma\Delta$. If nonzero as an
integer vector and $q>4\Gamma\Delta$, it is also nonzero modulo $q$. Its
hardness requires the actual structured-SIS certificate and extractor budget;
the inequality alone is not a security claim.

Define $\mathsf{Int}_p$ as the event that a commitment-fixed rational source
exists, is unique up to the charged comparison failure, and is integral at
every prime above $p$. Equivalently it admits a denominator invertible modulo
$p$, so its prime reduction $\operatorname{Dec}_p(f^*)$ is defined. Any caller
predicate or claim reduction over that table is conditioned on this event.
The backend must charge
$\Pr[\neg\mathsf{Int}_p\land\mathrm{accept}]$ once in its extraction-failure
ledger. A later successful fork cannot silently justify an earlier unconditioned
fixed-source argument; the composed extraction/checkpoint proof must establish
that interpretation.

The prime plane uses the full response and an $S_E$-linear map, where
$S_E=E[Z]/\Phi$ and $E$ is the selected prime checking field. With public row
weights $e_r$, it has the form

$$L_p(v)=\sum_r e_r v_r,\qquad L_p(z)=\sum_b c_b EP_b.$$

The planes $EP_b$ are bound behind D before the column challenges. Subtraction
and cancellation by a unit $d_b$ yield
$EP_b=L_p(\operatorname{Dec}_p(f_b^*))$. Only then may a coefficient functional
be applied to obtain the prime claim. An arbitrary scalar functional cannot
replace the $S_E$-linear map before cancellation. The binary plane requires
the analogous unit property modulo two.

**Proved conditional agreement.** The caller must prove bit membership and
required zero support over every unpacked committed prime coefficient,
including committed padding, and authenticate the resulting terminal through
the root. Virtual coordinates are zero by definition and are distinct from
committed padding. Undoing the packing signs produces the unsigned prime
table. Restoring them after the predicate yields an integer vector $b$ with
$\|b\|_\infty\le1$. Prime extraction then implies

$$a_b=\iota(d_b)b\pmod p.$$

If $2\Gamma+\Delta<p$, every coefficient of
$\iota(d_b)b-a_b$ is a multiple of $p$ of magnitude less than $p$, hence is zero
as an integer. Therefore $f_b^*=b$, and reduction modulo two recovers precisely
the same bits. This lift adds no error after its premises hold. Checking only
selected payload endpoints is insufficient; an unchecked coefficient equal to
two would read as binary zero and prime two. The caller owns these predicates
and their claim reductions. Their terminal may be fused into one prime point;
Akita owns authentication and the exact-prime extraction contract.

### The degree-148 algebra

**Proved algebra.** Set

$$\Phi(Z)=\Phi_{149}(Z)=\sum_{j=0}^{148}Z^j,\qquad
\Psi(Y)=\Phi(-Y^4)=\Phi_{1192}(Y),$$

$$S=\mathbb Z[Z]/\Phi,\qquad R=\mathbb Z[Y]/\Psi,\qquad
B=\mathbb F_2[Z]/\Phi.$$

A root of $\Phi(-Y^4)$ has order $8\cdot149=1192$; both monic polynomials
have degree 592, proving the identity. The embedding $\iota(Z)=-Y^4$ makes
$R$ free of rank four over $S$, with basis $1,Y,Y^2,Y^3$. Scalar coefficient
$j$ in component $a$ occupies carrier coefficient $4j+a$ with sign $(-1)^j$.

The prime 149 has $149-1=148=4\cdot37$. Modulo 149,

$$2^{74}\equiv-1,\quad 2^4\equiv16\ne1,\quad p\equiv32,
\quad32^{74}\equiv-1,\quad32^4\equiv63\ne1.$$

These tests against the prime divisors of 148 prove that both characteristics
have multiplicative order 148. The cyclotomic factor-degree criterion therefore
makes $\Phi$ irreducible over both $\mathbb F_2$ and $\mathbb F_p$.

**Proved unit-difference lemma.** Choose supports in $\{0,\ldots,147\}$ of
weight at most $w$, with exactly one deterministic signing of all nonzero
coefficients per support. Distinct challenges have different supports. Their
difference is nonzero modulo two and has degree below 148. A coefficient in
the symmetric difference is $\pm1$, so the difference is also nonzero modulo
$p$. Both scalar residue algebras are fields, hence every such difference is
a unit in both. Its image acts invertibly on the rank-four carrier module.
The condition on signs is essential: allowing both $Z^j$ and $-Z^j$ for one
support gives the zero difference modulo two. This proof concerns the new
algebra; it does not repair the old trinomial by renaming a profile.

The 128-bit frontend still uses an injective additive basis map

$$\phi\left(\sum_{u<128}b_uX^u\right)=\sum_{u<128}b_uZ^u.$$

Since $128\nmid148$, there is no field embedding from $\mathrm{GF}(2^{128})$
into $B$. The [linear field switch](../crates/akita-types/src/binary_source/field_switch.rs)
uses only $\mathbb F_2$-linearity. It retains 128 partials and seven selector
coordinates; for $n$ carrier axes its selector and degree-two round error is
$(7+2n)/2^{148}$ under the same fixed-source premises. A canonical element has
148 bits, with unused encoding bits required to be zero.

All profile dimensions must change together: scalar degree 148, carrier degree
592, committed padding coefficients 128 through 147, and virtual coefficients
148 through 255 for an eight-axis coefficient domain. A retained 1024-slot
carrier stride has virtual slots 592 through 1023. The plane codec, setup labels,
source maps, adjoints and transcript version must bind this interpretation;
mixing old and new constants rejects.

### Conservative single-owner response and carries

**Proved operator bounds.** In $S$, reduce by $Z^{149}=1$, then eliminate
$Z^{148}=-\sum_{j<148}Z^j$. Multiplication by a monomial first places each
input in one of 149 slots. Each surviving output receives at most one direct
coefficient and one eliminated coefficient, giving infinity operator norm at
most two. In $R$, reduce by $Y^{596}=-1$, then eliminate slots 592 through 595.
Each eliminated slot affects one residue class modulo four, giving the same
bound. Thus a weight-$w$ challenge has $\Gamma\le2w$ and a difference has
bound $2\Gamma$. General carrier multiplication has coefficient bound
$1184\|u\|_\infty\|v\|_\infty$: each negacyclic slot collects at most 592
products and elimination adds at most one such slot. A scalar product output
collects at most three unreduced product coefficients.

**Research envelope.** For $C=256$ columns, the smallest support cap meeting
the illustrative unfiltered per-address target $|S_w|\ge C2^{128}$ is $w=53$:

$$|S_{53}|=\sum_{j=0}^{53}\binom{148}{j}
=124512087410009418767006149192377079957728>2^{136}.$$

Weight 52 fails this target. Admission must specify a uniform support sampler
and bind its deterministic sign rule to the profile. This target is not a
whole-query security certificate. Here $\Gamma=106$. For one binary owner with one
source polynomial per column
and source coefficient magnitude at most one,

$$\left\|\sum_{b<C}\iota(c_b)s_b\right\|_\infty\le C\Gamma=27136.$$

Balanced base-eight digits in $\{-4,\ldots,3\}$ have positive reach and
maximum negative magnitude

$$\operatorname{Pos}(d)=\frac{3(8^d-1)}7,\qquad
\operatorname{Neg}(d)=\frac{4(8^d-1)}7.$$

Six digits give the interval $[-149796,112347]$, containing every such honest
response without response rejection. Its malicious diameter and resulting
bounds are

$$\Delta=262143,\qquad4\Gamma\Delta=111148632,\qquad
2\Gamma+\Delta=262355<p.$$

These are single-owner bounds. A shared response across $o$ identical owners
requires the honest bound $oC\Gamma$; heterogeneous owners require
$\sum_a C_a\Gamma_a M_a$, with source magnitude $M_a$. Separate responses
retain their own bounds. The grouped layout, including any ordinary response
components, must be specified before reusing these numbers. Aggregation may
require new response depths, diameters, carries, quotients and SIS bounds.

**Modulus constraints, not a selection.** Require an odd prime

$$111148632<q<2^{64}.$$

The lower inequality makes nonzero comparison vectors nonzero modulo $q$; it
does not establish hardness. The upper inequality is a sufficient arithmetic
cap for the strict lift below and fits an unsigned 64-bit modulus. It is not a
kernel or intermediate-product guarantee. A different cap must satisfy the
strict-lift inequality directly. Adequate separation from the comparison
radius and a structured-SIS certificate over $\Phi_{1192}$ remain open.
A transform using primitive 1192nd roots requires $q\equiv1\pmod{1192}$;
the chosen arithmetic implementation must state its actual constraints.
No numerical $q$ or rank is selected, and the old q24 table cannot certify this
profile.

**Coefficient quotient.** Let $N=4mC$ be the total padded carrier count over all
$C$ columns ($4m$ per column), and $r_A$ the as-yet unselected source-image rank. Put
$A_{\max}=(q-1)/2$. Choose the least image depth $d_T$ with
$\operatorname{Pos}(d_T)\ge(q-1)/2$ and set
$T_{\max}=\operatorname{Neg}(d_T)$. For the source residual
$\mathcal R=Az-\sum_b\iota(c_b)T_b$, a sufficient coefficient bound is

$$L=1184mA_{\max}\cdot149796+C\cdot106T_{\max},\qquad
H_K=\lfloor L/q\rfloor.$$

Reuse the coefficient-quotient design of
[#365](https://github.com/LayerZero-Labs/akita/pull/365) and
[#366](https://github.com/LayerZero-Labs/akita/pull/366): commit balanced digits
of $K=\mathcal R/q$ in the outgoing witness, range-check them, and batch all
$592r_A$ coefficient equations independently over $\mathbb F_p$. With
$B_K=\operatorname{Neg}(d_K)$, require

$$\operatorname{Pos}(d_K)\ge H_K,\qquad L+qB_K<p.$$

This recovers the exact integer equation from its prime-field check. The
variant has no random-projection J/y messages. Minimal image depth gives
$T_{\max}/q<16/3+4/q<6$. For $m\le131072$ and $C=256$,

$$L/q<592\cdot131072\cdot149796+256\cdot106\cdot6
=11623364459520.$$

Fifteen quotient digits suffice, with $B_K=20105355479332$, and

$$L+qB_K<2^{64}(11623364459520+20105355479332)
=585291576498408217930419857784832<p.$$

This establishes an arithmetic envelope, not admission of its SIS geometry.

**Parity and value carries.** The root equations retain their form with scalar
degree 148. A sufficient parity bound is

$$H_{\rm par}=148(N/C)\cdot4+C\cdot53\cdot4,\qquad
|K_{\rm par}|\le\lfloor3H_{\rm par}/2\rfloor.$$

Two illustrative geometries within the arithmetic envelope give:

| $N$ | $m$ | $H_{\rm par}$ | Carry bound | Carry digits |
| --- | --- | --- | --- | --- |
| $2^{20}$ | $1024$ | $2479104$ | $3718656$ | $8$ |
| $2^{27}$ | $131072$ | $310432768$ | $465649152$ | $11$ |

Positive reach covers each honest carry; the corresponding malicious no-wrap
inequalities are

$$3\cdot2479104+2\operatorname{Neg}(8)=26611272<p,$$
$$3\cdot310432768+2\operatorname{Neg}(11)=10748366408<p.$$

The honest value carry has bound
$\lfloor(3\cdot148C+1)/2\rfloor=56832$, so six digits suffice. Its malicious
no-wrap inequality is

$$3\cdot148\cdot256\cdot4+1+2\operatorname{Neg}(6)=754249<p.$$

The prime plane uses 43 centered base-eight digits to cover $(p-1)/2$, and its
scalar-algebra-linear equation uses the unit lemma above. These carry and
quotient results are conditional on the stated single-owner layout; neither
illustrative geometry is admitted by this document.

### Reuse of the shared root

The [multigroup specification](binary-source-multigroup.md), introduced by
[#363](https://github.com/LayerZero-Labs/akita/pull/363), supplies the intended
group roster, translated segments, shared outgoing witness, composed prefix
and single successor. Its approved design is not evidence that this new
profile or the complete driver is implemented.

[#364](https://github.com/LayerZero-Labs/akita/pull/364) supplies field-profile
and rank-derived payload work. The quotient layout and independent coefficient
batching from #365 and #366, and the coordinate-wise adjoint and typed Stage 3
work from #367, are reusable design components. The new algebra still needs
its own arithmetic and adjoints. At the pinned base, the typed prefix explicitly
rejects p64 prime planes; that support must not be inferred from Stage 3 alone.

Profile-generic obligations include group ownership and order, single-point
manifests, rank-derived payloads, witness offsets, independent row batching,
setup-prefix placement, one successor, and indexed transcript/retry accounting.
Profile-specific obligations include both moduli, cyclotomic reduction, source
signs and dimensions, unit proof, challenge sampler, carry bounds, codecs,
setup identities and security tables. A degree-148 profile is a new algebra,
not merely another field-profile enum value. Any future ordinary-prime migration
is a separate compatibility decision outside this proposal.

### Alternatives considered

**Unchanged family with probabilistic invertibility.** Keep $\Phi_0$ and the
actual deterministic support-dependent sign rule. For its 162 primitive roots
$\Omega\subset\mathbb F_p$, define

$$\mu=\max_{\zeta\in\Omega,v\in\mathbb F_p}
\frac{\#\{c\in S_{46}:c(\zeta)=v\}}{M}.$$

For a fixed central challenge, an independent uniform neighbour has a nonunit
difference with probability at most $\min(1,162\mu)$, including equality.
The zero support and 162 singletons have distinct evaluations at every root:
a power of an odd-order root cannot equal $-1$. Consequently

$$1/p\le\mu\le1-162/M.$$

This unconditional upper bound is unusable for cryptographic extraction.
Near-uniform evaluation is a heuristic, not a certificate. The current
[sign rule](../crates/akita-challenges/src/binary/sign.rs) depends on the whole
support; a theorem for independent coefficient signs cannot be imported
without a reduction.

A certified $\bar\mu\ge\mu$ with $D=\lceil162\bar\mu M\rceil<M$ would bound
the bad set, including the centre, by $D$. Among $D$ distinct accepted
neighbours at a coordinate, at most $D-1$ can be bad, so a good pair exists.
Under a coordinate-wise forking theorem producing that star for $I$ coordinates,
the proposed interactive debit is $ID/M$ and the arity is $1+ID$, rather than
$1+I$. The theorem's execution model, expected query budget and indexed-stream
composition must be justified for the selected protocol. Do not treat accepted
neighbours as independent uniform samples. A conservative ledger for
$T_{\rm fork}$ such invocations adds $T_{\rm fork}ID/M$, alongside binding and
suffix terms. The resulting budget must remain polynomial and be included in
SIS sizing. Even the unproved ideal $\bar\mu=1/p$ suggests a root debit near
$256\cdot162/p$, not a complete 128-bit guarantee. Useful anti-concentration
and a viable composed extractor certificate remain open.

**Cross-prime bit-circuit reader.** Retain the existing $P$ root and binary
frontend. In its one outgoing witness, commit a Boolean source copy, the
existing authenticated image digits, canonical bit encodings of complete
ordinary commitment preimages, and wires of a deterministic arithmetic circuit.
The circuit must check:

1. Bitness, canonical encodings and all source descriptors.
2. The signed source image $Ab=T\pmod q$, against the same authenticated $T$.
3. Every original ordinary commitment from its full preimage and exact setup.
4. Every requested ordinary and source prime functional using canonical integer
   arithmetic modulo $p$, not a cast into $P$.

Commit all inputs before circuit-check challenges, authenticate setup-dependent
coefficients through the composed prefix, and reduce circuit terminals to the
same outgoing witness and one ordinary $P$ successor. Under ordinary extraction,
the binary weak opening $f^*=a/\iota(d)$ and the circuit copy satisfy

$$A(a-\iota(d)b)=0\pmod q.$$

For the existing root bounds, the difference has norm at most
$4095+184=4279$. A nonzero difference gives a short SIS vector; otherwise the
native binary source equals the circuit's bit source. This finite conditional
agreement argument needs no unit modulo $p$.

The full ordinary-preimage grammar, exact commitment/evaluation circuits,
coefficient compiler, setup encoding, widened witness and complete schedule
remain unspecified for production. Recommitting all ordinary preimages also
requires an explicit decision about compatibility with a caller's one-session,
no-second-PCS requirement. One successor does not settle that restriction.
Converting only the binary source while leaving ordinary groups in another
characteristic does not close the shared-session contract.

## Documentation

Keep this record proposed until the maintainers decide its scope. Once a
supported construction and its security contract are implemented, fold the
stable explanation into [security](../book/src/how/security.md) and update this
record under the [spec lifecycle](PRUNING.md). This documentation change does
not alter the current runtime behavior or mark the research profile approved.

## Execution

The next gate is independent review of the displayed algebra, extraction
premises and arithmetic. Then resolve profile ownership and the grouped layout,
select exact moduli and ranks, and certify the complete structured-SIS,
extractor/query and completeness bounds. New binary/carrier arithmetic, prime
planes, setup identities, the complete ordinary roster and independent security
review all remain required. No production admission follows from the unit lemma
or the illustrative arithmetic envelope.

## References

- [binary-source-checkpoint](binary-source-checkpoint.md): current binary-source
  root and its unit-difference theorem over $P$.
- [binary-source-groups](binary-source-groups.md) and
  [binary-source-multigroup](binary-source-multigroup.md): group roster, shared
  witness, composed prefix and successor reused here.
- LayerZero-Labs/akita #362 to #367: single prime point per binary group,
  multi-group spec, p64/E64 profile and coefficient-quotient work.
