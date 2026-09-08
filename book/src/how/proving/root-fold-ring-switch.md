# Root fold and ring switching

One folding step builds a batched relation and switches it into the next ring
witness. The schedule also selects how each commitment group is opened.

## Subring coefficient packing

The preceding pages established the two semantic facts needed by
`SubringCoefficientPacking`:

1. the [direct packed scalar
   row](./field-ring-reduction.md#subring-coefficient-packing-shorter-partials)
   recombines the packed opening digits into the original scalar claim; and
2. [packing commutes with the sparse source
   fold](./akita-fold.md#subring-coefficient-packing-consistency), so the
   packed partial and folded source describe the same random combination.

This section starts from those facts and explains their physical realization:
the packing quotient, its coordinate planes, and evaluation at the ring-switch
challenge.

### Native geometry

Let $F$ be the base field and $E$ the opening field, with $k=[E:F]$. The
schedule chooses $D$, $s$, and a packing factor $\eta$ satisfying

$$
D=k\eta s.
$$

The implementation and live specification call these values `d_A`, `s`, and
`h`, respectively. The three relevant rings are

| Ring | Definition | Physical role |
|---|---|---|
| A ring | $R=F[X]/(X^D+1)$ | source witness and A rows |
| challenge subring | $S=F[U]/(U^s+1)$ | sparse fold challenges |
| extension opening ring | $C=E[U]/(U^s+1)$ | packed partial and packing quotient |

The embedding used by the A relation is

$$
\iota:S\hookrightarrow R,
\qquad
\iota(U)=X^{k\eta}.
$$

The [two-dimensional source
table](./field-ring-reduction.md#subring-coefficient-packing-shorter-partials)
already showed how the $u$ columns are contracted and the $j$ rows are kept.
This page starts from its output. One packed partial has $s$ coefficients in
$E$. Expanding each coefficient in the canonical $E/F$ basis gives the
following physical representation:

| Object | Logical shape | Base-field storage | Width |
|---|---|---|---:|
| all packed partials $e_i(U)$ | one $s$-coefficient element of $C$ per claim/block pair | `[claim][block][extension coordinate][subring coefficient]` | $ks$ per pair |
| one group's $Q_{\mathrm{pack}}(U)$ | one degree-below-$s$ polynomial in $E[U]$ | `[extension coordinate][subring coefficient]` | $ks$ |

Within either object, one extension-coordinate plane contains all $s$
coefficients in increasing $j$ order. Its width $ks=D/\eta$ is **not** the
dimension of one larger ring. A packed partial is one logical $C$-valued
element represented by $k$ base-field planes. The quotient has the same
coefficient shape, but is kept as a degree-below-$s$ representative in
$E[U]$. The D commitment, or its compressed H realization, binds the
digit-decomposed partial planes before the fold challenge is sampled.

### The packing consistency quotient

Let $e_i(U)\in C$ be the packed partial for a live claim/block pair and let
$c_i(U)\in S$ be its fold challenge. For the folded-source digits, use the
shorthand

$$
G\hat{\mathbf z}
:=
\left(
\sum_fG_f^{\mathrm{fold}}\hat z_{p,a,f}
\right)_{p,a}.
$$

This notation preserves the $(p,a)$ vector: the sum recomposes only the fold
digit index $f$. Applying the packing map gives $L(G\hat{\mathbf z})(U)$.
Semantic consistency says

$$
\sum_i c_i(U)e_i(U)=L(G\hat{\mathbf z})(U)
\qquad\text{in }C.
$$

Canonical representatives of the two sides need not be equal as ordinary
polynomials. Their difference is divisible by the cyclotomic modulus, so the
prover supplies

$$
Q_{\mathrm{pack}}(U)\in E[U],
\qquad
\deg Q_{\mathrm{pack}}<s,
$$

such that the following identity holds in $E[U]$:

$$
\boxed{
\sum_i c_i(U)e_i(U)-L(G\hat{\mathbf z})(U)
=
(U^s+1)Q_{\mathrm{pack}}(U).
}
\tag{1}
$$

The quotient also has $s$ coefficients in $E$, hence $k$ coordinate planes of
length $s$. Its digit coordinates occupy the consistency-row slot in the
shared pre-compression quotient range. In Stage 2, the packed E and Q events
join the common relation-weight factorization, while the folded-source side is
the separate packing-Z structured term. These contributions replace the
legacy `EvaluationTrace` consistency coefficients; they do not add a second
copy of the obligation.

### Evaluate the native relations at one challenge

After the quotient and next witness are bound, the transcript samples the
ring-switch challenge $\alpha$. Evaluating Equation (1) at $U=\alpha$ gives

$$
\sum_i c_i(\alpha)e_i(\alpha)
-L(G\hat{\mathbf z})(\alpha)
=
(\alpha^s+1)Q_{\mathrm{pack}}(\alpha).
\tag{2}
$$

The A rows use the same challenge polynomials through the embedding. For a
challenge

$$
c_i(U)=\sum_{j=0}^{s-1}c_{i,j}U^j,
$$

the two required evaluations are

$$
c_i(\alpha)
\qquad\text{and}\qquad
\iota(c_i)(\alpha)
=
c_i(\alpha^{k\eta}).
\tag{3}
$$

Equation (3) does not introduce a second transcript challenge. It evaluates
one coefficient list in the two native geometries required by packing
consistency and the A relation.

### One packing fold in transcript order

1. The schedule fixes $D$, $s$, the canonical extension basis, and the sparse
   challenge family.
2. The prover forms each packed partial and digit-decomposes its $ks$
   base-field coordinates.
3. The prover binds the complete D payload, or its compressed H payload.
4. The transcript samples one $c_i(U)$ for each live claim/block pair.
5. The prover folds the A-ring sources with $c_i(X^{k\eta})$ and computes
   $Q_{\mathrm{pack}}$.
6. The prover binds $Q_{\mathrm{pack}}$ and the next witness; only then does
   the transcript sample $\alpha$.
7. Stage 2 checks the scalar opening, Equation (2), and the A rows using the
   two evaluations in Equation (3).

This ordering fixes the packed digits before the subring challenges and fixes
the quotient and next witness before the evaluation point.

### Worked production geometry

For the fp32 candidate

```text
d_A = 1024,
k   = 4,
s   = 128,
h   = 2,
k h = 8,
```

the A-ring coefficient index is `a + 8j`. For every fixed `j`, the partial
contracts eight base-field coefficients into one element of the degree-four
extension field. That value occupies four base-field coordinates:

```text
evaluation-trace partial:  1024 coordinates in F
packed partial:              128 values in E
                             128 * 4 = 512 coordinates in F
```

The packed partial and $Q_{\mathrm{pack}}$ are each half as wide as their
full-A-ring counterparts. The challenge embeds at exponents
`0, 8, 16, ..., 1016`.

Choosing `s = 64` with the same `d_A` and `k` would instead give `h = 4` and
256 base-field coordinates per packed object. That smaller subring needs a
different sparse challenge family and can increase the A response bound or
change the recursive suffix. The planner therefore prices the complete
schedule rather than minimizing `s` alone.

### Scope and schedule placement

Packing gives two direct savings at an eligible fold: it removes EOR for a
proper extension-field opening, and it reduces each partial and packing
quotient from $D$ to $ks$ base-field coordinates before digit decomposition.
It does not shrink every proof component by $\eta$; gadget depths, matrix
ranks, compression payloads, response bounds, and later folds can change.

Current generated schedules use packing only at existing nonterminal absolute
fold levels 0 and 1. Every group in such a fold must have a feasible packing
assignment, and one fold cannot mix packing with `EvaluationTrace`. Later
recursive folds and the terminal use `EvaluationTrace`; packing adds neither
an EOR payload nor a packing terminal.

This chapter documents the implemented relation and schedule boundary. The
active design record gives the formal planner and soundness requirements,
including the implemented coordinatewise CWSS accounting. The equations here
are protocol relations; by themselves, they are not an end-to-end soundness
theorem.

## The root fold

`OpeningClaimsLayout` routes polynomial groups to claims. Each group keeps its
own public point, commitment profile, and opening geometry. The relation order
is final group followed by precommitted groups. A recursive fold uses the same
group rules for its folded witness and an incoming setup prefix.

## Ring switching

The schedule selects how each fold turns its native-ring relations into field
relations for Stage 2. In `QuotientLift` mode, the prover supplies a unique
polynomial-modulus quotient for every physical relation row. The derivation
below describes this mode.

`ReducedEvaluation` instead moves negacyclic reduction into public coefficient
weights and creates no polynomial-modulus quotient rows or quotient digits.
Production schedules admit it only as a monotone, setup-direct
`EvaluationTrace` suffix beginning at absolute level 2. Coefficient packing
remains quotient-lift-only. The [ring-relation realization
chapter](./akita-fold-realizations.md#reduced-evaluation) gives the reduced
weights and complete schedule restrictions.

This operation is distinct from EOR. EOR changes an extension-valued opening
claim before the lattice relation is formed. In `QuotientLift` mode, ring
switching lifts the resulting lattice relation out of its quotient ring so
that sumcheck can prove it over a field.

### Recover the quotient from two convolutions

Let `a(X)` and `s(X)` have degree less than `D`, and write their ordinary
product as

$$
a(X)s(X)=L(X)+X^D H(X),
$$

where both `L` and `H` have degree less than `D`. Reducing this product modulo
the cyclic and negacyclic moduli gives

$$
\begin{aligned}
[as]_{X^D-1} &= L+H,\\
[as]_{X^D+1} &= L-H.
\end{aligned}
$$

The field has odd characteristic, so division by two is defined. The high half
of the ordinary convolution is therefore

$$
H=\frac{[as]_{X^D-1}-[as]_{X^D+1}}{2}.
\tag{4}
$$

Equation (4) is exactly the quotient in

$$
a(X)s(X)-[as]_{X^D+1}=(X^D+1)H(X).
$$

For a complete row of the relation, let

$$
P_i(X)=\sum_j M_{i,j}(X)w_j(X).
$$

The quotient-ring equation says that `[P_i]_(X^D+1) = h_i`. Consequently, the
ordinary-polynomial identity used by Stage 2 is

$$
P_i(X)-h_i(X)=(X^D+1)r_i(X),
$$

with

$$
r_i=\frac{[P_i]_{X^D-1}-[P_i]_{X^D+1}}{2}
   =\frac{[P_i]_{X^D-1}-h_i}{2}.
\tag{5}
$$

The prover digit-decomposes each `r_i` and appends those digits to the recursive
witness. After the verifier substitutes `X = alpha`, the factor `X^D + 1`
becomes the public scalar `alpha^D + 1`. This turns every lifted row into a
field relation suitable for the fused Stage-2 sumcheck.

### Preserve each row's native ring

Akita does not enlarge every relation to one common ring dimension before
computing Equation (5). Consistency and A rows use `d_A`, B rows use `d_B`, and
D rows use `d_D`. Their quotients retain those same native dimensions. This is
both the mathematical layout and the physical recursive-witness layout; the
row geometry records the native dimension and the number of coordinate planes.

The coefficient-packing consistency row is the one nonstandard geometry. It is
an equation over `E[Y]/(Y^s + 1)`, represented as `k` base-field coordinate
planes of length `s`. Its quotient therefore has `k s` physical coordinates.
It is not reinterpreted as one ring of dimension `k s`. Its modulus evaluates
to `alpha^s + 1`, as in the packing identity in Equation (2).

### Compute only the coefficients that survive

The matrix rows use paired cyclic and negacyclic transforms to obtain Equation
(5). Akita performs both convolutions through the same CRT profiles used for
ring multiplication, then converts their difference back to the base field and
multiplies by `1/2`.

Sparse challenge products need less work. If a nonzero challenge coefficient
is at position `p`, only source coefficients `D-p` through `D-1` can reach the
high half of the ordinary convolution. The quotient kernel visits only those
coefficients and accumulates directly into `r`; it does not form the low half
that negacyclic reduction would discard. The same rule applies when the
consistency row combines folded opening material with sparse challenges.

Compressed commitments introduce additional F and H relation rows. Their
quotients use the same cyclic-versus-negacyclic identity and remain attached to
the compression layer that owns them. Compression changes the row layout, not
the algebra of Equation (5).

### Cached and streamed execution are equivalent

The CPU backend chooses between two execution plans after the complete row
geometry has been validated:

- A retained operation reuses the exact transformed setup prefix held by the
  prepared setup.
- A large operation transforms CRT-safe chunks of the same logical prefix as
  it proceeds and releases each chunk afterward.

Both plans cover the same rows, columns, transform domains, and quotient
coordinates. The CRT capacity bound limits how many products may be accumulated
before reconstruction in either plan. This choice affects time and memory only;
it does not change setup identity, proof bytes, transcript order, or the
quotient checked by the verifier.

## Implementation map

- `crates/akita-prover/src/protocol/ring_relation.rs` assembles ordinary
  relation terms.
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs` computes
  the role-native ordinary quotients and sparse high-half contributions.
- `crates/akita-prover/src/compute/cpu/ring_switch.rs` selects the retained or
  streamed CPU kernels.
- `crates/akita-prover/src/protocol/ring_switch.rs` assembles the ring-switch
  witness and proof state.
- `crates/akita-prover/src/protocol/coefficient_packing.rs` forms packed
  partials and the packing quotient.
- `crates/akita-types/src/subring_coefficient_packing.rs` defines and validates
  the packing geometry.
- `crates/akita-types/src/proof/coefficient_packing_relation.rs` supplies the
  factorized Stage-2 packing relation.
- `crates/akita-verifier/src/protocol/core/fold/` replays the relation and
  rejects a proof whose dimensions or quotient structure do not match the
  selected schedule.
