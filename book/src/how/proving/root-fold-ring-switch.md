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
U\longmapsto X^{k\eta}.
$$

The source index split from the semantic derivation is
$\ell=u+k\eta j$. The physical coefficient grid is therefore

```text
                         contracted by the opening
                    u = 0   1   ...   k eta - 1
                  +-----------------------------
       j = 0      |  f     f             f      | -> e_i[0]   in E
       j = 1      |  f     f             f      | -> e_i[1]   in E
         ...      |  ...                     ...|
       j = s - 1  |  f     f             f      | -> e_i[s-1] in E
                  +-----------------------------

                    retained as the U coefficient index
```

One packed partial has $s$ coefficients in $E$. In the canonical $E/F$ basis,
it is stored as $k$ coordinate planes of length $s$, in the order

```text
[claim][block][extension coordinate][subring coefficient].
```

Its physical width is therefore $ks=D/\eta$. This is **not** one ring of
dimension $ks$: it is one logical $C$-valued object represented by $k$
base-field planes, each with modulus $U^s+1$. The D commitment, or its
compressed H realization, binds the digit-decomposed planes before the fold
challenge is sampled.

### The packing consistency quotient

Let $e_i(U)\in C$ be the packed partial for a live claim/block pair, let
$c_i(U)\in S$ be its fold challenge, and let $L(G\hat z)(U)$ be the packed
image of the recomposed folded-source digits. Semantic consistency says

$$
\sum_i c_i(U)e_i(U)=L(G\hat z)(U)
\qquad\text{in }C.
$$

Canonical representatives of the two sides need not be equal as ordinary
polynomials. Their difference is divisible by the cyclotomic modulus, so the
prover supplies $Q_{\mathrm{pack}}(U)\in C$ such that

$$
\boxed{
\sum_i c_i(U)e_i(U)-L(G\hat z)(U)
=
(U^s+1)Q_{\mathrm{pack}}(U).
}
\tag{1}
$$

The quotient also has $s$ coefficients in $E$, hence $k$ coordinate planes of
length $s$. The current Stage-2 implementation realizes Equation (1) through
packed E/Q coordinate-plane events plus the structured packing-Z term. It does
not add the legacy ordinary `consistency` row as a second copy of the same
obligation.

### Evaluate the native relations at one challenge

After the quotient and next witness are bound, the transcript samples the
ring-switch challenge $\alpha$. Evaluating Equation (1) at $U=\alpha$ gives

$$
\sum_i c_i(\alpha)e_i(\alpha)
-L(G\hat z)(\alpha)
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
live specification remains `active` because its full-vector-fork extraction
argument is still an open review item; the equations here should not be read
as a completed end-to-end soundness theorem.

## The root fold

`OpeningClaimsLayout` routes polynomial groups to claims. Each group keeps its
own public point, commitment profile, and opening geometry. The relation order
is final group followed by precommitted groups. A recursive fold uses the same
group rules for its folded witness and an incoming setup prefix.

## Ring switching

Every physical cyclotomic-ring row is lifted through its own unique quotient
before evaluation at $\alpha$. An ordinary row in
$F[X]/(X^{d_i}+1)$ uses denominator $X^{d_i}+1$ and evaluates it as
$\alpha^{d_i}+1$. A packing consistency relation instead uses $k$ coordinate
planes with denominator $U^s+1$, producing the factor $\alpha^s+1$ in
Equation (2).

This ring switch is distinct from EOR. EOR changes an extension-valued opening
claim before the lattice relation. Ring switching proves the polynomial
quotients of the physical lattice relations themselves.

## Implementation map

- `crates/akita-prover/src/protocol/ring_relation.rs` assembles ordinary
  relation terms.
- `crates/akita-prover/src/protocol/ring_switch.rs` computes native-ring
  quotients and their evaluations.
- `crates/akita-prover/src/protocol/coefficient_packing.rs` forms packed
  partials and the packing quotient.
- `crates/akita-types/src/subring_coefficient_packing.rs` defines and validates
  the packing geometry.
- `crates/akita-types/src/proof/coefficient_packing_relation.rs` supplies the
  factorized Stage-2 packing relation.
- `crates/akita-verifier/src/protocol/core/fold/` reconstructs the scheduled
  relations and rejects inconsistent dimensions or quotient structure.
