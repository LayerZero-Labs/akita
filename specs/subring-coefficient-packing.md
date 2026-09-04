# Spec: subring coefficient packing

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-10 |
| Status | active |
| PR | [#394](https://github.com/LayerZero-Labs/akita/pull/394) |
| Supersedes | The assumption that every extension-field opening first uses extension-opening reduction |
| Superseded-by | |
| Book-chapter | book/src/how/proving/root-fold-ring-switch.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Review status

The comprehensive review of PR #394 found three concrete blockers. All three
are now repaired.

1. The old one-cursor challenge vector did not provide the coordinatewise
   forks assumed by the extraction argument. PR #417 keeps one group root but
   derives each claim-major block coordinate through a separate indexed query.
2. Recursive-prefix guided pruning could discard the suffix needed by the
   payload projection. It now requires compatible strict dominance in both
   projections and covers the crossing and proof-tie cases directly.
3. The old catalog checker compared the current compiled catalog with a current
   regeneration. The generator now keeps that drift guard separate from stable
   revision snapshots and a checked base-to-head logical-key-union report.

Draft PR [#417](https://github.com/LayerZero-Labs/akita/pull/417) owns the
concrete repair for item 1. It keeps one transcript-derived root seed per
commitment group, but replaces the shared vector XOF with one fixed-width
indexed query per claim-major block coordinate. The same change applies to
`EvaluationTrace`, including operator-norm-rejected draws.

### Reader guide

This document is the formal design record. It defines the coefficient layout,
relations, planner rules, and acceptance criteria. For a step by step
explanation, start with the Book chapter
[Root fold and ring switching](../book/src/how/proving/root-fold-ring-switch.md#subring-coefficient-packing).
That chapter shows the coefficient grid, explains why the challenge acts on
only one axis, and works through the production fp32 geometry
`d_A = 1024`, `k = 4`, `s = 128`, and `h = 2`.

## Decision

Akita will support the **subring coefficient packing** opening method for base
field committed tables opened at extension field points. The method keeps one
coefficient axis explicit in an extension opening ring and contracts the other
coefficient axes over the extension field. Fold challenges live in a challenge
subring and embed sparsely into the A ring. The challenge subring dimension and
the A ring dimension are independent schedule choices, subject to the
divisibility condition below.

Generated schedules use `SubringCoefficientPacking` at absolute fold levels 0
and 1 when those nonterminal folds exist. This rule applies to every field
profile. Every group at such a fold MUST have a complete, statically feasible
packing assignment. Otherwise the schedule is unsupported. The planner MUST
NOT fall back to `EvaluationTrace` at an early packing fold and MUST NOT mix the
two method families within one fold. For extension fields, EOR is not available
at packing levels. For degree one fields, the same method may reduce the partial
opening and D or H source even though there is no EOR payload to remove.

Subring coefficient packing folds at levels 0 and 1 MUST use the Linf A
security route. This keeps PR 369's rule that physical L2 admission starts at
fold level 3. The planner
MUST derive the Linf response bound through the same policy as PR 369. At the
root, this includes the unconditional adversarial bound for caller controlled
dense inputs. At level 1, it uses the propagated typed response model when the
selected profile provides one. This feature does not add another response
model.

Later recursive folds and the terminal use `EvaluationTrace`. The feature does
not add a subring packing terminal and does not extend subring packing search
past the existing two level adaptive prefix. A short schedule uses
`SubringCoefficientPacking` at each existing nonterminal fold with absolute
index 0 or 1. The planner MUST NOT add a fold only to obtain two subring packing
levels.

The planner searches the challenge subring dimension `s` independently of the
A ring dimension `d_A` at levels 0 and 1. Both values are candidate
coordinates. They are not explicit optimization components. The planner
retains the current adaptive catalog objective, which minimizes first-direct
padded setup capacity, proof payload, total setup, root output-witness length,
and the canonical descriptor in that order.

This implementation includes the selective L2 fold sizing work merged in
[PR #369](https://github.com/LayerZero-Labs/akita/pull/369). It also assumes the
dyadic B slicing merged in
[PR #388](https://github.com/LayerZero-Labs/akita/pull/388).

## Why this change

Akita currently commits base-field coefficients but uses an extension field for
opening points and sum-check challenges. EOR converts that extension-field
opening into a form accepted by the ring relation. It is correct, but it adds a
degree-two sum-check and transcript-bound partial evaluations at every
extension-valued recursive opening.

The conversion is avoidable. A coefficient table can instead be viewed as a
polynomial in the extension opening ring, whose coefficients already lie in
the extension field. This viewpoint has three useful consequences:

1. The original extension-field point can be used directly, so L0 and L1 need
   no EOR proof.
2. Each partial opening and its consistency quotient can use fewer base field
   coordinates than a full A ring element.
3. The current commitment type, A relation, B commitment, setup matrices, and
   most NTT caches can remain over the A ring.

The mechanism is an axis split. Write each A coefficient index as
`a + k h j`. The partial evaluation contracts `a` and keeps `j`. The fold
challenge is restricted to the `j` axis by sampling it in
`K[Y]/(Y^s+1)` and embedding `Y` as `X^(k h)`. Challenge multiplication
therefore commutes with the partial evaluation. A general A ring challenge
would mix both axes, so the shorter partial would not contain enough
information to reproduce the folded source.

The change is not unconditionally cheaper. A smaller challenge subring requires
a different sparse challenge family. Meeting the same entropy target can
increase
the challenge's norm, which can enlarge the folded witness, the secure A rank,
and the `t` part of the next witness. Near the recursive tail, a subring
coefficient packing opening may also require a larger A ring. This does not by itself
make the candidate worse: a larger ring can need proportionally fewer A rows.
The planner must compare exact field coordinate counts and the complete suffix.
Neither `s` nor `d_A` is an optimization objective by itself.

## Scope and notation

The `SubringCoefficientPacking` equations below describe one polynomial and one commitment
group. Existing claim batching and multi-group row coefficients apply outside
these equations and do not change them.

| Symbol | Meaning |
|---|---|
| `K` | Base coefficient field `F_q` |
| `E` | Challenge and evaluation field, with extension degree `k = [E:K]` |
| `d_A` | A ring dimension |
| `n_A` | Secure output rank of the A matrix |
| `s` | Challenge subring dimension selected by the schedule |
| `h` | Packing factor `d_A / (k s)` |
| `R` | A ring `K[X]/(X^{d_A}+1)` |
| `S` | Challenge subring `K[Y]/(Y^s+1)` |
| `C` | Extension opening ring `E[Y]/(Y^s+1) = E tensor_K S` |
| `beta_t` | Element `t` of Akita's fixed canonical `K`-basis of `E` |

Every admitted subring packing candidate satisfies

```text
d_A = k h s,
h >= 1,
d_A, k, h, and s are powers of two.
```

The embedding from the challenge subring into the A ring is

```text
S -> R,       Y -> X^(k h).
```

It preserves coefficient support and the coefficient `l1`, `l2`, and `linf`
norms:

```text
c(Y) = sum_(j < s) c_j Y^j
  maps to
c(X^(k h)) = sum_(j < s) c_j X^(k h j).
```

The implementation MUST use this canonical embedding. It MUST NOT search over
coefficient permutations or alternative subring embeddings.

### Protocol boundary

There is one subring coefficient packing opening method. This implementation
uses it only at nonterminal fold levels 0 and 1. Its partial digits become part
of the ordinary flat witness committed for the next fold. The descriptor MUST
bind `k`, `s`, and the extension coordinate order.

The stored witness and its commitment do not have a subring packing type. A
later fold may open that flat base field witness with `EvaluationTrace`. The
same rule applies to a precommitted group and to a setup prefix. The commitment fixes the
polynomial layout and commitment matrices. The consuming fold fixes the
opening method and challenge family.

The current tensor EOR and Hachi terminal remain unchanged. The terminal
analysis below explains why a hidden subring packing tail would need another
opening argument. That analysis is a security boundary, not implementation
scope.

### Why later folds and the terminal stay unchanged

The current audited sparse challenge ladder starts at dimension 64. A subring
packing candidate using that smallest challenge therefore contains

```text
k s = 64       for fp128,
k s = 128      for fp64,
k s = 256      for fp32.
```

This is independent of `d_A`. Thus fp32 can keep `s = 64` and use `d_A = 256`,
while fp128 can keep `s = 64` with any admitted `d_A` in `{64, 128, 256}`. A
larger A ring can be useful when its secure rank falls enough. It can also be
worse when the secure rank falls by too little. The planner prices the exact
candidate under its existing objective.

The commitment slicing tables also show why the planner cannot always raise
`d_A`. A representative fp32 fold has 128 A input
columns at coefficient bound `2^20 - 1`. The audited Q32 table gives:

| A dimension | Secure rank | A image width |
|---:|---:|---:|
| 128 | 8 | 1,024 |
| 256 | 5 | 1,280 |
| 512 | 2 | 1,024 |

At dimension 256, rank 4 supports only 124 columns, so the fifth row makes that
candidate worse. Dimension 512 recovers the same A image width as dimension
128 and can carry the fp32 `s = 64` challenge. This example supports exact
pricing at levels 0 and 1. It does not justify a new search policy after level
1.

## Why this feature does not change the terminal

Suppose the prover decomposes the extension coordinate arrays into short digits
`ehat`, binds

```text
v_D = D ehat,
```

and then runs range and relation sum-checks without revealing `ehat`. Those
sum-checks end at a random point `r_sc` and require the claimed value
`ehat(r_sc)`. The verifier cannot derive that value from `v_D`.

This is not repaired by including `D ehat = v_D` inside the same sum-check. A
cheating prover can choose the last value after seeing `r_sc` and make the
univariate transcript close without committing to one global low-degree table.
Stage 1 has the same final-oracle problem. The D image gives computational
binding *if two short preimages are already known*; it is not by itself a
polynomial-opening protocol.

The evaluation trace method supplies the missing step by putting `ehat`
inside the recursively committed next witness and opening that commitment at
the Stage 2 point. The current terminal supplies the needed authentication
through the existing EOR and Hachi path.

A future hidden subring packing terminal would have to do all of the following:

1. bind the extension coordinates before the block fold challenges;
2. prove the bound witness is short;
3. authenticate every final witness evaluation requested by the range and
   relation proofs; and
4. enforce packing consistency and the exact `E`-valued opening on that
   same authenticated witness.

Until item 3 has a concrete proof, no future planner may price the 512 byte D
image and local sum check messages as a complete replacement for raw `e`. The
previous 1,744 byte estimate omitted this opening argument and is invalid.

Projecting the `k` extension coordinates to one coordinate does not supply the
missing opening either. For example, an `S` linear projection

```text
p_i = e_(i,0) + eta e_(i,1) + ... + eta^(k-1) e_(i,k-1)
```

preserves the packing consistency equation, but it does not preserve the
extension-field MLE opening equation. Multiplication by `eta` acts on the
subring coefficient index, while the extension-field opening weights act on
the MLE indices and mix the `k` field coordinates. In general,

```text
Open(eta e) != eta Open(e).
```

The projected coordinate therefore authenticates only a projection of the
packing consistency relation. It is not a complete terminal opening.

For one claim, six live blocks, extension degree four, challenge subring
dimension 64, and four byte base field elements, a transparent path would send

```text
1 * 6 * 4 * 64 * 4 = 6,144 bytes
```

of raw partial openings. The representative EOR and Hachi baseline sends 3,072
raw partial opening bytes and a 544 byte EOR proof, for 3,616 opening bytes.
This comparison explains why the current terminal remains in scope and a
subring packing terminal does not.

## Evaluation trace method

For each live claim and block pair, the current prover produces a partial opening
`e_i` as a full element of `R`, hence `d_A` base-field coordinates. At a
nonterminal fold, it gadget-decomposes those coordinates into `e_hat`, commits
the D image, and absorbs that payload before sampling the sparse fold
challenges `c_i`.

The two A-native relations are, schematically,

```text
sum_i c_i e_i = a z                         in R
[A G z_hat]_r = sum_i c_i [G t_hat_i]_r     in R, for every A row r.
```

Here `a` is the current ring opening multiplier and `G` denotes the applicable
gadget recomposition. The first equation is the consistency row. The remaining
equations are the A rows. Their polynomial representatives need quotient rows
for divisibility by `X^{d_A}+1`.

When `k > 1`, EOR first changes the opening claim and protocol point. The
evaluation-trace row then uses the ring-subfield trace/Galois construction to
connect the ring partials to the original scalar opening. The ring-switch
verifier evaluates every sparse challenge once at `X = alpha` and reuses that
value in the consistency and A contractions.

At levels 0 and 1, this specification changes four facts. A partial opening is
not a full `R` element. The consistency row uses the subring modulus. The
scalar row uses coefficient packing weights. The packing consistency and A
relations use different evaluations of the same challenge. Later folds and the
terminal use the evaluation trace method without these changes.

## Subring coefficient packing

### Canonical coefficient layout

Write one A ring element at block `i` and position `x` as

```text
F_(i,x)(X)
  = sum_(j < s) sum_(a < k h) f_(i,x,a,j) X^(a + k h j).
```

Thus `a` is the low coefficient index and `j` is the subring coefficient index.
The physical A ring coefficient index is exactly

```text
a + k h j.
```

The opening point's coefficient variables are split in the same order:

```text
r_pack     has log2(k h) coordinates and contracts a;
r_tail     has log2(s) coordinates and later contracts j.
```

The remaining existing axes are the position point `r_M` and block point
`r_B`. The point order and descriptor MUST bind this split. Prover and verifier
MUST derive it from `(k, d_A, s)`; it is not caller-selected layout metadata.

Let `w_basis(r, u)` denote the Boolean tensor-product weight selected by the
protocol's authenticated `BasisMode`:

```text
w_Lagrange(r, u) = eq(r, u);
w_Monomial(r, u) = product over bit positions ell with u_ell = 1 of r_ell.
```

Coefficient packing supports both existing opening bases. The basis changes
only these point-derived weights. It does not change the coefficient layout,
challenge subring, extension-coordinate basis, fold relation, or transcript
order. Every point axis below uses the same schedule-bound basis mode.

### Partial opening

For each live block, define

```text
e_i(Y)
  = sum_(x,a,j)
      w_basis(r_M, x) w_basis(r_pack, a) f_(i,x,a,j) Y^j
  in C = E[Y]/(Y^s+1).
```

No trace map is used. Each of the `s` coefficients of `e_i` is one ordinary
element of `E`. Fix the implementation's canonical `K`-basis
`beta_0, ..., beta_(k-1)` of `E` and write

```text
e_i(Y)
  = sum_(j < s) (sum_(t < k) beta_t e_(i,t,j)) Y^j.
```

The physical base-field layout is

```text
[claim][block][extension coordinate t][subring coefficient j].
```

It contains exactly `k s` base-field coordinates per claim/block. Backends MAY
temporarily use packed `E` values, but transcript encoding, gadget
decomposition, commitment input, range checking, and witness sizing MUST use
the canonical base-field layout above.

`Y` is the formal indeterminate of the extension opening ring, not an opening
point coordinate. The scalar opening below contracts the coefficient table
with `eq(r_tail,j)`. Ring switching later evaluates the extension opening ring
polynomial at `Y = alpha`; those are different operations with different
purposes. The scalar contraction uses `w_basis(r_tail,j)`.

### Scalar opening equation

For one polynomial with claimed opening `v`, the coefficient packing equation
is

```text
sum_i w_basis(r_B, i)
  sum_(j < s) w_basis(r_tail, j)
  sum_(t < k) beta_t e_(i,t,j)
  = v.
```

After gadget decomposition at opening basis `b_open`, this becomes

```text
sum_i w_basis(r_B, i)
  sum_(j < s) w_basis(r_tail, j)
  sum_(t < k) beta_t
  sum_l b_open^l e_hat_(i,l,t,j)
  = v.
```

This replaces the current evaluation trace formula for a subring coefficient
packing group. It remains one logical field level Stage 2 row, with the
existing claim batching coefficient applied outside the displayed equation.
It has no cyclotomic quotient. The implementation SHOULD name its prepared
weights as coefficient packing opening weights rather than trace weights.

This equation is evaluated from digits authenticated through the next recursive
witness. A grouped root has one schedule owned opening geometry per group, in
canonical root group order. The precommitted profile freezes the commitment
geometry. The root schedule separately freezes how that group is opened. The
verifier MUST NOT apply one group's coefficient layout or challenge subring
dimension to another group.

## Fold challenges and the two relation rings

### Challenge subring and embedding

Each fold challenge is sampled as

```text
c_i(Y) in S = K[Y]/(Y^s+1)
```

using the challenge configuration audited for dimension `s`, not the one for
`d_A`. The transcript sampler MUST bind the opening method, `s`, the challenge
configuration, group identity, live block count, and claim count before
expansion.

The same challenge is used in two rings:

```text
packing consistency relation: c_i(Y)        in S;
A relation:                   c_i(X^(k h))  in R.
```

No second challenge is sampled. The A ring form is the coefficient preserving
embedding of the subring form.

### Folded source and `S` linearity

For each position `x`, the A ring folded source is

```text
Z_x(X) = sum_i c_i(X^(k h)) F_(i,x)(X)    in R.
```

Let the coefficient packing map be

```text
L(F_i)(Y)
  = sum_(x,a,j)
      w_basis(r_M, x) w_basis(r_pack, a) f_(i,x,a,j) Y^j.
```

Because multiplying by `Y` advances only the `j` index, and because wrapping
`j = s` gives the same minus sign as `X^{d_A} = -1`, this map is `S`-linear:

```text
L(c_i(X^(k h)) F_i) = c_i(Y) L(F_i)    in C.
```

Therefore honest witnesses satisfy the packing consistency equation

```text
L(Z)(Y) = sum_i c_i(Y) e_i(Y)          in C.
```

This identity is the algebraic reason the subring coefficient packing method is
complete.

### Packing consistency quotient

Use ordinary polynomial representatives of degree below `s`. Define

```text
N_pack(Y) = sum_i c_i(Y) e_i(Y) - L(G z_hat)(Y).
```

The consistency equation is equivalent to the existence of one quotient

```text
Q_pack(Y) in E[Y],  degree(Q_pack) < s,

N_pack(Y) = (Y^s + 1) Q_pack(Y).
```

This is **one quotient over `C`**, not `k` independent relation rows. If

```text
Q_pack(Y) = sum_(j < s) (sum_(t < k) beta_t q_(t,j)) Y^j,
```

then its physical witness layout is the `k` extension coordinate arrays

```text
[extension coordinate t][subring coefficient j].
```

The quotient contributes `k s` base-field coordinates before its ordinary
gadget decomposition. Relation layout types MUST distinguish:

- one logical row selector;
- subring modulus dimension `s`; and
- physical coordinate width `k s`.

Treating the row as a base-field ring of dimension `k s` is incorrect: it would
use the modulus `Y^{k s}+1` and the denominator `alpha^{k s}+1`.

Since `L(G z_hat)` has degree below `s`, the quotient is just the high half of
the challenge products:

```text
Q_pack = high_s(sum_i c_i e_i).
```

The prover SHOULD compute this coordinatewise over the `k` extension
coordinates, sharing the sparse challenge positions across all coordinates.

### A rows remain in the A ring

The A rows remain in the A ring:

```text
[A G z_hat]_r
  = sum_i c_i(X^(k h)) [G t_hat_i]_r
  in R, for every A row r.
```

They keep `n_A` logical rows, A ring dimension `d_A`, the existing A matrix,
and the existing `t_hat` layout. Only the challenge support changes. Subring
coefficient position `j` appears at A ring position `k h j`.

The sparse challenge-times-`t` quotient can be viewed as `k h` independent
length-`s` lanes. An implementation MAY exploit those lanes, but the result
MUST match multiplication by the embedded challenge in `R` exactly.

### B and D rows

B slicing from PR #388 is unchanged. B continues to bind `t_hat` using one
physical matrix reused across its selected dyadic slices.

D binds the gadget digits of the partial openings at levels 0 and 1.
The first implementation requires

```text
d_D divides k s.
```

This avoids a second padding convention. The number of D-role subcolumns per
partial is `selected_partial_width / d_D`. D ranks, compression source widths,
and H compression geometry MUST be recomputed from that exact width. They MUST
NOT be obtained by scaling an old `d_A` price after rank selection.

### Relation witness and setup projection layouts

The Stage 2 relation witness and the Stage 3 setup projection use different
coefficient layouts. They MUST have distinct checked geometry types.

The Stage 2 relation coefficient block must factor every row modulus used in
the combined relation, including the subring modulus `s`. It may therefore be
smaller than each A, B, and D role dimension. This remains valid when `d_D`
divides `k s` but does not divide `s`. The `k` extension coordinates remain
semantically separate even when the flat address calculation groups their
coefficients into smaller common blocks.

The Stage 3 setup projection keeps its existing base derived from the A, B, and
D role dimensions. It does not shrink merely because Stage 2 includes a
packing consistency row. The implementation MUST prove that both checked layouts
produce the same flat coefficient weights for every shared A, B, and D scan.
It MUST NOT strengthen the subring packing admission rule from `d_D | k s` to
`d_D | s` to make the two layouts equal.

### Stage 2 uses a sum of structured linear terms

The current Stage 2 relation weights use one shared low coefficient factor.
This works because every native ring contribution uses consecutive powers of
`alpha` on that factor.

Coefficient packing adds a different weight on the same `z_hat` coefficients.
Write a physical A coefficient as

```text
p = a + k h j.
```

The native A and setup contribution uses

```text
alpha^(a + k h j).
```

The packing consistency contribution uses

```text
w_basis(r_pack, a) alpha^j.
```

For a general point and a general `alpha`, these two coefficient vectors are
not scalar multiples of each other. One shared low coefficient factor therefore
cannot represent their sum. Changing the coefficient order does not fix this.
It only changes which of the two vectors fails to share the factor.

Stage 2 MUST represent its linear weight as an ordered sum of structured terms.
Each term remains factored into a short coefficient vector and sparse lane
weights. The prover folds every term under the same sum check challenges. The
verifier evaluates the same terms at the final point without constructing a
full witness sized table.

The ordinary native term keeps all A, B, D, setup, and quotient contributions.
It also keeps the packed E and packing quotient contributions because their
subring coefficients use consecutive powers of `alpha`. Each packing group adds
one separate `z_hat` consistency term with the coefficient weights above. The
scalar opening term for that group uses

```text
gamma_claim w_basis(r_B, block) beta_t w_basis(r_tail, j) G_open[digit]
```

on the packed E digit at `[claim][block][t][j]`. Here `beta_t` is the canonical
extension basis element. This scalar opening is the claimed opening itself. It
does not use the EvaluationTrace trace map or EOR.

The protocol and its public types use the name sum of structured Stage 2 weight
terms. Every contribution remains factored, and a grouped opening can have more
than two terms, so a fixed term count would be misleading.

## Ring switching

### Two evaluations of each challenge

The ring-switch challenge `alpha` remains one element of `E`. The verifier MUST
derive two values from every subring challenge:

```text
subring_challenge_at_alpha = c_i(alpha)
  = sum_(j < s) c_(i,j) alpha^j;

embedded_challenge_at_alpha = c_i(alpha^(k h))
  = sum_(j < s) c_(i,j) alpha^(k h j).
```

The packing consistency row uses `subring_challenge_at_alpha`. Every A row uses
`embedded_challenge_at_alpha`.

The current single `c_alphas` cache MUST be split or typed so that these values
cannot be interchanged. Computing one and reusing it for both relations is a
protocol error except in the degenerate case `k h = 1`.

### Evaluating the packing consistency quotient

In subring coefficient packing method, the consistency check at `Y = alpha` is

```text
sum_i c_i(alpha) e_i(alpha)
  - L(G z_hat)(alpha)
  - (alpha^s + 1) Q_pack(alpha)
  = 0 in E.
```

For an extension coordinate representation,

```text
Q_pack(alpha)
  = sum_(t < k) beta_t sum_(j < s) q_(t,j) alpha^j.
```

This fixed basis combination does not need an additional random row-batching
challenge. Before evaluation, the `beta_t` form a `K`-basis, so a nonzero set of
coordinate polynomials gives one nonzero polynomial in `E[Y]`. Random `alpha`
then tests that single polynomial. Cancellation at a particular `alpha` is
already covered by its root bound.

The prepared relation point MUST use the subring powers
`1, alpha, ..., alpha^(s-1)` and denominator `alpha^s+1` for this row. It MUST
continue to use A ring powers and `alpha^{d_A}+1` for A rows.

### Cyclic and negacyclic products

For any degree-below-`s` product written as

```text
c(Y)e(Y) = L(Y) + Y^s H(Y),
```

the cyclic and negacyclic reductions are

```text
cyclic     = L + H,
negacyclic = L - H,
H          = (cyclic - negacyclic) / 2.
```

These identities still apply because the base characteristic is odd. They do
not, by themselves, make a new persistent cache useful. Current sparse
challenge products already compute only the high half. The
`SubringCoefficientPacking` method SHOULD extend that high half kernel to `k`
length `s` extension coordinate arrays.

The existing cyclic and negacyclic setup caches for `A z` remain useful and
remain in the A ring. This change MUST NOT replace them with extension field
setup matrices. D side cache widths change with the selected partial opening,
and setup and cache requirements MUST be derived from the selected opening
method.

## Soundness requirements

This section states the security obligations introduced by the new method. It
does not replace the existing MSIS binding proof for A, B, D, F, and H.

### Transcript order

For each subring coefficient packing group, the transcript MUST enforce this
dependency order:

1. Bind the instance, schedule, opening method, dimensions, coefficient layout, group
   layout, opening point, and original commitment.
2. Bind the complete D or H payload that commits to every base field coordinate
   of every `e_i`.
3. Sample the subring challenges `c_i` at dimension `s`.
4. Bind the challenge dependent folded witness, A/B data, packing consistency
   quotient, and next witness commitment.
5. Sample `alpha`, relation-row coefficients, and later sum-check challenges.

No coordinate of `e_i` may remain unbound when `c_i` is sampled. No coordinate
of `Q_pack`, `z_hat`, or `t_hat` may remain unbound when `alpha` is sampled.
Existing labels MAY be retained only when the serialized descriptor makes the
opening method and dimensions unambiguous. Otherwise new domain separated labels are
REQUIRED.

### Challenge entropy and unit differences

Every admitted subring challenge configuration MUST meet the configured
per-draw Fiat-Shamir min-entropy target. The complete proof MUST use Akita's
existing schedule error accounting, including the number of challenge
coordinates and the public random oracle query bound. A raw 128-bit family is
not by itself a 128-bit schedule.

Pairwise difference invertibility follows from the fixed prime profiles, not
from a separate challenge certificate. Let `ell` denote the cyclotomic split
count in LS18. This is distinct from Akita's extension degree `k`. LS18
Corollary 1.2 states that, for powers of two `s >= ell > 1`, if

```text
q = 2 ell + 1 mod 4 ell,
```

then every nonzero `delta` in `K[Y]/(Y^s+1)` with

```text
||delta||_inf < q^(1/ell) / sqrt(ell)
```

is a unit. Akita fixes these production field invariants:

| Modulus profile | Prime `q` | LS18 split count `ell` | Congruence |
|---|---:|---:|---|
| `Q32Offset99` | `2^32 - 99` | 2 | `q = 5 mod 8` |
| `Q64Offset59` | `2^64 - 59` | 2 | `q = 5 mod 8` |
| `Q128OffsetA7F7` | `2^128 - 2^32 + 22537` | 4 | `q = 9 mod 16` |

Every production sparse challenge family has `c_max <= 2`. Therefore every
coefficient of a challenge difference has magnitude at most
`2 c_max <= 4`. For each fixed profile above,

```text
2 c_max < q^(1/ell) / sqrt(ell),
```

so every nonzero difference of two production challenges is a unit in `S`.
This is a build time production field invariant. It is not schedule metadata or
candidate admission. The implementation MUST NOT add a per-candidate unit
certificate, registry, or planner gate. A future production field profile or
challenge family must establish the same LS18 condition during review.

If `delta(Y)` is a unit in `S`, then `delta(X^(k h))` is a unit in `R`: the
subring embedding maps the inverse of `delta` to an A ring inverse. The same
`delta` is also a unit after scalar extension to `C`.

### L2 norm under the subring embedding

Write an A ring coefficient index as `r + k h j`, where `r < k h` and
`j < s`. Multiplication by `c(X^(k h))` preserves `r`. For each fixed `r`, its
action is negacyclic multiplication by `c(Y)` on the `s` coefficients indexed
by `j`. A coefficient permutation therefore writes the A ring multiplication
matrix as a block diagonal matrix with `k h` identical subring blocks.

It follows that

```text
||M_(c(X^(k h)))||_2 = ||M_(c(Y))||_2.
```

Tests MUST compare this block reduction with direct A ring multiplication and
MUST cover the sign on negacyclic wraparound. The operator norm equality is
retained as an algebraic fact for later work. It does not admit an L2 subring
packing candidate at levels 0 or 1 in this feature.

### Forking extraction

The implemented transcript structure is specified normatively in
[`specs/transcript-grinding.md`](transcript-grinding.md). For a fixed group root
and the fixed shared fold-response nonce, coordinate `(claim, block)` is a
separate indexed random-oracle query. Reprogramming it leaves every other fold
coordinate and the live transcript state unchanged. Thus the implementation
supplies the coordinatewise CWSS transcripts below without relying
on extraction from full-vector forks.

Consider two accepting transcripts with the same pre-challenge commitments and
different challenge at one claim/block position. Let

```text
delta = c_j - c'_j.
```

After subtracting the accepted A relations,

```text
A (z - z') = delta(X^(k h)) t_j       in R^(n_A).
```

Under `SubringCoefficientPacking`, subtracting the accepted packing consistency
relations gives

```text
L(G(z - z')) = delta(Y) e_j           in C.
```

Because `delta` is a unit in both rings, these equations determine the opened
`t_j` and `e_j` from the fork. The existing B/F binding of `t_hat`, D/H binding
of `e_hat`, A binding of the folded source, range proof for all digit planes,
and quotient checks then give the same weak-opening/MSIS reduction as the
current fold.

The extractor takes one central accepting vector and one coordinatewise fork
for every claim and block position. The CWSS sum charges the support of every
coordinate. The online random-oracle reduction separately charges group-root
queries, indexed coordinate queries, and repeated roots caused by the shared
fold-response nonce. Root collisions, root prequeries, sum-check errors, the
`(2s - 1)/|E|` ring-switch error, and all A/B/D/F/H MSIS terms remain additive.
This accounting does not claim extraction from challenge entropy alone or from
two arbitrary full-vector forks.

### Ring-switch polynomial check

For an honest witness, the packing consistency numerator is identically zero
after the quotient is included. For a false witness, it is one nonzero polynomial over
`E` of degree at most `2s-1`. Sampling `alpha` after the quotient is bound
detects it except with probability at most

```text
(2s - 1) / |E|,
```

before accounting for the existing row batching and other sum-check errors.
The coordinate basis does not multiply this error by `k`: basis independence
shows that a nonzero coordinate vector is a nonzero coefficient in `E`, and
the verifier tests the resulting single `E` polynomial.

The final theorem statement for `SubringCoefficientPacking` MUST include:

- binding of the original and partial commitments;
- the subring challenge entropy condition and the fixed LS18 prime and
  shortness conditions;
- the subring polynomial root bound;
- the existing A/B/D/F/H MSIS assumptions;
- the existing range and sum-check soundness errors; and
- random-oracle forking loss for the central vector and every indexed
  coordinate fork.

## Historical EOR evidence at the commitment slicing baseline

### Exact current EOR formula

The current serialized EOR payload contains challenge-field partials and a
compressed degree-two sum-check. Let

```text
k  = [E:K],
P  = total number of root polynomials,
n0 = maximum root num_vars,
W1 = field-element length entering L1.
```

All current fp32/fp64 challenge fields serialize to 16 bytes. When EOR is
enabled, the exact header-free payload is

```text
L0 bytes = 16 * (k P + 2 * (n0 - log2(k)));

L1 bytes = 16 * (k + 2 * (ceil(log2(W1)) - log2(k))).
```

For one polynomial and `k` equal to 2 or 4, these simplify to

```text
L0 bytes = 32 * n0;
L1 bytes = 32 * ceil(log2(W1)).
```

These formulas are the canonical
`extension_opening_reduction_level_bytes` calculation, which is tested against
the serialized EOR payload. Removing EOR does not remove the fold-response
entry in the proof-level packed nonce stream; the numbers below count only
bytes that actually disappear with the EOR proof.

### Complete current catalog census

The table expands every fp32 and fp64 row at the commitment slicing baseline
that later merged in PR 388. It applies the canonical sizing function.
`Current proof` is that planner's exact payload estimate. `L0+L1` is the gross
saving if those two EOR payloads are removed while everything else remains
fixed. The implementation must regenerate these numbers on the selective L2
base before using them as current performance evidence.

| Catalog row | Current proof | L0 EOR | L1 EOR | L0+L1 | Current proof share |
|---|---:|---:|---:|---:|---:|
| fp32 dense, nv20, P=1 | 79,840 | 640 | 672 | 1,312 | 1.64% |
| fp32 dense, nv26, P=1 | 83,172 | 832 | 768 | 1,600 | 1.92% |
| fp32 one-hot, nv14, P=1 | 66,484 | 448 | 544 | 992 | 1.49% |
| fp32 one-hot, nv16, P=1 | 67,624 | 512 | 544 | 1,056 | 1.56% |
| fp32 one-hot, nv16, P=2 | 67,688 | 576 | 544 | 1,120 | 1.65% |
| fp32 one-hot, nv20, P=1 | 74,572 | 640 | 608 | 1,248 | 1.67% |
| fp32 one-hot, nv20, P=2, two groups | 77,740 | 704 | 608 | 1,312 | 1.69% |
| fp32 one-hot, nv28, P=1 | 82,388 | 896 | 736 | 1,632 | 1.98% |
| fp32 one-hot, nv30, P=1 | 83,300 | 960 | 768 | 1,728 | 2.07% |
| fp64 dense, nv14, P=1 | 79,976 | 448 | 576 | 1,024 | 1.28% |
| fp64 dense, nv20, P=1 | 86,160 | 640 | 704 | 1,344 | 1.56% |
| fp64 dense, nv26, P=1 | 88,900 | 832 | 800 | 1,632 | 1.84% |
| fp64 one-hot, nv28, P=1 | 87,232 | 896 | 736 | 1,632 | 1.87% |
| fp64 one-hot, nv30, P=1 | 87,568 | 960 | 768 | 1,728 | 1.97% |

At that baseline, the catalogs spend 992 to 1,728 bytes on level 0 and level 1
EOR, or 1.28% to 2.07% of the complete proof estimate. These are historical
gross savings. They are not the final selective L2 planner result.
`SubringCoefficientPacking` also changes the next witness, ranks, sum check
domains, and selected schedule.

### Base field coordinate savings

Before digits, one partial opening and its consistency quotient each change from

```text
d_A base-field coordinates
```

to

```text
k s = d_A / h base-field coordinates.
```

The exact reduction factor is `h`. For `B` live claim/block pairs and opening
digit depth `delta_open`, the D input changes from

```text
B * (d_A / d_D) * delta_open    D-ring elements
```

to

```text
B * (k s / d_D) * delta_open    D-ring elements.
```

The packing consistency quotient's base field coordinate count changes by the
same factor before quotient digit decomposition. Compression output payloads may have fixed
sizes, so the planner MUST propagate the shorter witness through ranks,
compression chains, relation domains, successor dimensions, and proof sizing;
it MUST NOT report `h` as an automatic proof-size factor.

### Concrete fp32, `d_A = 1024`, `k = 4`

The candidates induced by the current production challenge ladder expose the
main tradeoff. The fixed fp32 prime and every family in this table satisfy the
LS18 condition above.

| `s` | `h` | `k h` subring embedding stride | coordinates per partial | production sparse family at `s` | challenge `l1` mass |
|---:|---:|---:|---:|---|---:|
| 64 | 4 | 16 | 256 | 31 coefficients in `±1`, 10 in `±2` | 51 |
| 128 | 2 | 8 | 512 | 31 coefficients in `±1` | 31 |
| 256 | 1 | 4 | 1,024 | 23 coefficients in `±1` | 23 |

For the middle choice, coefficient index `a + 8j` maps subring coefficient
position `j` to A ring position `8j`. Every partial and packing consistency
quotient uses four length 128 extension coordinate arrays, or 512 base field
coordinates total, instead of 1,024. The ring switch verifier computes
`c(alpha)` for the packing consistency row and `c(alpha^8)` for the A rows.

The `s=64` choice gives a fourfold smaller partial than `s=256`, but it uses a
heavier subring challenge. At levels 0 and 1 it changes the digit count, D
width, A response bound, and successor witness. The planner must compare the
complete candidate rather than assuming that the smallest `s` wins.

## Planner contract

### The consuming fold owns the opening plan

Each opening group has one schedule owned opening plan. The exact type name may
differ, but it must express the following choice.

```rust
enum OpeningMethod {
    EvaluationTrace,
    SubringCoefficientPacking {
        challenge_subring_dimension: usize,
    },
}
```

`EvaluationTrace` uses full A ring partial openings and evaluation trace
weights. It first runs EOR when `k > 1`. `SubringCoefficientPacking` uses the
coefficient packing map and draws the fold challenge from the challenge
subring.

The methods coincide algebraically when `k = 1`, `h = 1`, and `s = d_A`. In
that case `S = C = R`, up to the indeterminate name, the subring embedding is
the identity, and the degree one trace is the identity. The schedule and
transcript still bind one method tag. Generated levels 0 and 1 use
`SubringCoefficientPacking` when a complete compatible assignment exists;
later folds use `EvaluationTrace`. The planner MUST NOT enumerate both methods
as separate candidates at one level merely because this overlap exists.

The implementation expresses this ownership split with the following shape.

```rust
struct GroupOpeningPlan {
    opening_method: OpeningMethod,
    fold_challenge_config: SparseChallengeConfig,
    log_basis_open: u32,
    num_digits_open: usize,
    num_digits_fold: usize,
}

struct GroupOpenPhaseParams {
    profile: GroupCommitPhaseParams,
    opening: GroupOpeningPlan,
    setup_natural_len: Option<usize>,
}
```

There is one canonical owner for each field. `GroupCommitPhaseParams` owns the
frozen commit phase identity. `GroupOpeningPlan` owns the consuming fold's
opening policy.

The opening method and challenge subring dimension are protocol data. Runtime
schedules, generated rows, canonical descriptors, catalog identity, proof size
reports, and the transcript MUST bind them. The terminal always uses
`EvaluationTrace` in this feature.

A scalar recursive fold has one opening group. A grouped root stores one entry
per group in `OpeningClaimsLayout::root_group_order`. All level 0 entries use
`SubringCoefficientPacking`, but their `d_A`, `s`, and derived `h` may differ.
The planner does not pad them to one level wide `s`.

The consuming fold owns this plan. The commitment profile does not. A
`GroupCommitPhaseParams` or setup prefix commitment fixes the physical source
encoding, the source polynomial layout, the A and B matrices, and the
commitment bytes. It does not fix whether a later fold uses `EvaluationTrace`
or `SubringCoefficientPacking` when that opening method supports the frozen
source encoding.

The physical source encoding is separate commitment metadata:

```rust
enum CommittedSourceEncoding {
    CanonicalCoefficientTable,
    TensorSubfieldProjection { extension_degree: usize },
}
```

Coefficient packing in this feature supports `CanonicalCoefficientTable`.
The existing EOR path may use `TensorSubfieldProjection`. These encodings are
different commitment identities because the tensor subfield projection does
not commute with the coefficient packing challenge action
`Y -> X^(k h)`. The implementation MUST reject coefficient packing over a
tensor projected source before transcript mutation. It MUST NOT choose a
physical encoding by reading `OpeningMethod` after a commitment profile has
already been fixed.

The schedule fixes the opening method, `s`, and the sparse challenge family
before proving begins. The transcript draws the actual sparse challenge at
runtime after the D or H payload is bound. Neither the runtime draw nor a
change of opening method changes the bytes of a commitment with a fixed source
encoding. A producer may create a distinct commitment under another source
encoding, but that encoding is then part of commitment identity.

The implementation MUST keep commitment identity separate from opening
admission data. In particular, a setup prefix registry key MUST NOT create two
different commitments only because two consuming schedules choose different
opening methods for the same content, source encoding, and matrix geometry. A
composite schedule object may contain both kinds of data, but its commitment
identity and opening plan must remain separate fields with separate
validation.

The consuming fold validates the frozen A and B matrices against its selected
challenge bounds. A commitment that is too narrow for the selected challenge
is not admissible even though its bytes do not depend on the challenge draw.

`h` and the subring embedding stride are derived from `(k, d_A, s)`. They MUST
NOT be serialized as independent choices.

### Transition after level 1

A subring packing fold outputs one canonical flat base field witness. Its
partial opening digits and packing consistency quotient coordinates are
ordinary fields in that witness. Their earlier meaning is checked while
verifying the producing fold.
The next fold commits to and opens the flat witness selected by its own
schedule. It does not reuse the previous fold's challenge subring.

For example, suppose level 0 and level 1 both offload setup contributions.
Level 1 consumes the first setup prefix with its scheduled
`SubringCoefficientPacking` opening. It then produces a flat witness and a
second setup prefix for level 2. Level 2 opens both its witness and that second
prefix with `EvaluationTrace`. No conversion is needed. The level 2 schedule
binds the method and its conditional EOR, and it checks the frozen commitment
geometry against the level 2 challenge family.

Whether level 2 runs EOR depends on the field configuration. The fp128 presets
have extension degree one, so evaluation trace opens the prefix and witness
without EOR. The fp32 and fp64 presets use extension degrees four and two. In
those presets, one EOR batches a dense term for the setup prefix with the
recursive suffix term. This is not a limit on the number of setup-offloading
levels. A later fold may create another prefix when the complete schedule and
setup objective select that edge.

The selective L2 branch currently rejects setup prefix derivation from an A
commitment that has no SIS table key. Subring packing folds use Linf, so they do not
enter that L2 path. The existing restriction remains unchanged for later
evaluation trace folds that consider L2.

### Candidate admission

A subring packing candidate is admitted only when all of the following
conditions hold.

1. `k`, `d_A`, and `s` are powers of two.
2. `k s` divides `d_A`.
3. `h = d_A/(k s)` is positive.
4. An audited sparse challenge configuration exists at dimension `s`.
5. That configuration meets the existing challenge entropy and total schedule
   error requirements.
6. The field and ring dispatcher supports the A ring dimension and subring
   packing kernels.
7. `d_D` divides `k s` in the first implementation.
8. Every D or H compression source satisfies its current byte cap.
9. A, B, and D matrix widths have secure ranks at the exact candidate bounds.
10. The next witness and relation address geometry can be represented without
    unchecked padding or allocation.

The planner searches the existing production challenge dimensions
`{64, 128, 256, 512, 1024, 2048}`. It keeps only values satisfying
`k s | d_A`. Thus `s` is at least 64 and at most `d_A/k`. The algebra and
kernels must be generic over checked `s`. The planner MUST NOT scan arbitrary
integers or create a challenge family during schedule search.

For `k = 1`, EOR is not valid and contributes no bytes. When a complete
packing assignment exists, the planner still uses `SubringCoefficientPacking`
at levels 0 and 1. It may choose `s < d_A` to reduce partial and quotient
coordinates. The overlap with `EvaluationTrace` is `s = d_A` and `h = 1` when
that `s` exists in the audited registry.

### Linf security route and the preserved L2 identity

Every subring packing candidate at levels 0 and 1 uses the Linf A security
route. The planner applies PR 369's current Linf derivation to the exact
subring coefficient packing geometry. The root uses the current root policy. A
recursive subring packing fold uses the typed response model when its profile
provides one. Otherwise it uses the current universal Linf bound. The resulting
bound determines the secure A rank at the A ring dimension `d_A`.

Multiplication by the embedded challenge `c(X^(k h))` preserves each residue
class modulo `k h`. After a coefficient permutation, the A ring
multiplication map is a direct sum of `k h` copies of multiplication by
`c(Y)` in the challenge subring. Its L2 operator norm is therefore exactly the
subring operator norm at dimension `s`. Physical response length, SIS rank,
and A geometry still depend on `d_A` and the complete response shape.

The implementation MUST encode this reduction and compare it with a direct
A ring multiplication reference for every admitted geometry. The identity is
useful for later work, but this feature MUST NOT use it to select an L2 route
at level 0 or 1.

### Level policy

The policy applies to every field profile.

1. Absolute nonterminal levels 0 and 1 first enumerate complete subring packing
   assignments across every group and admissible A dimension.
2. Absolute nonterminal levels 2 and later use `EvaluationTrace`.
3. The terminal uses `EvaluationTrace` and Hachi, with EOR where it applies.
4. A schedule with only root and terminal uses `SubringCoefficientPacking`
   only at root.
5. A schedule with root, one recursive fold, and terminal uses
   `SubringCoefficientPacking` at both nonterminal folds.
6. The planner does not add a fold to reach the two level subring packing
   scope.
7. If the complete level has no statically feasible packing assignment, the
   schedule is unsupported. Frozen tensor-encoded root profiles are not
   reinterpreted as canonical coefficient commitments.

A precommitted root group freezes its commitment profile. Its root opening plan
may choose any certified `s` compatible with the frozen `d_A` and security
bounds. Level 1 sees the root output as one flat recursive witness and chooses
one new `(d_A, s)` for that witness. It does not preserve one subring
coefficient packing geometry per original root group.

A setup prefix is another precommitted group at the fold that consumes it. It
uses that consuming fold's opening method. A prefix consumed at level 1 uses
`SubringCoefficientPacking`. A prefix consumed at level 2 or later uses
`EvaluationTrace`.

### Independent challenge subring and A ring dimensions

The challenge subring dimension `s` controls challenge entropy and the `k s`
partial and quotient widths. The A ring dimension `d_A` controls the A rows.
Once `s` is fixed, changing `d_A` changes the packing factor and A ring
geometry. It does not enlarge the partial opening or quotient.

The planner enumerates every admitted `(d_A, s)` pair at levels 0 and 1 under
the incoming dimension ceiling. It preserves the selective L2 branch's current
dimension policy and uniform suffix after level 1. This feature does not add A
dimension search beyond level 1. A terminal reached inside that adaptive
prefix may use an admitted adaptive dimension.

B and D remain independent role dimensions and may stay below `d_A`. The
planner does not raise them merely because a larger A ring has a lower rank.
It does not add a semantic preference for smaller `s` or larger `d_A`.

### Objective and exact pricing

Adaptive direct catalogs retain `MinFirstDirectSetupThenPayloadV2`:
first-direct padded setup capacity, proof payload, exact total setup field
elements, root output-witness length, and the canonical descriptor. Recursive
catalogs use `MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3`, which first
compares the next-power-of-two capacity covering the total setup envelope.
Exact setup differences within one recursive capacity bucket are tolerated
before comparing first-direct capacity, proof payload, and first-direct
output-witness length. A numeric tie then goes directly to the canonical descriptor.
No objective component for `s`, `d_A`, rank, fold count, or prover time is
added.

For every subring packing candidate, the planner MUST recompute at least the
following values.

1. Removed EOR bytes at levels 0 and 1.
2. Partial coordinate count and opening gadget depth.
3. D input width, secure D rank, H source, and compression geometry.
4. Packing consistency quotient coordinates and quotient digit count.
5. Sparse challenge `l1`, `l2`, and `linf` bounds at `s`.
6. The Linf folded `z` bound and secure A rank under PR 369's applicable root
   or recursive response model.
7. `t_hat`, B input width, B slicing candidates, and F compression geometry.
8. Logical row count, physical row dimensions, relation address length, and
   sum check rounds.
9. Setup prefix eligibility and setup matrix field elements.
10. Successor witness length and the complete suffix cost.

The report MUST show `opening_method`, `challenge_subring_dimension`,
`a_ring_dimension`, `packing_factor`, `partial_base_field_width`,
`packing_quotient_base_field_width`, the secure A route, first-direct padded
setup capacity, total setup field elements, proof bytes, and successor witness
length. It must list every catalog regression against the selective L2
baseline. A minor row may regress when the target presets and the overall
catalog improve. Per row nonregression is not an acceptance condition.

### Selective L2 base-to-head catalog evidence

The checked revision report compares base
`5a4f72bce3920ecb753751187cd2eaab3f915b8b` with this branch. The immutable
[base snapshot](evidence/subring-coefficient-packing/base.tsv),
[head snapshot](evidence/subring-coefficient-packing/head.tsv), and
[complete comparison](evidence/subring-coefficient-packing/comparison.tsv)
contain a 68-row base and an 83-row head across all 13 generated families. The
logical-key union has 16 additions and one intentional removal. The additions
are three bounded-dense rows, one grouped one-hot row with a bounded-dense
precommit, four fp128 grouped nv34 rows, and eight fp32/fp64 scalar coverage
rows. The removed row is the unsupported fp128 dense `final=44:1` catalog
stress point. All 67 retained rows change exact schedule identity, as expected
for this breaking protocol and planner-policy cutover.

Across the 67 retained logical keys, setup improves on 45 rows, is equal on
six, and regresses on 16. Its sum falls from 7,022,341,504 to 3,483,724,544
field elements, a 50.39% reduction. Proof payload improves on 66 rows and
regresses on one. Its sum falls from 4,965,430 to 4,433,327 bytes, a 10.72%
reduction. Fourteen rows use fewer fold levels, 41 retain their count, and 12
use more levels.

The first-direct setup capacity improves on 64 retained rows, is equal on three,
and regresses on none. Its sum falls from 10,278,666,240 to 5,087,879,168 field
elements, a 50.50% reduction. These baseline values were
reconstructed at the pinned base commit from each exact materialized schedule;
the comparator treats a missing coordinate as drift rather than a wildcard.

This closes the missing-row, objective, and comparison-report evidence gaps.
The retained catalog improves in aggregate under both setup coordinates and
proof payload. The two proof regressions remain explicit in the per-row review
data; setup-primary selection does not imply per-row proof nonregression.

At the checked head, the fp32 dense nv20 adaptive direct rows use
`MinFirstDirectSetupThenPayloadV2`. Their first-direct padded
capacities are 131,072 and 262,144 fields. Their six-level schedules use
458,752 and 524,288 total setup fields and produce 62,447 and 63,254 proof
bytes across six fold levels. The checked decision is recorded in the
[catalog evidence note](evidence/subring-coefficient-packing/README.md#current-fp32-nv20-adaptive-objective).

### B slicing interaction

Subring packing search and B slicing both apply at levels 0 and 1. Subring
coefficient packing shortens `e`, the D or H source, and the consistency
quotient. B slicing shortens the
physical B matrix and may add logical rows.

B width depends on `t_hat`. The `t_hat` geometry depends on the subring
challenge norm, the selected A dimension, and secure A rank. Candidate
construction MUST use this order.

1. Choose `d_A`, `s`, and the subring challenge family.
2. Derive the Linf response bound, A rank, and `t_hat` through PR 369's current
   root or recursive response model.
3. Enumerate and prune the bounded B slice counts from PR 388.
4. Derive D or H and B or F compression plans.
5. Construct the next witness and score the complete suffix.

The planner MUST NOT choose a B slice count from geometry computed before the
subring packing candidate. The bounded slice set `{1, 2, 4, 8}` and its current
local pruning rule remain unchanged.

### Search control

The planner MUST keep the search bounded in the following ways.

1. Derive `h` from `(k, d_A, s)`.
2. Use one canonical coefficient layout and embedding.
3. Use the fixed audited list of `s` values.
4. Apply geometry, challenge entropy, dispatcher support, and divisibility
   admission before rank lookup.
5. Search subring packing candidates only at levels 0 and 1.
6. Apply B slicing only after A and `t_hat` geometry is known.
7. Keep the existing deterministic frontier and memo state objective.
8. Keep the current uniform suffix after the adaptive prefix.
9. Compare the pruned result with an unpruned oracle on small fixtures.

## Implementation boundaries

### `akita-types`

- Add the schedule owned opening method and checked
  `SubringCoefficientPackingGeometry`.
- Add canonical descriptor encoding for the opening method and
  `challenge_subring_dimension`.
- Represent the packing consistency row as one logical row with challenge
  subring dimension `s` and extension coordinate width `k`.
- Extend witness and address layouts for `k s` partial and quotient coordinates.
- Give the Stage 2 relation witness and Stage 3 setup projection distinct
  checked layout types. Prove their flat A, B, and D weight equivalence.
- Separate precommitted commitment identity from the opening plan selected by
  the consuming fold. Apply the same split to setup prefix slot identity.
- Generalize proof size and successor witness sizing at levels 0 and 1.
- Keep malformed verifier inputs on typed `AkitaError` or
  `SerializationError` paths.

### `akita-challenges`

- Reuse the signed-sparse sampler at dimension `s`.
- Reuse the existing challenge entropy audit. Do not add a separate unit
  difference certificate or subring embedding L2 admission path.
- Keep the LS18 prime congruence and shortness condition with the fixed
  production field profiles rather than making it planner state.
- Bind the challenge subring dimension and opening method in the draw domain.
- Split the packing draw into fixed batches of 128 challenges. Derive each
  batch from the transcript seed, the packing batch domain, and its canonical
  batch index. Keep challenge order independent of the worker count.
- Do not create a second A ring challenge draw.

### `akita-prover`

- Add dense, one-hot, and recursive kernels that compute the `s` extension
  coefficients directly from the canonical coefficient split.
- Decompose the resulting `k s` base-field coordinates into D-role `e_hat`.
- Compute `Q_pack` with shared challenge high half accumulation over `k`
  extension coordinate arrays.
- Keep A quotients over `d_A` with challenges embedded at stride `k h`.
- Split subring and embedded challenge evaluations in ring switch preparation.
- Replace the trace specific Stage 2 term with the coefficient packing opening
  term at levels 0 and 1.
- Use the `EvaluationTrace` prover path at later folds and at the terminal.
- Preserve current cyclic/negacyclic A setup caches.

### `akita-verifier`

- Reconstruct the coefficient packing scalar opening row from `r_B`, `r_tail`, the canonical
  extension basis, and opening gadget weights.
- Evaluate the packing consistency quotient with denominator `alpha^s+1`.
- For subring coefficient packing groups, evaluate the same challenge at `alpha` and
  `alpha^(k h)` for its two roles.
- Use the `EvaluationTrace` verifier path at later folds and at the terminal.
- Reject opening method, dimension, and layout mismatches before allocation.
- Preserve the no-panic verifier contract.

### `akita-planner`, `akita-schedules`, and `akita-config`

- Add bounded subring packing candidates and the level policy above.
- Extend the existing two level adaptive search with the bounded `s` registry.
  Keep the current uniform suffix after that prefix.
- Recompute exact ranks, setup, compression, proof bytes, and successors.
- Regenerate every affected catalog on top of the selective L2 branch.
- Add report columns for opening method, challenge subring dimension, packing
  factor, security route, first-direct padded capacity, total setup field
  elements, and proof bytes.
- Preserve the shared first-direct, proof-payload, total-setup, output-witness
  objective without adding an `s` or `d_A` objective component.
- Report every catalog regression. Do not reject the feature only because a
  minor row regresses.

## Acceptance criteria

### Algebra and completeness

- [x] `SubringCoefficientPackingGeometry` accepts exactly supported
      `(k,d_A,s)` triples and derives `h` and stride without independent
      metadata.
- [x] Coefficient index `a + k h j` round-trips for every supported geometry.
- [x] Dense, one-hot, and recursive partial openings match a flat MLE reference.
- [x] `L(c(X^(k h))F) = c(Y)L(F)` holds against a naive reference for random
      small fixtures and every supported field tier.
- [x] A ring multiplication by `c(X^(k h))` matches `k h` permuted subring
      blocks, including negacyclic wraparound, and preserves the L2 operator
      norm.
- [x] Coefficient packing scalar opening weights reproduce the claimed opening, including
      partial final blocks, multiple polynomials, and multiple groups.
- [x] `Q_pack = high_s(sum_i c_i e_i)` satisfies the full ordinary-polynomial
      divisibility identity in `E[Y]`.
- [x] The `k` extension coordinate arrays evaluate to the same `E` value as a
      packed-extension reference.
- [x] Honest subring coefficient packing proofs verify at levels 0 and 1 for every supported field
      tier.
- [x] A level 2 fold opens the flat output of a subring packing level 1 fold with the
      evaluation trace method without conversion or subring metadata reuse.
- [x] Stage 2 accepts `d_D | k s` when `d_D` does not divide `s`, and its flat
      A, B, and D weights match the separate Stage 3 setup projection.

### Soundness and transcript

- [x] Every admitted challenge family meets Akita's per-draw entropy condition,
      and the complete proof meets the existing total schedule error budget.
- [x] The fixed production field table above remains part of security review,
      not schedule metadata or candidate admission.
- [x] Generated and supplied subring packing schedules at levels 0 and 1 use the Linf A
      security route and cannot select physical L2.
- [x] Nonterminal partial D or H payloads are transcript bound before their
      subring challenge draws.
- [x] Packing consistency quotient and next-witness data are bound before `alpha`.
- [x] Opening method, `s`, challenge configuration, coefficient layout, and group identity
      change the descriptor/transcript bytes.
- [x] The verifier computes distinct `c(alpha)` and `c(alpha^(k h))` values and
      tests fail when either is substituted for the other.
- [x] A nonzero extension coordinate numerator is detected by the packed
      `E[Y]` ring-switch oracle.
- [x] The subring coefficient packing theorem adds no `1/|K|` coordinate projection term.
- [x] Multi-fork extraction and total soundness-error accounting are documented
      alongside the implementation.
- [x] Malformed opening method, dimension, or coordinate counts return typed errors without
      panic or unbounded allocation.

### Planner and sizing

- [x] Generated schedules for every field tier use `SubringCoefficientPacking`
      at existing nonterminal levels 0 and 1. Rows without a complete compatible
      packing assignment are unsupported.
- [x] Extension field packing schedules contain no EOR at levels 0 and 1.
- [x] Later folds and the terminal retain `EvaluationTrace`.
- [x] Short schedule tests cover root to terminal and root to one recursive
      fold to terminal without inserting another fold.
- [x] The planner searches every admitted `(d_A, s)` pair only inside the two
      level adaptive prefix and keeps the current uniform suffix.
- [x] Adaptive direct catalogs minimize first-direct setup capacity, proof
      payload, exact total setup, root output-witness length, and the canonical
      descriptor. Recursive catalogs minimize padded total setup-envelope
      capacity, first-direct setup capacity, proof payload, first-direct
      output-witness length, and then the canonical descriptor. The objective has no
      explicit `s`, `d_A`, or fold-count component.
- [x] `d_D` not dividing the selected native or hidden-digit width rejects
      before matrix/rank construction.
- [x] Exact D/H and A/B/F ranks are recomputed from subring coefficient packing geometry and norms.
- [x] PR 388 B slicing is enumerated only after subring challenge derived A and `t`
      geometry.
- [x] Bounded DP output matches an unpruned oracle on small search fixtures.
- [x] Reports reproduce the historical EOR census and show new results on the
      selective L2 baseline.
- [x] At least one fp32 and one fp64 production row demonstrate the expected
      L0/L1 EOR removal in actual serialized proof breakdowns.
- [x] The one removed fp128 dense nv44 stress row and every retained-row
      regression are explicit in the checked base-to-head report.
- [x] Both checked snapshots contain the exact padded first-direct setup
      capacity, and the complete comparison reports it for every retained row.
- [x] The checked head contains 83 rows across 13 families. The complete
      comparison contains 16 additions, one removal, and 67 changed rows.
- [x] Subring packing setup prefix edges use Linf. Reports for later evaluation trace
      edges preserve PR 369's existing L2 admission result.

### Precommitment and setup prefixes

- [x] Commitment identity excludes the consuming opening method, `s`, and
      challenge draw.
- [x] Schedule and transcript identity include the consuming opening plan.
- [x] The same frozen precommitted commitment can be admitted under
      `SubringCoefficientPacking` or `EvaluationTrace` when its matrices meet
      the selected method's security
      checks.
- [x] A setup prefix consumed at level 1 uses `SubringCoefficientPacking`. A
      setup prefix consumed at level 2 uses `EvaluationTrace`.
- [x] A two level recursive offloading test verifies the transition from a
      subring packing consumer at level 1 to an evaluation trace consumer at level 2.
- [x] Changing the opening plan does not duplicate identical setup prefix
      commitment bytes or alter the committed polynomial layout.

### Performance and caches

- [x] Partial opening and packing consistency quotient allocations contain exactly `k s`
      base-field coordinates per semantic item before digits.
- [x] Packing quotient high half construction does not materialize full extension field
      convolution tables.
- [x] Existing A cyclic/negacyclic setup caches remain shared and correct.
- [x] D/H cache requirements use the selected partial opening width and do not retain
      old `d_A`-wide buffers.
- [x] Profile output records prover time, verifier time, peak memory, setup
      field elements, proof bytes, and per level witness sizes against the
      selective L2 baseline.
- [x] A packed-`E` verifier Horner loop is adopted only if it beats the canonical
      extension coordinate loop without changing bytes or arithmetic results.

### Repository validation

- [x] Generated schedule tables are clean after regeneration.
- [x] Focused algebra, prover, verifier, planner, and catalog tests pass.
- [x] All required feature-graph Clippy jobs pass.
- [x] `./scripts/check-doc-guardrails.sh` passes.

## Non-goals

- No packing consistency relation after absolute fold level 1.
- No subring packing terminal or change to the evaluation trace and Hachi
  terminal protocol.
- No A dimension search beyond the current two level adaptive prefix.
- No arbitrary integer challenge subring dimensions or coefficient layout search.
- No second independent challenge for the A rows.
- No pure extension-field commitment or setup matrix.
- No claim that the smallest challenge subring is always optimal.
- No global objective component based on `s` or `d_A`.
- No requirement that every catalog row improve. Minor regressions are allowed
  and must be reported.
- No claim that flattening a `k s` extension opening ring value and adding four
  public rows yields a sound `s` coordinate prechallenge opening.
- No claim that a D image plus local sum-checks forms a complete opening proof;
  short-preimage binding does not authenticate a final multilinear evaluation.
- No use of the rejected ring-valued interpolation described below: its
  opening operators are only `K`-linear and do not preserve degree over `S`.
- No change to PR 388's B slice count set, dyadic partition, or 8 KiB
  compression-source limit.
- No backward-compatible decoding of schedules or proofs that predate this
  opening method. Akita remains in development; affected catalogs and descriptors are
  regenerated rather than aliased.

## Documentation outcome

The implementation folded the stable protocol prose into:

- `book/src/how/proving/root-fold-ring-switch.md` for the packing consistency relation and
  two challenge evaluations;
- `book/src/how/proving/extension-opening-reduction.md` for L0/L1 removal and
  the unchanged later fold and terminal boundary;
- `book/src/how/configuration.md` for planner candidates, setup prefix
  ownership, and reports;
- `book/src/foundations/rings-and-fields.md` for the subring embedding and unit
  condition; and
- `book/src/how/security.md` for the forking and polynomial-root arguments.

Those chapters contain the current durable explanation. This design record is
active again under the workflow in [`specs/PRUNING.md`](PRUNING.md) until the
open review blockers are resolved.

## References

- [Lyubashevsky and Seiler, EUROCRYPT 2018](https://doi.org/10.1007/978-3-319-78381-9_8),
  Corollary 1.2 for partial splitting and short invertibility.
- [B commitment slicing](archive/2026-Q3/commitment-slicing.md), PR 388 baseline and B planner
  interaction.
- [Selective L2 fold sizing](https://github.com/LayerZero-Labs/akita/pull/369),
  planner objective, response sizing, and operator norm certificates.
- [Extension-field opening batching](archive/2026-Q3/extension-field-opening-batching.md),
  tensor EOR and the transformed-commitment soundness boundary.
- [Root fold and ring switch](../book/src/how/proving/root-fold-ring-switch.md),
  current production sparse families and role dimensions.
- [EOR streamed prover](archive/2026-Q3/eor-streamed-prover.md), historical EOR prover path and
  performance context.
- [`crates/akita-types/src/layout/proof_size.rs`](../crates/akita-types/src/layout/proof_size.rs),
  canonical current EOR byte formula.
- [`crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`](../crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs),
  current high-half, consistency, and A quotient construction.
- [`crates/akita-prover/src/protocol/ring_switch/relation_weights.rs`](../crates/akita-prover/src/protocol/ring_switch/relation_weights.rs),
  current structured relation weights and challenge reuse.
- [`crates/akita-verifier/src/protocol/ring_switch.rs`](../crates/akita-verifier/src/protocol/ring_switch.rs),
  current `c_alphas` preparation.
- [`crates/akita-verifier/src/protocol/evaluation_trace.rs`](../crates/akita-verifier/src/protocol/evaluation_trace.rs),
  current trace-based scalar-opening contraction.
