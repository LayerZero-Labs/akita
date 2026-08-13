# Spec: coefficient-carrier openings and subring fold challenges

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-10 |
| Status | proposed |
| PR | [#394](https://github.com/LayerZero-Labs/akita/pull/394) |
| Supersedes | The assumption that every extension-field opening first uses extension-opening reduction |
| Superseded-by | |
| Book-chapter | book/src/how/proving/root-fold-ring-switch.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Decision

Akita will support a direct opening mode for base-field committed tables opened
at extension-field points. The mode keeps one coefficient axis explicit as a
smaller cyclotomic **carrier ring** and contracts the other coefficient axes
directly over the extension field. Fold challenges live in that smaller ring
and embed sparsely into the ambient A ring. The challenge dimension and the
ambient A-ring dimension are independent schedule choices, subject to the
divisibility condition below.

Generated schedules MUST use this direct mode at absolute fold levels 0 and 1
when those nonterminal folds exist. This rule applies to every field profile.
For extension fields, EOR is not available at those levels. For degree one
fields, the same mode may reduce the partial opening and D or H source even
though there is no EOR payload to remove.

Later recursive folds and the terminal keep the current opening protocol. The
feature does not add a carrier terminal and does not extend carrier search past
the existing two level adaptive prefix. A short schedule uses carrier mode at
each existing nonterminal fold with absolute index 0 or 1. The planner MUST NOT
add a fold only to obtain two carrier levels.

The planner searches the carrier dimension `s` independently of the ambient A
dimension `d_A` at levels 0 and 1. Both values are candidate coordinates. They
are not explicit optimization components. The planner retains the current
catalog objective, which minimizes setup matrix field elements first and proof
payload second, followed by its current canonical deterministic ordering.

This specification is stacked on the selective L2 fold sizing work in
[PR #369](https://github.com/LayerZero-Labs/akita/pull/369), at commit
`8637b85472349c5cd8d2178221399b5dea3773ef`. It also assumes the dyadic B
slicing merged in [PR #388](https://github.com/LayerZero-Labs/akita/pull/388).

## Why this change

Akita currently commits base-field coefficients but uses an extension field for
opening points and sum-check challenges. EOR converts that extension-field
opening into a form accepted by the ring relation. It is correct, but it adds a
degree-two sum-check and transcript-bound partial evaluations at every
extension-valued recursive opening.

The conversion is avoidable. A coefficient table can instead be viewed as a
polynomial over a smaller carrier ring whose coefficients already lie in the
extension field. This viewpoint has three useful consequences:

1. The original extension-field point can be used directly, so L0 and L1 need
   no EOR proof.
2. Each partial opening and its consistency quotient can use fewer base-field
   coordinates than a full ambient-ring element.
3. The current commitment type, A relation, B commitment, setup matrices, and
   most NTT caches can remain over the ambient base field ring.

The change is not unconditionally cheaper. A smaller carrier requires a
different sparse challenge family. Meeting the same entropy target can increase
the challenge's norm, which can enlarge the folded witness, the secure A rank,
and the `t` part of the next witness. Near the recursive tail, a direct
small-field opening may also require a larger A ring. This does not by itself
make the candidate worse: a larger ring can need proportionally fewer A rows.
The planner must compare exact field coordinate counts and the complete suffix.
Neither `s` nor `d_A` is an optimization objective by itself.

## Scope and notation

The direct-mode equations below describe one polynomial and one commitment
group. Existing claim batching and multi-group row coefficients apply outside
these equations and do not change them.

| Symbol | Meaning |
|---|---|
| `K` | Base coefficient field `F_q` |
| `E` | Challenge and evaluation field, with extension degree `k = [E:K]` |
| `d_A` | Ambient A-ring dimension |
| `n_A` | Secure output rank of the A matrix |
| `s` | Carrier-ring dimension selected by the schedule |
| `h` | Native-mode packing gain `d_A / (k s)` |
| `R` | Ambient ring `K[X]/(X^{d_A}+1)` |
| `S` | Challenge carrier `K[Y]/(Y^s+1)` |
| `C` | Extension carrier `E[Y]/(Y^s+1)` |
| `beta_t` | Element `t` of Akita's fixed canonical `K`-basis of `E` |

Every admitted direct candidate satisfies

```text
d_A = k h s,
h >= 1,
d_A, k, h, and s are powers of two.
```

The embedding from the carrier into the ambient ring is

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
coefficient permutations or alternative carrier embeddings.

### Protocol boundary

There is one coefficient carrier opening relation. This implementation uses it
only at nonterminal fold levels 0 and 1. Its partial digits become part of the
ordinary flat witness committed for the next fold. The descriptor MUST bind
`k`, `s`, and the extension coordinate order.

The stored witness and its commitment do not have a carrier type. A later fold
may open that flat base field witness with the current EOR path. The same rule
applies to a precommitted group and to a setup prefix. The commitment fixes the
polynomial layout and commitment matrices. The consuming fold fixes the
opening reduction and challenge family.

The current tensor EOR and Hachi terminal remain unchanged. The terminal
analysis below explains why a hidden carrier tail would need another opening
argument. That analysis is a security boundary, not implementation scope.

### Why later folds and the terminal stay unchanged

The current audited sparse challenge ladder starts at dimension 64. A native
candidate using that smallest challenge therefore carries

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

Suppose the prover decomposes the native planes into short digits `ehat`, binds

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

The current nonterminal protocol supplies the missing step by putting `ehat`
inside the recursively committed next witness and opening that commitment at
the Stage 2 point. The current terminal supplies the needed authentication
through the existing EOR and Hachi path.

A future hidden carrier terminal would have to do all of the following:

1. bind the coefficient planes before the block-fold challenges;
2. prove the bound witness is short;
3. authenticate every final witness evaluation requested by the range and
   relation proofs; and
4. enforce native carrier consistency and the exact `E`-valued opening on that
   same authenticated witness.

Until item 3 has a concrete proof, no future planner may price the 512 byte D
image and local sum check messages as a complete replacement for raw `e`. The
previous 1,744 byte estimate omitted this opening argument and is invalid.

Projecting the `k` coordinate planes to one carrier plane does not supply the
missing opening either. For example, an `S`-linear projection

```text
p_i = e_(i,0) + eta e_(i,1) + ... + eta^(k-1) e_(i,k-1)
```

preserves the carrier-consistency equation, but it does not preserve the
extension-field MLE opening equation. Multiplication by `eta` acts on the
carrier coefficient index, while the extension-field opening weights act on
the MLE indices and mix the `k` field coordinates. In general,

```text
Open(eta e) != eta Open(e).
```

The projected plane therefore authenticates only a projection of the carrier
relation. It is not a complete terminal opening.

For one claim, six live blocks, extension degree four, carrier dimension 64,
and four byte base field elements, a transparent path would send

```text
1 * 6 * 4 * 64 * 4 = 6,144 bytes
```

of raw partial openings. The representative EOR and Hachi baseline sends 3,072
raw partial opening bytes and a 544 byte EOR proof, for 3,616 opening bytes.
This comparison explains why the current terminal remains in scope and a
carrier terminal does not.

## Current protocol

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

At levels 0 and 1, this specification changes four facts. A direct partial is
not a full `R` element. The consistency row uses the carrier modulus. The
scalar row uses direct coefficient weights. The carrier and A relations use
different evaluations of the same challenge. Later folds and the terminal use
the current protocol without these changes.

## Direct coefficient packing

### Canonical coefficient layout

Write one ambient ring element at block `i` and position `x` as

```text
F_(i,x)(X)
  = sum_(j < s) sum_(a < k h) f_(i,x,a,j) X^(a + k h j).
```

Thus `a` is the low coefficient index and `j` is the carrier index. The physical
ambient coefficient index is exactly

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

### Partial opening

For each live block, define

```text
e_i(Y)
  = sum_(x,a,j)
      eq(r_M, x) eq(r_pack, a) f_(i,x,a,j) Y^j
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
[claim][block][extension coordinate t][carrier coefficient j].
```

It contains exactly `k s` base-field coordinates per claim/block. Backends MAY
temporarily use packed `E` values, but transcript encoding, gadget
decomposition, commitment input, range checking, and witness sizing MUST use
the canonical base-field layout above.

`Y` is a formal carrier indeterminate, not an opening-point coordinate. The
scalar opening below contracts the coefficient table with `eq(r_tail,j)`.
Ring switching later evaluates the carrier polynomial at `Y = alpha`; those are
different operations with different purposes.

### Scalar opening equation

For one polynomial with claimed opening `v`, the direct equation is

```text
sum_i eq(r_B, i)
  sum_(j < s) eq(r_tail, j)
  sum_(t < k) beta_t e_(i,t,j)
  = v.
```

After gadget decomposition at opening basis `b_open`, this becomes

```text
sum_i eq(r_B, i)
  sum_(j < s) eq(r_tail, j)
  sum_(t < k) beta_t
  sum_l b_open^l e_hat_(i,l,t,j)
  = v.
```

This replaces the current evaluation-trace formula for a direct-carrier group.
It remains one logical field-level Stage-2 row, with the existing claim-batching
coefficient applied outside the displayed equation. It has no cyclotomic
quotient. The implementation SHOULD name its prepared weights as direct
coefficient-opening weights rather than trace weights.

This equation is evaluated from digits authenticated through the next recursive
witness. A grouped root has one schedule owned opening geometry per group, in
canonical root group order. The precommitted profile freezes the commitment
geometry. The root schedule separately freezes how that group is opened. The
verifier MUST NOT apply one group's coefficient layout or carrier dimension to
another group.

## Fold challenges and the two relation rings

### Carrier challenge

Each fold challenge is sampled as

```text
c_i(Y) in S = K[Y]/(Y^s+1)
```

using the challenge configuration audited for dimension `s`, not the one for
`d_A`. The transcript sampler MUST bind the opening mode, `s`, the challenge
configuration, group identity, live block count, and claim count before
expansion.

The same challenge is used in two rings:

```text
carrier relation:  c_i(Y)          in S;
A relation:        c_i(X^(k h))    in R.
```

No second challenge is sampled. The ambient form is a coefficient-preserving
embedding of the carrier form.

### Folded source and carrier linearity

For each position `x`, the ambient folded source is

```text
Z_x(X) = sum_i c_i(X^(k h)) F_(i,x)(X)    in R.
```

Let the direct coefficient contraction be

```text
L(F_i)(Y)
  = sum_(x,a,j)
      eq(r_M, x) eq(r_pack, a) f_(i,x,a,j) Y^j.
```

Because multiplying by `Y` advances only the `j` index, and because wrapping
`j = s` gives the same minus sign as `X^{d_A} = -1`, this map is `S`-linear:

```text
L(c_i(X^(k h)) F_i) = c_i(Y) L(F_i)    in C.
```

Therefore honest witnesses satisfy the carrier consistency equation

```text
L(Z)(Y) = sum_i c_i(Y) e_i(Y)          in C.
```

This identity is the algebraic reason the direct protocol is complete.

### Carrier consistency quotient

Use ordinary polynomial representatives of degree below `s`. Define

```text
N_eval(Y) = sum_i c_i(Y) e_i(Y) - L(G z_hat)(Y).
```

The consistency equation is equivalent to the existence of one quotient

```text
Q_eval(Y) in E[Y],  degree(Q_eval) < s,

N_eval(Y) = (Y^s + 1) Q_eval(Y).
```

This is **one quotient over `C`**, not `k` independent relation rows. If

```text
Q_eval(Y) = sum_(j < s) (sum_(t < k) beta_t q_(t,j)) Y^j,
```

then its physical witness layout is the `k` base-field coordinate planes

```text
[extension coordinate t][carrier coefficient j].
```

The quotient contributes `k s` base-field coordinates before its ordinary
gadget decomposition. Relation layout types MUST distinguish:

- one logical row selector;
- carrier modulus dimension `s`; and
- physical coordinate width `k s`.

Treating the row as a base-field ring of dimension `k s` is incorrect: it would
use the modulus `Y^{k s}+1` and the denominator `alpha^{k s}+1`.

Since `L(G z_hat)` has degree below `s`, the quotient is just the high half of
the challenge products:

```text
Q_eval = high_s(sum_i c_i e_i).
```

The prover SHOULD compute this coordinatewise over the `k` base-field planes,
sharing the sparse challenge positions across all planes.

### A rows remain ambient

The A rows do not move to the carrier. They remain

```text
[A G z_hat]_r
  = sum_i c_i(X^(k h)) [G t_hat_i]_r
  in R, for every A row r.
```

They keep `n_A` logical rows, ambient dimension `d_A`, the existing A matrix,
and the existing `t_hat` layout. Only the challenge support changes: carrier
position `j` appears at ambient position `k h j`.

The sparse challenge-times-`t` quotient can be viewed as `k h` independent
length-`s` lanes. An implementation MAY exploit those lanes, but the result
MUST match multiplication by the embedded challenge in `R` exactly.

### B and D rows

B slicing from PR #388 is unchanged. B continues to bind `t_hat` using one
physical matrix reused across its selected dyadic slices.

D binds the gadget digits of the carrier partial openings at levels 0 and 1.
The first implementation requires

```text
d_D divides k s.
```

This avoids a second padding convention. The number of D-role subcolumns per
partial is `selected_partial_width / d_D`. D ranks, compression source widths,
and H compression geometry MUST be recomputed from that exact width. They MUST
NOT be obtained by scaling an old `d_A` price after rank selection.

## Ring switching

### Two evaluations of each challenge

The ring-switch challenge `alpha` remains one element of `E`. The verifier MUST
derive two values from every carrier challenge:

```text
c_carrier_alpha = c_i(alpha)
  = sum_(j < s) c_(i,j) alpha^j;

c_ambient_alpha = c_i(alpha^(k h))
  = sum_(j < s) c_(i,j) alpha^(k h j).
```

The carrier consistency row uses `c_carrier_alpha`. Every A row uses
`c_ambient_alpha`.

The current single `c_alphas` cache MUST be split or typed so that these values
cannot be interchanged. Computing one and reusing it for both relations is a
protocol error except in the degenerate case `k h = 1`.

### Evaluating the carrier quotient

In native mode, the consistency check at `Y = alpha` is

```text
sum_i c_i(alpha) e_i(alpha)
  - L(G z_hat)(alpha)
  - (alpha^s + 1) Q_eval(alpha)
  = 0 in E.
```

For a coordinate-plane representation,

```text
Q_eval(alpha)
  = sum_(t < k) beta_t sum_(j < s) q_(t,j) alpha^j.
```

This fixed basis combination does not need an additional random row-batching
challenge. Before evaluation, the `beta_t` form a `K`-basis, so a nonzero set of
coordinate polynomials gives one nonzero polynomial in `E[Y]`. Random `alpha`
then tests that single polynomial. Cancellation at a particular `alpha` is
already covered by its root bound.

The prepared relation point MUST use the carrier powers
`1, alpha, ..., alpha^(s-1)` and denominator `alpha^s+1` for this row. It MUST
continue to use ambient powers and `alpha^{d_A}+1` for A rows.

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
challenge products already compute only the high half. Native mode SHOULD
extend that high-half kernel to `k` length-`s` coordinate planes.

The existing cyclic and negacyclic setup caches for `A z` remain useful and remain
ambient. This change MUST NOT replace them with extension-field setup matrices.
D-side cache widths change with the shorter direct partial, and setup/cache
requirements MUST be derived from the selected mode.

## Soundness requirements

This section states the security obligations introduced by the new mode. It
does not replace the existing MSIS binding proof for A, B, D, F, and H.

### Transcript order

For each native group, the transcript MUST enforce this dependency order:

1. Bind the instance, schedule, mode, dimensions, coefficient layout, group
   layout, opening point, and original commitment.
2. Bind the complete D or H payload that commits to every base field coordinate
   of every `e_i`.
3. Sample the carrier challenges `c_i` at dimension `s`.
4. Bind the challenge-dependent folded witness, A/B data, carrier quotient, and
   next-witness commitment.
5. Sample `alpha`, relation-row coefficients, and later sum-check challenges.

No coordinate of `e_i` may remain unbound when `c_i` is sampled. No coordinate
of `Q_eval`, `z_hat`, or `t_hat` may remain unbound when `alpha` is sampled.
Existing labels MAY be retained only when the serialized descriptor makes the
mode and dimensions unambiguous. Otherwise new domain-separated labels are
REQUIRED.

### Challenge entropy and unit differences

Every admitted carrier challenge configuration MUST satisfy both conditions:

1. one draw has at least the configured 128-bit Fiat-Shamir min-entropy target;
2. the difference of any two distinct challenges in the family is a unit in
   `S` under Akita's audited short-invertibility bound.

The second condition MUST be checked for the **difference** family, including
its doubled coefficient and norm bounds, not merely for one sampled challenge.
The proof and parameter checker MUST use the factorization/invertibility bound
for `Y^s+1`. Entropy validation alone is insufficient.

If `delta(Y)` is a unit in `S`, then `delta(X^(k h))` is a unit in `R`: the
carrier embedding maps the inverse of `delta` to an ambient inverse. The same
`delta` is also a unit after scalar extension to `C`.

### L2 norm under the carrier embedding

Write an ambient coefficient index as `r + k h j`, where `r < k h` and
`j < s`. Multiplication by `c(X^(k h))` preserves `r`. For each fixed `r`, its
action is negacyclic multiplication by `c(Y)` on the `s` coefficients indexed
by `j`. A coefficient permutation therefore writes the ambient multiplication
matrix as a block diagonal matrix with `k h` identical carrier blocks.

It follows that

```text
||M_(c(X^(k h)))||_2 = ||M_(c(Y))||_2.
```

The selective L2 response route may use the operator norm certificate at
dimension `s` after checking this embedding. It may not select a certificate by
ambient dimension `d_A`. Tests MUST compare the block reduction with direct
ambient multiplication and MUST cover the sign on negacyclic wraparound.

### Forking extraction

Consider two accepting transcripts with the same pre-challenge commitments and
different challenge at one claim/block position. Let

```text
delta = c_j - c'_j.
```

After subtracting the accepted A relations,

```text
A (z - z') = delta(X^(k h)) t_j       in R^(n_A).
```

In native mode, subtracting the accepted carrier consistency relations gives

```text
L(G(z - z')) = delta(Y) e_j           in C.
```

Because `delta` is a unit in both rings, these equations determine the opened
`t_j` and `e_j` from the fork. The existing B/F binding of `t_hat`, D/H binding
of `e_hat`, A binding of the folded source, range proof for all digit planes,
and quotient checks then give the same weak-opening/MSIS reduction as the
current fold.

The implementation security note MUST spell out how the standard multi-fork
argument isolates all claim/block positions. It MUST NOT claim extraction from
challenge entropy alone.

### Ring-switch polynomial check

For an honest witness, the carrier numerator is identically zero after the
quotient is included. For a false witness, it is one nonzero polynomial over
`E` of degree at most `2s-1`. Sampling `alpha` after the quotient is bound
detects it except with probability at most

```text
(2s - 1) / |E|,
```

before accounting for the existing row batching and other sum-check errors.
The coordinate basis does not multiply this error by `k`: basis independence
shows that a nonzero coordinate vector is a nonzero coefficient in `E`, and
the verifier tests the resulting single `E` polynomial.

The final theorem statement for direct mode MUST include:

- binding of the original and partial commitments;
- the carrier challenge entropy and unit-difference assumptions;
- the carrier polynomial root bound;
- the existing A/B/D/F/H MSIS assumptions;
- the existing range and sum-check soundness errors; and
- random-oracle forking loss for the complete vector of fold challenges.

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
the serialized EOR payload. Removing EOR does not remove the fold grind nonce;
the numbers below count only bytes that actually disappear with the EOR proof.

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
gross savings. They are not the final selective L2 planner result. Carrier mode
also changes the next witness, ranks, sum check domains, and selected schedule.

### Carrier-coordinate savings

Before digits, one direct partial and its consistency quotient each change from

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

The carrier quotient's base-field coordinate count changes by the same factor
before quotient-digit decomposition. Compression output payloads may have fixed
sizes, so the planner MUST propagate the shorter witness through ranks,
compression chains, relation domains, successor dimensions, and proof sizing;
it MUST NOT report `h` as an automatic proof-size factor.

### Concrete fp32, `d_A = 1024`, `k = 4`

The candidates induced by the current production challenge ladder expose the
main tradeoff. Direct-mode admission still requires the new unit-difference
certificate specified above.

| `s` | `h` | `k h` ambient stride | coordinates per partial | production sparse family at `s` | challenge `l1` mass |
|---:|---:|---:|---:|---|---:|
| 64 | 4 | 16 | 256 | 31 coefficients in `±1`, 10 in `±2` | 51 |
| 128 | 2 | 8 | 512 | 31 coefficients in `±1` | 31 |
| 256 | 1 | 4 | 1,024 | 23 coefficients in `±1` | 23 |

For the middle choice, coefficient index `a + 8j` maps carrier position `j`
to ambient position `8j`. Every partial and carrier quotient uses four
length-128 base-field coordinate planes, or 512 coordinates total, instead of
1,024. The ring-switch verifier computes `c(alpha)` for the carrier row and
`c(alpha^8)` for the A rows.

The `s=64` choice gives a fourfold smaller partial than `s=256`, but it uses a
heavier carrier challenge. At levels 0 and 1 it changes the digit count, D
width, A response bound, and successor witness. The planner must compare the
complete candidate rather than assuming that the smallest `s` wins.

## Planner contract

### The consuming fold owns the opening plan

Each opening group has one schedule owned opening plan. The exact type name may
differ, but it must express the following choice.

```rust
enum OpeningReductionMode {
    Current,
    CoefficientCarrier { carrier_dimension: usize },
}
```

The intended ownership split is equivalent to the following shape.

```rust
struct PrecommittedCommitmentProfile {
    layout: CommittedGroupProfile,
}

struct OpeningGroupPlan {
    mode: OpeningReductionMode,
    fold_challenge_config: SparseChallengeConfig,
    log_basis_open: u32,
    num_digits_open: usize,
    num_digits_fold: usize,
}

struct ScheduledPrecommittedGroup {
    commitment: PrecommittedCommitmentProfile,
    opening: OpeningGroupPlan,
}
```

Equivalent types are acceptable. There must still be one canonical owner for
each field. The implementation should split the current
`PrecommittedLevelParams` fields along this boundary instead of adding a second
copy of them.

The mode and carrier dimension are protocol data. Runtime schedules, generated
rows, canonical descriptors, catalog identity, proof size reports, and the
transcript MUST bind them. The terminal always uses `Current` in this feature.

A scalar recursive fold has one opening group. A grouped root stores one entry
per group in `OpeningClaimsLayout::root_group_order`. All level 0 entries use
the carrier mode, but their `d_A`, `s`, and derived `h` may differ. The planner
does not pad them to one level wide `s`.

The consuming fold owns this plan. The commitment profile does not. A
`CommittedGroupProfile` or setup prefix commitment fixes the source polynomial
layout, the A and B matrices, and the commitment bytes. It does not fix whether
a later fold uses EOR or a carrier opening.

The schedule fixes the opening mode, `s`, and the sparse challenge family
before proving begins. The transcript draws the actual sparse challenge at
runtime after the D or H payload is bound. Neither the runtime draw nor the
choice of opening reduction changes the earlier commitment bytes.

The implementation MUST keep commitment identity separate from opening
admission data. In particular, a setup prefix registry key MUST NOT create two
different commitments only because two consuming schedules choose different
opening modes for the same content and matrix geometry. A composite schedule
object may contain both kinds of data, but its commitment identity and opening
plan must remain separate fields with separate validation.

The consuming fold validates the frozen A and B matrices against its selected
challenge bounds. A commitment that is too narrow for the selected challenge
is not admissible even though its bytes do not depend on the challenge draw.

`h` and the ambient stride are derived from `(k, d_A, s)`. They MUST NOT be
serialized as independent choices.

### Transition after level 1

A carrier fold outputs one canonical flat base field witness. Its partial
opening digits and carrier quotient coordinates are ordinary fields in that
witness. Their earlier meaning is checked while verifying the producing fold.
The next fold commits to and opens the flat witness selected by its own
schedule. It does not reuse the previous fold's carrier ring.

For example, suppose level 0 and level 1 both offload setup contributions.
Level 1 consumes the first setup prefix with its scheduled carrier opening. It
then produces a flat witness and a second setup prefix for level 2. Level 2
opens both its witness and that second prefix with the current EOR path. No
conversion is needed. The level 2 schedule binds EOR, and it checks the frozen
commitment geometry against the level 2 challenge family.

The selective L2 branch currently rejects setup prefix derivation from an A
commitment that has no SIS table key. Carrier search does not weaken this rule.
If an L2 candidate cannot create the required prefix, the planner must use an
admissible Linf candidate, use direct setup, or reject that schedule edge.

### Candidate admission

A carrier candidate is admitted only when all of the following conditions hold.

1. `k`, `d_A`, and `s` are powers of two.
2. `k s` divides `d_A`.
3. `h = d_A/(k s)` is positive.
4. An audited sparse challenge configuration exists at dimension `s`.
5. That configuration meets the entropy and unit difference requirements.
6. The field and ring dispatcher supports the ambient A dimension and carrier
   kernels.
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

For `k = 1`, EOR is not valid and contributes no bytes. The planner still uses
carrier mode at levels 0 and 1. It may choose `s < d_A` to reduce partial and
quotient coordinates. The full ring candidate is `s = d_A` and `h = 1` when
that `s` exists in the audited registry.

### L2 and Linf security routes

The L2 response calculation is keyed by `s`, not by `d_A`. Multiplication by
the embedded challenge `c(X^(k h))` preserves each residue class modulo `k h`.
After a coefficient permutation, the ambient multiplication map is a direct
sum of `k h` copies of multiplication by `c(Y)` in the carrier ring. Its L2
operator norm is therefore exactly the carrier operator norm.

The implementation MUST encode this reduction and compare it with a direct
ambient multiplication reference for every admitted geometry. An L2 candidate
also needs an operator norm certificate for its exact carrier challenge
family. The selective L2 branch currently has such certificates at dimensions
64 and 128. A carrier dimension without that certificate may still use the
audited Linf security route. It MUST NOT reuse an L2 certificate from another
dimension.

The unit difference certificate is separate from the Linf or L2 response
certificate. The former admits the carrier challenge family. The latter prices
the A response and rank.

### Level policy

The policy applies to every field profile.

1. Absolute nonterminal levels 0 and 1 enumerate carrier candidates only.
2. Absolute nonterminal levels 2 and later use the current opening protocol.
3. The terminal uses the current EOR and Hachi protocol where EOR applies.
4. A schedule with only root and terminal uses carrier mode only at root.
5. A schedule with root, one recursive fold, and terminal uses carrier mode at
   both nonterminal folds.
6. The planner does not add a fold to reach the two level carrier scope.

A precommitted root group freezes its commitment profile. Its root opening plan
may choose any certified `s` compatible with the frozen `d_A` and security
bounds. Level 1 sees the root output as one flat recursive witness and chooses
one new `(d_A, s)` for that witness. It does not preserve one carrier geometry
per original root group.

A setup prefix is another precommitted group at the fold that consumes it. It
uses that consuming fold's opening mode. A prefix consumed at level 1 uses
carrier mode. A prefix consumed at level 2 or later uses the current opening
protocol.

### Independent carrier and A dimensions

The carrier dimension `s` controls challenge entropy and the `k s` partial and
quotient widths. The A dimension `d_A` controls the ambient A rows. Once `s` is
fixed, changing `d_A` changes the packing gain and ambient geometry. It does
not enlarge the carrier partial or quotient.

The planner enumerates every admitted `(d_A, s)` pair at levels 0 and 1 under
the incoming dimension ceiling. It preserves the selective L2 branch's current
dimension policy and uniform suffix after level 1. This feature does not add A
dimension search to later folds or the terminal.

B and D remain independent role dimensions and may stay below `d_A`. The
planner does not raise them merely because a larger A ring has a lower rank.
It does not add a semantic preference for smaller `s` or larger `d_A`.

### Objective and exact pricing

The planner keeps the selective L2 branch's catalog objective. It minimizes
setup matrix field elements first and proof payload second. Exact ties use the
branch's current canonical deterministic ordering. No new objective component
for `s`, `d_A`, rank, or prover time is added.

For every carrier candidate, the planner MUST recompute at least the following
values.

1. Removed EOR bytes at levels 0 and 1.
2. Partial coordinate count and opening gadget depth.
3. D input width, secure D rank, H source, and compression geometry.
4. Carrier quotient coordinates and quotient digit count.
5. Sparse challenge `l1`, `l2`, and `linf` bounds at `s`.
6. Folded `z` bounds and the secure A rank under every available route.
7. `t_hat`, B input width, B slicing candidates, and F compression geometry.
8. Logical row count, physical row dimensions, relation address length, and
   sum check rounds.
9. Setup prefix eligibility and setup matrix field elements.
10. Successor witness length and the complete suffix cost.

The report MUST show the selected mode, `s`, `d_A`, `h`, secure A route,
partial coordinates, quotient coordinates, setup field elements, proof bytes,
and successor witness length. It must list every catalog regression against the
selective L2 baseline. A minor row may regress when the target presets and the
overall catalog improve. Per row nonregression is not an acceptance condition.

### B slicing interaction

Carrier search and B slicing both apply at levels 0 and 1. The carrier shortens
`e`, the D or H source, and the consistency quotient. B slicing shortens the
physical B matrix and may add logical rows.

B width depends on `t_hat`. The `t_hat` geometry depends on the carrier
challenge norm, the selected A dimension, and secure A rank. Candidate
construction MUST use this order.

1. Choose `d_A`, `s`, and the carrier challenge family.
2. Evaluate the available Linf and L2 routes, then derive A rank and `t_hat`.
3. Enumerate and prune the bounded B slice counts from PR 388.
4. Derive D or H and B or F compression plans.
5. Construct the next witness and score the complete suffix.

The planner MUST NOT choose a B slice count from geometry computed before the
carrier candidate. The bounded slice set `{1, 2, 4, 8}` and its current local
pruning rule remain unchanged.

### Search control

The planner MUST keep the search bounded in the following ways.

1. Derive `h` from `(k, d_A, s)`.
2. Use one canonical coefficient layout and embedding.
3. Use the fixed audited list of `s` values.
4. Apply security and divisibility admission before rank lookup.
5. Search carrier candidates only at levels 0 and 1.
6. Apply B slicing only after A and `t_hat` geometry is known.
7. Keep the existing deterministic frontier and memo state objective.
8. Keep the current uniform suffix after the adaptive prefix.
9. Compare the pruned result with an unpruned oracle on small fixtures.

## Implementation boundaries

### `akita-types`

- Add the schedule owned opening mode and checked carrier geometry.
- Add canonical descriptor encoding for mode and `s`.
- Represent the carrier consistency row as one logical row with carrier
  dimension `s` and extension coordinate width `k`.
- Extend witness and address layouts for `k s` partial and quotient coordinates.
- Separate precommitted commitment identity from the opening plan selected by
  the consuming fold. Apply the same split to setup prefix slot identity.
- Generalize proof size and successor witness sizing at levels 0 and 1.
- Keep malformed verifier inputs on typed `AkitaError` or
  `SerializationError` paths.

### `akita-challenges`

- Reuse the signed-sparse sampler at dimension `s`.
- Add or expose a parameter certificate that covers entropy and the complete
  pairwise-difference invertibility bound.
- Expose L2 operator norm certificates by carrier dimension and exact challenge
  family. Keep the Linf route for other certified carrier dimensions.
- Bind carrier dimension and mode in the draw domain.
- Do not create a second ambient challenge draw.

### `akita-prover`

- Add dense, one-hot, and recursive kernels that compute the `s` extension
  coefficients directly from the canonical coefficient split.
- Decompose the resulting `k s` base-field coordinates into D-role `e_hat`.
- Compute `Q_eval` with shared-challenge high-half accumulation over `k`
  coordinate planes.
- Keep A quotients over `d_A` with challenges embedded at stride `k h`.
- Split carrier and ambient challenge evaluations in ring-switch preparation.
- Replace the trace specific Stage 2 term with the direct coefficient opening
  term at levels 0 and 1.
- Use the current prover path at later folds and at the terminal.
- Preserve current cyclic/negacyclic A setup caches.

### `akita-verifier`

- Reconstruct the direct scalar-opening row from `r_B`, `r_tail`, the canonical
  extension basis, and opening gadget weights.
- Evaluate carrier quotient planes with denominator `alpha^s+1`.
- For native groups, evaluate the same challenge at `alpha` and
  `alpha^(k h)` for its two roles.
- Use the current verifier path at later folds and at the terminal.
- Reject mode/dimension/layout mismatches before allocation.
- Preserve the no-panic verifier contract.

### `akita-planner`, `akita-schedules`, and `akita-config`

- Add bounded carrier candidates and the level policy above.
- Extend the existing two level adaptive search with the bounded `s` registry.
  Keep the current uniform suffix after that prefix.
- Recompute exact ranks, setup, compression, proof bytes, and successors.
- Regenerate every affected catalog on top of the selective L2 branch.
- Add report columns for opening mode, carrier geometry, security route, setup
  field elements, and proof bytes.
- Preserve the current setup first and proof payload second objective without
  adding an `s` or `d_A` objective component.
- Report every catalog regression. Do not reject the feature only because a
  minor row regresses.

## Acceptance criteria

### Algebra and completeness

- [ ] Checked carrier geometry accepts exactly supported `(k,d_A,s)` triples and
      derives `h` and stride without independent metadata.
- [ ] Coefficient index `a + k h j` round-trips for every supported geometry.
- [ ] Dense, one-hot, and recursive direct partials match a flat MLE reference.
- [ ] `L(c(X^(k h))F) = c(Y)L(F)` holds against a naive reference for random
      small fixtures and every supported field tier.
- [ ] Ambient multiplication by `c(X^(k h))` matches `k h` permuted carrier
      lanes, including negacyclic wraparound, and preserves the L2 operator
      norm.
- [ ] Direct scalar-opening weights reproduce the claimed opening, including
      partial final blocks, multiple polynomials, and multiple groups.
- [ ] `Q_eval = high_s(sum_i c_i e_i)` satisfies the full ordinary-polynomial
      divisibility identity in `E[Y]`.
- [ ] The `k` base-field quotient planes evaluate to the same `E` value as a
      packed-extension reference.
- [ ] Honest carrier proofs verify at levels 0 and 1 for every supported field
      tier.
- [ ] A level 2 fold opens the flat output of a carrier level 1 fold with the
      current protocol without conversion or carrier metadata reuse.

### Soundness and transcript

- [ ] Every admitted challenge family has a reviewable 128-bit entropy and
      pairwise-difference unit certificate for `S`.
- [ ] The certificate checks the difference family's exact coefficient/norm
      envelope.
- [ ] Each selective L2 carrier candidate has an operator norm certificate for
      its exact `s` and challenge family. Other carrier candidates use Linf.
- [ ] Nonterminal partial D or H payloads are transcript bound before their
      carrier challenge draws.
- [ ] Carrier quotient and next-witness data are bound before `alpha`.
- [ ] Mode, `s`, challenge configuration, coefficient layout, and group identity
      change the descriptor/transcript bytes.
- [ ] The verifier computes distinct `c(alpha)` and `c(alpha^(k h))` values and
      tests fail when either is substituted for the other.
- [ ] A nonzero coordinate-plane numerator is detected by the packed
      `E[Y]` ring-switch oracle.
- [ ] The direct-mode theorem adds no `1/|K|` coordinate-projection term.
- [ ] Multi-fork extraction and total soundness-error accounting are documented
      alongside the implementation.
- [ ] Malformed mode/dimension/coordinate counts return typed errors without
      panic or unbounded allocation.

### Planner and sizing

- [ ] Generated schedules for every field tier use carrier mode at existing
      nonterminal levels 0 and 1.
- [ ] Extension field schedules contain no EOR at levels 0 and 1.
- [ ] Later folds and the terminal retain their current opening protocol.
- [ ] Short schedule tests cover root to terminal and root to one recursive
      fold to terminal without inserting another fold.
- [ ] The planner searches every admitted `(d_A, s)` pair only inside the two
      level adaptive prefix and keeps the current uniform suffix.
- [ ] The catalog objective remains setup matrix field elements first and proof
      payload second, followed by the current canonical deterministic ordering.
      It has no explicit `s` or `d_A` objective component.
- [ ] `d_D` not dividing the selected native or hidden-digit width rejects
      before matrix/rank construction.
- [ ] Exact D/H and A/B/F ranks are recomputed from carrier geometry and norms.
- [ ] PR 388 B slicing is enumerated only after carrier derived A and `t`
      geometry.
- [ ] Bounded DP output matches an unpruned oracle on small search fixtures.
- [ ] Reports reproduce the historical EOR census and show new results on the
      selective L2 baseline.
- [ ] At least one fp32 and one fp64 production row demonstrate the expected
      L0/L1 EOR removal in actual serialized proof breakdowns.
- [ ] No generated catalog silently drops a previously supported row. Every
      regression is listed, but minor per row regressions are allowed.
- [ ] The schedule report identifies each setup prefix edge that cannot use an
      L2 A route and records the selected Linf, direct setup, or rejected path.

### Precommitment and setup prefixes

- [ ] Commitment identity excludes the consuming opening mode, `s`, and
      challenge draw.
- [ ] Schedule and transcript identity include the consuming opening plan.
- [ ] The same frozen precommitted commitment can be admitted under carrier or
      current opening mode when its matrices meet both security checks.
- [ ] A setup prefix consumed at level 1 uses carrier mode. A setup prefix
      consumed at level 2 uses the current protocol.
- [ ] A two level recursive offloading test verifies the transition from a
      carrier consumer at level 1 to a current protocol consumer at level 2.
- [ ] Changing the opening plan does not duplicate identical setup prefix
      commitment bytes or alter the committed polynomial layout.

### Performance and caches

- [ ] Direct partial and carrier quotient allocations contain exactly `k s`
      base-field coordinates per semantic item before digits.
- [ ] Carrier high-half construction does not materialize full extension-field
      convolution tables.
- [ ] Existing A cyclic/negacyclic setup caches remain shared and correct.
- [ ] D/H cache requirements use the selected carrier width and do not retain
      old `d_A`-wide buffers.
- [ ] Profile output records prover time, verifier time, peak memory, setup
      field elements, proof bytes, and per level witness sizes against the
      selective L2 baseline.
- [ ] A packed-`E` verifier Horner loop is adopted only if it beats the canonical
      coordinate-plane loop without changing bytes or arithmetic results.

### Repository validation

- [ ] Generated schedule tables are clean after regeneration.
- [ ] Focused algebra, prover, verifier, planner, and catalog tests pass.
- [ ] All required feature-graph Clippy jobs pass.
- [ ] `./scripts/check-doc-guardrails.sh` passes.

## Non-goals

- No implementation code lands in the initial spec-only commit.
- No carrier relation after absolute fold level 1.
- No carrier terminal or change to the current terminal protocol.
- No A dimension search beyond the current two level adaptive prefix.
- No arbitrary integer carrier dimensions or coefficient layout search.
- No second independent challenge for the A rows.
- No pure extension-field commitment or setup matrix.
- No claim that the smallest carrier is always optimal.
- No global objective component based on `s` or `d_A`.
- No requirement that every catalog row improve. Minor regressions are allowed
  and must be reported.
- No claim that flattening a native `k s` carrier and adding four public rows
  yields a sound `s`-coordinate prechallenge carrier.
- No claim that a D image plus local sum-checks forms a complete opening proof;
  short-preimage binding does not authenticate a final multilinear evaluation.
- No use of the rejected ring-valued interpolation described below: its
  opening operators are only `K`-linear and do not preserve degree over `S`.
- No change to PR 388's B slice count set, dyadic partition, or 8 KiB
  compression-source limit.
- No backward-compatible decoding of schedules or proofs that predate this
  mode. Akita remains in development; affected catalogs and descriptors are
  regenerated rather than aliased.

## Documentation follow-up

The implementation PR must fold stable protocol prose into:

- `book/src/how/proving/root-fold-ring-switch.md` for the carrier relation and
  two challenge evaluations;
- `book/src/how/proving/extension-opening-reduction.md` for L0/L1 removal and
  the unchanged later fold and terminal boundary;
- `book/src/how/configuration.md` for planner candidates, setup prefix
  ownership, and reports;
- `book/src/foundations/rings-and-fields.md` for the carrier embedding and unit
  condition; and
- `book/src/how/security.md` for the forking and polynomial-root arguments.

Once those chapters own the durable explanation and the implementation ships,
this spec moves through `implemented` to the normal archive workflow in
[`specs/PRUNING.md`](PRUNING.md).

## References

- [B commitment slicing](commitment-slicing.md), PR 388 baseline and B planner
  interaction.
- [Selective L2 fold sizing](https://github.com/LayerZero-Labs/akita/pull/369),
  planner objective, response sizing, and operator norm certificates.
- [Extension-field opening batching](extension-field-opening-batching.md),
  tensor EOR and the transformed-commitment soundness boundary.
- [Ring-dimension and challenge cutover](ring-dim-challenge-cutover.md), current
  production sparse families and role dimensions.
- [EOR streamed prover](eor-streamed-prover.md), current EOR prover path and
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
