# Symphony Norm Check

This note extracts the norm-check mechanism from the Symphony paper and rewrites it in the same spirit as `LABRADOR_LINEAR_ONLY.md` and `LATTICEFOLD_PLUS_NORM_CHECK.md`: isolate the real algebra, explain what the protocol is actually certifying, and make the tradeoffs legible.

The short version is:

- `LatticeFold+` proves a full coefficient range statement by monomial encoding.
- `LaBRADOR`-style random projection reduces a long norm statement to a shorter one.
- `Symphony` combines them: first project, then run a LatticeFold+-style monomial lookup on the projected object.

So yes, it is a hybrid of the two ideas. But it is not literally "the best of both worlds" with no downside. The key tradeoff is that Symphony's result is an approximate norm proof for the original witness, not an exact one.

Main paper anchors:

- Algebra background, monomial embedding, and random projection: Section 2.1, pages 10-11
- Fine-grained norm relation `VfyOpen_{\ell_h,B}`: Section 2.2, page 13
- Monomial RoK imported from LatticeFold+: Section 3.3, pages 20-21
- Approximate range proof `\Pi_{\mathrm{rg}}`: Section 3.4, pages 21-25
- Norm-check complexity: Proposition 3.3, page 25
- Candidate parameters: Section 7, pages 39-40

## Scope

This note focuses on the part of Symphony that proves the witness opening is low norm.

The most important first fact is that Symphony is not proving the exact same statement as LatticeFold+.

LatticeFold+ proves an exact coefficientwise bound on a ring vector `f \in R_q^n`:

$$
\|f\|_\infty < B.
$$

Symphony instead starts from a finer blockwise `\ell_2` condition on the coefficient matrix `\mathrm{cf}(f) \in \mathbb{Z}_q^{n \times d}`.

Let `\ell_h | n`. Partition `\mathrm{cf}(f)` into blocks

$$
F_{i,j} \in \mathbb{Z}_q^{\ell_h \times 1},
\qquad
(i,j) \in [n / \ell_h] \times [d].
$$

Then the input norm relation is

$$
\forall (i,j): \|F_{i,j}\|_2 \le B.
$$

This is exactly the `VfyOpen_{\ell_h,B}` relation in the paper.

So Symphony's norm check should be read as:

1. The witness is committed correctly.
2. Every coefficient-column block of height `\ell_h` has `\ell_2` norm at most `B`.

That blockwise statement is stronger than a generic global bound in the way the protocol needs, because the random projection is applied independently to those blocks.

## Symbol Table

| Symbol | Meaning |
| --- | --- |
| `f \in R_q^n` | Witness vector |
| `\mathrm{cf}(f) \in \mathbb{Z}_q^{n \times d}` | Coefficient matrix of `f` |
| `F_{i,j} \in \mathbb{Z}_q^{\ell_h \times 1}` | One coefficient-column block of `\mathrm{cf}(f)` |
| `\ell_h` | Projection input length, i.e. block height |
| `\lambda_{\mathrm{pj}} = 256` | Projection output length |
| `J \in \{0,\pm 1\}^{\lambda_{\mathrm{pj}} \times \ell_h}` | Small random projection matrix |
| `M_J = I_{n/\ell_h} \otimes J` | Structured block-diagonal projection |
| `H = (I_{n/\ell_h} \otimes J)\,\mathrm{cf}(f)` | Projected coefficient matrix |
| `d` | Ring dimension |
| `d'` | In Symphony, `d' := d - 2` |
| `B_{d,k_g}` | Largest magnitude representable by `k_g` signed base-`d'` digits |
| `H^{(1)}, \dots, H^{(k_g)}` | Signed base-`d'` digit matrices of `H` |
| `h^{(i)} = \mathrm{flt}(H^{(i)})` | Flattened digit matrix |
| `g^{(i)} = \mathrm{Exp}(h^{(i)}) \in M^n` | Monomial encoding of flattened digit vector |
| `t(X)` | Table polynomial used to decode monomials back to integers |
| `\Pi_{\mathrm{mon}}` | Monomial reduction of knowledge |
| `\Pi_{\mathrm{rg}}` | Symphony's approximate norm / range proof |
| `B'` | Relaxed norm bound extracted in soundness |

## Big Picture

At a high level, Symphony's norm check is:

```text
original witness f in R_q^n
        |
        | coefficient matrix cf(f)
        v
  blockwise l2 bound on F_{i,j}
        |
        | structured random projection with J
        v
  H = (I \otimes J) cf(f)
        |
        | if the original blocks are small,
        | entries of H are small with high probability
        v
  H has small integer entries
        |
        | decompose each entry in signed base d'
        v
  H = H^(1) + d' H^(2) + ... + d'^(k_g-1) H^(k_g)
        |
        | monomial-encode each digit matrix
        v
  g^(1), ..., g^(k_g) in M^n
        |
        | prove monomial-ness via Pi_mon
        v
  random evaluations u^(i)
        |
        | decode with t(X) and recombine digits
        v
  recover a random evaluation of H
        |
        | tie that back to the committed witness f
        v
  projected witness is small
        |
        | random projection lemma
        v
  original witness blocks are small up to slack B'
```

This is the whole hybrid:

- the front half is random projection;
- the back half is LatticeFold+-style monomial lookup.

## What Symphony Is Actually Proving

The input norm relation is not

$$
\|f\|_\infty < B
$$

and not even directly

$$
\|f\|_2 < B.
$$

Instead, Symphony defines the fine-grained opening check

$$
\mathrm{VfyOpen}_{\ell_h,B}(pp_{cm}, c, f) = 1
$$

to mean:

$$
Af = c
\qquad\text{and}\qquad
\forall (i,j): \|F_{i,j}\|_2 \le B,
$$

where the `F_{i,j}` are the `\ell_h \times 1` blocks in the coefficient matrix `\mathrm{cf}(f)`.

Visually:

```text
cf(f) =
[ F_{1,1}  F_{1,2}  ...  F_{1,d} ]
[ F_{2,1}  F_{2,2}  ...  F_{2,d} ]
[   ...      ...    ...    ...   ]
[ F_{n/lh,1} ...         F_{n/lh,d} ]

each F_{i,j} is a column of height ell_h
and every one of them must satisfy ||F_{i,j}||_2 <= B
```

Why this particular shape?

Because the projection matrix acts on each `F_{i,j}` block independently. The norm statement is chosen to match the geometry of that projection.

## Phase 1: Structured Random Projection

This is the LaBRADOR-like half of the construction.

The verifier samples

$$
J \in \{0,\pm 1\}^{\lambda_{\mathrm{pj}} \times \ell_h},
\qquad
\lambda_{\mathrm{pj}} = 256,
$$

and the prover computes

$$
H := (I_{n/\ell_h} \otimes J)\,\mathrm{cf}(f)
\in
\mathbb{Z}_q^{m \times d},
\qquad
m = n\lambda_{\mathrm{pj}} / \ell_h.
$$

So each original block `F_{i,j}` is replaced by a shorter projected block

$$
H_{i,j} = J F_{i,j} \in \mathbb{Z}_q^{\lambda_{\mathrm{pj}}}.
$$

### Intuition

This is the same geometric idea behind JL-style norm sketches:

- if `F_{i,j}` is small, then a random sign projection of it stays small;
- if `F_{i,j}` is large, then the projection is unlikely to look small.

But Symphony uses a structured block-diagonal projection, not one giant dense `256 \times N` matrix.

That is the first "best of both worlds" moment:

- the verifier only samples a small `J`;
- the prover gets a much shorter object to range-check;
- the resulting checks stay compatible with folding and field arithmetic.

### The actual projection lemma

The relevant paper facts are:

1. For a random sign vector `u`, one coordinate projection satisfies

$$
|\langle u, v \rangle| \le 9.5 \|v\|_2
$$

with overwhelming probability.

2. Conversely, if `\|v\|_2 > B`, then

$$
\|Jv \bmod q\|_2 \le \sqrt{30}\,B
$$

happens only with tiny probability.

So projection gives a completeness direction and a soundness direction:

- honest small blocks project to small coordinates;
- dishonest large blocks are unlikely to hide inside a small projection.

## Phase 2: Why Projection Comes Before Monomials

This is the key design difference from LatticeFold+.

In LatticeFold+, the witness is a full ring vector `f \in R_q^n`, so the exact range proof has to deal with all `d` coefficients of each ring element. If done naively, that means `d` monomial commitments, which is too much. LatticeFold+ solves that using a double-commitment and commitment-transformation layer.

Symphony dodges that entire machinery by projecting first.

After projection, the prover only needs to range-check the projected coefficient matrix `H`, not the full coefficient matrix `\mathrm{cf}(f)`.

Under the paper's simplifying choice `md = n`, flattening `H` gives a vector of length exactly `n`, so the monomial layer works with `O(1)` monomial vectors of length `n`, not `d` separate families of them.

This is the second "best of both worlds" moment:

- the projection shrinks the range-proof target;
- the monomial layer then becomes simple enough that no double commitment is needed.

## Phase 3: Signed Base-`d'` Decomposition Of The Projected Entries

This is where Symphony imports the LatticeFold+ logic, but with slightly different notation.

The paper sets

$$
d' := d - 2.
$$

This looks odd if you are coming from LatticeFold+, where `d'` denoted `d/2`. Here the purpose is different:

- the table polynomial can decode integers in `(-d/2, d/2)`;
- therefore each digit must lie in `[-(d/2-1), d/2-1]`;
- and `d'/2 = (d-2)/2 = d/2 - 1` is exactly that safe digit range.

So Symphony represents each entry of `H` in signed base `d'`, with digits bounded by `d'/2`.

The representable magnitude using `k_g` such digits is

$$
B_{d,k_g}
:=
\frac{d'}{2}\,(1 + d' + \cdots + d'^{k_g-1}).
$$

They choose the minimal `k_g` such that

$$
B_{d,k_g} \ge 9.5 B.
$$

That way, any honest projected entry, which is at most about `9.5 B`, can be expanded as

$$
H = H^{(1)} + d' H^{(2)} + \cdots + d'^{k_g-1} H^{(k_g)}
$$

with

$$
\|H^{(i)}\|_\infty \le d'/2.
$$

### Concrete intuition for `d = 64`

With `d = 64`, Symphony has

$$
d' = 62,
\qquad
d'/2 = 31.
$$

So every projected entry is written using signed base `62` digits in `[-31,31]`.

For `k_g = 3`,

$$
B_{64,3}
=
31(1 + 62 + 62^2)
=
121117,
$$

which is exactly the value listed in the paper's candidate parameters.

This explains why they only need `k_g = 3` monomial vectors in the concrete instantiation.

## Phase 4: Monomial-Encoding The Projected Digits

Flatten each digit matrix:

$$
h^{(i)} := \mathrm{flt}(H^{(i)}) \in \mathbb{Z}_q^n.
$$

Then encode it coordinatewise as monomials:

$$
g^{(i)} := \mathrm{Exp}(h^{(i)}) \in M^n.
$$

This is now exactly the LatticeFold+ monomial embedding move:

- each small signed digit becomes one monomial;
- the table polynomial `t(X)` will decode it back later.

But Symphony only does this for the projected digits, not for the full original witness coefficients.

That is the central simplification.

## Phase 5: The Monomial RoK `\Pi_{\mathrm{mon}}`

This is the LatticeFold+ half of the construction.

For each `i \in [k_g]`, the prover commits to the monomial vector `g^{(i)}`:

$$
c^{(i)} := A g^{(i)}.
$$

Then the parties run the monomial reduction of knowledge `\Pi_{\mathrm{mon}}`.

At a high level, `\Pi_{\mathrm{mon}}` proves that every entry of every `g^{(i)}` is actually in the monomial set

$$
M = \{0,1,X,\dots,X^{d-1}\}.
$$

The same core identity from LatticeFold+ is used:

$$
a(X^2) = a(X)^2
\qquad\Longleftrightarrow\qquad
a \text{ is a monomial}.
$$

After batching with sumcheck, the verifier only keeps one random evaluation summary per monomial vector:

$$
u^{(i)} = \langle ts(r \| s), g^{(i)} \rangle \in E.
$$

So the nonlinear "is this really a monomial?" burden is again pushed into the monomial subprotocol.

### Why this part is cheap

The paper's monomial lemma says the cost beyond the sumcheck is:

- prover: `O(n k_g)` field additions and `O(n)` field ops;
- verifier: `O(k_g d + \log n)` field ops.

This is cheap for the same reason as in LatticeFold+:

- monomial vectors are easy to commit to;
- the monomial check reduces to one degree-3 sumcheck over the field `K`.

## Phase 6: Decode With The Table Polynomial

Now comes the actual "range proof" step.

The paper defines the same table polynomial shape as LatticeFold+:

$$
t(X) := \sum_{i=1}^{d/2-1} i \cdot (X^i + X^{-i}) \in R_q.
$$

And it uses the same decoder property:

$$
\mathrm{ct}(\mathrm{Exp}(a)\cdot t(X)) = a
\qquad
\text{for } a \in (-d/2,d/2).
$$

This is where the tensor-of-rings object `E` matters.

For each `i`, the verifier has the random evaluation summary

$$
u^{(i)} = \langle ts(r \| s), g^{(i)} \rangle \in E.
$$

It multiplies by `t(X)` on the `R_q` side:

$$
u^{(i)} \cdot t(X) \in E.
$$

Then it views the result as a `K`-vector `[ut^{(i)}_1, \dots, ut^{(i)}_d]`.

The first coordinate `ut^{(i)}_1` is the batched constant term.

That is the subtle but beautiful point:

- multiplication by `t(X)` happens in the ring direction;
- reading the first `K`-coordinate extracts the constant term;
- this decodes the monomials back to the projected digits.

The verifier checks

$$
ut^{(i)}_1 = \langle ts(s), v^{(i)} \rangle,
$$

where

$$
v^{(i)} := H^{(i)\top} ts(r) \in K^d.
$$

Intuition:

- `u^{(i)}` is the batched monomial view of the digits;
- `v^{(i)}` is the batched honest digit view;
- the equality above says those are the same objects after decoding.

This is the exact monomial-lookup layer from LatticeFold+, but now it is being applied only to the projected matrix `H`.

## Phase 7: Recombine The Digits

After digitwise decoding, the verifier recombines them in base `d'`:

$$
v :=
\left(
v^{(1)} + d' v^{(2)} + \cdots + d'^{k_g-1} v^{(k_g)}
\right)^\top
\in E.
$$

This `v` is supposed to be the random evaluation of the full projected matrix `H`.

So the range proof outputs two foldable linear statements:

1. a linear statement about the original witness `f` and its projection `H`;
2. a batch-linear statement about the monomial helper vectors `g^{(1)}, \dots, g^{(k_g)}`.

That is exactly why this norm check is suitable as a subroutine inside a folding scheme.

## Why The Verifier Can Trust It

The paper's soundness proof is easiest to understand through its three bad events.

### Bad Event 1: The original witness is large, but the projection looks small

This is the projection failure event.

If some original block `F_{i,j}` has norm above the relaxed threshold `B'`, random projection should make `H_{i,j}` large. If all projected entries still look small, the projection lemma says this happens with tiny probability.

So this is the "approximate JL" part of the proof.

### Bad Event 2: The helper vectors are not really monomials

This is handled by `\Pi_{\mathrm{mon}}`.

If the prover's `g^{(i)}` are not monomials, the monomial RoK fails except with negligible probability.

So this is the "LatticeFold+" part of the proof.

### Bad Event 3: The helper vectors are monomials, but they do not decode to the true projected matrix

This is handled by the random evaluation checks.

If the prover tries to use monomials that decode to some fake small digit matrices `H^{(i)}` whose weighted sum is not the true projection `H`, then:

- the monomial decode checks still force those `H^{(i)}` to match the claimed `v^{(i)}`;
- the linear relation for the original witness still forces `v` to match the true `H`;
- unless those two coincide, the random challenge `r` catches the inconsistency with overwhelming probability.

So the protocol is sound only if all three layers line up:

```text
projection
   +
monomial validity
   +
random evaluation consistency
```

## The Crucial Tradeoff: Exact On The Projection, Approximate On The Original Witness

This is the most important conceptual point in the whole note.

Symphony does **not** give an exact original-witness norm proof the way LatticeFold+ does.

What it gives is:

1. an exact algebraic proof that the projected matrix `H` is representable by small digits;
2. a probabilistic argument that if `H` is small, then the original block norms are at most some larger `B'`.

The paper states the relaxed soundness bound as

$$
B' = \frac{16 B_{d,k_g}}{\sqrt{30}}.
$$

So the extracted witness is guaranteed to satisfy the relaxed relation `R_{\mathrm{rg}}^{\ell_h,B'}`, not necessarily the original relation `R_{\mathrm{rg}}^{\ell_h,B}`.

This is why the authors explicitly call it an **approximate** range proof.

## Is This "Best Of Both Worlds"?

My answer is:

- yes, in the architectural sense;
- no, in the literal strongest-security sense.

### Yes: the hybrid really does combine the two key advantages

From the random-projection side, Symphony gets:

- a short object `H` to norm-check instead of the full witness;
- only `k_g = O(1)` monomial helper vectors in practice;
- a verifier that is polylogarithmic rather than linear in the witness length.

From the LatticeFold+ side, Symphony gets:

- an algebraic range proof with monomial lookup;
- no need for integer-heavy verifier logic outside sumcheck;
- cheap monomial commitments and clean linear reduced statements.

Most importantly, by projecting first, Symphony avoids the whole double-commitment / commitment-transformation layer that LatticeFold+ needed for full ring-vector norm checking.

### No: there are real tradeoffs

The hybrid gives up some things:

- the original witness guarantee is approximate, not exact;
- the bound extracted in soundness is `B'`, not the original `B`;
- the statement is blockwise `\ell_2`, not raw coefficientwise `\ell_\infty`;
- completeness has a small failure probability because honest random projection can occasionally overshoot.

So "best of both worlds" is true if you mean:

- short verifier,
- cheap prover,
- monomial lookup instead of bit decomposition,
- no double commitments.

It is false if you mean:

- exact original norm proof with no relaxation.

## Relation To LaBRADOR

Since you are already thinking in LaBRADOR terms, here is the clean comparison.

LaBRADOR-style intuition:

- use a random projection to reduce a large norm statement to a smaller one.

Symphony keeps that idea, but changes almost everything around it:

- the projection is structured as `I \otimes J`, so the verifier only needs the small `J`;
- the projected object is then certified by monomial lookup, not by the LaBRADOR/JL consistency machinery;
- the result is polylog verifier complexity, not linear-time verifier complexity.

So it is fair to say Symphony is "JL-like plus monomial lookup", but technically it is closer to the random-projection line `[GHL22; BS23; KLNO25]` than to LaBRADOR's exact transcript-derived JL sketching mechanism.

## Efficiency

This is where the hybrid really pays off.

### 1. Monomial layer cost

For the `k_g` monomial vectors:

- one degree-3 sumcheck over size `n`;
- prover work beyond the sumcheck:

$$
O(n k_g)
$$

field additions and

$$
O(n)
$$

field ops;

- verifier work beyond the sumcheck:

$$
O(k_g d + \log n)
$$

field ops.

This is the cheap LatticeFold+ part.

### 2. Projection layer cost

The approximate range proof `\Pi_{\mathrm{rg}}` adds:

- projection computation:

$$
n d \lambda_{\mathrm{pj}}
$$

low-norm integer additions;

- helper commitment work:

$$
O(k_g \kappa n)
$$

ring additions;

- plus the monomial-layer costs above.

The verifier cost beyond the sumcheck is:

- `k_g t` ring ops and `d` field ops for the decode checks;
- `k_g d` scalar multiplications between `\mathbb{Z}_q` and `K`;
- plus the monomial verifier cost.

So the expensive part is still very light compared to committing to arbitrary witnesses.

### 3. Why this is better than plain LatticeFold+

LatticeFold+ had to handle the full coefficient matrix of the original ring witness.

That forced:

- either `d` monomial commitments naively, or
- double commitments plus commitment transformation to compress them.

Symphony projects first, then only proves range on the projected matrix `H`.

So in practice it needs only `k_g` monomial vectors, where `k_g` is a small constant. That is the main asymptotic and concrete simplification.

### 4. Why this is better than plain LaBRADOR-style projection

A pure projection-based norm check still has to prove that the projected object is honestly small in a verifier-friendly way.

Symphony's monomial lookup gives exactly that:

- cheap prover,
- algebraic verifier,
- folding-friendly linear output statement.

So projection shrinks the problem, and monomials solve the smaller problem elegantly.

### 5. Concrete parameter intuition

In the candidate instantiation table, the paper gives:

- `d = 64`,
- `k_g = 3`,
- `B_{d,k_g} = 121117`,
- and relaxed bound `B' = 353806`.

The important practical point is:

- only **three** monomial vectors are needed.

That is dramatically simpler than a `d = 64`-column full coefficient proof.

But the equally important caveat is:

- the relaxed bound `B'` is much larger than the intended input bound `B`.

That is the price of the projection-plus-digits hybrid. The paper accepts this because the folding depth is small, so some slack is tolerable.

## The Cleanest Mental Model

If I had to compress the whole construction into one sentence, it would be:

> Symphony proves the original witness is low norm by proving that a structured random projection of its coefficient matrix is low, and it proves that projected low-norm statement using LatticeFold+'s monomial lookup machinery.

Or even shorter:

```text
project first,
prove exact smallness on the projection,
inherit approximate smallness for the original witness
```

That is the right way to think about it.

## What I Would Remember

1. Symphony's norm check is genuinely a hybrid of random projection and LatticeFold+-style monomial lookup.
2. The projection front-end is what eliminates the need for LatticeFold+'s double-commitment machinery.
3. The monomial back-end is what makes the projected norm proof algebraic and cheap.
4. The result is not an exact original norm proof. It is an approximate one with relaxed bound `B'`.
5. So the construction really does get many of the best engineering properties of both worlds, but not the strongest exact statement of either one.
