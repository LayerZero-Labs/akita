# LatticeFold+ Norm Check

This note extracts the LatticeFold+ norm check from the paper and rewrites it in the same spirit as `LABRADOR_LINEAR_ONLY.md`: isolate the core algebra, explain why the protocol is built that way, and separate the real costs from the bookkeeping.

The paper's "norm check" is really an `\ell_\infty` range proof for a committed witness vector `f \in R_q^n`. Concretely, it proves that every coefficient of every ring element in `f` lies in `(-B, B)`. Since the commitment is only binding for low-norm openings, this is the mechanism that makes folding sound.

Main paper anchors:

- Monomial encoding and decoding: Section 2.1, pages 8-10
- Monomial set check `\Pi_{\mathrm{mon}}`: Section 4.2, pages 17-20
- Warm-up range check for `(-d/2, d/2)`: Section 4.3, page 21
- Full range / norm check `\Pi_{\mathrm{rgchk}}`: Section 4.3, pages 22-24
- Commitment transformation and communication caveat: Section 4.4, pages 25-30
- End-to-end efficiency estimate: Section 5.3, pages 40-42

## Scope

This note focuses on the part of LatticeFold+ that proves

$$
\|f\|_\infty < B
$$

for a committed `f \in R_q^n`, where the norm is coefficient-wise over the ring

$$
R_q = \mathbb{Z}_q[X] / \langle X^d + 1 \rangle.
$$

Equivalently, if

$$
f_i = \sum_{o=0}^{d-1} f_{i,o} X^o,
$$

then the goal is to prove

$$
f_{i,o} \in (-B, B)
\qquad
\text{for every } i \in [n],\ o \in [d].
$$

The high-level reduction is:

1. Reduce a coefficient bound to a small-digit bound.
2. Encode each small digit as a monomial.
3. Prove monomial membership efficiently.
4. Decode the monomials back to the claimed digits using one fixed ring element `\psi`.
5. Conclude the original coefficients are in range, hence `\|f\|_\infty < B`.

So the "norm check reduces to the monomial check" in a very literal sense: once the prover convinces the verifier that certain helper objects are genuine monomials, the rest of the range proof is mostly linear algebra plus constant-term extraction.

## Symbol Table

| Symbol | Meaning |
| --- | --- |
| `d` | Ring dimension, a power of two |
| `d'` | Half-dimension `d / 2` |
| `M` | Monomial set `{0, 1, X, ..., X^{d-1}} \subset R_q` |
| `\exp(a)` | Monomial encoding of a small integer `a \in (-d, d)` |
| `\mathrm{EXP}(a)` | Allowed monomial encodings of `a`; singleton except at `a = 0` |
| `\psi` | Fixed ring element whose constant term decodes monomials back to integers |
| `f \in R_q^n` | Witness vector whose norm must be bounded |
| `\mathrm{cf}(f) \in \mathbb{Z}_q^{n \times d}` | Coefficient matrix of `f` |
| `B = (d')^k` | Target coefficient bound |
| `D_f \in \mathbb{Z}_q^{n \times dk}` | Base-`d'` digit decomposition of `\mathrm{cf}(f)` |
| `M_f \in M^{n \times dk}` | Monomial encoding of all digits in `D_f` |
| `CM_f = \mathrm{dcom}(M_f)` | Double commitment to `M_f` |
| `\tau_D = \mathrm{split}(\mathrm{com}(M_f)) \in (-d', d')^n` | Short decomposition of the linear commitment to `M_f` |
| `m_\tau \in \mathrm{EXP}(\tau_D)` | Monomial encoding of `\tau_D` |
| `\Pi_{\mathrm{mon}}` | Monomial set check |
| `\Pi_{\mathrm{rgchk}}` | Full coefficient range / norm check |

## Big Picture

At a high level, the norm check looks like this:

```text
committed witness f in R_q^n
        |
        | coefficient expansion
        v
  cf(f) in Z_q^{n x d}
        |
        | base-d' decomposition, where B = (d')^k
        v
  D_f in (-d', d')^{n x d x k}
        |
        | encode each small digit a as one monomial exp(a)
        v
  M_f in M^{n x d x k}
        |
        | prove M_f really is monomial-valued
        | via Pi_mon
        v
  random linear summaries of the digit blocks
        |
        | decode monomials back to integers using ct(psi * -)
        v
  recover the claimed digits of cf(f)
        |
        | recombine digits in base d'
        v
  cf(f) lies in (-B, B)^{n x d}
        |
        v
  ||f||_infty < B
```

The key trick is the middle one:

- bounded integers are converted into monomials;
- monomial-ness is easy to test algebraically;
- one fixed ring element `\psi` decodes those monomials back to the original integers.

That is why the protocol feels much more algebraic than the bit-decomposition range proof in LatticeFold.

## Why This Check Exists

Ajtai-style lattice commitments are only binding for low-norm openings. Folding, however, combines witnesses linearly. If you keep folding without controlling witness size, the accumulated witness can leave the binding regime and soundness breaks.

LatticeFold+ therefore needs a proof that each committed witness is coefficient-wise bounded. Since the paper uses the coefficient `\ell_\infty` norm, this reduces to checking a range on every coefficient.

So throughout this note:

- "range check" means "each coefficient is in `(-B, B)`";
- "norm check" means exactly the same thing, because

$$
\|f\|_\infty < B
\iff
\mathrm{cf}(f) \in (-B, B)^{n \times d}.
$$

## Phase 1: Encode A Small Integer As A Monomial

This is the conceptual heart of the construction.

### 1.1 The monomial set

The paper defines

$$
M = \{0, 1, X, X^2, \dots, X^{d-1}\} \subset R_q.
$$

These are the simplest possible ring elements:

- zero, or
- a single monomial with coefficient `1`.

The encoding of an integer `a \in (-d, d)` is

$$
\exp(a) := \mathrm{sgn}(a) X^a \in M.
$$

This is well-defined inside `R_q` because if `a < 0`, then using `X^d = -1`,

$$
-X^a = X^{a+d}.
$$

So a negative signed monomial is still represented by an ordinary monomial in `M`.

The set-valued version `\mathrm{EXP}(a)` is:

- a singleton `{ \exp(a) }` when `a \neq 0`;
- a small three-element set when `a = 0`.

That zero-case ambiguity is harmless. It just means the proof system does not insist on one unique monomial representative for zero.

### 1.2 Why monomials are nice

Monomials are attractive for two different reasons:

1. They are easy to commit to.
2. They are easy to test.

For commitments, multiplying a matrix `A` by a monomial entry does not look like a general ring multiplication. It only rotates coefficients and maybe flips signs. So commitments to monomial vectors/matrices are much cheaper than commitments to arbitrary ring vectors/matrices.

For testing, monomials satisfy a special polynomial identity:

$$
a(X^2) = a(X)^2.
$$

This is obvious for `a(X) = X^t`, because both sides equal `X^{2t}`.

It is false for a generic polynomial because cross-terms appear. For example,

$$
(1 + X)^2 = 1 + 2X + X^2
\neq
1 + X^2 = a(X^2).
$$

Lemma 2.1 says, over `\mathbb{Z}_q[X]`, this identity characterizes monomials exactly:

$$
a(X^2) = a(X)^2
\iff
a \in \{0, 1, X, X^2, \dots \}.
$$

That is the algebraic test the protocol exploits.

## Phase 2: Decode A Monomial Back To Its Integer

The second heart of the construction is the decoder polynomial

$$
\psi := \sum_{i=1}^{d'-1} i \cdot (X^{-i} + X^i),
\qquad d' := d/2.
$$

The constant term map

$$
\mathrm{ct} : R_q \to \mathbb{Z}_q
$$

extracts the coefficient of `X^0`.

The crucial lemma is:

$$
a \in (-d', d')
\quad \Longleftrightarrow \quad
\exists b \in M \text{ such that } \mathrm{ct}(b \cdot \psi) = a,
$$

and in fact such a `b` must lie in `\mathrm{EXP}(a)`.

This is Lemma 2.2 in the paper.

### 2.1 Intuition for `\psi`

Think of `\psi` as a "lookup wheel" whose coefficients are the integers

```text
exponent:  ...  -3  -2  -1   0   1   2   3  ...
coeff:     ...   3   2   1   0   1   2   3  ...
```

Multiplying by a monomial rotates this wheel. The constant term then reads off whichever label lands on exponent `0`.

Examples:

- If `a = 2`, then `b = X^2`, and the term `2 X^{-2}` in `\psi` rotates into the constant term. So `\mathrm{ct}(X^2 \psi) = 2`.
- If `a = -2`, then `b = \exp(-2) = -X^{-2} = X^{d-2}` in `R_q`, and the same mechanism yields `\mathrm{ct}(b \psi) = -2`.
- If `a = 0`, then the allowed representatives all give constant term `0`.

So `b` behaves like an encoding of `a`, and `\mathrm{ct}(b \psi)` decodes it.

This is the reason the whole range proof works:

- monomial membership says "you are a valid codeword";
- the constant-term identity says "you decode to the claimed integer".

Together those imply a true range statement.

## Phase 3: The Monomial Set Check `\Pi_{\mathrm{mon}}`

Suppose the prover claims that a committed matrix

$$
M \in R_q^{n \times m}
$$

has every entry in the monomial set `M`.

The verifier does not want to inspect all `nm` entries directly. Instead, LatticeFold+ turns that claim into one batched algebraic test.

### 3.1 The local identity

For each entry `M_{i,j}`, monomial membership implies

$$
\mathrm{ev}_{M_{i,j}}(\beta)^2 = \mathrm{ev}_{M_{i,j}}(\beta^2)
$$

for every field point `\beta`.

If an entry is not a monomial, Corollary 4.1 says this identity fails for a random `\beta` with high probability.

So at the local level, the verifier would like to check

$$
\mathrm{ev}_{M_{i,j}}(\beta)^2 - \mathrm{ev}_{M_{i,j}}(\beta^2) = 0
$$

for all `(i,j)`.

### 3.2 From local checks to one batched sumcheck

The protocol batches these local checks column-by-column and row-by-row.

Fix a column `j`. Define two length-`n` vectors over the sumcheck field:

$$
m^{(j)} = \bigl(\mathrm{ev}_{M_{0,j}}(\beta), \dots, \mathrm{ev}_{M_{n-1,j}}(\beta)\bigr),
$$

and

$$
m'^{(j)} = \bigl(\mathrm{ev}_{M_{0,j}}(\beta^2), \dots, \mathrm{ev}_{M_{n-1,j}}(\beta^2)\bigr).
$$

The verifier sends a random row-selector challenge `c \in C^{\log n}`. The sumcheck claim for column `j` is

$$
\sum_{i \in [n]}
\mathrm{eq}(c, \langle i \rangle)
\cdot
\left(
\widetilde{m^{(j)}}(\langle i \rangle)^2
-
\widetilde{m'^{(j)}}(\langle i \rangle)
\right)
= 0.
$$

Then all `m` column-claims are randomly combined into one degree-3 sumcheck.

Intuition:

- `\beta` checks the monomial identity inside each ring element.
- `c` picks out a random location in the Boolean hypercube view of the rows.
- random linear combination across columns prevents the prover from hiding one bad column behind many good ones.

After the sumcheck, the verifier only needs a few evaluations of multilinear extensions rather than the whole matrix.

### 3.3 What the prover sends at the end

After reduction, the prover sends for each column

$$
e_j = M_{*,j}^\top \cdot \mathrm{tensor}(r) \in R_q,
$$

where `r` is the sumcheck challenge.

The verifier then checks

$$
\mathrm{eq}(c, r)
\cdot
\sum_{j=1}^m \alpha^j
\left(
\mathrm{ev}_{e_j}(\beta)^2 - \mathrm{ev}_{e_j}(\beta^2)
\right)
= v,
$$

where `v` is the claimed sumcheck evaluation.

So the output of `\Pi_{\mathrm{mon}}` is not "I have checked every monomial directly." Instead, it is:

- the matrix remains committed,
- but the verifier now holds one random evaluation summary `(r, e)`,
- and the knowledge soundness theorem says any prover who answers correctly must know a matrix whose entries are all monomials.

## Why `\Pi_{\mathrm{mon}}` Is So Efficient

This is where LatticeFold+ wins.

Remark 4.3 gives the main cost picture for a monomial matrix `M \in M^{n \times m}`:

1. The sumcheck is degree `3` and runs over the field `C`, not over the ring `R_q`.
2. The prover can compute all multilinear evaluations `e_j` using only `O(n)` field multiplications plus `O(nm)` field additions.
3. The commitment `\mathrm{com}(M)` is cheap because each entry is a monomial.

More concretely:

- computing `\mathrm{tensor}(r)` takes `O(n)` scalar multiplications;
- each `e_j = \langle M_{*,j}, \mathrm{tensor}(r) \rangle` is computed by scattering row weights into coefficient slots, so it uses additions rather than arbitrary ring multiplications;
- committing to `M` costs about `n \kappa m` ring additions, equivalently about `n \kappa d m` scalar additions.

This is much cheaper than committing to a dense arbitrary ring matrix, which would require true ring multiplications.

That observation is the engine of the whole norm check: once digits are turned into monomials, the expensive part becomes cheap.

## Phase 4: Warm-Up Range Check For `(-d', d')`

Before the full norm check, the paper gives a warm-up: prove that

$$
\tau \in (-d', d')^n.
$$

The prover does this by sending a monomial encoding

$$
m_\tau \in \mathrm{EXP}(\tau) \subset M^n.
$$

Then:

1. Run `\Pi_{\mathrm{mon}}` on `m_\tau`.
2. Let the output include

$$
b = \langle m_\tau, \mathrm{tensor}(r) \rangle \in R_q.
$$

3. The prover separately sends

$$
a = \langle \tau, \mathrm{tensor}(r) \rangle \in C.
$$

4. The verifier checks

$$
\mathrm{ct}(\psi \cdot b) = a.
$$

Why does this work? Because if every coordinate is honestly encoded, then

$$
\mathrm{ct}(\psi \cdot m_{\tau,i}) = \tau_i
$$

coordinate-wise, and linearity gives

$$
\mathrm{ct}(\psi \cdot \langle m_\tau, \mathrm{tensor}(r) \rangle)
=
\langle \tau, \mathrm{tensor}(r) \rangle.
$$

If some coordinate is bad, then the difference

$$
\mathrm{ct}(\psi \cdot m_\tau) - \tau
$$

is a nonzero vector, and the random tensor challenge catches it with Schwartz-Zippel type probability.

So the warm-up range proof is:

```text
prove helper is monomial-coded
        +
prove helper decodes to tau
        =
prove tau is in (-d', d')^n
```

This is already the whole philosophy of the final norm check.

## Phase 5: Full Norm Check For `\|f\|_\infty < B`

Now we lift the warm-up from one small integer per coordinate to one ring element per coordinate.

### 5.1 Decompose coefficients in base `d'`

Write

$$
B = (d')^k.
$$

The coefficient matrix of `f` is

$$
\mathrm{cf}(f) \in \mathbb{Z}_q^{n \times d}.
$$

Decompose every coefficient in base `d'`:

$$
D_f = [D_{f,0}, \dots, D_{f,k-1}]
=
G^{-1}_{d',k}(\mathrm{cf}(f))
\in
\mathbb{Z}_q^{n \times dk},
$$

with each digit bounded by `(-d', d')`.

Conceptually, each coefficient satisfies

$$
f_{i,o}
=
\sum_{t=0}^{k-1} (d')^t \cdot D_{f,t}[i,o].
$$

So the hard statement "`f_{i,o}` is in `(-B, B)`" is reduced to the easier statement "each digit `D_{f,t}[i,o]` is in `(-d', d')`."

### 5.2 Encode every digit as a monomial

Replace every digit by its monomial encoding:

$$
M_f \in \mathrm{EXP}(D_f) \subset M^{n \times dk}.
$$

Think of `M_f` as `k` blocks, each block being an `n \times d` monomial matrix:

```text
M_f = [ M_{f,0} | M_{f,1} | ... | M_{f,k-1} ]
          nxd      nxd             nxd
```

Each block corresponds to one base-`d'` digit position.

### 5.3 Why a double commitment is needed

Naively, the prover could send `\mathrm{com}(M_f)`, which has `dk` committed columns. That is too bulky for recursive folding.

Instead the paper compresses `\mathrm{com}(M_f)` into one short vector:

$$
\tau_D := \mathrm{split}(\mathrm{com}(M_f)) \in (-d', d')^n,
$$

and then sends the double commitment

$$
CM_f := \mathrm{dcom}(M_f) = \mathrm{com}(\tau_D).
$$

This turns the large matrix commitment into one ordinary Ajtai commitment to a short vector.

Because `\tau_D` itself must also be low-norm and well-formed, the prover additionally encodes it as

$$
m_\tau \in \mathrm{EXP}(\tau_D).
$$

So the full range proof manipulates two monomial objects:

- `M_f`, which encodes the digits of `f`;
- `m_\tau`, which encodes the short decomposition of `\mathrm{com}(M_f)`.

## Phase 6: The Full Protocol `\Pi_{\mathrm{rgchk}}`

The full range check is now easy to summarize.

### 6.1 What it proves

Given commitments

$$
(\mathrm{cm}_f, CM_f, \mathrm{cm}_{m_\tau}),
$$

the prover wants to convince the verifier that:

1. `f` opens `\mathrm{cm}_f`;
2. `M_f` is a valid monomial encoding of the base-`d'` digits of `\mathrm{cf}(f)`;
3. `\tau_D` is the short decomposition of `\mathrm{com}(M_f)`;
4. `m_\tau` is a valid monomial encoding of `\tau_D`;
5. therefore all coefficients of `f` lie in `(-B, B)`.

### 6.2 What the protocol actually does

Construction 4.4 can be read as:

1. Run a batched `\Pi_{\mathrm{mon}}` on both `M_f` and `m_\tau`.
2. This yields:
   - `b = \langle m_\tau, \mathrm{tensor}(r) \rangle`,
   - and for each digit block `t`, a vector

$$
u_t = M_{f,t}^\top \cdot \mathrm{tensor}(r) \in R_q^d.
$$

3. The prover sends

$$
v = \mathrm{cf}(f)^\top \cdot \mathrm{tensor}(r) \in C^d
$$

and

$$
a = \langle \tau_D, \mathrm{tensor}(r) \rangle \in C.
$$

4. The verifier checks two decoding identities:

$$
\mathrm{ct}(\psi \cdot b) = a
$$

and

$$
\mathrm{ct}
\left(
\psi \cdot
\left(
u_0 + d' u_1 + \cdots + (d')^{k-1} u_{k-1}
\right)
\right)
= v.
$$

That second equation is the whole norm check in one line.

It says:

- each `u_t` is a random linear summary of the monomial encodings of the `t`-th digit block;
- multiplying by `\psi` and taking constant terms decodes those monomials back to the actual small digits;
- weighting by powers of `d'` recombines the digits into the original coefficients of `f`.

So the verifier does not check all coefficients individually. It checks one random linear summary of the coefficient table, but in a way that is binding to a committed witness and knowledge-sound.

## Why This Really Reduces To The Monomial Check

This is the core logical structure:

1. `\Pi_{\mathrm{mon}}` proves the helper objects are genuine monomials.
2. Lemma 2.2 says genuine monomials plus the `\psi` constant-term relation imply the claimed integers are in the right range.
3. Base-`d'` recombination turns small digits into the full coefficient vector.

So the "hard nonlinear part" of the range proof is almost entirely pushed into Step 1.

Everything after that is linear:

- inner products with `\mathrm{tensor}(r)`,
- linear recombination with powers of `d'`,
- constant-term extraction after multiplication by the fixed public element `\psi`.

That is exactly why the protocol is simpler than LatticeFold's bit-decomposition range proof.

Another way to say it:

```text
norm check
  =
  monomial set check
  +
  "decode monomials with psi"
  +
  "recombine base-d' digits"
```

The monomial set check is the only place where the verifier needs a genuine proof that a nonlinear property holds entry-wise.

## Visual Summary

Here is the whole proof as a pipeline.

```text
small integer a in (-d', d')
        |
        | encode
        v
   exp(a) in M
        |
        | monomial identity a(X^2) = a(X)^2
        v
   efficient monomial check
        |
        | decode with psi
        v
   ct(psi * exp(a)) = a
```

Now tensor that picture over all digits of all coefficients:

```text
cf(f)
  -> base-d' digits D_f
  -> monomial encodings M_f
  -> batched Pi_mon
  -> decoded digit summaries via ct(psi * -)
  -> recombine with 1, d', ..., (d')^(k-1)
  -> recover cf(f)
  -> conclude ||f||_infty < B
```

## Efficiency

This is the part that matters most in practice.

### 1. Monomial check cost

For a monomial matrix `M \in M^{n \times m}`, Remark 4.3 gives:

- degree-3 sumcheck over the field `C`;
- prover work for the multilinear evaluations:
  - `O(n)` scalar multiplications,
  - `O(nm)` scalar additions;
- commitment work for `\mathrm{com}(M)`:

$$
\approx n \kappa m
$$

ring additions, equivalently about

$$
n \kappa d m
$$

scalar additions.

The key point is not the exact constant. The key point is that these are additions and cheap field operations, not full dense ring multiplications.

### 2. Warm-up range check cost

For `\tau \in (-d', d')^n`, the protocol needs:

- one monomial set check on `m_\tau`;
- one extra transmitted scalar `a`;
- one constant-term check `\mathrm{ct}(\psi b) = a`.

So once monomial membership is available, the extra overhead is tiny.

### 3. Full norm check cost

Let

$$
k = \log_{d'} B.
$$

Then the full range proof uses:

- one batched monomial set check over `dk + 1` monomial columns/objects:
  - the `dk` columns of `M_f`,
  - plus the vector `m_\tau`;
- one double commitment `CM_f = \mathrm{dcom}(M_f)`;
- one monomial commitment `\mathrm{cm}_{m_\tau} = \mathrm{com}(m_\tau)`;
- one transmitted `a \in C`;
- one transmitted `v \in C^d`;
- and the decoded summaries `u_0, ..., u_{k-1}` that come out of the batched monomial proof.

The knowledge error from Lemma 4.7 is

$$
\epsilon_{\mathrm{rg}}
=
\epsilon_{\mathrm{mon}, dk+1}
+
\epsilon_{\mathrm{bind}}
+
\frac{\log n}{|C|},
$$

and using

$$
\epsilon_{\mathrm{mon},m}
=
\frac{2d + m + 4 \log n}{|C|}
+
\epsilon_{\mathrm{bind}},
$$

this becomes

$$
\epsilon_{\mathrm{rg}}
=
\frac{2d + dk + 1 + 5 \log n}{|C|}
+
2 \epsilon_{\mathrm{bind}}.
$$

So the norm check has clean linear dependence on `dk`, which is the number of base-`d'` digits across all `d` coefficients of one ring element.

### 4. Why this beats bit decomposition

This is the biggest conceptual win.

Old LatticeFold range proof:

- decomposes coefficients into bits;
- commits to many bit-decomposed vectors;
- pays roughly `\log_2(B)` helper commitments per witness.

LatticeFold+ range proof:

- decomposes coefficients in base `d' = d/2`, not base `2`;
- encodes each digit as one monomial;
- compresses the resulting `dk` monomial columns using a double commitment;
- uses only cheap monomial commitments plus sumchecks over the field `C`.

Since

$$
k = \log_{d'} B = \frac{\log B}{\log d'},
$$

this is already much smaller than `\log_2(B)` when `d` is moderate.

For the paper's concrete parameter example:

- `d = 64`,
- so `d' = 32`,
- and `B = 2^{10}`,

we get

$$
k = \log_{32}(2^{10}) = 2.
$$

So each coefficient uses only `2` base-`32` digits instead of `10` bits.

That is an enormous structural simplification.

### 5. The caveat: raw `\Pi_{\mathrm{rgchk}}` still exposes a `dk` term

There is one subtlety.

As a standalone protocol, `\Pi_{\mathrm{rgchk}}` outputs the values

$$
u_0, \dots, u_{k-1} \in R_q^d,
$$

so communication still contains a `dk` ring-element term.

The paper immediately addresses this in Section 4.4 and Remark 4.7:

- first transform the double-commitment statement into one about an ordinary linear commitment;
- then compress the bulky `dk` evaluation payload the same way they compressed `\mathrm{com}(M_f)`.

So the right conclusion is:

- the raw algebraic norm check is already much cheaper than LatticeFold's bit proof;
- but the full folding scheme still needs the commitment-transformation layer to make the transcript truly compact.

### 6. End-to-end effect inside folding

By Theorem 5.3, once all optimizations are applied, the full folding reduction has:

- prover time dominated by

$$
L n \kappa
$$

ring multiplications plus

$$
O(L n \kappa d k)
$$

ring additions;

- verifier time dominated by

$$
O(L d k)
$$

ring multiplications, excluding hashing.

The paper explicitly contrasts this with LatticeFold, where the prover is dominated by an `n`-sized degree-4 sumcheck plus `L \log_2(B)` decomposed commitments. Their summary claim is that LatticeFold+ is `\Omega(\log B)` faster asymptotically, and they estimate about a `5x` concrete prover speedup from benchmark data.

## What I Would Remember

If you want the short version, it is this:

1. LatticeFold+ proves a coefficient bound by expressing every small digit as a monomial.
2. Monomial membership is checked by the identity `a(X^2) = a(X)^2`, batched with sumcheck.
3. The fixed public polynomial `\psi` decodes monomials back to integers via the constant term.
4. Base-`d'` digit recombination upgrades the small-digit proof into a full `\|f\|_\infty < B` proof.
5. Efficiency comes from replacing bit commitments with monomial commitments and from using base `d'` digits, so the expensive part scales with `k = \log_{d'} B`, not with `\log_2 B`.

In other words, the norm check is efficient because it turns a generic range-proof problem into:

- a cheap monomial-membership problem, plus
- a very cheap linear decoding problem.

That is the main technical idea.
