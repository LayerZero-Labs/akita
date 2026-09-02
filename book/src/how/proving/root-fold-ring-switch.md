# Root fold and ring switching

One folding step builds a batched relation and switches it into the next ring
witness. The schedule also selects how each commitment group is opened.

## Subring coefficient packing

This section explains why a challenge from a smaller ring lets Akita use a
shorter partial evaluation. The main fact is a commutation rule. Akita can
first fold the source and then take a partial evaluation, or it can first take
the partial evaluations and then fold them. Both orders give the same result.

The rule holds because the challenge acts only on the coefficient axis that the
partial evaluation keeps.

### The three rings

Let `K` be the base field. Let `E` be the field that contains the opening
point, with extension degree `k = [E:K]`. The schedule selects an A ring
dimension `d_A` and a challenge subring dimension `s` that satisfy

```text
d_A = k h s
```

for a power of two `h`. The protocol then uses three rings.

| Ring | Definition | Job |
|---|---|---|
| A ring | `R = K[X]/(X^d_A + 1)` | Holds the committed source and the A relation |
| Challenge subring | `S = K[Y]/(Y^s + 1)` | Holds each sparse fold challenge |
| Extension opening ring | `C = E[Y]/(Y^s + 1)` | Holds each shortened partial evaluation |

Only `S` embeds into `R`. The embedding is

```text
Y -> X^(k h).
```

This is a ring embedding because

```text
(X^(k h))^s = X^d_A = -1 in R.
```

The relation `Y^s = -1` in `S` is therefore preserved in `R`. The protocol does
not sample a general A ring challenge and then delete most of its coefficients.
It samples an element of the smaller ring `S`. After embedding into `R`, those
coefficients occupy regularly spaced positions.

The opening ring `C` has the same polynomial modulus as `S`, but its
coefficients lie in `E`. It is useful to view it as `k` base field coordinate
planes, each of length `s`.

The number `s` is the polynomial dimension of the challenge ring. It is not
the extension degree and it is not the number of challenges. Akita samples one
challenge for each live claim and block pair. Each such challenge has `s`
coefficient positions:

```text
c_i(Y) = c_(i,0) + c_(i,1)Y + ... + c_(i,s-1)Y^(s-1).
```

The challenge is sparse, so the schedule selects how many of those positions
are nonzero. The dimension `s` determines the challenge family and the number
of extension field coefficients retained by each partial.

### Split the A ring coefficient index

Every A ring coefficient index has one unique form

```text
ell = a + k h j,

0 <= a < k h,
0 <= j < s.
```

The index `a` is the part that the partial evaluation contracts. The index `j`
is the part that remains. For one block `i` and one position `x`, write

```text
F_(i,x)(X)
  = sum_(j < s) sum_(a < k h)
      f_(i,x,a,j) X^(a + k h j).
```

The coefficient variables in the opening point split in the same order.
`r_pack` has `log2(k h)` coordinates and supplies the weights for `a`.
`r_tail` has `log2(s)` coordinates and later supplies the weights for `j`.

The following grid shows the layout. Each cell is one coefficient in `K`.

```text
                         contracted by r_pack
                    a = 0   1   2   ...   k h - 1
                  +--------------------------------
       j = 0      |  f     f   f           f       | -> e_i[0]   in E
       j = 1      |  f     f   f           f       | -> e_i[1]   in E
       j = 2      |  f     f   f           f       | -> e_i[2]   in E
         ...      |  ...                         ...|
       j = s - 1  |  f     f   f           f       | -> e_i[s-1] in E
                  +--------------------------------

                    k h coefficients in K              s values in E
```

The position point `r_M` also contracts the position index `x`. After these two
contractions, the prover has one polynomial

```text
e_i(Y)
  = sum_(x,a,j)
      w(r_M, x) w(r_pack, a) f_(i,x,a,j) Y^j
  in C.
```

Here and below, `E` is the extension field and lowercase `e_i` is a partial
evaluation.

### Why the partial is shorter

An evaluation trace partial uses one full A ring element. It therefore has
`d_A` base field coordinates.

The packed partial has `s` coefficients in `E`. Each coefficient in `E` has
`k` coordinates in `K`, so the packed partial has

```text
k s = d_A / h
```

base field coordinates. The exact reduction factor is `h`.

The contraction consumes `k h` base field coefficients for each fixed `j` and
returns one value in `E`. That value still needs `k` base field coordinates.
The contraction therefore removes a factor of `h`, not a factor of `k h`.

The implementation stores the packed partial in this order:

```text
[claim][block][extension coordinate][subring coefficient].
```

At a nonterminal fold, the prover decomposes these `k s` coordinates into
opening digits. The D commitment, or its compressed H form, binds those digits
before the fold challenge is sampled. The shorter coordinate list reduces the
D or H input and the part of the next witness that carries the opening digits.

For an admitted geometry with `B` live claim and block pairs, opening digit
depth `delta_open`, and D ring dimension `d_D`, the number of D ring input
elements changes from

```text
evaluation trace:  B * (d_A / d_D) * delta_open
coefficient packing: B * (k s / d_D) * delta_open.
```

This is the direct reduction in the partial evaluation part of the relation.

### Why the challenge subring makes this valid

For each claim and block pair `i`, Akita samples

```text
c_i(Y) in S.
```

The A relation uses the embedded challenge

```text
c_i(X^(k h)) in R.
```

Its nonzero coefficients occur only at A ring positions

```text
0, k h, 2 k h, ..., (s - 1) k h.
```

Multiplication by this embedded challenge changes `j` but does not mix the
different values of `a`. Negacyclic wraparound in `j` produces the same minus
sign in `S` and `R`. The partial evaluation can therefore contract `a` before
or after challenge multiplication.

```text
source family {F_(i,x)}_x -- multiply by c_i(X^(k h)) --> {Z_x}_x
             |                                                  |
             | contract x and a                                | contract x and a
             v                                                  v
       partial e_i in C -- multiply by c_i(Y) ---------->    L(Z) in C
```

Writing this diagram as an equation gives

```text
L(c_i(X^(k h)) F_i) = c_i(Y) L(F_i).
```

After summing the blocks, an honest witness satisfies

```text
L(Z)(Y) = sum_i c_i(Y) e_i(Y)  in C.
```

This identity is the reason the protocol can replace a full A ring partial
with the shorter packed partial. A general challenge in `R` would mix `a` and
`j`. The contraction would then discard information needed to reproduce the
fold, and this identity would fail.

### Finish the claimed opening

The partial leaves the `j` index open. The scalar opening row contracts it with
the tail point and contracts the block index with `r_B`:

```text
sum_i w(r_B, i)
  sum_(j < s) w(r_tail, j) e_i[j]
  = v in E.
```

This is the original extension field opening claim. Coefficient packing does
not use a trace map. It therefore does not need extension opening reduction,
or EOR, when `k > 1`.

### The packing consistency quotient

Ordinary multiplication of two degree less than `s` polynomials can have
degree as high as `2s - 2`. Equality in `C` means that the difference is a
multiple of `Y^s + 1`. The prover supplies the quotient `Q_pack` such that

```text
sum_i c_i(Y) e_i(Y) - L(G z_hat)(Y)
  = (Y^s + 1) Q_pack(Y).
```

The quotient has `s` coefficients in `E`, so it also has exactly `k s` base
field coordinates before digit decomposition. It receives the same factor `h`
coordinate reduction as a packed partial.

The verifier checks this identity at `Y = alpha`. It evaluates the same
challenge at two different points:

```text
c_i(alpha)            for packing consistency,
c_i(alpha^(k h))      for the A relation.
```

There is one challenge and one coefficient list. The two evaluations follow
from the embedding `Y -> X^(k h)`.

Stage 2 checks the scalar opening and the packing consistency equation against
the same final witness evaluation point used by the range and native relation
terms. It keeps the packing weights in a factorized form. The prover and
verifier do not allocate one dense weight table with the size of the complete
witness.

### One packing fold in order

1. The schedule fixes `d_A`, `s`, the opening basis, and the challenge family.
2. The prover contracts the position axis and the `a` coefficient axis to form
   each `e_i(Y)`.
3. The prover decomposes the `k s` base field coordinates of each partial into
   digits and binds their D or H payload.
4. The transcript samples one `c_i(Y)` for every live claim and block pair.
5. The prover folds the A ring sources with `c_i(X^(k h))` and computes
   `Q_pack` from the high half of `sum_i c_i(Y)e_i(Y)`.
6. The prover binds `Q_pack` and the next witness, then the transcript samples
   `alpha`.
7. Stage 2 checks the claimed opening and the packing consistency equation.
   The A rows check the same challenges through their embedded A ring form.

### Worked production example

Consider the fp32 candidate

```text
d_A = 1024,
k   = 4,
s   = 128,
h   = 2,
k h = 8.
```

The A ring coefficient index is `a + 8j`. For each fixed `j`, the partial
evaluation contracts eight base field coefficients into one value in the
degree four extension field. That extension value uses four base field
coordinates.

```text
evaluation trace partial:  1024 coordinates in K
packed partial:              128 values in E
                            = 128 * 4
                            = 512 coordinates in K
```

The packed partial and `Q_pack` are each half as wide as their full A ring
counterparts. The challenge embeds at exponents `0, 8, 16, ..., 1016`.

Choosing `s = 64` with the same `d_A` and `k` would give `h = 4` and only 256
base field coordinates per partial. That choice needs a heavier sparse
challenge to retain the required entropy. It can therefore increase the A
response bound or change the next witness. The planner prices the complete
schedule rather than always choosing the smallest `s`.

### What the size claim does and does not mean

Coefficient packing gives two direct savings at an eligible fold.

1. It removes EOR when the opening point lies in a proper extension field.
2. It reduces each partial and each packing quotient from `d_A` to `k s`
   base field coordinates before digit decomposition.

It does not reduce every proof component by `h`. Gadget depths, matrix ranks,
compression output sizes, response bounds, and later folds can all change. A
fixed size H payload can also hide some of the raw coordinate reduction. The
planner therefore computes the complete setup and proof cost from the selected
geometry.

### Protocol placement

Production planning considers packing only at absolute fold levels 0 and 1.
Those folds use the coefficient `L∞` A security route. Later folds and the
terminal use evaluation trace. A nonterminal level 0 or 1 state without a
complete packing assignment is unsupported.

Commitment identity records the coefficient representation and the A and B
matrices. It does not record the consuming opening method, `s`, or challenge.
The schedule and transcript descriptor record that opening plan. This lets a
later evaluation trace fold consume a flat witness or setup prefix produced by
an earlier packing fold without changing its commitment identity.

The prover binds the complete D payload, or its compressed H payload, before it
draws the subring challenge. It binds `Q_pack` and the next witness before it
draws `alpha`. The verifier replays the same order.

## The root fold

`OpeningClaimsLayout` routes polynomial groups to claims. Each group keeps its
own public point, commitment profile, and opening geometry. The relation order
is final group followed by precommitted groups. A recursive fold uses the same
group rules for its folded witness and an incoming setup prefix.

## Ring switching

The fold relation is stated in a quotient ring, but Stage 2 evaluates ordinary
polynomials at a random field point. Ring switching supplies the missing link.
For every physical relation row, the prover gives the unique polynomial that
accounts for reduction modulo `X^D + 1`.

This operation is distinct from EOR. EOR changes an extension-valued opening
claim before the lattice relation is formed. Ring switching lifts the resulting
lattice relation out of its quotient ring so that sumcheck can prove it over a
field.

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
\tag{1}
$$

Equation (1) is exactly the quotient in

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
\tag{2}
$$

The prover digit-decomposes each `r_i` and appends those digits to the recursive
witness. After the verifier substitutes `X = alpha`, the factor `X^D + 1`
becomes the public scalar `alpha^D + 1`. This turns every lifted row into a
field relation suitable for the fused Stage-2 sumcheck.

### Preserve each row's native ring

Akita does not enlarge every relation to one common ring dimension before
computing Equation (2). Consistency and A rows use `d_A`, B rows use `d_B`, and
D rows use `d_D`. Their quotients retain those same native dimensions. This is
both the mathematical layout and the physical recursive-witness layout; the
row geometry records the native dimension and the number of coordinate planes.

The coefficient-packing consistency row is the one nonstandard geometry. It is
an equation over `E[Y]/(Y^s + 1)`, represented as `k` base-field coordinate
planes of length `s`. Its quotient therefore has `k s` physical coordinates.
It is not reinterpreted as one ring of dimension `k s`.

### Compute only the coefficients that survive

The matrix rows use paired cyclic and negacyclic transforms to obtain Equation
(2). Akita performs both convolutions through the same CRT profiles used for
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
the algebra of Equation (2).

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

- `crates/akita-prover/src/protocol/ring_relation.rs`.
- `crates/akita-prover/src/protocol/ring_switch.rs`.
- `crates/akita-prover/src/protocol/coefficient_packing.rs`.
- `crates/akita-types/src/subring_coefficient_packing.rs`.
- `crates/akita-types/src/proof/coefficient_packing_relation.rs`.
- `crates/akita-verifier/src/protocol/core/fold/` replays the relation and
  rejects a proof whose dimensions or quotient structure do not match the
  selected schedule.
