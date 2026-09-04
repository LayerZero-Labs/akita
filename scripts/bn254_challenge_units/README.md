# Exact BN254 D64 challenge-unit certificate

This directory certifies pairwise difference invertibility for Akita's
dimension-64 selective-L2 challenge shell over the BN254 scalar field.
The result applies even though `X^64 + 1` splits completely over that field.

Run the self-contained verifier with:

```bash
python3 scripts/bn254_challenge_units/check_d64_kernel.py
```

The expected final line is:

```text
verified: BN254 D64 evaluation kernel has no nonzero vector with squared norm <= 336 (1507538 enumeration nodes)
```

The verifier uses only the Python standard library and exact integer and
rational arithmetic. It does not trust the lattice reduction that produced
the certificate basis.

## Statement

Let

```text
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
S = F_r[Y] / (Y^64 + 1).
```

The certificate proves that every nonzero vector in the integer evaluation
kernel at a primitive 128th root has squared coefficient norm greater than
`336`.

Consequently, every challenge family contained in the coefficient ball
`||c||_2^2 <= 84` has pairwise-unit differences. This covers the current
selective-L2 shell and leaves room to redesign the shell or rejection policy.

Every challenge in the selective-L2 D64 shell has 31 coefficients of magnitude
one and 11 coefficients of magnitude two. Its squared coefficient norm is

```text
31 + 4 * 11 = 75.
```

For any two challenges `c` and `c'` in the current shell, the triangle
inequality gives

```text
||c - c'||_2^2 <= (||c||_2 + ||c'||_2)^2 = 4 * 75 = 300 < 336.
```

Consequently, a nonzero difference cannot vanish at the certified root. The
other primitive 128th roots are Galois conjugates. Substitution by an odd power
of `Y` acts as a signed permutation on the coefficient vector, so their kernel
lattices are isometric. A nonzero challenge difference therefore vanishes at
none of the 64 roots and is a unit in `S`.

Operator-norm rejection only takes a subset of this shell, so it preserves the
unit-difference property. If `64` divides an ambient power-of-two ring degree
`d`, substitution `Y = X^(d / 64)` embeds the inverse from `S` into
`F_r[X] / (X^d + 1)`. The same challenge set is therefore strong in every such
ambient ring.

## Native D128 candidates under investigation

The exact D64 result is already sufficient for an embedded challenge in every
larger power-of-two ring. The following candidates explore a different tradeoff:
use more than 64 coefficient positions in `F_r[X] / (X^128 + 1)` while keeping
the position set stable under every cyclotomic Galois automorphism.

For a nonzero exponent `i`, multiplication by an odd integer modulo 256
preserves the 2-adic valuation of `i`. Each row below therefore selects a union
of complete Galois orbits. A shortest-vector certificate at one primitive
256th root would apply to every root by signed coefficient permutation.

| candidate | selected 2-adic valuations | positions | shell `(±1, ±2)` | raw bits | `max ||c||_2^2` | difference bound |
|---|---:|---:|---:|---:|---:|---:|
| S96 | `{0, 1}` | 96 | `(35, 1)` | 129.219824 | 39 | 156 |
| S88 | `{0, 2, 3}` | 88 | `(35, 2)` | 129.224682 | 43 | 172 |
| S80 | `{0, 2}` | 80 | `(36, 3)` | 128.630976 | 48 | 192 |

The structural claims and support counts are exact. The S96 threshold-15 row
now also has an exact accepted-support certificate; the other operator-norm
figures remain Monte Carlo estimates. With 200,000 samples and seed
`20260903`, the estimated accepted support is:

| candidate | threshold 14 | threshold 15 | threshold 16 |
|---|---:|---:|---:|
| S96 | 128.683676 bits | 128.991760 bits | 129.129505 bits |
| S88 | 128.320471 bits | 128.815892 bits | 129.047958 bits |
| S80 | 127.079053 bits | 127.873634 bits | 128.274925 bits |

Reproduce the experiment in a Python environment that provides NumPy:

```bash
python3 scripts/bn254_challenge_units/explore_d128_subspaces.py --samples 200000
```

S96 now has the first artifact needed for protocol use:

```text
scripts/bn254_challenge_units/d128_s96_operator_norm/
```

Its degree-30 exact moment dual proves at least `2^128.497038674319` accepted
challenges under the q=48 runtime predicate at threshold 15. The exact
standalone checker binds the position mask, shell, fixed-point containment,
eight modular moment computations, polynomial positivity, union bound, and
support floor.

The remaining obligation for S96 is an exact evaluation-kernel certificate
excluding every nonzero vector through squared norm 156. S88 and S80 still
need both artifacts. In general, a native candidate needs:

1. an exhaustive evaluation-kernel certificate excluding every nonzero vector
   through the row's difference bound; and
2. an exact accepted-support certificate for the selected operator-norm
   threshold.

Once either artifact pair is complete in D128, substitution
`Y = X^(d / 128)` carries the same strong challenge set and inverse into every
ambient power-of-two ring whose degree `d` is a multiple of 128. A successful
D128 certificate therefore covers D256 and all larger production dimensions;
it does not require a separate rank-256 shortest-vector proof.

## Portability to other moduli

The proof method is not specific to BN254. For a prime modulus `q` and a
power-of-two degree `D`, complete splitting requires `q = 1 mod 2D`. Choose a
primitive `2D`-th root `omega` in `F_q` and form the integer lattice

```text
L(q, omega) = {a in Z^D : sum_i a_i omega^i = 0 mod q}.
```

This lattice has determinant `q`. A certificate basis and its shortest-vector
enumeration are specific to `(q, omega)`, but the checker construction applies
unchanged to every such modulus. Primitive roots for the same modulus give
isometric lattices when the challenge positions are Galois invariant.

There is also a modulus-independent sufficient bound. Let every challenge
satisfy `||c||_2^2 <= E`. Every difference then satisfies
`||c-c'||_2^2 <= 4E`. Parseval's identity over the `D` complex roots and the
arithmetic-geometric mean inequality give

```text
|Norm(c-c')| <= ||c-c'||_2^D <= (4E)^(D/2).
```

If a nonzero difference vanishes at a root modulo `q`, then `q` divides this
nonzero algebraic norm. Therefore

```text
q > (4E)^(D/2)
```

is a universal sufficient condition for pairwise-unit differences. It is
usually conservative. For D64 and `E = 84`, it requires more than 268.554
modulus bits. BN254 has 253.597 modulus bits, so the exact lattice certificate
recovers almost 15 bits beyond this generic resultant bound.

The following table counts the complete integer coefficient ball and finds its
smallest squared radius with at least `2^128` elements. The universal column is
a theorem. The Gaussian-heuristic column estimates where a typical determinant
`q` evaluation lattice should begin to clear the difference radius; it is not
a security bound.

| challenge dimension | minimum `E` | ball support | universal modulus bits | heuristic modulus bits |
|---:|---:|---:|---:|---:|
| 32 | 554 | 128.014448 | 177.819875 | 163.326933 |
| 64 | 65 | 128.240264 | 256.715770 | 195.729887 |
| 128 | 31 | 130.001092 | 445.068564 | 259.096799 |

Recompute the exact ball counts and both columns with:

```bash
python3 scripts/bn254_challenge_units/analyze_dimension_tradeoffs.py
```

The non-monotonicity across dimensions is the norm tradeoff. Lower-dimensional
challenge subrings need much larger coefficients to hold 128 bits. They permit
smaller field moduli but worsen the folding and operator-norm bounds. Higher
dimensions provide many low-energy coefficient vectors, but their evaluation
lattices have smaller determinant per dimension.

### Absolute and practical lower limits

For a fully split ring, evaluation at any one root must be injective on a
strong challenge set. Consequently `|C| <= q`. A challenge set with at least
`2^128` elements therefore requires `q >= 2^128`. No prime whose binary length
is at most 128 can meet this exact support floor; the smallest possible prime
modulus is just above `2^128` and has binary length 129.

This cardinality limit is attainable but not useful for Akita's norm bounds.
For example, scalar challenges `{0, ..., 2^128 - 1}` are strong over any prime
larger than `2^128`, but their operator norm is exponential. Very small
challenge subrings approach the same field-size floor with the same basic
problem: their coefficients become too large for efficient folding.

For D64 challenges with roughly the present coefficient energy, the practical
transition is much higher. An exploratory BKZ screen found exact kernel
vectors with the following squared norms. Each row uses the largest prime of
the given binary length that is `1 mod 128`.

| modulus bits | 128 | 160 | 176 | 192 | 200 | 204 | 208 | 224 | 254 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| found squared norm | 78 | 161 | 203 | 247 | 347 | 325 | 470 | 626 | 1228 |

A found norm at or below 336 is an exact obstruction to certifying the entire
`E = 84` ball for that modulus. A larger found norm is only an upper bound on
the lattice minimum, not proof that the ball is safe. The screen suggests that
favorable D64 moduli around 200--210 bits may work, while a modulus near 128
bits does not support the present low-norm construction. The irregular 200-
and 204-bit rows also show why every selected modulus needs its own exact
certificate.

Run the exploratory screen in an environment with `fpylll`, `cysignals`, and
SymPy:

```bash
python3 scripts/bn254_challenge_units/screen_d64_moduli.py \
  --bits 128 160 176 192 200 204 208 224 254
```

This command is a candidate filter. It does not replace the exact checker.

## Expected cost of the remaining exact work

The accepted-support obligation for S96 is complete. Each of its eight
degree-30 modular moment computations takes about five seconds when run in
parallel on the development machine. CRT reconstruction is immediate; rational
dual search and exact Bernstein construction take a few minutes. The final
standalone Sturm checker takes seconds. Degree 30 and the first-order union
bound already clear the 128-bit floor by 0.497 bits, so no higher dual degree is
needed for this candidate.

The rank-96 kernel exclusion is the expensive obligation. Basis construction
and BKZ screening take minutes on a laptop. Exact unpruned enumeration is
exponential in the worst case and did not finish during short interactive
trials with the current basis. The target radius is nevertheless well below
both the Gaussian scale and the shortest vectors found by BKZ. This makes a
tuned, parallel subtree enumeration plausible as a workstation-hours or
overnight computation, rather than a `2^128` exhaustive search. S88 and S80
provide fallback points with wider lattice margins if S96 proves inconvenient.

## What the checker verifies

The constant coefficient maps to one, so the integer evaluation kernel has
index and determinant `r`. The checker verifies that:

1. the recorded root has exact order 128 modulo `r`;
2. every recorded basis row evaluates to zero at that root;
3. the basis determinant has absolute value exactly `r`, making it a basis of
   the complete kernel;
4. an exact LDL/Gram--Schmidt decomposition is positive; and
5. exhaustive Fincke--Pohst enumeration contains no nonzero vector through
   squared radius `336`.

The certificate deliberately establishes only the bounded lattice statement.
Primality of the standardized BN254 scalar modulus is treated as part of the
field definition.
