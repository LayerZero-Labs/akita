# A single-challenge full-D128 unit image

## Statement

Let `C_S72` be the already certified S72 challenge set in
`F_q[X]/(X^128+1)`: 38 signed coefficients of magnitude one and five of
magnitude two on positions whose 2-adic valuation is zero or three, followed
by the strict operator-norm rejection at threshold 18. Its exact certificates
prove

```text
|C_S72| >= 2^128.632167631106
```

and prove that every difference of two distinct accepted challenges is a
unit. Define one new challenge set

```text
C_full = {(1 + X)c : c in C_S72}.
```

Then all of the following are exact:

- `C_full` is one ring-challenge set, not a tuple or repeated challenge;
- `|C_full|=|C_S72|>=2^128.632167631106`;
- every difference of distinct elements of `C_full` is a unit;
- the family covers all 128 coefficient positions and every challenge has
  contributions in both coefficient parities;
- its operator norm is strictly below 36;
- its squared coefficient norm is at most 168; and
- multiplication by a challenge can remain factored: first multiply by the
  sparse S72 challenge, then perform one negacyclic shift and addition.

This is a rigorous full-D128 wrapper around the completed S72 theorem. It is
not a proof for the native weight-35 or weight-40 full shells. The price of the
wrapper is norm expansion, not additional protocol rounds.

## Proof

Put `u=1+X`. In the integer cyclotomic ring,

```text
(1 + X) * sum_{i=0}^{127} (-X)^i = 1 - X^128 = 2.
```

The BN254 scalar modulus is odd, so two is invertible and `u` is a unit. Thus
multiplication by `u` is injective and preserves units. For distinct
`c,c' in C_S72`,

```text
uc - uc' = u(c-c')
```

is a product of two units. This proves the cardinality and pairwise-unit
claims without any distributional assumption.

Let `S` be the 72-position S72 mask. Direct orbit arithmetic gives

```text
|S intersect (S+1)| = 16,
S union (S+1) = {0,...,127}.
```

Consequently the coefficient support of the transformed family reaches every
D128 position. Every base challenge has 43 nonzero positions, while only 16
positions can overlap their shifts. At least 27 odd-position contributions and
27 even-position contributions therefore cannot cancel. Both parities occur
in every transformed challenge. The accepted base set is Galois invariant and
every base challenge uses at least one odd position, so its Galois orbit covers
all odd positions; multiplication by `1+X` also covers all even positions.

For every complex root `zeta` of `X^128+1`,

```text
|1+zeta| <= 2,
```

so the certified strict bound `Gamma(c)<18` gives
`Gamma((1+X)c)<36`.

The coefficient-energy bound is sharper than the generic factor-four bound.
The overlap graph contributing to `<c,Xc>` is eight disjoint paths with two
edges each. It has 16 edges. Giving all edge vertices magnitude one contributes
16, and each of the five magnitude-two coefficients can increase the total by
at most two. Hence

```text
<c,Xc> <= 26,
||(1+X)c||_2^2 = 2*58 + 2<c,Xc> <= 168.
```

The standalone structural checker exhausts the small magnitude-two placement
calculation and verifies an explicit inverse for `1+X` modulo BN254.

## Reproduction

The load-bearing result is the conjunction of three exact checks:

```bash
python3 scripts/bn254_challenge_units/check_s72_kernel.py --workers 4

cd scripts/bn254_challenge_units/d128_s72_operator_norm
python3 validate_moment_generator.py
python3 -u check_cert.py

cd ../../..
python3 scripts/bn254_challenge_units/check_full_d128_unit_image.py
```

The first checker proves pairwise invertibility of the base set, the second
pair proves its accepted-support floor, and the final checker proves the unit
image's new algebraic, coverage, and energy claims.

## Limit

This construction improves coefficient coverage and retains a single
challenge, but it does not remove the S72 algebraic origin. A genuinely native
full-shell result still requires a rigorous fixed-anchor bound for the BN254
rank-128 evaluation lattice. The current BKZ screen is not such a proof.
