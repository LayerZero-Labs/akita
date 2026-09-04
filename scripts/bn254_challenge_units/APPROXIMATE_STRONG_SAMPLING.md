# Approximate strong sampling over fully split BN254

## Result

The one-round full-D128 weight-35 and weight-40 candidates in this note do not
currently have a rigorous fixed-anchor collision bound. In particular, the
independent-edge calculation below is not a security result and must not be
used to instantiate Akita.

Pairwise-unit differences are sufficient but unnecessary. The property used by
Akita's coordinate-wise extractor is the fixed-anchor bound

```text
epsilon_C = max_{c0 in C} Pr_{c <- C}[c != c0 and c-c0 is a non-unit].
```

Write `d(c0)` for the number of other challenges whose difference from `c0`
is a non-unit, and `d_max = max d(c0)`. For a uniform finite set,

```text
epsilon_C = d_max / |C|.
```

If an extraction uses `M` coordinate forks, equality of the forked challenge
costs another `1/|C|` per coordinate. Under Akita's documented online
Fiat-Shamir bound with at most `Q` adversarial random-oracle queries, the
challenge contribution is therefore at most

```text
(Q + 1) * M * (1 + d_max) / |C|.                 (1)
```

This is the quantity to make small. There is no requirement that `d_max` be
zero.

The maximum over fixed anchors matters. Akita first fixes a central accepting
transcript and then samples each coordinate fork. This matches Definition 11
of [PikkuFold](https://eprint.iacr.org/2026/1809), which explicitly strengthens
the two-fresh-challenge definition in
[Cyclo](https://eprint.iacr.org/2026/359). An average collision probability for
two fresh challenges cannot simply replace the maximum in Akita's current
extractor.

## Two full-D128 operating points

Both rows use every one of the 128 coefficient positions and reject on Akita's
strict q=48 fixed-point operator-norm predicate.

| shell | energy | threshold | exact acceptance floor | exact accepted-support floor | expected trials |
|---|---:|---:|---:|---:|---:|
| signed weight 35 | 35 | 14 | 0.494326711912 | 2^138.653104393085 | 2.023 |
| signed weight 40 | 40 | 15 | 0.461570934375 | 2^149.857657644806 | 2.167 |

The weight-40 certificate uses a degree-30 exact moment dual, exact polynomial
positivity, nine modular moment computations, and an eight-prime CRT. It buys
11.205 additional certified support bits for five additional nonzero
coefficients and one unit of operator threshold.

If `d_max <= 2`, applying (1) at a 128-bit target gives the following maximum
query budgets:

| accepted shell | M=1 | M=8 | M=32 | M=128 | M=256 |
|---|---:|---:|---:|---:|---:|
| weight 35, threshold 14 | 535 | 66 | 15 | 3 | 1 |
| weight 40, threshold 15 | 1,266,744 | 158,342 | 39,584 | 9,895 | 4,947 |

These are challenge-term budgets, not end-to-end Akita security claims. A
production schedule must use its actual total coordinate count, query bound,
and every other additive knowledge-error term.

Recompute the table with:

```bash
python3 scripts/bn254_challenge_units/analyze_approximate_strong_budget.py
```

## Collision-graph heuristic

Model the evaluation of each challenge at each of the 128 roots as an
independent uniform value in the BN254 scalar field. Two distinct challenges
then have a non-unit difference with probability

```text
p = 1 - (1 - 1/q)^128 = approximately 2^-246.596691.
```

In the corresponding independent-edge graph, a union bound on factorial
moments gives

```text
Pr[max degree >= k] <= |C| * binom(|C|-1, k) * p^k.
```

Using the larger raw shell, rather than the operator-accepted subset, makes the
following model predictions conservative within that model:

| raw shell | model bound for degree >= 2 | model bound for degree >= 3 |
|---|---:|---:|
| weight 35 | 2^-75.185 | 2^-183.697 |
| weight 40 | 2^-41.274 | 2^-138.483 |

Thus the model predicts `d_max <= 2` at more than 128 bits for both candidates.
This prediction is not accepted as a theorem. Conditional on an independently
proved `d_max <= 2`, the query budgets above would follow exactly.

This is not a proof about BN254. The challenge supports have fewer elements
than one BN254 slot, so PikkuFold's `|C| >= q^e` near-uniform-slot heuristic
does not apply. Operator rejection can also change slot point masses.

The exact scaled experiment is encouraging but also demonstrates why the
random-slot model must not be presented as calibrated fact. For the fully
split ring of degree 16 and the complete signed weight-5 shell of 139,776
challenges, exhaustive enumeration found zero collision edges at four primes
from 24 to 31 bits. The independent random-slot model predicts between 146
and 18,632 edges. The algebraic challenge map was much more injective than a
random function in every tested case, but a different fixed modulus could have
different arithmetic.

Reproduce all scaled counts with:

```bash
python3 scripts/bn254_challenge_units/exhaustive_scaled_collision_model.py
```

## Routes to a theorem

The fixed-weight result in Lemma 34 of
[Boudgoust--Lapiha](https://eprint.iacr.org/2025/1080) is directly relevant.
In the fully split case its class degree is one, so its prescribed-class
fixed-weight distribution is the ordinary global fixed-weight shell. The
lemma gives a rigorous per-slot point-mass expression. Two issues remain:

1. its Fourier sum ranges over the BN254 field and still needs a sharp,
   efficiently checkable bound in this concrete subgroup; and
2. conditioning on operator acceptance can increase a point mass by at most
   the reciprocal of the certified acceptance floor.

A sharp evaluation of that expression would prove a fixed-anchor bound
directly and avoid shortest-vector enumeration.

The alternative is a deterministic list-size certificate. The existing
weight-35 reduction shows that a squared minimum distance of 76 caps every
single-root fiber at 12 and hence gives `d_max <= 128 * 11 = 1408`. That is
already approximate, not an all-pairs result, but its probability margin is
too narrow once many coordinates or Fiat-Shamir queries are included. A useful
next certificate should target the `d_max` allowed by (1) for a concrete
schedule rather than target zero collisions.

## Status

- Exact: shell cardinalities, acceptance floors, operator thresholds, and
  equation (1).
- Exact conditional statement: any proved `d_max` bound plugs directly into
  equation (1).
- Heuristic: the independent-edge degree bounds and `d_max <= 2`.
- Measured exactly only in scaled rings: the four degree-16 collision graphs.
- Open for BN254: a rigorous maximum-degree or per-slot point-mass bound for
  the operator-accepted full-D128 shell.
