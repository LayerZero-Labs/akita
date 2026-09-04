# Uniform D128 weight-40 challenge candidate

This directory certifies the support and multiplication norm of

```text
C_raw = {c in {-1, 0, 1}^128 : wt(c) = 40}.
```

The sampler uses all 128 coefficient positions and rejects unless Akita's
strict q=48 fixed-point operator-norm predicate accepts at runtime threshold
15. Every raw challenge has squared coefficient norm and L1 norm 40. The raw
support has `150.973033360930` bits.

The degree-30 exact moment certificate proves

```text
q1 <= 0.008412954150392326
Pr[Gamma(c) < 15 - 453 / 2^48] >= 0.461570934374891162
log2(N_raw * p0) >= 149.857657644806
```

The certified expected trial count is below `2.167`. Compared with the
weight-35 threshold-14 candidate, this adds five signed coefficients and one
unit of operator threshold in exchange for `11.205` extra certified support
bits. Those bits are useful when the approximate-strong-sampling term is
multiplied by coordinate and Fiat-Shamir query counts.

Run the standalone checks from this directory:

```bash
python3 validate_moment_generator.py
python3 -u check_cert.py
python3 -u check_cert.py --check-bernstein
python3 ../analyze_approximate_strong_budget.py
```

`validate_moment_generator.py` compiles the canonical D128 generator and
compares it with exhaustive enumeration of a small shell. `check_cert.py` uses
only the Python standard library. It verifies the nine residue files, exact
moments, rational dual expectation, Sturm positivity, q=48 containment, and
the accepted-support floor. The optional Bernstein replay verifies all 4096
exact subintervals.

The certificate does not prove a BN254 collision bound. The accompanying
[research note](../APPROXIMATE_STRONG_SAMPLING.md) separates the exact support
result from the collision-graph heuristic and derives the precise
coordinate/query budget for any future maximum-degree certificate.

## Reproduction pipeline

The canonical generator is
`scripts/operator_norm/d128/moments_mod.cpp`. Generate degree-30 residues with
arguments `PRIME GENERATOR 30 128 40` for these pairs:

| prime | primitive generator |
|---:|---:|
| 2305843009213689601 | 11 |
| 2305843009213689089 | 3 |
| 2305843009213687297 | 15 |
| 2305843009213683713 | 3 |
| 2305843009213682689 | 17 |
| 2305843009213675777 | 11 |
| 2305843009213673729 | 3 |
| 2305843009213666049 | 3 |
| 2305843009213663489 | 7 |

Then run:

```bash
python3 reconstruct_moments.py
python3 solve_dual.py
python3 make_cert.py
python3 -u check_cert.py
```

The first eight primes reconstruct all moment numerators; the ninth is an
independent check. Only reconstruction and dual search require SymPy, NumPy,
and SciPy. The emitted certificate is rational, and its standalone checker
replays every load-bearing claim exactly.
