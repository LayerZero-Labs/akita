# D128 S72 accepted-support certificate

This directory certifies the accepted support of the S72 native D128
challenge candidate. Its Galois-invariant position set is

```text
S72 = {i in {1, ..., 127} : v2(i) is zero or three}.
```

Every challenge has 38 coefficients in `{−1, 1}`, five coefficients in
`{−2, 2}`, and zero elsewhere. Its squared coefficient norm is 58, its L1 norm
is 48, and its raw support has 129.5121606480869 bits.

The q=48 runtime predicate uses threshold 18 and a root table whose coordinate
error is at most 4 units. The spectral coordinate error is therefore at most
192 units. The least integer strictly above `2 sqrt(2) * 192` is 544, so the
exact subset bounded here is

```text
Gamma(c) < 18 - 544 / 2^48.
```

The final exact result is

```text
q1 <= 0.007134842792044588
Pr[Gamma(c) < 18 - 544 / 2^48] >= 0.543370061309146335
log2(N_raw * p0) >= 128.632167631106
```

The exact support-floor cutoff is `q1 < 0.010147097315770123`.

## Final verification

From this directory, run:

```bash
python3 validate_moment_generator.py
python3 -u check_cert.py
```

The first command compiles the common masked modular generator, compares it
with exhaustive enumeration of a small mixed shell, and checks an independently
derived closed form for the S72 second moment. The second command checks all
eight modular residue files, the CRT size bound, moment admissibility, the
rational dual expectation, exact polynomial positivity through a Sturm
sequence, and the accepted-support inequality. It uses only the Python standard
library.

Pass `--check-bernstein` to replay the 4096-subinterval exact Bernstein
certificate used when constructing the rational dual.

## Why one spectral marginal suffices

Multiplication by an odd exponent modulo 256 preserves 2-adic valuation. It
therefore acts on S72 by a signed permutation after reduction modulo
`X^128 + 1`. The challenge distribution is invariant under all cyclotomic
Galois automorphisms, so all 64 conjugacy-distinct spectral magnitudes have the
same marginal distribution. The certificate bounds that marginal by `q1` and
applies a union bound.

## Reproduction pipeline

`../masked_moments_mod.cpp` computes the first 31 unnormalized spectral moments
modulo a prime with a primitive 256th root. The seven reconstruction primes have
a 427-bit product, above the 422-bit a priori bound
`C(72,43) * C(43,5) * 2304^30`. The eighth prime is omitted from reconstruction
and checks every recovered moment independently.

Compile the generator and regenerate each `moments_<prime>.txt` file with the
prime/generator pairs below:

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

For each pair, invoke the generator as follows:

```bash
./masked_moments_mod PRIME GENERATOR 30 128 38 5 9
```

The final argument is a bit mask of selected 2-adic valuations. `9` selects
valuations zero and three for S72.

Then run:

```bash
python3 reconstruct_moments.py
python3 solve_dual.py
python3 make_cert.py
python3 -u check_cert.py
```

`solve_dual.py` requires NumPy and SciPy only to search for a candidate. It
rounds the result to rational coefficients and establishes exact Bernstein
gaps before emitting it. Every load-bearing claim is replayed by the standalone
checker without NumPy or SciPy.
