# D128 S96 accepted-support certificate

This directory certifies the accepted support of the leading native D128
challenge candidate. The position set is

```text
S96 = {i in {1, ..., 127} : i is not divisible by 4}.
```

Every challenge has 35 coefficients in `{−1, 1}`, one coefficient in
`{−2, 2}`, and zero elsewhere. Its squared coefficient norm is 39, its L1 norm
is 37, and its raw support has 129.2198237937146 bits.

The q=48 runtime predicate uses threshold 15 and a root table whose coordinate
error is at most 4 units. The spectral coordinate error is therefore at most
148 units. The least integer strictly above `2 sqrt(2) * 148` is 419, so the
exact subset bounded here is

```text
Gamma(c) < 15 - 419 / 2^48.
```

The final exact result is

```text
q1 <= 0.006157397303872698
Pr[Gamma(c) < 15 - 419 / 2^48] >= 0.605926572552147324
log2(N_raw * p0) >= 128.497038674319
```

The exact support-floor cutoff is `q1 < 0.008916638239421765`.

## Final verification

From this directory, run:

```bash
python3 validate_moment_generator.py
python3 -u check_cert.py
```

The first command compiles the modular generator, compares it with exhaustive
enumeration of a small S6 mixed shell, and checks an independently derived
closed form for the S96 second moment. The second command checks all eight
modular residue files, the CRT size bound, moment admissibility, the rational
dual expectation, exact polynomial positivity through a Sturm sequence, and
the accepted-support inequality. It uses only the Python standard library.

Pass `--check-bernstein` to replay the 4096-subinterval exact Bernstein
certificate used when constructing the rational dual.

## Why one spectral marginal suffices

Multiplication by an odd exponent modulo 256 preserves whether a nonzero
position has 2-adic valuation zero or one. It therefore acts on S96 by a signed
permutation after reduction modulo `X^128 + 1`. The challenge distribution is
invariant under all cyclotomic Galois automorphisms, so all 64
conjugacy-distinct spectral magnitudes have the same marginal distribution.
The certificate bounds that marginal by `q1` and applies a union bound.

## Reproduction pipeline

`moments_mod.cpp` computes the first 31 unnormalized spectral moments modulo a
prime with a primitive 256th root. The seven reconstruction primes have a
427-bit product, above the 406-bit a priori bound
`C(96,36) * 36 * 1369^30`. The eighth prime is omitted from reconstruction and
checks every recovered moment independently.

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
./moments_mod PRIME GENERATOR 30 128 35 1
```

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
