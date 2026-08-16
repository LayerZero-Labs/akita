# D128 operator-norm accepted-support certificate

This directory certifies the production signed weight-31 shell in dimension 128
for a strict runtime threshold of 13. The q=48 root table has coordinate error
at most 4 units. Since this shell has L1 norm 31, the per-frequency coordinate
error is at most 124 units. The least integer strictly above
`2 sqrt(2) * 124` is 351. The exact contained subset is therefore
`Gamma(c) <= 13 - 351 / 2^48`.

The final exact result is:

```
q1 <= 0.008120720198295826
Pr[Gamma(c) < 13 - 351 / 2^48] >= 0.480273907309067161
log2(N_raw * p0) >= 128.563317131141
```

The exact support-floor cutoff is `q1 < 0.010546520951777437`.

## Final verification

```bash
python3 validate_moment_generator.py
python3 -u check_cert.py
```

The first command compiles the modular moment generator, compares it with a
separate exhaustive enumeration on a small shell, and checks the independently
derived closed form for the D128 second moment. The second command checks all
eight saved modular residues, moment admissibility, the rational dual
expectation, exact polynomial positivity through a rational Sturm sequence, and
the exact accepted-support inequality. Pass `--check-bernstein` to additionally
replay all 4096 rational Bernstein subintervals used by the emitted certificate.

## Reproduction pipeline

`moments_mod.cpp` computes unnormalized spectral moments modulo a prime that has
a primitive `2d`-th root. The seven reconstruction primes have a 427-bit
product, above the 396-bit a priori bound `C(128,31) * 961^30`. The eighth prime
is not used by CRT and checks every reconstructed moment independently.

Compile and regenerate the eight `moments_<prime>.txt` files with the prime and
primitive-generator pairs recorded in `reconstruct_moments.py`, then run:

```bash
python3 reconstruct_moments.py
python3 solve_dual.py
python3 make_cert.py
python3 -u check_cert.py
```

`solve_dual.py` uses floating point only to search for a candidate polynomial.
It rounds the coefficients to exact rationals and adds an exact rational upward
shift. The final checker uses rational arithmetic for every load-bearing step.
The certificate uses degree 30 and 4096 rational Bernstein subintervals. Its
tail interval starts at the exact square of the fixed-point containment
boundary, not at the integer value 169.

For a fixed-seed 100,000-sample validation, the empirical acceptance rates at
thresholds 11, 12, 13, and 14 were respectively 0.24010, 0.55498, 0.79173, and
0.91811. The formal floor at 13 is intentionally much lower than the observed
rate.
