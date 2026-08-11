# Operator-norm acceptance certificates

Machine-checkable certificates for the family-level obligation of Appendix C of
the Akita paper: a rigorous lower bound on the rejection-sampling acceptance
probability

```
p = Pr[ Gamma(c) <= Gamma ]
```

for a `d = 64` exact-shell challenge family, where `Gamma(c)` is the negacyclic
operator norm (largest singular value of multiplication by `c`). The bound feeds
the accepted-support floor `log2(N_raw * p) >= lambda`.

The method (paper appendix C, `sec:opnorm-moment-method`) has four steps:

1. **Exact moments.** `M_m = E[X^m]` of `X = |c(zeta_k)|^2` for one frequency,
   exact rationals from the sign-then-support generating function, computed
   without floating point via multi-modular CRT over primes `p = 1 mod 2d`.
2. **Dual majorant.** A rational polynomial `Q` with `Q >= 0` on `[0, B]` and
   `Q >= 1` on `[Gamma^2, B]` (here `B = ||c||_1^2`) gives
   `Pr(X > Gamma^2) <= E[Q(X)] = sum_i q_i M_i`.
3. **Bernstein certificate.** `Q` and `Q - 1` are shown nonnegative on the two
   intervals by subdividing into equal rational pieces and checking that every
   degree-`n` Bernstein coefficient is `>= 0` (exact).
4. **Union bound.** `p >= 1 - (d/2) * E[Q(X)]`, since there are `d/2` distinct
   magnitudes by conjugate symmetry.

## Files

- `cert_d64_a31_b11_gamma18.json` — certificate for the `(a, b) = (31, 11)`
  shell at cap `Gamma = 18`. Holds the parameters, `N_raw`, the exact moments
  `M_0 .. M_30`, the degree-30 dual polynomial `Q(x) = sum_j r_j T_j(2x/B - 1)`
  in the shifted-Chebyshev basis (exact rational coefficients `r_j`), the
  Bernstein subdivision spec, and the claimed `q1_star`, `p0`.
- `check_cert.py` — self-contained, floating-point-free checker (Python stdlib
  only). Binds every moment to eight exact CRT residue files and an independent
  ninth-prime residue, recomputes `q1_star`, re-verifies Bernstein
  nonnegativity, checks moment admissibility, and confirms
  `log2(N_raw * p0) >= lambda`.
- `moments_mod.cpp` and `reconstruct_moments.py` — the exact mixed-shell modular
  moment generator and eight-prime CRT reconstruction. The CRT product has 488
  bits against a 432-bit degree-30 numerator bound.
- `validate_moment_generator.py` — compares the generator with exhaustive
  enumeration on a small mixed shell and checks the D64 second moment against
  an independent closed form.

## Running

```bash
python3 check_cert.py                         # checks the bundled (31,11), Gamma=18 cert
python3 check_cert.py path/to/other_cert.json # checks another certificate
python3 validate_moment_generator.py          # optional generator implementation check
```

Exit code `0` iff all checks pass. Expected output for the bundled certificate:

```
Pr[Gamma(c) <= 18] >= p0 = 0.2349106543
log2(N_raw * p0) = 128.062439  (>= 128 ? yes)
```

## Adding a new parameter setting `(d, a, b, Gamma)`

1. Compute the exact moments `M_0 .. M_n` of `X = |c(zeta_0)|^2` for the new
   family (Step 1). Bumping `n` tightens the bound.
2. Solve the truncated Chebyshev-Markov moment LP for the optimal dual `Q` at
   `t = Gamma^2` and emit its exact rational coefficients.
3. Pick a subdivision count large enough that all Bernstein coefficients are
   `>= 0` (Step 3).
4. Write a JSON with the same schema and run `check_cert.py` on it.

### Notes and limits

- The normal checker does not compile the generator. It verifies that all
  supplied exact moments match the eight saved reconstruction residues and the
  independent ninth-prime residue. Run `validate_moment_generator.py` when
  changing the generator itself.
- The union bound can certify no floor above `1 - (d/2) * Pr(X > Gamma^2)`. When
  `(d/2) * Pr(X > Gamma^2) >= 1` the union bound is vacuous and a second-order
  Hunter-Worsley refinement (subtracting certified lower bounds on adjacent
  joint exceedances) is needed. For `(31, 11)` this rules out `Gamma <= 16` via
  this method; `Gamma = 18` clears `lambda = 128` first-order.
