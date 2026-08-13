# Operator-norm acceptance certificates

Machine-checkable certificates for the family-level obligation of Appendix C of
the Akita paper: a rigorous lower bound on the support accepted by the runtime
rejection sampler.

```
p = Pr[ Gamma(c) < 18 - 600 / 2^48 ]
```

for the `d = 64` exact-shell challenge family, where `Gamma(c)` is the
negacyclic operator norm. The strict runtime predicate now uses threshold 18.
The bound feeds the accepted-support floor `log2(N_raw * p) >= lambda`.

The small subtraction is the exact fixed-point containment margin. The q=48
root table has coordinate error at most 4 units. A shell element has L1 norm 53,
so each frequency coordinate has error at most `r = 53 * 4 = 212` units. The
least integer strictly above `2 sqrt(2) r` is 600. Therefore every challenge
with true norm at most `18 - 600 / 2^48` passes the strict threshold-18 runtime
predicate. The certificate proves at least 128 bits of support inside the
slightly smaller strict subset shown above.

The method (paper appendix C, `sec:opnorm-moment-method`) has four steps:

1. **Exact moments.** `M_m = E[X^m]` of `X = |c(zeta_k)|^2` for one frequency,
   exact rationals from the sign-then-support generating function, computed
   without floating point via multi-modular CRT over primes `p = 1 mod 2d`.
2. **Dual majorant.** A rational polynomial `Q` with `Q >= 0` on `[0, B]` and
   `Q >= 1` from the exact squared fixed-point containment boundary through
   `B` gives a marginal spectral tail bound.
3. **Bernstein certificate.** `Q` and `Q - 1` are shown nonnegative on the two
   intervals by subdividing into equal rational pieces and checking that every
   degree-`n` Bernstein coefficient is `>= 0` (exact).
4. **Union bound.** `p >= 1 - (d/2) * E[Q(X)]`, since there are `d/2` distinct
   magnitudes by conjugate symmetry.

## Files

- `cert_d64_a31_b11_gamma18.json` — certificate for the `(a, b) = (31, 11)`
  shell and strict runtime threshold 18. Holds the parameters, `N_raw`, the exact moments
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
python3 check_cert.py                         # checks the bundled threshold-18 cert
python3 check_cert.py path/to/other_cert.json # checks another certificate
python3 validate_moment_generator.py          # optional generator implementation check
```

Exit code `0` iff all checks pass. Expected output for the bundled certificate:

```
Pr[Gamma(c) < 18 - 600 / 2^48] >= p0 = 0.2349106543
log2(N_raw * p0) = 128.062439  (>= 128 ? yes)
```

## Adding a new parameter setting `(d, a, b, threshold)`

1. Compute the exact moments `M_0 .. M_n` of `X = |c(zeta_0)|^2` for the new
   family (Step 1). Bumping `n` tightens the bound.
2. Solve the truncated Chebyshev-Markov moment LP near the intended threshold
   and emit its exact rational coefficients.
3. Compute the exact fixed-point containment margin and prove `Q >= 1` from the
   squared contained-subset boundary, not merely from the integer threshold.
4. Pick a subdivision count large enough that all Bernstein coefficients are
   `>= 0` (Step 3).
5. Write a JSON with the same schema and run `check_cert.py` on it.

### Notes and limits

- The normal checker does not compile the generator. It verifies that all
  supplied exact moments match the eight saved reconstruction residues and the
  independent ninth-prime residue. Run `validate_moment_generator.py` when
  changing the generator itself.
- The union bound can certify no floor above the corresponding one-frequency
  tail union bound. When that union bound is vacuous, a second-order
  Hunter-Worsley refinement (subtracting certified lower bounds on adjacent
  joint exceedances) is needed. For `(31, 11)`, threshold 18 clears
  `lambda = 128` with the first-order method.
