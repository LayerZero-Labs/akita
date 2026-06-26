# Binary modular-JL security target (JL tail prototype)

Security lemma, proof strategy, and constant-generation notes for the binary-sign JL cutover in PR #191.
Mechanical prototype scope lives in [`akita-jl-tail-projection-prototype.md`](akita-jl-tail-projection-prototype.md).

The statement Akita should prove and cite internally is the following binary analogue of LaBRADOR Lemma 4.2.

**Target lemma (single-shot binary modular JL, Akita form).** Let `q` be an odd modulus, `J ∈ {-1,+1}^{m×d}` have independent Rademacher entries, and let `w ∈ [-q/2,q/2]^d` be fixed before `J` is sampled. For concrete constants `a_m`, `B_m`, and `Q_m`, if `||w||_2 >= b` and `b <= q / Q_m`, then

```text
Pr_J[ ||Jw mod q||_2 < sqrt(a_m) * b ] <= 2^-lambda.
```

For the PR #191 default, the intended row count is `m = 256` and `lambda = 128`. The proof task is to choose the largest usable `a_256` (for tight slack) and the smallest safe `Q_256` (for small-field headroom), while keeping the upper-tail completeness threshold `B_256` explicit:

```text
Pr_J[ ||Jw mod q||_2 > sqrt(B_256) * ||w||_2 ] <= eps_hi
```

The verifier then checks `||p||_2 <= T_p`. If the accepted image is consistent and `T_p = sqrt(a_m) * T_w`, soundness gives `||w||_2 <= T_w` except with the above failure probability. Completeness uses a larger honest threshold, or nonce regrind, sized from `B_m`.

**Do not reuse LaBRADOR constants blindly.** Ternary LaBRADOR uses `E[C^2]=1/2`, so one projection row has variance `||w||_2^2/2` and `E||Jw||_2^2 = 128||w||_2^2` at `m=256`. Binary signs have variance `||w||_2^2` per row and `E||Jw||_2^2 = 256||w||_2^2`. A binary version can be normalized either by:

```text
J_bin_raw ∈ {-1,+1}^{m×d};        E||J_bin_raw w||_2^2 = m ||w||_2^2
```

or by comparing against the ternary scale with `J_bin_scaled = J_bin_raw / sqrt(2)`. Implementation should keep raw integer signs and adjust thresholds/constants, not introduce irrational scaling into the protocol.

### Binary proof strategy

The proof should adapt LaBRADOR Appendix A, with the real binary tail from Achlioptas/GHL as the concentration input. The structure is:

1. **Real lower tail, no modulus.** Prove for fixed nonzero `w` that `Pr[||Jw||_2 < sqrt(a) ||w||_2]` is at most `2^-lambda` for binary `J`. The fastest conservative proof is Paley-Zygmund/small-ball per row plus Chernoff over `m` rows; the tight proof should use Achlioptas-style moment comparison or a numerical dynamic program over the extremal Rademacher sum to maximize `Pr[|<pi,w>| < t||w||]`. This replaces LaBRADOR Lemma 4.1.
2. **Case 1: small norm, wrap is rare.** If `||w||_2 < q/10`, then a small modular image implies either the real image is small or some row wrapped close to a nonzero multiple of `q`. Bound the first event by step 1. Bound the wrap event by the binary upper tail `Pr[|<pi,w>| > c q]`, unioned over `m` rows. Binary signs are subgaussian with parameter `||w||_2`, so this part is at least as clean as the ternary proof.
3. **Case 2: one huge coordinate.** If `||w||_∞ >= q/C_inf`, fix all row signs except the largest coordinate. For binary signs, toggling that one sign gives two values separated by `2|w_i|`; at most one can lie in a window of radius below `|w_i|`. Thus a single row has constant probability of escaping the small window, and Chernoff over `m` rows gives the `2^-lambda` lower-tail bound. This is the ternary proof's "large coordinate" case without the zero-sign complication.
4. **Case 3: spread-out large norm.** If `||w||_2 >= q/10` but `||w||_∞` is small, truncate to a subvector `v` with norm in a fixed interval below `q/10` and disjoint residual `w-v`. Apply Berry-Esseen to `<pi,v> = sum_i epsilon_i v_i`, where each summand has variance `v_i^2` and third moment `|v_i|^3`. The Berry-Esseen error is proportional to `||v||_∞ / ||v||_2`, and is smaller than the ternary case after matching thresholds because binary signs have twice the row variance. Condition on the residual `<pi,w-v> mod q`; the bad set is an interval of length `2 sqrt(a) b`, and anti-concentration of the approximating normal bounds one-row failure away from one. Raise to `m`.

The constants should be produced by a small checked script, committed under `scripts/` or `crates/akita-challenges/benches/` only if it is part of the reproducible security accounting. Inputs:

- `m` row count, initially 256.
- target `lambda`, initially 128.
- candidate lower threshold `a`.
- modular precondition constant `Q`.
- large-coordinate cutoff `C_inf`.
- Berry-Esseen constant, using the explicit `0.56` or better published constant chosen by the writeup.

Outputs:

- `a_m`: lower-tail threshold used for soundness.
- `B_m`: upper-tail threshold / honest acceptance bucket used for completeness and regrind.
- `Q_m`: modulus precondition `b <= q/Q_m`.
- individual failure terms for the three cases, so the final lemma is auditable rather than a black-box simulation.

The first pass may be conservative; the final pass should optimize `a_m` and `Q_m` jointly. Greyhound's binary `±1` implementation is precedent for the distribution, not a proof of these constants; Grand Danois inherits the ternary LaBRADOR constants and should not be cited for binary constants.
