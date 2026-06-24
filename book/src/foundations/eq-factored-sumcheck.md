# Equality-factored sum-check

> **Status:** stub. Part of the initial Akita Book scaffold.

An optimization for sum-check instances carrying an equality factor,
\\( T = \sum_x \mathrm{eq}(\tau,x)\,p(x) \\), which appear in Akita's zero-checks
and norm-checks. Folds from paper §2.4. The prover sends one-degree-lower round
messages and the verifier accounts for the equality factor itself.

## The eq-factored round message

Because \\( \mathrm{eq}(\tau,x) \\) factors across variables, the prover sends
only the data factor \\( q_j \\) (degree \\( \le \ell-1 \\), one lower than the
unfactored message), and the inter-round check folds in the current variable's
equality factor. Combined with linear-coefficient omission, only
\\( (q_{j,0}, q_{j,2}, \dots) \\) are sent.

**Sources to fold in**

- Paper §2.4 `sec:prelim-eq-factored` (the data-factor message).
- `crates/akita-sumcheck/src/compact_fold.rs`, `crates/akita-algebra/src/split_eq.rs`.

## The inversion-free verifier

Recovering the omitted linear term normally divides by \\( \tau_j \\); the
equivalent inversion-free form carries the claim in scaled form
\\( (\widetilde{T}_j, \Pi_j) \\) and defers the divisions into one accumulated
scale applied at the final check.

**Sources to fold in**

- Paper §2.4 ("An inversion-free verifier", `fig:eq-factored-sumcheck`).
- External: Gruen (ePrint 2024/1210), speeding-up-sumcheck.

## Batched use and scope

Coefficient omission needs a common verifier-known equality factor, so it
applies alone or batched only with instances sharing the same
\\( \mathrm{eq}(\tau,\cdot) \\) (e.g. the per-fold relation sum-check); a mixed
batch reverts to the standard compressed message and keeps only the prover-side
table savings.

**Sources to fold in**

- Paper §2.4 ("Batched use"; ties to §3.5 relation sum-check and §4 offloaded verifier).
