#!/usr/bin/env python3
"""Compute log2 of the Fiat-Shamir challenge-space size |C| for each
production sparse-challenge family, and the resulting tensor CWSS knowledge
error per level: ε_tensor = 4 * 2^(r/2) / |C|. Compares to main's pre-tensor
families where available."""
from math import lgamma, log, log2


def log2_factorial(n: int) -> float:
    # lgamma(n+1) = ln(n!); convert natural log -> log2 by dividing by ln(2).
    return lgamma(n + 1) / log(2.0)


def log2_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return log2_factorial(n) - log2_factorial(k) - log2_factorial(n - k)


def log2_multinom(n: int, ks: list[int]) -> float:
    if sum(ks) > n:
        return float("-inf")
    out = log2_factorial(n) - log2_factorial(n - sum(ks))
    for k in ks:
        out -= log2_factorial(k)
    return out


def exact_shell_entropy(d: int, count_mag1: int, count_mag2: int) -> float:
    """|C| for ExactShell{count_mag1, count_mag2}.
    Pick 30+12 positions from 64, assign 30 of them mag-1 ± and 12 mag-2 ±."""
    total_positions = count_mag1 + count_mag2
    if total_positions > d:
        return float("-inf")
    n_positions = log2_binom(d, total_positions)
    n_split = log2_binom(total_positions, count_mag1) if count_mag2 > 0 else 0.0
    n_signs = count_mag1 + count_mag2  # 2 signs per nonzero position
    return n_positions + n_split + n_signs


def uniform_entropy(d: int, weight: int, num_signs: int = 2) -> float:
    """|C| for Uniform{weight, nonzero_coeffs}."""
    if weight > d:
        return float("-inf")
    return log2_binom(d, weight) + weight * log2(num_signs)


def cwss_eps_tensor(log2_C: float, r: int) -> float:
    # ε_tensor = 4 * 2^(r/2) / |C|. log2(ε) = 2 + r/2 - log2|C|.
    return 2 + r / 2 - log2_C


def cwss_eps_flat(log2_C: float, r: int) -> float:
    # ε_flat = 2 * 2^r / |C|. log2(ε) = 1 + r - log2|C|.
    return 1 + r - log2_C


PRESETS = [
    # (preset name, |C| computation, max r per generated table, shape)
    ("D=32 (branch + main, BoundedL1Norm{M=8, B=121})", 128.0, 23, "Flat"),
    ("D=64 branch + main (ExactShell{30, 12})",
     exact_shell_entropy(64, 30, 12), 14, "Tensor"),
    ("D=128 branch (Uniform{weight=32, +-1})",
     uniform_entropy(128, 32), 14, "Tensor"),
    # Historical: pre-cutover branch values, kept for reference.
    ("(historical) D=64 pre-cutover (ExactShell{18, 0})",
     exact_shell_entropy(64, 18, 0), 14, "Tensor"),
    ("(historical) D=128 pre-cutover (Uniform{weight=13, +-1})",
     uniform_entropy(128, 13), 14, "Tensor"),
]

# Project security target. Per Hachi paper Lemma 3, the CWSS knowledge error
# eps = O(2^r / |C|) is "negligible since |C| is exponential in λ". The
# concrete-security interpretation is that the challenge-space size itself
# must be >= 2^128: the per-level eps is then 2^-(128 - r/2 - 2) for tensor
# (or 2^-(128 - r - 1) for flat), with the small r-dependent shortfall being
# the standard sumcheck/CWSS slack that the paper accepts.
SECURITY_LAMBDA = 128

print(
    f"{'preset':62} {'shape':>8} {'log2|C|':>10} {'r':>4} "
    f"{'eps_tensor':>12} {'eps_flat':>12}   {'verdict'}"
)
for name, log2_C, r, shape in PRESETS:
    eps_t = cwss_eps_tensor(log2_C, r)
    eps_f = cwss_eps_flat(log2_C, r)
    eps_actual = eps_t if shape == "Tensor" else eps_f
    # The actual-vs-target gauge is |C| >= 2^lambda. With |C| = 2^lambda the
    # per-level eps is at the lambda-bit floor minus the r/2 (or r) slack.
    pass_c = log2_C >= SECURITY_LAMBDA
    verdict = (
        f"|C| >= 2^{SECURITY_LAMBDA}  ✓  (eps_actual=2^{eps_actual:.1f})"
        if pass_c
        else f"|C| <  2^{SECURITY_LAMBDA}  ✗  (eps_actual=2^{eps_actual:.1f}, "
        f"deficit={SECURITY_LAMBDA - log2_C:.1f} bits)"
    )
    print(
        f"{name:62} {shape:>8} {log2_C:>10.1f} {r:>4} "
        f"{f'2^{eps_t:.1f}':>12} {f'2^{eps_f:.1f}':>12}   {verdict}"
    )

print()
print("Project target: |C| >= 2^128 so that per-level CWSS error is negligible")
print("at the 128-bit security parameter. The exact per-level eps is the")
print("standard r/2 + small slack below 2^-lambda; it is the SAME slack the")
print("Hachi paper accepts in Lemma 3 (asymptotic negligibility).")
