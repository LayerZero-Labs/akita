#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    label: str
    side_l1: int
    side_linf: int
    side_bits: float

    @property
    def overall_bits(self) -> float:
        return 2.0 * self.side_bits

    @property
    def effective_l1_mass(self) -> int:
        return self.side_l1 * self.side_l1

    @property
    def extraction_proxy(self) -> int:
        return 4 * self.side_l1 * self.side_linf


def log2_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def exact_shell_bits(d: int, count_mag1: int, count_mag2: int) -> float:
    weight = count_mag1 + count_mag2
    if weight <= 0 or weight > d:
        return float("-inf")
    return log2_comb(d, weight) + log2_comb(weight, count_mag2) + weight


def uniform_pm1_bits(d: int, weight: int) -> float:
    if weight <= 0 or weight > d:
        return float("-inf")
    return log2_comb(d, weight) + weight


def bounded_l1_bits(d: int, max_coeff: int, l1_bound: int) -> float:
    dp = [0] * (l1_bound + 1)
    dp[0] = 1
    for _ in range(d):
        next_dp = [0] * (l1_bound + 1)
        for current_l1, count in enumerate(dp):
            if count == 0:
                continue
            next_dp[current_l1] += count
            for magnitude in range(1, min(max_coeff, l1_bound - current_l1) + 1):
                next_dp[current_l1 + magnitude] += 2 * count
        dp = next_dp
    return math.log2(sum(dp))


def best_exact_shell(d: int, threshold_bits: float) -> Candidate:
    best: tuple[int, int, float, int, int] | None = None
    for count_mag1 in range(d + 1):
        for count_mag2 in range(d + 1 - count_mag1):
            side_bits = exact_shell_bits(d, count_mag1, count_mag2)
            if side_bits < threshold_bits:
                continue
            side_l1 = count_mag1 + 2 * count_mag2
            side_linf = 2 if count_mag2 else 1
            candidate = (side_l1, side_linf, -side_bits, count_mag1, count_mag2)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError(f"no exact-shell candidate reaches {threshold_bits} bits for D={d}")
    side_l1, side_linf, neg_bits, count_mag1, count_mag2 = best
    return Candidate(
        label=f"ExactShell {{ count_mag1: {count_mag1}, count_mag2: {count_mag2} }}",
        side_l1=side_l1,
        side_linf=side_linf,
        side_bits=-neg_bits,
    )


def best_uniform_pm1(d: int, threshold_bits: float) -> Candidate:
    for weight in range(1, d + 1):
        side_bits = uniform_pm1_bits(d, weight)
        if side_bits >= threshold_bits:
            return Candidate(
                label=f"Uniform {{ weight: {weight}, nonzero_coeffs: [-1, 1] }}",
                side_l1=weight,
                side_linf=1,
                side_bits=side_bits,
            )
    raise RuntimeError(f"no uniform +/-1 candidate reaches {threshold_bits} bits for D={d}")


def best_bounded_l1(d: int, threshold_bits: float) -> Candidate:
    best: tuple[int, int, float] | None = None
    for max_coeff in range(1, 9):
        for l1_bound in range(1, 129):
            side_bits = bounded_l1_bits(d, max_coeff, l1_bound)
            if side_bits < threshold_bits:
                continue
            candidate = (l1_bound, max_coeff, -side_bits)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError(f"no bounded-L1 candidate reaches {threshold_bits} bits for D={d}")
    l1_bound, max_coeff, neg_bits = best
    return Candidate(
        label=f"BoundedL1 {{ max_coeff: {max_coeff}, l1_bound: {l1_bound} }}",
        side_l1=l1_bound,
        side_linf=max_coeff,
        side_bits=-neg_bits,
    )


def print_candidate(name: str, candidate: Candidate) -> None:
    print(f"{name}:")
    print(f"  {candidate.label}")
    print(f"  side_bits        = {candidate.side_bits:.3f}")
    print(f"  overall_bits     = {candidate.overall_bits:.3f}")
    print(f"  side_l1          = {candidate.side_l1}")
    print(f"  effective_l1     = {candidate.effective_l1_mass}")
    print(f"  extraction_proxy = {candidate.extraction_proxy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search near-minimum tensor stage-1 challenge parameters."
    )
    parser.add_argument(
        "--threshold-bits-per-side",
        type=float,
        default=64.0,
        help="Minimum support bits required per tensor side.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold = args.threshold_bits_per_side

    current_defaults = {
        "D32 current": Candidate("BoundedL1Norm (M=8, B=121)", 121, 8, 128.13315865886818),
        "D64 current": Candidate(
            "ExactShell { count_mag1: 30, count_mag2: 12 }", 54, 2, exact_shell_bits(64, 30, 12)
        ),
        "D128 current": Candidate(
            "Uniform { weight: 31, nonzero_coeffs: [-1, 1] }", 31, 1, uniform_pm1_bits(128, 31)
        ),
    }

    absolute_minima = {
        "D32 minimum": best_exact_shell(32, threshold),
        "D64 minimum": best_exact_shell(64, threshold),
        "D128 minimum": best_uniform_pm1(128, threshold),
        "D32 bounded comparison": best_bounded_l1(32, threshold),
        "D64 bounded comparison": best_bounded_l1(64, threshold),
    }

    safe_defaults = {
        "D32 shipped": Candidate("BoundedL1Norm (M=8, B=121)", 121, 8, 128.13315865886818),
        "D64 shipped": Candidate(
            "ExactShell { count_mag1: 18, count_mag2: 0 }",
            18,
            1,
            exact_shell_bits(64, 18, 0),
        ),
        "D128 shipped": Candidate(
            "Uniform { weight: 13, nonzero_coeffs: [-1, 1] }",
            13,
            1,
            uniform_pm1_bits(128, 13),
        ),
    }

    print(f"Tensor side search threshold: {threshold:.1f} bits per side")
    print()
    print("Current defaults")
    for name, candidate in current_defaults.items():
        print_candidate(name, candidate)
        print()

    print("Absolute minima used for the planner rerun")
    for name, candidate in absolute_minima.items():
        print_candidate(name, candidate)
        print()

    print("Note: D32 exact-shell minima are not currently shipped.")
    print(
        "  A direct schedule/audit probe rejected the low-mass D32 exact-shell candidates,"
    )
    print("  so the shipped D32 default remains BoundedL1Norm.")
    print()

    print("Final shipped defaults")
    for name, candidate in safe_defaults.items():
        print_candidate(name, candidate)
        print()


if __name__ == "__main__":
    main()
