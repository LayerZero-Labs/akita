#!/usr/bin/env python3
"""Explore Galois-invariant sparse challenge subspaces for BN254 D128.

This script is not a security certificate. It verifies the exact structural
and support-count claims, then uses deterministic Monte Carlo sampling to rank
candidate operator-norm rejection policies. NumPy is required for the batched
negacyclic Fourier transforms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import comb, log2


RING_DIMENSION = 128
FULL_CYCLOTOMIC_ORDER = 2 * RING_DIMENSION


@dataclass(frozen=True)
class Candidate:
    name: str
    valuations: frozenset[int]
    count_pm1: int
    count_pm2: int

    def positions(self) -> list[int]:
        return [
            exponent
            for exponent in range(1, RING_DIMENSION)
            if two_adic_valuation(exponent) in self.valuations
        ]

    def raw_support_bits(self) -> float:
        dimension = len(self.positions())
        weight = self.count_pm1 + self.count_pm2
        support = (
            comb(dimension, weight)
            * comb(weight, self.count_pm1)
            * 2**weight
        )
        return log2(support)

    def challenge_l2_squared(self) -> int:
        return self.count_pm1 + 4 * self.count_pm2


CANDIDATES = (
    Candidate("S96", frozenset((0, 1)), 35, 1),
    Candidate("S88", frozenset((0, 2, 3)), 35, 2),
    Candidate("S80", frozenset((0, 2)), 36, 3),
    Candidate("S72", frozenset((0, 3)), 38, 5),
)


def two_adic_valuation(value: int) -> int:
    assert value > 0
    return (value & -value).bit_length() - 1


def signed_representative(exponent: int) -> int:
    reduced = exponent % FULL_CYCLOTOMIC_ORDER
    return reduced if reduced < RING_DIMENSION else reduced - RING_DIMENSION


def verify_galois_invariance(candidate: Candidate) -> None:
    positions = set(candidate.positions())
    assert len(positions) == int(candidate.name[1:])
    for automorphism in range(1, FULL_CYCLOTOMIC_ORDER, 2):
        image = {
            abs(signed_representative(automorphism * exponent))
            for exponent in positions
        }
        assert image == positions


def sample_candidate(candidate: Candidate, samples: int, seed: int, batch: int) -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise SystemExit(
            "NumPy is required for Monte Carlo sampling; install it outside "
            "the repository and rerun this script"
        ) from error

    generator = np.random.default_rng(seed)
    positions = candidate.positions()
    weight = candidate.count_pm1 + candidate.count_pm2
    twist = np.exp(1j * np.pi * np.arange(RING_DIMENSION) / RING_DIMENSION)
    thresholds = range(13, 25)
    accepted = {threshold: 0 for threshold in thresholds}

    for start in range(0, samples, batch):
        batch_size = min(batch, samples - start)
        coefficients = np.zeros((batch_size, RING_DIMENSION), dtype=np.int8)
        for row in range(batch_size):
            selected = generator.choice(positions, weight, replace=False)
            count_pm1 = candidate.count_pm1
            coefficients[row, selected[:count_pm1]] = generator.choice(
                (-1, 1), count_pm1
            )
            coefficients[row, selected[count_pm1:]] = 2 * generator.choice(
                (-1, 1), candidate.count_pm2
            )
        evaluations = np.fft.fft(coefficients * twist, axis=1)
        operator_norms = np.max(np.abs(evaluations), axis=1)
        for threshold in thresholds:
            accepted[threshold] += int(
                np.count_nonzero(operator_norms < threshold)
            )

    raw_bits = candidate.raw_support_bits()
    energy = candidate.challenge_l2_squared()
    print(
        f"{candidate.name}: dimension={len(positions)}, "
        f"shell=({candidate.count_pm1},{candidate.count_pm2}), "
        f"raw_bits={raw_bits:.9f}, l2_sq={energy}, "
        f"difference_l2_sq_bound={4 * energy}"
    )
    for threshold in thresholds:
        probability = accepted[threshold] / samples
        accepted_bits = raw_bits + log2(probability) if probability else float("-inf")
        print(
            f"  Gamma<{threshold}: accepted={probability:.9f}, "
            f"estimated_bits={accepted_bits:.9f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_903)
    parser.add_argument("--batch", type=int, default=1_000)
    arguments = parser.parse_args()
    if arguments.samples <= 0 or arguments.batch <= 0:
        parser.error("--samples and --batch must be positive")

    for offset, candidate in enumerate(CANDIDATES):
        verify_galois_invariance(candidate)
        sample_candidate(
            candidate,
            samples=arguments.samples,
            seed=arguments.seed + offset,
            batch=arguments.batch,
        )


if __name__ == "__main__":
    main()
