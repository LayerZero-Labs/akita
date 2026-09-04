#!/usr/bin/env python3
"""Count integer coefficient balls and report modulus-size tradeoffs."""

from __future__ import annotations

from math import e, isqrt, log2, pi


TARGET_SUPPORT = 1 << 128
DIMENSIONS = (32, 64, 128)
MAX_SQUARED_NORM = 600


def minimum_energy(dimension: int) -> tuple[int, int]:
    one_coordinate = [(0, 1)] + [
        (magnitude * magnitude, 2)
        for magnitude in range(1, isqrt(MAX_SQUARED_NORM) + 1)
    ]
    counts = [0] * (MAX_SQUARED_NORM + 1)
    counts[0] = 1
    for _ in range(dimension):
        next_counts = [0] * (MAX_SQUARED_NORM + 1)
        for current_energy, current_count in enumerate(counts):
            if current_count == 0:
                continue
            for added_energy, multiplicity in one_coordinate:
                total_energy = current_energy + added_energy
                if total_energy > MAX_SQUARED_NORM:
                    break
                next_counts[total_energy] += current_count * multiplicity
        counts = next_counts

    cumulative = 0
    for energy, count in enumerate(counts):
        cumulative += count
        if cumulative >= TARGET_SUPPORT:
            return energy, cumulative
    raise RuntimeError("increase MAX_SQUARED_NORM")


def universal_modulus_bits(dimension: int, energy: int) -> float:
    return dimension * log2(4 * energy) / 2


def gaussian_heuristic_modulus_bits(dimension: int, energy: int) -> float:
    hermite_scale = dimension / (2 * pi * e)
    return dimension * log2(4 * energy / hermite_scale) / 2


def main() -> None:
    print("dimension,min_energy,support_bits,universal_q_bits,heuristic_q_bits")
    for dimension in DIMENSIONS:
        energy, support = minimum_energy(dimension)
        print(
            f"{dimension},{energy},{log2(support):.9f},"
            f"{universal_modulus_bits(dimension, energy):.9f},"
            f"{gaussian_heuristic_modulus_bits(dimension, energy):.9f}"
        )


if __name__ == "__main__":
    main()
