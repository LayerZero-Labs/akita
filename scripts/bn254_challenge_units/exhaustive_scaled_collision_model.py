#!/usr/bin/env python3
"""Exhaustively compare a scaled fixed-weight shell with a random-slot model."""

from __future__ import annotations

import itertools
import math


DIMENSION = 16
WEIGHT = 5
MODULI = (
    8_388_673,
    33_554_593,
    134_218_081,
    1_073_741_857,
)
EXPECTED_UNION_EDGES = (0, 0, 0, 0)


def prime_factors(value: int) -> list[int]:
    factors = []
    candidate = 2
    while candidate * candidate <= value:
        if value % candidate == 0:
            factors.append(candidate)
            while value % candidate == 0:
                value //= candidate
        candidate += 1 if candidate == 2 else 2
    if value > 1:
        factors.append(value)
    return factors


def primitive_root(modulus: int) -> int:
    factors = prime_factors(modulus - 1)
    for candidate in itertools.count(2):
        if all(
            pow(candidate, (modulus - 1) // factor, modulus) != 1
            for factor in factors
        ):
            return candidate
    raise AssertionError("unreachable")


def signed_shell() -> list[tuple[tuple[int, ...], int]]:
    return [
        (positions, signs)
        for positions in itertools.combinations(range(DIMENSION), WEIGHT)
        for signs in range(1 << WEIGHT)
    ]


def collision_statistics(
    modulus: int, challenges: list[tuple[tuple[int, ...], int]]
) -> tuple[int, int, int]:
    generator = primitive_root(modulus)
    root = pow(generator, (modulus - 1) // (2 * DIMENSION), modulus)
    assert pow(root, DIMENSION, modulus) == modulus - 1
    roots = [pow(root, 2 * index + 1, modulus) for index in range(DIMENSION)]
    count = len(challenges)
    edges: set[int] = set()
    maximum_fiber = 1

    for evaluation_root in roots:
        powers = [pow(evaluation_root, exponent, modulus) for exponent in range(DIMENSION)]
        values = []
        for challenge_id, (positions, signs) in enumerate(challenges):
            value = sum(
                (1 if signs >> index & 1 else -1) * powers[position]
                for index, position in enumerate(positions)
            ) % modulus
            values.append((value, challenge_id))
        values.sort()
        start = 0
        while start < count:
            end = start + 1
            while end < count and values[end][0] == values[start][0]:
                end += 1
            ids = [entry[1] for entry in values[start:end]]
            maximum_fiber = max(maximum_fiber, len(ids))
            for left, right in itertools.combinations(ids, 2):
                low, high = sorted((left, right))
                edges.add(low * count + high)
            start = end

    degrees = [0] * count
    for edge in edges:
        left, right = divmod(edge, count)
        degrees[left] += 1
        degrees[right] += 1
    return len(edges), maximum_fiber, max(degrees)


def main() -> None:
    challenges = signed_shell()
    count = len(challenges)
    print(
        f"D={DIMENSION}, weight={WEIGHT}, |C|={count}, "
        f"support_bits={math.log2(count):.6f}"
    )
    for modulus, expected_edges in zip(MODULI, EXPECTED_UNION_EDGES):
        edges, maximum_fiber, maximum_degree = collision_statistics(
            modulus, challenges
        )
        assert edges == expected_edges
        random_probability = 1 - (1 - 1 / modulus) ** DIMENSION
        random_edges = math.comb(count, 2) * random_probability
        print(
            f"q={modulus}: exact_edges={edges}, random_edges={random_edges:.3f}, "
            f"max_fiber={maximum_fiber}, max_degree={maximum_degree}"
        )
    print("PASS: all scaled fully split collision counts match")


if __name__ == "__main__":
    main()
