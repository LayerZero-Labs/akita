#!/usr/bin/env python3
"""Check the exact structural bounds for the full-D128 unit image.

Run the existing S72 kernel and accepted-support checkers separately.  This
script checks the new transformation: the unit identity, full coefficient
coverage, overlap graph, and coefficient-energy bound.
"""

from __future__ import annotations

from itertools import combinations


BN254_SCALAR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
RING_DIMENSION = 128
S72_MAGNITUDE_ONE_COUNT = 38
S72_MAGNITUDE_TWO_COUNT = 5
S72_ENERGY = 58
SHIFT = 1


def two_adic_valuation(value: int) -> int:
    assert value > 0
    return (value & -value).bit_length() - 1


def main() -> None:
    modulus = BN254_SCALAR_MODULUS
    assert modulus % 2 == 1

    positions = {
        exponent
        for exponent in range(1, RING_DIMENSION)
        if two_adic_valuation(exponent) in {0, 3}
    }
    shifted_positions = {
        (exponent + SHIFT) % RING_DIMENSION for exponent in positions
    }
    assert len(positions) == 72
    assert positions | shifted_positions == set(range(RING_DIMENSION))
    assert len(positions & shifted_positions) == 16
    assert sum(exponent % 2 == 0 for exponent in positions & shifted_positions) == 8
    assert sum(exponent % 2 == 1 for exponent in positions & shifted_positions) == 8

    # (1 + X) * sum_{i=0}^{127} (-X)^i = 1 - X^128 = 2 in
    # Z[X]/(X^128 + 1).  The inverse therefore exists modulo every odd q.
    inverse = [
        pow(2, -1, modulus) * (-1 if index % 2 else 1) % modulus
        for index in range(RING_DIMENSION)
    ]
    product = [0] * RING_DIMENSION
    for left_exponent in (0, 1):
        for right_exponent, coefficient in enumerate(inverse):
            exponent = left_exponent + right_exponent
            if exponent >= RING_DIMENSION:
                exponent -= RING_DIMENSION
                coefficient = -coefficient
            product[exponent] = (product[exponent] + coefficient) % modulus
    assert product == [1] + [0] * (RING_DIMENSION - 1)

    # The overlap terms in <s, Xs> form eight disjoint length-two paths.
    edges = set()
    for exponent in positions:
        shifted = (exponent + SHIFT) % RING_DIMENSION
        if shifted in positions:
            edges.add(tuple(sorted((exponent, shifted))))
    assert len(edges) == 16
    degrees = {exponent: 0 for exponent in positions}
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    assert sum(degree == 2 for degree in degrees.values()) == 8
    assert sum(degree == 1 for degree in degrees.values()) == 16
    assert sum(degree == 0 for degree in degrees.values()) == 48

    # All 24 edge vertices may carry nonzero coefficients because the shell
    # weight is 43.  Exhaust the placements of the five magnitude-two values;
    # all other edge vertices have magnitude one in the maximizing support.
    edge_vertices = sorted(
        exponent for exponent, degree in degrees.items() if degree
    )
    maximum_absolute_correlation = 0
    for doubled in combinations(edge_vertices, S72_MAGNITUDE_TWO_COUNT):
        doubled_set = set(doubled)
        magnitudes = {
            exponent: 2 if exponent in doubled_set else 1
            for exponent in edge_vertices
        }
        correlation = sum(
            magnitudes[left] * magnitudes[right] for left, right in edges
        )
        maximum_absolute_correlation = max(
            maximum_absolute_correlation, correlation
        )
    assert maximum_absolute_correlation == 26

    transformed_energy_bound = (
        2 * S72_ENERGY + 2 * maximum_absolute_correlation
    )
    assert transformed_energy_bound == 168
    assert S72_MAGNITUDE_ONE_COUNT + S72_MAGNITUDE_TWO_COUNT == 43

    print("verified: 1 + X is a unit in the BN254 D128 ring")
    print("verified: S72 union (S72 + 1) covers all 128 positions")
    print(
        "verified: transformed squared coefficient norm is at most "
        f"{transformed_energy_bound}"
    )


if __name__ == "__main__":
    main()
