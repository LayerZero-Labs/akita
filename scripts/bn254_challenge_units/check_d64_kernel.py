#!/usr/bin/env python3
"""Verify the exact BN254 D64 challenge-unit certificate.

The certificate gives a basis for the integer lattice of coefficient vectors
whose evaluation at one primitive 128th root vanishes modulo the BN254 scalar
modulus.  This checker uses only Python integer and rational arithmetic.  It
verifies that the certificate is a full kernel basis, then exhaustively proves
that the lattice has no nonzero vector of squared Euclidean norm at most 336.
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import isqrt
from pathlib import Path

BN254_SCALAR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
DIMENSION = 64
SQUARED_NORM_EXCLUSION_BOUND = 336
EXPECTED_ENUMERATION_NODES = 1_507_538
D64_PM1_COUNT = 31
D64_PM2_COUNT = 11
MAX_CERTIFIED_CHALLENGE_L2_SQUARED = SQUARED_NORM_EXCLUSION_BOUND // 4


def bareiss_determinant(matrix: list[list[int]]) -> int:
    """Return the exact determinant using fraction-free elimination."""

    values = [row[:] for row in matrix]
    dimension = len(values)
    sign = 1
    previous_pivot = 1
    for column in range(dimension - 1):
        if values[column][column] == 0:
            pivot_row = next(
                row
                for row in range(column + 1, dimension)
                if values[row][column] != 0
            )
            values[column], values[pivot_row] = values[pivot_row], values[column]
            sign = -sign
        pivot = values[column][column]
        for row in range(column + 1, dimension):
            for next_column in range(column + 1, dimension):
                numerator = (
                    values[row][next_column] * pivot
                    - values[row][column] * values[column][next_column]
                )
                assert numerator % previous_pivot == 0
                values[row][next_column] = numerator // previous_pivot
        previous_pivot = pivot
        for row in range(column + 1, dimension):
            values[row][column] = 0
    return sign * values[-1][-1]


def exact_ldl(
    basis: list[list[int]],
) -> tuple[list[list[Fraction]], list[Fraction]]:
    """Compute the exact Gram--Schmidt coefficients and squared lengths."""

    dimension = len(basis)
    gram = [
        [sum(x * y for x, y in zip(basis[row], basis[column])) for column in range(dimension)]
        for row in range(dimension)
    ]
    mu = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    diagonal = [Fraction(0) for _ in range(dimension)]
    for row in range(dimension):
        mu[row][row] = Fraction(1)
        for column in range(row):
            numerator = Fraction(gram[row][column])
            for prior in range(column):
                numerator -= mu[row][prior] * mu[column][prior] * diagonal[prior]
            mu[row][column] = numerator / diagonal[column]
        squared_length = Fraction(gram[row][row])
        for prior in range(row):
            squared_length -= mu[row][prior] ** 2 * diagonal[prior]
        assert squared_length > 0
        diagonal[row] = squared_length
    return mu, diagonal


def ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def contains_short_nonzero_vector(
    mu: list[list[Fraction]],
    diagonal: list[Fraction],
    squared_norm_bound: int,
) -> tuple[bool, int]:
    """Exhaustively enumerate every lattice vector inside the given ball."""

    dimension = len(diagonal)
    coefficients = [0] * dimension
    nodes = 0
    found = False

    def visit(level: int, remaining: Fraction) -> None:
        nonlocal found, nodes
        if found:
            return
        if level < 0:
            found = any(coefficients)
            return

        center_offset = sum(
            (
                mu[higher][level] * coefficients[higher]
                for higher in range(level + 1, dimension)
            ),
            Fraction(0),
        )
        allowance = remaining / diagonal[level]
        if allowance < 0:
            return

        center_numerator = center_offset.numerator
        center_denominator = center_offset.denominator
        integer_limit = (
            allowance.numerator * center_denominator**2 // allowance.denominator
        )
        radius = isqrt(integer_limit)
        lower = ceil_div(-radius - center_numerator, center_denominator)
        upper = (radius - center_numerator) // center_denominator

        candidates = list(range(lower, upper + 1))
        candidates.sort(key=lambda value: abs(Fraction(value) + center_offset))
        for value in candidates:
            nodes += 1
            coefficients[level] = value
            offset = Fraction(value) + center_offset
            visit(level - 1, remaining - diagonal[level] * offset**2)
            if found:
                return
        coefficients[level] = 0

    visit(dimension - 1, Fraction(squared_norm_bound))
    return found, nodes


def main() -> None:
    certificate_path = Path(__file__).with_name("d64_kernel_basis.json")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    modulus = certificate["modulus"]
    dimension = certificate["dimension"]
    omega = certificate["omega"]
    squared_norm_bound = certificate["squared_norm_exclusion_bound"]
    basis = certificate["basis"]

    assert modulus == BN254_SCALAR_MODULUS
    assert dimension == DIMENSION
    assert squared_norm_bound == SQUARED_NORM_EXCLUSION_BOUND
    assert len(basis) == dimension
    assert all(len(row) == dimension for row in basis)

    # omega has exact order 128 because omega^64 = -1.
    assert pow(omega, dimension, modulus) == modulus - 1
    assert pow(omega, 2 * dimension, modulus) == 1
    powers = [pow(omega, exponent, modulus) for exponent in range(dimension)]

    # Every row lies in the evaluation kernel.  The kernel has index modulus
    # because the constant coefficient maps to 1.  Determinant modulus thus
    # proves that these rows form a basis of the complete kernel.
    assert all(
        sum(coefficient * power for coefficient, power in zip(row, powers)) % modulus
        == 0
        for row in basis
    )
    assert abs(bareiss_determinant(basis)) == modulus

    production_challenge_l2_squared = D64_PM1_COUNT + 4 * D64_PM2_COUNT
    assert production_challenge_l2_squared == 75
    assert production_challenge_l2_squared <= MAX_CERTIFIED_CHALLENGE_L2_SQUARED
    assert 4 * MAX_CERTIFIED_CHALLENGE_L2_SQUARED == squared_norm_bound

    mu, diagonal = exact_ldl(basis)
    found, nodes = contains_short_nonzero_vector(mu, diagonal, squared_norm_bound)
    assert not found
    assert nodes == EXPECTED_ENUMERATION_NODES
    print(
        "verified: BN254 D64 evaluation kernel has no nonzero vector with "
        f"squared norm <= {squared_norm_bound} ({nodes} enumeration nodes)"
    )


if __name__ == "__main__":
    main()
