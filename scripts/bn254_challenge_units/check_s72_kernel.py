#!/usr/bin/env python3
"""Verify the exact BN254 S72 evaluation-kernel certificate.

The checker uses only Python integer and rational arithmetic. It verifies that
the recorded rank-72 basis is the complete evaluation kernel on the selected
D128 coefficient positions, then performs an exhaustive Fincke--Pohst search
through squared norm 232. Use multiple workers to divide exact subtrees without
changing the enumerated set or node count.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from fractions import Fraction
from math import isqrt
from pathlib import Path

BN254_SCALAR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
RING_DIMENSION = 128
CHALLENGE_DIMENSION = 72
SQUARED_NORM_EXCLUSION_BOUND = 232
EXPECTED_ENUMERATION_NODES = 127_185_682
S72_PM1_COUNT = 38
S72_PM2_COUNT = 5
TARGET_SUBTREE_NODES = 10_000

WORKER_MU: list[list[Fraction]]
WORKER_DIAGONAL: list[Fraction]


def two_adic_valuation(value: int) -> int:
    assert value > 0
    return (value & -value).bit_length() - 1


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
    """Compute exact Gram--Schmidt coefficients and squared lengths."""

    dimension = len(basis)
    gram = [
        [
            sum(x * y for x, y in zip(basis[row], basis[column]))
            for column in range(dimension)
        ]
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


def candidate_interval(
    center_offset: Fraction,
    remaining: Fraction,
    squared_length: Fraction,
) -> list[int]:
    allowance = remaining / squared_length
    if allowance < 0:
        return []
    center_numerator = center_offset.numerator
    center_denominator = center_offset.denominator
    integer_limit = (
        allowance.numerator * center_denominator**2 // allowance.denominator
    )
    radius = isqrt(integer_limit)
    lower = ceil_div(-radius - center_numerator, center_denominator)
    upper = (radius - center_numerator) // center_denominator
    return sorted(
        range(lower, upper + 1),
        key=lambda value: abs(value * center_denominator + center_numerator),
    )


def enumerate_frontier(
    mu: list[list[Fraction]],
    diagonal: list[Fraction],
    squared_norm_bound: int,
) -> tuple[list[tuple[int, tuple[int, ...], Fraction]], int]:
    """Enumerate exact top-level prefixes that partition the search tree."""

    dimension = len(diagonal)
    coefficients = [0] * dimension
    frontier = []
    nodes = 0
    floating_diagonal = [float(value) for value in diagonal]

    def estimated_work(level: int, remaining: Fraction) -> float:
        # This heuristic decides only where to split a subtree. Both branches
        # of that decision are exhaustively enumerated with exact arithmetic.
        squared_radius = float(remaining)
        if squared_radius <= 0:
            return 1
        total = 0.0
        log_diagonal_product = 0.0
        for subtree_dimension in range(1, level + 2):
            diagonal_index = level - subtree_dimension + 1
            log_diagonal_product += math.log(floating_diagonal[diagonal_index])
            log_nodes = (
                (subtree_dimension / 2) * math.log(math.pi * squared_radius)
                - math.lgamma(subtree_dimension / 2 + 1)
                - 0.5 * log_diagonal_product
            )
            if log_nodes > math.log(TARGET_SUBTREE_NODES):
                return TARGET_SUBTREE_NODES + 1
            total += math.exp(log_nodes)
            if total > TARGET_SUBTREE_NODES:
                return total
        return total

    def visit(level: int, remaining: Fraction) -> None:
        nonlocal nodes
        if level < 0 or estimated_work(level, remaining) <= TARGET_SUBTREE_NODES:
            frontier.append((level, tuple(coefficients[level + 1 :]), remaining))
            return
        center_offset = sum(
            (
                mu[higher][level] * coefficients[higher]
                for higher in range(level + 1, dimension)
            ),
            Fraction(0),
        )
        for value in candidate_interval(center_offset, remaining, diagonal[level]):
            nodes += 1
            coefficients[level] = value
            offset = Fraction(
                value * center_offset.denominator + center_offset.numerator,
                center_offset.denominator,
            )
            visit(level - 1, remaining - diagonal[level] * offset**2)
        coefficients[level] = 0

    visit(dimension - 1, Fraction(squared_norm_bound))
    return frontier, nodes


def initialize_worker(
    mu: list[list[Fraction]], diagonal: list[Fraction]
) -> None:
    global WORKER_MU, WORKER_DIAGONAL
    WORKER_MU = mu
    WORKER_DIAGONAL = diagonal


def enumerate_subtree(
    task: tuple[int, tuple[int, ...], Fraction],
) -> tuple[list[int] | None, int]:
    """Exhaust one exact subtree and return a witness or its node count."""

    initial_level, suffix, initial_remaining = task
    dimension = len(WORKER_DIAGONAL)
    coefficients = [0] * dimension
    coefficients[initial_level + 1 :] = suffix
    nodes = 0
    witness = None

    def visit(level: int, remaining: Fraction) -> None:
        nonlocal nodes, witness
        if witness is not None:
            return
        if level < 0:
            if any(coefficients):
                witness = coefficients[:]
            return
        center_offset = sum(
            (
                WORKER_MU[higher][level] * coefficients[higher]
                for higher in range(level + 1, dimension)
            ),
            Fraction(0),
        )
        for value in candidate_interval(
            center_offset, remaining, WORKER_DIAGONAL[level]
        ):
            nodes += 1
            coefficients[level] = value
            offset = Fraction(
                value * center_offset.denominator + center_offset.numerator,
                center_offset.denominator,
            )
            visit(
                level - 1,
                remaining - WORKER_DIAGONAL[level] * offset**2,
            )
            if witness is not None:
                return
        coefficients[level] = 0

    visit(initial_level, initial_remaining)
    return witness, nodes


def contains_short_nonzero_vector(
    mu: list[list[Fraction]],
    diagonal: list[Fraction],
    squared_norm_bound: int,
    workers: int,
) -> tuple[list[int] | None, int]:
    """Exhaustively enumerate the lattice ball using exact arithmetic."""

    frontier, prefix_nodes = enumerate_frontier(
        mu, diagonal, squared_norm_bound
    )
    print(
        f"exact frontier: {len(frontier)} subtrees, {prefix_nodes} prefix nodes",
        flush=True,
    )
    total_nodes = prefix_nodes
    if workers == 1:
        initialize_worker(mu, diagonal)
        for completed, task in enumerate(frontier, start=1):
            witness, nodes = enumerate_subtree(task)
            total_nodes += nodes
            if witness is not None:
                return witness, total_nodes
            if completed % 500 == 0:
                print(
                    f"exact progress: {completed}/{len(frontier)} subtrees, "
                    f"{total_nodes} nodes",
                    flush=True,
                )
        return None, total_nodes

    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        context = multiprocessing.get_context()
    with context.Pool(
        workers,
        initializer=initialize_worker,
        initargs=(mu, diagonal),
    ) as pool:
        results = pool.imap_unordered(enumerate_subtree, frontier, chunksize=1)
        for completed, (witness, nodes) in enumerate(results, start=1):
            total_nodes += nodes
            if witness is not None:
                pool.terminate()
                return witness, total_nodes
            if completed % 500 == 0:
                print(
                    f"exact progress: {completed}/{len(frontier)} subtrees, "
                    f"{total_nodes} nodes",
                    flush=True,
                )
    return None, total_nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.workers <= 0:
        parser.error("--workers must be positive")

    certificate_path = Path(__file__).with_name("s72_kernel_basis.json")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    modulus = certificate["modulus"]
    ring_dimension = certificate["ring_dimension"]
    challenge_dimension = certificate["challenge_dimension"]
    omega = certificate["omega"]
    positions = certificate["positions"]
    squared_norm_bound = certificate["squared_norm_exclusion_bound"]
    basis = certificate["basis"]

    assert modulus == BN254_SCALAR_MODULUS
    assert ring_dimension == RING_DIMENSION
    assert challenge_dimension == CHALLENGE_DIMENSION
    assert squared_norm_bound == SQUARED_NORM_EXCLUSION_BOUND
    assert certificate["valuations"] == [0, 3]
    assert positions == [
        exponent
        for exponent in range(1, ring_dimension)
        if two_adic_valuation(exponent) in {0, 3}
    ]
    assert len(basis) == challenge_dimension
    assert all(len(row) == challenge_dimension for row in basis)

    assert pow(omega, ring_dimension, modulus) == modulus - 1
    assert pow(omega, 2 * ring_dimension, modulus) == 1
    powers = [pow(omega, exponent, modulus) for exponent in positions]
    assert all(
        sum(coefficient * power for coefficient, power in zip(row, powers))
        % modulus
        == 0
        for row in basis
    )
    assert abs(bareiss_determinant(basis)) == modulus

    challenge_l2_squared = S72_PM1_COUNT + 4 * S72_PM2_COUNT
    assert challenge_l2_squared == 58
    assert 4 * challenge_l2_squared == squared_norm_bound

    mu, diagonal = exact_ldl(basis)
    witness, nodes = contains_short_nonzero_vector(
        mu, diagonal, squared_norm_bound, arguments.workers
    )
    assert witness is None, witness
    assert nodes == EXPECTED_ENUMERATION_NODES
    print(
        "verified: BN254 S72 evaluation kernel has no nonzero vector with "
        f"squared norm <= {squared_norm_bound} ({nodes} enumeration nodes)"
    )


if __name__ == "__main__":
    main()
