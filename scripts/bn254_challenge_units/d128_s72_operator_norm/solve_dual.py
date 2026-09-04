#!/usr/bin/env python3
"""Search for, rationalize, and exactly bound the S72 moment dual."""

import json
import math
from fractions import Fraction as F
from math import comb, log2
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

DIMENSION = 128
SELECTED_POSITIONS = 72
MAG1_COUNT = 38
MAG2_COUNT = 5
WEIGHT = MAG1_COUNT + MAG2_COUNT
L1_NORM = MAG1_COUNT + 2 * MAG2_COUNT
SUPPORT_CAP = L1_NORM**2
GAMMA = 18
DEGREE = 30
NFREQ = DIMENSION // 2
FIXED_POINT_Q = 48
ROOT_COORDINATE_ERROR_UNITS = 4


def fixed_point_tail_start_sq():
    scale = 1 << FIXED_POINT_Q
    root_error = L1_NORM * ROOT_COORDINATE_ERROR_UNITS
    squared_error = 8 * root_error * root_error
    margin = math.isqrt(squared_error)
    if margin * margin <= squared_error:
        margin += 1
    upper = F(GAMMA * scale - margin, scale)
    return upper * upper


def chebyshev_monomials(degree):
    polynomials = [[F(1)], [F(0), F(1)]]
    for current_degree in range(2, degree + 1):
        current = [F(0)] * (current_degree + 1)
        for index, coefficient in enumerate(polynomials[current_degree - 1]):
            current[index + 1] += 2 * coefficient
        for index, coefficient in enumerate(polynomials[current_degree - 2]):
            current[index] -= coefficient
        polynomials.append(current)
    return polynomials


def chebyshev_values(points, degree):
    values = np.zeros((len(points), degree + 1))
    values[:, 0] = 1
    values[:, 1] = points
    for current_degree in range(2, degree + 1):
        values[:, current_degree] = (
            2 * points * values[:, current_degree - 1]
            - values[:, current_degree - 2]
        )
    return values


def bernstein_min(coefficients, lo, hi):
    length = hi - lo
    reparameterized = [F(0)] * (DEGREE + 1)
    for index, coefficient in enumerate(coefficients):
        for power in range(index + 1):
            reparameterized[power] += (
                coefficient
                * comb(index, power)
                * lo ** (index - power)
                * length**power
            )
    return min(
        sum(
            F(comb(k, index), comb(DEGREE, index)) * reparameterized[index]
            for index in range(k + 1)
        )
        for k in range(DEGREE + 1)
    )


def to_x_monomials(rational, chebyshev):
    coefficients = [F(0)] * (DEGREE + 1)
    for degree in range(DEGREE + 1):
        for power, coefficient in enumerate(chebyshev[degree]):
            for index in range(power + 1):
                coefficients[index] += (
                    rational[degree]
                    * coefficient
                    * comb(power, index)
                    * F(2, SUPPORT_CAP) ** index
                    * (-1) ** (power - index)
                )
    return coefficients


def certificate_gap(coefficients, lo, hi, subdivisions):
    return min(
        bernstein_min(
            coefficients,
            lo + (hi - lo) * F(index, subdivisions),
            lo + (hi - lo) * F(index + 1, subdivisions),
        )
        for index in range(subdivisions)
    )


def main():
    moments = {
        int(key): F(value)
        for key, value in json.loads(
            Path("moments_d128_s72_a38_b5.json").read_text()
        ).items()
    }
    chebyshev = chebyshev_monomials(DEGREE)
    normalized_moments = [
        sum(
            comb(power, index)
            * F(2, SUPPORT_CAP) ** index
            * (-1) ** (power - index)
            * moments[index]
            for index in range(power + 1)
        )
        for power in range(DEGREE + 1)
    ]
    objective_exact = [
        sum(
            chebyshev[degree][power] * normalized_moments[power]
            for power in range(len(chebyshev[degree]))
        )
        for degree in range(DEGREE + 1)
    ]
    objective = np.array([float(value) for value in objective_exact])

    tail_start_sq = fixed_point_tail_start_sq()
    threshold = float(F(2, SUPPORT_CAP) * tail_start_sq - 1)
    nonnegative_points = set(np.linspace(-1, 1, 600))
    tail_points = set(np.linspace(threshold, 1, 600))
    result = None
    for iteration in range(100):
        points0 = np.array(sorted(nonnegative_points))
        points1 = np.array(sorted(tail_points))
        matrix0 = chebyshev_values(points0, DEGREE)
        matrix1 = chebyshev_values(points1, DEGREE)
        result = linprog(
            objective,
            A_ub=np.vstack([-matrix0, -matrix1]),
            b_ub=np.concatenate([np.zeros(len(points0)), -np.ones(len(points1))]),
            bounds=[(-100.0, 100.0)] * (DEGREE + 1),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"iteration {iteration}: {result.message}")
        derivative_roots = np.polynomial.Chebyshev(result.x).deriv().roots()
        real_roots = [
            float(root.real) for root in derivative_roots if abs(root.imag) < 1e-8
        ]
        candidates0 = [-1.0, 1.0] + [root for root in real_roots if -1 < root < 1]
        candidates1 = [threshold, 1.0] + [
            root for root in real_roots if threshold < root < 1
        ]
        minimum0 = float(
            np.min(np.polynomial.chebyshev.chebval(candidates0, result.x))
        )
        minimum1 = float(
            np.min(np.polynomial.chebyshev.chebval(candidates1, result.x) - 1)
        )
        print(
            f"iteration {iteration + 1}: q={result.fun:.12f}, "
            f"minima={minimum0:.3e},{minimum1:.3e}"
        )
        if minimum0 >= -1e-6 and minimum1 >= -1e-6:
            break
        nonnegative_points.update(round(value, 12) for value in candidates0)
        tail_points.update(round(value, 12) for value in candidates1)
    else:
        raise RuntimeError("exchange algorithm did not converge")

    denominator = 2**60
    rational = [F(round(value * denominator), denominator) for value in result.x]
    x_coefficients = to_x_monomials(rational, chebyshev)
    subdivisions = 4096
    gap0 = certificate_gap(x_coefficients, F(0), F(SUPPORT_CAP), subdivisions)
    minus_one = x_coefficients[:]
    minus_one[0] -= 1
    gap1 = certificate_gap(
        minus_one, tail_start_sq, F(SUPPORT_CAP), subdivisions
    )
    shift = max(F(0), -gap0, -gap1) + F(1, 10**10)
    rational[0] += shift
    x_coefficients[0] += shift
    minus_one[0] += shift
    gap0 = certificate_gap(x_coefficients, F(0), F(SUPPORT_CAP), subdivisions)
    gap1 = certificate_gap(
        minus_one, tail_start_sq, F(SUPPORT_CAP), subdivisions
    )
    assert gap0 > 0 and gap1 > 0

    marginal_tail = sum(
        rational[index] * objective_exact[index] for index in range(DEGREE + 1)
    )
    support_count = comb(SELECTED_POSITIONS, WEIGHT) * comb(WEIGHT, MAG2_COUNT)
    raw_support = support_count * 2**WEIGHT
    acceptance_floor = 1 - NFREQ * marginal_tail
    cutoff = F(1, NFREQ) * (1 - F(2**128, raw_support))
    print(f"shift: {float(shift):.12e}")
    print(f"post-shift Bernstein gaps: {float(gap0):.12e}, {float(gap1):.12e}")
    print(f"q: {float(marginal_tail):.18f}")
    print(f"cutoff: {float(cutoff):.18f}")
    print(f"p0: {float(acceptance_floor):.18f}")
    print(
        "accepted bits: "
        f"{log2(raw_support) + log2(float(acceptance_floor)):.12f}"
    )

    Path("dual_coefficients.json").write_text(
        json.dumps(
            {
                str(index): f"{value.numerator}/{value.denominator}"
                for index, value in enumerate(rational)
            },
            indent=2,
        )
        + "\n"
    )
    Path("dual_summary.json").write_text(
        json.dumps(
            {
                "degree": DEGREE,
                "subdivisions": subdivisions,
                "q": f"{marginal_tail.numerator}/{marginal_tail.denominator}",
                "p0": f"{acceptance_floor.numerator}/{acceptance_floor.denominator}",
                "cutoff": f"{cutoff.numerator}/{cutoff.denominator}",
                "gap0": f"{gap0.numerator}/{gap0.denominator}",
                "gap1": f"{gap1.numerator}/{gap1.denominator}",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
