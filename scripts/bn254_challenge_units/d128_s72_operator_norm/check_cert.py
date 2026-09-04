#!/usr/bin/env python3
"""Verify the exact D128 S72 accepted-support certificate with stdlib only."""

import json
import math
import sys
from fractions import Fraction as F
from math import comb
from pathlib import Path


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


def trim(polynomial):
    while polynomial and polynomial[-1] == 0:
        polynomial.pop()
    return polynomial


def polynomial_remainder(dividend, divisor):
    remainder = dividend[:]
    while remainder and len(remainder) >= len(divisor):
        factor = remainder[-1] / divisor[-1]
        shift = len(remainder) - len(divisor)
        for index, coefficient in enumerate(divisor):
            remainder[index + shift] -= factor * coefficient
        trim(remainder)
    return remainder


def sturm_sequence(polynomial):
    derivative = [index * polynomial[index] for index in range(1, len(polynomial))]
    sequence = [trim(polynomial[:]), trim(derivative)]
    while sequence[-1]:
        remainder = [
            -coefficient
            for coefficient in polynomial_remainder(sequence[-2], sequence[-1])
        ]
        if not remainder:
            break
        scale = abs(remainder[-1])
        sequence.append([coefficient / scale for coefficient in remainder])
    return sequence


def evaluate(polynomial, point):
    value = F(0)
    for coefficient in reversed(polynomial):
        value = value * point + coefficient
    return value


def sign_variations(sequence, point):
    signs = []
    for polynomial in sequence:
        value = evaluate(polynomial, point)
        if value != 0:
            signs.append(value > 0)
    return sum(left != right for left, right in zip(signs, signs[1:]))


def strictly_positive_on_interval(polynomial, lo, hi):
    if evaluate(polynomial, lo) <= 0 or evaluate(polynomial, hi) <= 0:
        return False, None
    sequence = sturm_sequence(polynomial)
    root_count = sign_variations(sequence, lo) - sign_variations(sequence, hi)
    return root_count == 0, root_count


def positive_definite(matrix):
    work = [row[:] for row in matrix]
    pivots = []
    for pivot_index in range(len(work)):
        pivot = work[pivot_index][pivot_index]
        pivots.append(pivot)
        if pivot <= 0:
            return False, pivot
        for row in range(pivot_index + 1, len(work)):
            factor = work[row][pivot_index] / pivot
            for column in range(pivot_index, len(work)):
                work[row][column] -= factor * work[pivot_index][column]
    return True, min(pivots)


def bernstein_min(coefficients, lo, hi, degree):
    length = hi - lo
    reparameterized = [F(0)] * (degree + 1)
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
            F(comb(k, index), comb(degree, index)) * reparameterized[index]
            for index in range(k + 1)
        )
        for k in range(degree + 1)
    )


def interval_certificate_min(coefficients, lo, hi, subdivisions, degree):
    return min(
        bernstein_min(
            coefficients,
            lo + (hi - lo) * F(index, subdivisions),
            lo + (hi - lo) * F(index + 1, subdivisions),
            degree,
        )
        for index in range(subdivisions)
    )


def load_residues(path):
    lines = path.read_text().splitlines()
    header = lines[0].split()
    return int(header[1]), {
        int(line.split()[0]): int(line.split()[1]) for line in lines[1:]
    }


def two_adic_valuation(value):
    assert value > 0
    return (value & -value).bit_length() - 1


def main():
    arguments = sys.argv[1:]
    check_bernstein = "--check-bernstein" in arguments
    arguments = [argument for argument in arguments if argument != "--check-bernstein"]
    certificate_path = (
        Path(arguments[0])
        if arguments
        else Path(__file__).with_name("cert_d128_s72_a38_b5_gamma18.json")
    )
    certificate = json.loads(certificate_path.read_text())
    params = certificate["params"]
    dimension = params["d"]
    selected_count = params["selected_positions"]
    mag1_count = params["a_mag1"]
    mag2_count = params["b_mag2"]
    weight = params["w_nonzero"]
    support_cap = params["support_cap_B"]
    nominal_gamma = params["nominal_Gamma"]
    frequencies = params["num_distinct_freqs"]
    security_bits = params["lambda_bits"]
    support_count = comb(selected_count, weight) * comb(weight, mag2_count)
    raw_support = support_count * 2**weight
    moments = {
        int(key): F(value)
        for key, value in certificate["moments_M0_to_M30"].items()
    }
    degree = certificate["dual_poly"]["degree"]
    rational = {
        int(key): F(value)
        for key, value in certificate["dual_poly"]["coeffs_r0_to_r30"].items()
    }
    ok = True

    def check(name, condition, detail=""):
        nonlocal ok
        ok &= condition
        suffix = f": {detail}" if detail else ""
        print(f"[{'PASS' if condition else 'FAIL'}] {name}{suffix}")

    positions = [
        index
        for index in range(1, dimension)
        if two_adic_valuation(index) in {0, 3}
    ]
    check(
        "S72 structural parameters",
        dimension == 128
        and selected_count == len(positions) == 72
        and mag1_count == 38
        and mag2_count == 5
        and weight == 43,
    )
    check(
        "selected positions are stable under every odd Galois action",
        all(
            {(odd * position) % dimension for position in positions} == set(positions)
            for odd in range(1, 2 * dimension, 2)
        ),
    )
    check("B=l1^2", support_cap == params["l1_norm"] ** 2 == 2304)
    check("challenge energy is 58", params["l2_sq"] == 58)
    check("nominal threshold is 18", nominal_gamma == 18)
    check("64 conjugacy-distinct frequencies", frequencies == dimension // 2 == 64)
    check("raw support", str(raw_support) == certificate["N_raw"])
    expected_second = (
        2 * params["l2_sq"] ** 2
        - (mag1_count + 16 * mag2_count)
        - F(
            params["l2_sq"] ** 2 - (mag1_count + 16 * mag2_count),
            selected_count - 1,
        )
    )
    check(
        "M0, M1, and independent closed-form M2",
        moments[0] == 1 and moments[1] == 58 and moments[2] == expected_second,
        f"M2={moments[2]}",
    )

    containment = certificate["fixed_point_containment"]
    fixed_point_q = containment["fractional_bits"]
    coordinate_error = containment["root_coordinate_error_units"]
    runtime_threshold = containment["runtime_strict_threshold"]
    root_error = params["l1_norm"] * coordinate_error
    squared_error = 8 * root_error * root_error
    margin = math.isqrt(squared_error)
    if margin * margin <= squared_error:
        margin += 1
    scale = 1 << fixed_point_q
    true_norm_upper = F(runtime_threshold * scale - margin, scale)
    tail_start_sq = true_norm_upper * true_norm_upper
    check(
        "runtime table contract uses q=48 and root error at most 4",
        fixed_point_q == 48 and coordinate_error == 4,
    )
    check("runtime threshold matches nominal threshold", runtime_threshold == nominal_gamma)
    check(
        "fixed-point margin is the least strict integer upper bound",
        containment["rounding_margin_units"] == margin
        and margin * margin > squared_error
        and (margin - 1) * (margin - 1) <= squared_error,
        f"h={margin}, 8r^2={squared_error}",
    )
    check(
        "certified true-norm boundary",
        F(containment["certified_true_norm_upper_bound"]) == true_norm_upper,
        str(true_norm_upper),
    )
    check(
        "certified tail boundary",
        F(containment["certified_tail_start_sq"]) == tail_start_sq,
    )

    generation = certificate["moment_generation"]
    reconstruction_primes = generation["reconstruction_primes"]
    check_prime = generation["independent_check_prime"]
    modulus_product = math.prod(reconstruction_primes)
    check(
        "CRT product exceeds the a priori degree-30 numerator bound",
        modulus_product > support_count * support_cap**degree,
        f"{modulus_product.bit_length()} > {(support_count * support_cap**degree).bit_length()} bits",
    )
    residue_checks = True
    for prime in reconstruction_primes + [check_prime]:
        path = certificate_path.parent / f"moments_{prime}.txt"
        if not path.exists():
            residue_checks = False
            break
        header_prime, residues = load_residues(path)
        residue_checks &= header_prime == prime
        for moment in range(degree + 1):
            numerator_sum = moments[moment] * support_count
            if (
                numerator_sum.denominator != 1
                or numerator_sum.numerator % prime != residues[moment]
            ):
                residue_checks = False
                break
    check("all moments match eight modular generator outputs", residue_checks)

    half = degree // 2
    matrices = [
        [[moments[i + j] for j in range(half + 1)] for i in range(half + 1)],
        [[moments[i + j + 1] for j in range(half)] for i in range(half)],
        [
            [
                support_cap * moments[i + j] - moments[i + j + 1]
                for j in range(half)
            ]
            for i in range(half)
        ],
    ]
    for label, matrix in zip(["Hankel", "x localizer", "B-x localizer"], matrices):
        is_positive, minimum_pivot = positive_definite(matrix)
        check(
            f"moment admissibility {label}",
            is_positive,
            f"minimum LDL pivot {float(minimum_pivot):.3e}",
        )

    chebyshev = chebyshev_monomials(degree)
    normalized_moments = [
        sum(
            comb(power, index)
            * F(2, support_cap) ** index
            * (-1) ** (power - index)
            * moments[index]
            for index in range(power + 1)
        )
        for power in range(degree + 1)
    ]
    expected_chebyshev = [
        sum(
            chebyshev[current_degree][power] * normalized_moments[power]
            for power in range(len(chebyshev[current_degree]))
        )
        for current_degree in range(degree + 1)
    ]
    marginal_tail = sum(
        rational[index] * expected_chebyshev[index]
        for index in range(degree + 1)
    )
    acceptance_floor = 1 - frequencies * marginal_tail
    check(
        "claimed marginal tail bound",
        marginal_tail == F(certificate["claimed"]["q1_star"]),
        f"q={float(marginal_tail):.15f}",
    )
    check(
        "claimed acceptance floor",
        acceptance_floor == F(certificate["claimed"]["p0"]),
        f"p0={float(acceptance_floor):.15f}",
    )

    z_coefficients = [F(0)] * (degree + 1)
    for current_degree in range(degree + 1):
        for power, coefficient in enumerate(chebyshev[current_degree]):
            z_coefficients[power] += rational[current_degree] * coefficient
    positive, root_count = strictly_positive_on_interval(
        z_coefficients, F(-1), F(1)
    )
    check("Q>0 exact Sturm certificate on [-1,1]", positive, f"roots={root_count}")
    z_minus_one = z_coefficients[:]
    z_minus_one[0] -= 1
    threshold_z = F(2, support_cap) * tail_start_sq - 1
    positive, root_count = strictly_positive_on_interval(
        z_minus_one, threshold_z, F(1)
    )
    check(
        "Q-1>0 exact Sturm certificate on the tail",
        positive,
        f"roots={root_count}",
    )

    if check_bernstein:
        x_coefficients = [F(0)] * (degree + 1)
        for current_degree in range(degree + 1):
            for power, coefficient in enumerate(chebyshev[current_degree]):
                for index in range(power + 1):
                    x_coefficients[index] += (
                        rational[current_degree]
                        * coefficient
                        * comb(power, index)
                        * F(2, support_cap) ** index
                        * (-1) ** (power - index)
                    )
        subdivisions = certificate["bernstein_certificate"]["num_subintervals"]
        gap0 = interval_certificate_min(
            x_coefficients, F(0), F(support_cap), subdivisions, degree
        )
        x_minus_one = x_coefficients[:]
        x_minus_one[0] -= 1
        gap1 = interval_certificate_min(
            x_minus_one, tail_start_sq, F(support_cap), subdivisions, degree
        )
        check("Q>=0 exact Bernstein certificate", gap0 >= 0, f"minimum={float(gap0):.3e}")
        check("Q-1>=0 exact Bernstein certificate", gap1 >= 0, f"minimum={float(gap1):.3e}")
        check(
            "claimed Bernstein minima",
            gap0 == F(certificate["claimed"]["minimum_bernstein_Q"])
            and gap1 == F(certificate["claimed"]["minimum_bernstein_Q_minus_1"]),
        )

    cutoff = F(1, frequencies) * (1 - F(2**security_bits, raw_support))
    check(
        "q below exact support-floor cutoff",
        marginal_tail < cutoff,
        f"{float(marginal_tail):.9f} < {float(cutoff):.9f}",
    )
    check(
        "accepted support is at least 2^128",
        F(raw_support) * acceptance_floor >= 2**security_bits,
    )
    accepted_bits = (
        math.log2(raw_support)
        + math.log2(acceptance_floor.numerator)
        - math.log2(acceptance_floor.denominator)
    )
    print(f"RESULT: {'ALL CHECKS PASS' if ok else 'FAILURE'}")
    print(f"Pr[Gamma(c)<{true_norm_upper}] >= {float(acceptance_floor):.12f}")
    print(f"log2(N_raw*p0) = {accepted_bits:.12f}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
