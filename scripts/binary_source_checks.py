#!/usr/bin/env python3
"""Integer and adjoint checks for the proposed binary-source checkpoint.

This is an independent standard-library reference, not a parameter admission
certificate. Run without arguments for the fast checks, or with --full for
additional full-degree randomized adjoint checks. Compatible with Python 3.9.
"""

import argparse
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from math import comb, gcd, isqrt, log2
import random


P = (1 << 128) - 275
Q = 14253409
D = 648
SCALAR_DEGREE = 162
WIDTH = 256
WEIGHT = 46


def balanced_interval(depth, base=8):
    """Exact interval obtained from every digit in [-base/2, base/2-1]."""
    if depth < 1 or base < 2 or base % 2:
        raise ValueError("positive depth and positive even base required")
    geometric_sum = (base**depth - 1) // (base - 1)
    return -(base // 2) * geometric_sum, (base // 2 - 1) * geometric_sum


def ceil_sqrt(value):
    root = isqrt(value)
    return root + (root * root < value)


def is_prime(value):
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, isqrt(value) + 1, 2))


def multiplicative_order(value, modulus):
    if gcd(value, modulus) != 1:
        raise ValueError("value must be a unit modulo modulus")
    residue = value % modulus
    order = 1
    while residue != 1:
        residue = residue * value % modulus
        order += 1
        if order > modulus:
            raise ValueError("order search failed")
    return order


@lru_cache(maxsize=None)
def monomial_reduction(degree, middle_sign, exponent):
    """Sparse remainder of X^exponent modulo X^degree-sign X^(degree/2)+1.

    middle_sign=1 gives the source ring; -1 gives the scalar ring. The
    recursive reference does not assume either ring's special closed form.
    """
    if degree < 2 or degree % 2 or middle_sign not in (-1, 1) or exponent < 0:
        raise ValueError("even degree, sign +/-1, nonnegative exponent required")
    if exponent < degree:
        return ((exponent, 1),)
    coefficients = {}
    for reduced_exponent, multiplier in (
        (exponent - degree // 2, middle_sign),
        (exponent - degree, -1),
    ):
        for index, coefficient in monomial_reduction(
            degree, middle_sign, reduced_exponent
        ):
            coefficients[index] = coefficients.get(index, 0) + multiplier * coefficient
    return tuple(sorted((index, value) for index, value in coefficients.items() if value))


def convolution(left, right):
    if not left or not right:
        return []
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def reduce_polynomial(coefficients, degree, middle_sign=1):
    """Independent descending long division, including exact integer signs."""
    result = list(coefficients) + [0] * max(0, degree - len(coefficients))
    for exponent in range(len(result) - 1, degree - 1, -1):
        value = result[exponent]
        result[exponent] = 0
        result[exponent - degree // 2] += middle_sign * value
        result[exponent - degree] -= value
    return result[:degree]


def ring_product(left, right, middle_sign=1):
    if len(left) != len(right):
        raise ValueError("ring operands must have equal dimensions")
    return reduce_polynomial(convolution(left, right), len(left), middle_sign)


def multiplication_row_bounds(degree, middle_sign=1):
    """Exhaust every input pair, not merely a sample of monomial shifts."""
    pair_l1 = [0] * degree
    maximum_shift_l1 = 0
    for u in range(degree):
        shift_l1 = [0] * degree
        for v in range(degree):
            for index, coefficient in monomial_reduction(degree, middle_sign, u + v):
                pair_l1[index] += abs(coefficient)
                shift_l1[index] += abs(coefficient)
        maximum_shift_l1 = max(maximum_shift_l1, max(shift_l1))
    unreduced_l1 = [0] * degree
    for exponent in range(2 * degree - 1):
        for index, coefficient in monomial_reduction(degree, middle_sign, exponent):
            unreduced_l1[index] += abs(coefficient)
    return {
        "product_multiplier": max(pair_l1),
        "monomial_multiplier": maximum_shift_l1,
        "reduction_multiplier": max(unreduced_l1),
    }


def adjoint_direct(output_weights, right, middle_sign=1):
    """Transpose of multiplication by right in the coefficient dot product."""
    degree = len(right)
    if len(output_weights) != degree:
        raise ValueError("weight and operand dimensions must agree")
    extended = [
        sum(output_weights[index] * coefficient for index, coefficient in
            monomial_reduction(degree, middle_sign, exponent))
        for exponent in range(2 * degree - 1)
    ]
    return [sum(right[t] * extended[s + t] for t in range(degree))
            for s in range(degree)]


def adjoint_split_convolution(output_weights, right, middle_sign=1):
    """Same transpose using two ordinary products of degree < degree.

    A literal convolution with the entire extended row has length 3D-2.
    Splitting that row gives two products of length 2D-1 instead.
    """
    degree = len(right)
    if len(output_weights) != degree:
        raise ValueError("weight and operand dimensions must agree")
    extended = [
        sum(output_weights[index] * coefficient for index, coefficient in
            monomial_reduction(degree, middle_sign, exponent))
        for exponent in range(2 * degree - 1)
    ]
    reverse = right[::-1]
    low = convolution(extended[:degree], reverse)
    high = convolution(extended[degree:] + [0], reverse)
    return [low[degree - 1 + s] + (high[s - 1] if s else 0)
            for s in range(degree)]


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("dot-product dimensions must agree")
    return sum(a * b for a, b in zip(left, right))


def response_sizing(n, width=WIDTH, weight=WEIGHT):
    rows = (1 << n) // width
    ring_rows = rows // 4
    ceiling_argument = (16 * ring_rows * D + 6) // 7
    ceiling_log = (ceiling_argument - 1).bit_length()
    tail_log = (71 * ceiling_log + 99) // 100
    return {"tail_log": tail_log,
            "cap": ceil_sqrt(2 * width * weight * 4 * tail_log)}


def minimum_support_weight(arity, target_bits):
    """Minimum weight by exact integer comparison, with deterministic signs."""
    support = 0
    for weight in range(SCALAR_DEGREE + 1):
        support += comb(SCALAR_DEGREE, weight)
        if support >= arity * (1 << target_bits):
            return weight
    raise ValueError("target exceeds the full support space")


def required_work_bits(degree, modulus=P, target_bits=128):
    if degree < 1 or modulus < 2 or target_bits < 0:
        raise ValueError("positive degree, modulus >= 2, nonnegative bits required")
    bits = 0
    while degree * (1 << target_bits) > modulus * (1 << bits):
        bits += 1
    return bits


def quotient_fallback_bounds(n, multiplier):
    """Honest centered-image carry and malicious twelve-digit lift bounds."""
    ring_rows = (1 << n) // WIDTH // 4
    centered_bound = (Q - 1) // 2
    response_bound = -balanced_interval(4)[0]
    honest_fraction = Fraction(centered_bound, Q) * (
        multiplier * ring_rows * response_bound + WIDTH * 2 * WEIGHT
    )
    honest_carry = -(-honest_fraction.numerator // honest_fraction.denominator)
    carry_min, carry_max = balanced_interval(12)
    residual = geometry_bounds(n, 3, multiplier)["residual"]
    return {
        "honest_carry_fraction": str(honest_fraction),
        "honest_carry": honest_carry,
        "carry_interval": [carry_min, carry_max],
        "exact_lift_bound": residual - Q * carry_min,
    }


def geometry_bounds(n, rank, multiplier, width=WIDTH, weight=WEIGHT):
    rows = (1 << n) // width
    ring_rows = rows // 4
    image_min, _ = balanced_interval(8)
    response_min, response_max = balanced_interval(4)
    image_bound, response_bound = -image_min, -response_min
    gamma = 2 * weight
    residual = (multiplier * ring_rows * ((Q - 1) // 2) * response_bound
                + width * gamma * image_bound)
    projection = (Q - 1) * rank * D * residual
    carry_depth = 9 if n <= 23 else 10
    carry_min, carry_max = balanced_interval(carry_depth)
    parity_raw = SCALAR_DEGREE * rows * 4 + width * weight * 4
    parity_left = 3 * parity_raw
    return {
        "n": n, "rank": rank, "ring_rows": ring_rows,
        "residual": residual, "projection": projection,
        "projection_exact_lift": 2 * projection < P,
        "projection_signed_bytes": (projection.bit_length() + 1 + 7) // 8,
        "parity_raw": parity_raw, "carry_depth": carry_depth,
        "carry_bound": parity_left // 2,
        "carry_positive_reach": carry_max,
        "parity_exact_lift_bound": parity_left - 2 * carry_min,
        "nine_digit_carry_fits": parity_left // 2 <= balanced_interval(9)[1],
        "response_diameter": response_max - response_min,
    }


def arithmetic_report():
    source_kernel = multiplication_row_bounds(D)
    scalar_kernel = multiplication_row_bounds(SCALAR_DEGREE, -1)
    support = sum(comb(SCALAR_DEGREE, i) for i in range(WEIGHT + 1))
    gamma = 2 * WEIGHT
    response_min, response_max = balanced_interval(4)
    order = multiplicative_order(P, 243)
    coefficient_bits = (6 * WEIGHT).bit_length()
    value_min, value_max = balanced_interval(6)
    value_honest_left = 3 * SCALAR_DEGREE * WIDTH
    value_lift = 3 * SCALAR_DEGREE * WIDTH * 4 + 1 - 2 * value_min
    geometries = [geometry_bounds(n, rank, source_kernel["product_multiplier"])
                  for n in range(22, 25) for rank in (3, 4)]
    image_min, image_max = balanced_interval(8)
    transform_modulus = 23328
    centered_cap = 2 * image_max + 1
    admissible_primes = [candidate for candidate in range(1, centered_cap + 1,
                                                        transform_modulus)
                         if is_prime(candidate)]
    support_profiles = {}
    for weight in (46, 64, 77):
        count = sum(comb(SCALAR_DEGREE, i) for i in range(weight + 1))
        support_profiles[str(weight)] = {
            "support_size": count, "support_bits": log2(count),
            "fold_debit_bits": log2(count) - log2(WIDTH),
            "source_comparison_radius": 4 * 2 * weight * (response_max - response_min),
        }
    return {
        "q_prime": is_prime(Q), "q_transform_remainder": Q % transform_modulus,
        "largest_centered_congruent_prime": max(admissible_primes),
        "image_interval": [image_min, image_max], "centered_cap": centered_cap,
        "source_kernel": source_kernel, "scalar_kernel": scalar_kernel,
        "prime_order_mod_243": order,
        "unit_resultant_lower_bits": order * (P.bit_length() - 1),
        "unit_resultant_upper_bits": 81 * coefficient_bits,
        "support_size": support, "support_bits": log2(support),
        "support_profiles": support_profiles,
        "minimum_support_weights": {
            "arity256_target152": minimum_support_weight(256, 152),
            "arity512_target128": minimum_support_weight(512, 128),
            "arity1024_target128": minimum_support_weight(1024, 128),
        },
        "fold_debit_bits": log2(support) - log2(WIDTH),
        "fold_target_met": support >= WIDTH * (1 << 128),
        "projection_error": str(Fraction(1, Q**7)),
        "projection_error_bits": 7 * log2(Q),
        "geometry": geometries,
        "response_sizing": {str(n): response_sizing(n) for n in range(22, 25)},
        "candidate_width512": response_sizing(24, width=512, weight=47),
        "weight77_response_sizing": {
            str(n): response_sizing(n, weight=77) for n in (22, 23, 26)
        },
        "work_bits": {str(degree): required_work_bits(degree)
                      for degree in (1, 2, 647, 1294)},
        "quotient_fallback": {
            str(n): quotient_fallback_bounds(n, source_kernel["product_multiplier"])
            for n in range(22, 25)
        },
        "value_honest_carry_bound": (value_honest_left + 1) // 2,
        "value_carry_interval": [value_min, value_max],
        "value_exact_lift_bound": value_lift,
        "agreement_bound": 2 * gamma + response_max - response_min,
        "source_comparison_radius": 4 * gamma * (response_max - response_min),
        "completeness_failure_bits": -4096 * log2(Fraction(7, 8)),
        "literal_adjoint_convolution_length": 3 * D - 2,
        "split_adjoint_convolution_length": 2 * D - 1,
    }


def check_adjoint(full=False):
    """Exhaust small analogues and run reproducible full-degree checks."""
    count = 0
    for degree in (2, 4):
        basis = [tuple(int(i == j) for i in range(degree)) for j in range(degree)]
        for sign in (-1, 1):
            # Basis triples exhaust all coefficients of the trilinear identity.
            for weights, left, right in product(basis, repeat=3):
                direct = adjoint_direct(weights, right, sign)
                split = adjoint_split_convolution(weights, right, sign)
                assert direct == split
                assert dot(weights, ring_product(left, right, sign)) == dot(direct, left)
                count += 1
    # Exhaust the whole signed coefficient cube at the smallest analogue.
    vectors = list(product((-1, 0, 1), repeat=2))
    for weights, left, right in product(vectors, repeat=3):
        for sign in (-1, 1):
            direct = adjoint_direct(weights, right, sign)
            assert direct == adjoint_split_convolution(weights, right, sign)
            assert dot(weights, ring_product(left, right, sign)) == dot(direct, left)
            count += 1
    rng = random.Random(0xB1A648)
    for _ in range(24 if full else 3):
        weights = [rng.randrange(P) for _ in range(D)]
        left = [rng.randrange(-4, 4) for _ in range(D)]
        right = [rng.randrange(P) for _ in range(D)]
        direct = adjoint_direct(weights, right)
        assert direct == adjoint_split_convolution(weights, right)
        assert dot(weights, ring_product(left, right)) % P == dot(direct, left) % P
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="more full-degree random checks")
    args = parser.parse_args()
    report = arithmetic_report()
    assert report["q_prime"]
    assert report["largest_centered_congruent_prime"] == Q
    assert report["fold_target_met"]
    assert Q**7 >= 1 << 152
    assert report["unit_resultant_lower_bits"] >= report["unit_resultant_upper_bits"]
    assert report["value_exact_lift_bound"] < P
    assert report["agreement_bound"] < P
    for geometry in report["geometry"]:
        assert geometry["projection_exact_lift"]
        assert geometry["projection_signed_bytes"] <= 12
        assert geometry["carry_bound"] <= geometry["carry_positive_reach"]
        assert geometry["parity_exact_lift_bound"] < P
    for fallback in report["quotient_fallback"].values():
        assert fallback["honest_carry"] <= fallback["carry_interval"][1]
        assert fallback["exact_lift_bound"] < P
    report["adjoint_cases"] = check_adjoint(args.full)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
