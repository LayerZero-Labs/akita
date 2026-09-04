#!/usr/bin/env python3
"""Check the exact bounded-fiber reduction for the uniform D128 candidate."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path


DIMENSION = 128
WEIGHT = 35
MINIMUM_SQUARED_NORM_TARGET = 76
ROOT_COUNT = 128
SECURITY_BITS = 128


def log2_fraction(value: Fraction) -> float:
    return math.log2(value.numerator) - math.log2(value.denominator)


def main() -> None:
    certificate_path = Path(__file__).with_name(
        "cert_d128_uniform_w35_gamma14.json"
    )
    certificate = json.loads(certificate_path.read_text())
    params = certificate["params"]
    assert params["d"] == DIMENSION
    assert params["w_nonzero"] == WEIGHT
    assert params["l2_sq"] == WEIGHT
    assert params["lambda_bits"] == 136

    raw_support = int(certificate["N_raw"])
    acceptance_floor = Fraction(certificate["claimed"]["p0"])
    accepted_support_floor = raw_support * acceptance_floor
    accepted_bits = log2_fraction(accepted_support_floor)

    pairwise_inner_product_upper = (
        2 * WEIGHT - MINIMUM_SQUARED_NORM_TARGET
    ) // 2
    assert pairwise_inner_product_upper == -3

    fiber_cap = 1
    while (
        (fiber_cap + 1) * WEIGHT
        + (fiber_cap + 1)
        * fiber_cap
        * pairwise_inner_product_upper
        >= 0
    ):
        fiber_cap += 1
    assert fiber_cap == 12

    conditional_failure_bound = Fraction(
        ROOT_COUNT * (fiber_cap - 1), 1
    ) / accepted_support_floor
    conditional_security_bits = -log2_fraction(conditional_failure_bound)
    assert conditional_failure_bound < Fraction(1, 2**SECURITY_BITS)

    modulus = int(
        "21888242871839275222246405745257275088548364400416034343698204186575808495617"
    )
    radius = MINIMUM_SQUARED_NORM_TARGET - 1
    log2_ball_volume_over_determinant = (
        (DIMENSION / 2) * math.log2(math.pi * radius)
        - math.lgamma(DIMENSION / 2 + 1) / math.log(2)
        - math.log2(modulus)
    )

    print(f"certified accepted support: {accepted_bits:.12f} bits")
    print(f"certified expected trials: {float(1 / acceptance_floor):.12f}")
    print(
        "conditional kernel target: minimum nonzero squared norm "
        f">= {MINIMUM_SQUARED_NORM_TARGET}"
    )
    print(f"conditional per-root fiber cap: {fiber_cap}")
    print(
        "conditional 128-root failure bound: "
        f"2^-{conditional_security_bits:.12f}"
    )
    print(
        "heuristic log2(radius-75 ball volume / det(L)): "
        f"{log2_ball_volume_over_determinant:.12f}"
    )
    print("status: support and norm are exact; kernel lower bound remains open")


if __name__ == "__main__":
    main()
