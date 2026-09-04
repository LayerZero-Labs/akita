#!/usr/bin/env python3
"""Account for approximate-strong challenge loss in a coordinate-wise fork."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from math import comb
from pathlib import Path


BN254_SCALAR = int(
    "21888242871839275222246405745257275088548364400416034343698204186575808495617"
)
ROOTS = 128
TARGET_BITS = 128
HEURISTIC_DEGREE_CAP = 2
COORDINATE_COUNTS = (1, 8, 32, 128, 256)


def log2_fraction(value: Fraction) -> float:
    return math.log2(value.numerator) - math.log2(value.denominator)


def load_candidate(directory: str, certificate_name: str) -> dict[str, object]:
    path = Path(__file__).with_name(directory) / certificate_name
    certificate = json.loads(path.read_text())
    raw_support = int(certificate["N_raw"])
    acceptance_floor = Fraction(certificate["claimed"]["p0"])
    return {
        "name": directory,
        "raw_support": raw_support,
        "accepted_floor": raw_support * acceptance_floor,
        "threshold": certificate["params"]["nominal_Gamma"],
        "weight": certificate["params"]["w_nonzero"],
    }


def max_query_budget(
    support: Fraction, coordinates: int, degree_cap: int
) -> int:
    denominator = 2**TARGET_BITS * coordinates * (1 + degree_cap)
    max_q_plus_one = (support.numerator - 1) // (
        support.denominator * denominator
    )
    return max(-1, max_q_plus_one - 1)


def union_factorial_moment(
    vertices: int, edge_probability: Fraction, degree: int
) -> Fraction:
    return vertices * comb(vertices - 1, degree) * edge_probability**degree


def main() -> None:
    candidates = (
        load_candidate(
            "d128_uniform_w35_operator_norm",
            "cert_d128_uniform_w35_gamma14.json",
        ),
        load_candidate(
            "d128_uniform_w40_operator_norm",
            "cert_d128_uniform_w40_gamma15.json",
        ),
    )
    pair_probability = 1 - Fraction(BN254_SCALAR - 1, BN254_SCALAR) ** ROOTS
    print("fixed-anchor definition: epsilon_C = max_c0 d(c0) / |C|")
    print(
        "Fiat-Shamir challenge loss: "
        "(Q+1) * M * (1+d_max) / |C|"
    )
    print(
        "random-slot pair probability over 128 roots: "
        f"2^{log2_fraction(pair_probability):.6f} (heuristic model only)"
    )

    for candidate in candidates:
        raw_support = candidate["raw_support"]
        accepted_floor = candidate["accepted_floor"]
        assert isinstance(raw_support, int)
        assert isinstance(accepted_floor, Fraction)
        print()
        print(
            f"{candidate['name']}: weight={candidate['weight']}, "
            f"Gamma={candidate['threshold']}"
        )
        print(
            "  exact accepted support floor: "
            f"2^{log2_fraction(accepted_floor):.12f}"
        )
        for degree in (2, 3):
            failure = union_factorial_moment(
                raw_support, pair_probability, degree
            )
            print(
                "  independent-edge model Pr[max degree >= "
                f"{degree}]: <= 2^{log2_fraction(failure):.6f}"
            )
        print(
            "  maximum Q preserving a 2^-128 challenge term if d_max<=2:"
        )
        for coordinates in COORDINATE_COUNTS:
            maximum = max_query_budget(
                accepted_floor, coordinates, HEURISTIC_DEGREE_CAP
            )
            print(f"    M={coordinates:3d}: Q<={maximum}")

    print()
    print("status: support bounds are exact; degree-tail bounds are heuristic")


if __name__ == "__main__":
    main()
