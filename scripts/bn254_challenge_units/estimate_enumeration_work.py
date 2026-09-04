#!/usr/bin/env python3
"""Estimate exact Fincke--Pohst node counts from a reduced basis profile.

This is a workload estimator, not a security certificate. It sums the volumes
of the projected search balls divided by the corresponding Gram--Schmidt
covolumes. The D64 row provides a calibration against a completed exact search.
"""

import json
import math
from pathlib import Path

from check_d64_kernel import bareiss_determinant, exact_ldl

DIRECTORY = Path(__file__).resolve().parent


def projected_log2_nodes(basis, squared_radius):
    _, diagonal = exact_ldl(basis)
    dimension = len(diagonal)
    log_terms = []
    suffix_log_determinant = 0.0
    for projected_dimension in range(1, dimension + 1):
        suffix_log_determinant += math.log(
            float(diagonal[dimension - projected_dimension])
        )
        log_terms.append(
            (projected_dimension / 2) * math.log(math.pi * squared_radius)
            - math.lgamma(projected_dimension / 2 + 1)
            - 0.5 * suffix_log_determinant
        )
    maximum = max(log_terms)
    log_total = maximum + math.log(
        sum(math.exp(value - maximum) for value in log_terms)
    )
    return log_total / math.log(2)


def main():
    cases = [
        ("D64 exact certificate", "d64_kernel_basis.json", "squared_norm_exclusion_bound"),
        ("S72 exact candidate", "s72_kernel_basis.json", "squared_norm_exclusion_bound"),
        ("S96 BKZ-44 screen", "s96_screening_basis.json", "squared_norm_screening_bound"),
    ]
    for label, filename, bound_key in cases:
        artifact = json.loads((DIRECTORY / filename).read_text())
        basis = artifact["basis"]
        dimension = len(basis)
        positions = artifact.get("positions", list(range(dimension)))
        modulus = artifact["modulus"]
        powers = [pow(artifact["omega"], exponent, modulus) for exponent in positions]
        assert len(positions) == dimension
        assert all(
            sum(coefficient * power for coefficient, power in zip(row, powers))
            % modulus
            == 0
            for row in basis
        )
        assert abs(bareiss_determinant(basis)) == modulus
        estimate = projected_log2_nodes(artifact["basis"], artifact[bound_key])
        print(f"{label}: projected log2(nodes) = {estimate:.6f}")
    observed = [
        ("D64 observed", 1_507_538),
        ("S72 observed", 127_185_682),
    ]
    for label, nodes in observed:
        print(f"{label}: log2(nodes) = {math.log2(nodes):.6f}")


if __name__ == "__main__":
    main()
