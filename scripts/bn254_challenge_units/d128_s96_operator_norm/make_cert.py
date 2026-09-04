#!/usr/bin/env python3
"""Assemble the exact S96 accepted-support certificate."""

import json
import math
from fractions import Fraction as F
from math import comb
from pathlib import Path

DIMENSION = 128
SELECTED_POSITIONS = 96
MAG1_COUNT = 35
MAG2_COUNT = 1
WEIGHT = MAG1_COUNT + MAG2_COUNT
L1_NORM = MAG1_COUNT + 2 * MAG2_COUNT
L2_SQ = MAG1_COUNT + 4 * MAG2_COUNT
SUPPORT_CAP = L1_NORM**2
GAMMA = 15
LAMBDA = 128
NFREQ = DIMENSION // 2
FIXED_POINT_Q = 48
ROOT_COORDINATE_ERROR_UNITS = 4
PRIMES = [
    2305843009213689601,
    2305843009213689089,
    2305843009213687297,
    2305843009213683713,
    2305843009213682689,
    2305843009213675777,
    2305843009213673729,
    2305843009213666049,
]


def fixed_point_containment():
    scale = 1 << FIXED_POINT_Q
    root_error = L1_NORM * ROOT_COORDINATE_ERROR_UNITS
    squared_error = 8 * root_error * root_error
    margin = math.isqrt(squared_error)
    if margin * margin <= squared_error:
        margin += 1
    true_norm_upper = F(GAMMA * scale - margin, scale)
    return margin, true_norm_upper, true_norm_upper * true_norm_upper


def main():
    moments = json.loads(Path("moments_d128_s96_a35_b1.json").read_text())
    coefficients = json.loads(Path("dual_coefficients.json").read_text())
    summary = json.loads(Path("dual_summary.json").read_text())
    support_count = comb(SELECTED_POSITIONS, WEIGHT) * comb(WEIGHT, MAG2_COUNT)
    raw_support = support_count * 2**WEIGHT
    margin, true_norm_upper, tail_start_sq = fixed_point_containment()
    certificate = {
        "description": (
            "Exact accepted-support certificate for the D128 S96 mixed shell: "
            "35 signed coefficients of magnitude one and one signed coefficient "
            "of magnitude two on positions i in [1,127] with i not divisible by 4."
        ),
        "method": "degree-30 exact moment dual, exact polynomial positivity, union bound",
        "params": {
            "d": DIMENSION,
            "selected_positions": SELECTED_POSITIONS,
            "position_predicate": "1 <= i < 128 and i mod 4 != 0",
            "a_mag1": MAG1_COUNT,
            "b_mag2": MAG2_COUNT,
            "w_nonzero": WEIGHT,
            "l1_norm": L1_NORM,
            "l2_sq": L2_SQ,
            "support_cap_B": SUPPORT_CAP,
            "nominal_Gamma": GAMMA,
            "lambda_bits": LAMBDA,
            "num_distinct_freqs": NFREQ,
        },
        "fixed_point_containment": {
            "fractional_bits": FIXED_POINT_Q,
            "root_coordinate_error_units": ROOT_COORDINATE_ERROR_UNITS,
            "runtime_strict_threshold": GAMMA,
            "rounding_margin_units": margin,
            "certified_true_norm_upper_bound": str(true_norm_upper),
            "certified_tail_start_sq": str(tail_start_sq),
        },
        "N_raw": str(raw_support),
        "moments_M0_to_M30": moments,
        "dual_poly": {
            "basis": "shifted_Chebyshev",
            "def": "Q(x)=sum_j r_j * T_j(2x/B - 1)",
            "degree": 30,
            "coeffs_r0_to_r30": coefficients,
        },
        "bernstein_certificate": {
            "num_subintervals": summary["subdivisions"],
            "nonneg_interval_for_Q": [0, SUPPORT_CAP],
            "ge1_interval_for_Q_minus_1": [str(tail_start_sq), SUPPORT_CAP],
        },
        "claimed": {
            "q1_star": summary["q"],
            "p0": summary["p0"],
            "cutoff_q1_for_floor": summary["cutoff"],
            "minimum_bernstein_Q": summary["gap0"],
            "minimum_bernstein_Q_minus_1": summary["gap1"],
        },
        "moment_generation": {
            "algorithm": (
                "exact truncated mixed-shell bivariate EGF over the 96 selected "
                "positions, seven-prime CRT plus independent eighth-prime check"
            ),
            "reconstruction_primes": PRIMES[:7],
            "independent_check_prime": PRIMES[7],
            "a_priori_numerator_bound": "C(96,36)*36*1369^m for moment m",
        },
    }
    Path("cert_d128_s96_a35_b1_gamma15.json").write_text(
        json.dumps(certificate, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
