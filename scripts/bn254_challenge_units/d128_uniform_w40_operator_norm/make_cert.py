#!/usr/bin/env python3
import json
import math
from fractions import Fraction as F
from math import comb
from pathlib import Path

D = 128
W = 40
GAMMA = 15
LAMBDA = 145
NFREQ = D // 2
B = W * W
FIXED_POINT_Q = 48
ROOT_COORDINATE_ERROR_UNITS = 4


def fixed_point_containment():
    scale = 1 << FIXED_POINT_Q
    r_error = W * ROOT_COORDINATE_ERROR_UNITS
    squared_error = 8 * r_error * r_error
    margin = math.isqrt(squared_error)
    if margin * margin <= squared_error:
        margin += 1
    true_norm_upper = F(GAMMA * scale - margin, scale)
    return margin, true_norm_upper, true_norm_upper * true_norm_upper


def main():
    moments = json.loads(Path("moments_d128_w40.json").read_text())
    coefficients = json.loads(Path("dual_coefficients.json").read_text())
    summary = json.loads(Path("dual_summary.json").read_text())
    nraw = comb(D, W) * 2**W
    margin, true_norm_upper, tail_start_sq = fixed_point_containment()
    cert = {
        "description": (
            "Exact accepted-support certificate for a true-operator-norm subset of the "
            "experimental uniform d=128, weight-40 signed-unit challenge family that is contained "
            "in the strict q=48 fixed-point predicate at runtime threshold 14."
        ),
        "method": "degree-30 exact moment dual, exact Bernstein positivity, union bound",
        "params": {
            "d": D,
            "a_mag1": W,
            "b_mag2": 0,
            "w_nonzero": W,
            "l1_norm": W,
            "l2_sq": W,
            "support_cap_B": B,
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
        "N_raw": str(nraw),
        "moments_M0_to_M30": moments,
        "dual_poly": {
            "basis": "shifted_Chebyshev",
            "def": "Q(x)=sum_j r_j * T_j(2x/B - 1)",
            "degree": 30,
            "coeffs_r0_to_r30": coefficients,
        },
        "bernstein_certificate": {
            "num_subintervals": summary["subdivisions"],
            "nonneg_interval_for_Q": [0, B],
            "ge1_interval_for_Q_minus_1": [str(tail_start_sq), B],
        },
        "claimed": {
            "q1_star": summary["q"],
            "p0": summary["p0"],
            "cutoff_q1_for_floor": summary["cutoff"],
            "minimum_bernstein_Q": summary["gap0"],
            "minimum_bernstein_Q_minus_1_at_nominal_threshold": summary["gap1"],
        },
        "moment_generation": {
            "algorithm": "exact truncated bivariate EGF over finite-field primitive 256th roots, eight-prime CRT plus independent ninth-prime check",
            "reconstruction_primes": [
                2305843009213689601,
                2305843009213689089,
                2305843009213687297,
                2305843009213683713,
                2305843009213682689,
                2305843009213675777,
                2305843009213673729,
                2305843009213666049,
            ],
            "independent_check_prime": 2305843009213663489,
            "a_priori_numerator_bound": "C(128,40)*1600^m for moment m",
        },
    }
    Path("cert_d128_uniform_w40_gamma15.json").write_text(json.dumps(cert, indent=1) + "\n")


if __name__ == "__main__":
    main()
