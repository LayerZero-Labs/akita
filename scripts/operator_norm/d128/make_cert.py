#!/usr/bin/env python3
import json
from fractions import Fraction as F
from math import comb
from pathlib import Path

D = 128
W = 31
GAMMA = 13
LAMBDA = 128
NFREQ = D // 2
B = W * W


def main():
    moments = json.loads(Path("moments_d128_w31.json").read_text())
    coefficients = json.loads(Path("dual_coefficients.json").read_text())
    summary = json.loads(Path("dual_summary.json").read_text())
    nraw = comb(D, W) * 2**W
    cert = {
        "description": (
            "Exact accepted-support certificate for the production d=128, weight-31, "
            "signed unit sparse challenge family and true operator-norm subset Gamma<=13."
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
            "Gamma": GAMMA,
            "threshold_Gamma_sq": GAMMA * GAMMA,
            "lambda_bits": LAMBDA,
            "num_distinct_freqs": NFREQ,
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
            "ge1_interval_for_Q_minus_1": [GAMMA * GAMMA, B],
        },
        "claimed": {
            "q1_star": summary["q"],
            "p0": summary["p0"],
            "cutoff_q1_for_floor": summary["cutoff"],
            "minimum_bernstein_Q": summary["gap0"],
            "minimum_bernstein_Q_minus_1": summary["gap1"],
        },
        "moment_generation": {
            "algorithm": "exact truncated bivariate EGF over finite-field primitive 256th roots, seven-prime CRT plus independent eighth-prime check",
            "reconstruction_primes": [
                2305843009213689601,
                2305843009213689089,
                2305843009213687297,
                2305843009213683713,
                2305843009213682689,
                2305843009213675777,
                2305843009213673729,
            ],
            "independent_check_prime": 2305843009213666049,
            "a_priori_numerator_bound": "C(128,31)*961^m for moment m",
        },
    }
    Path("cert_d128_w31_gamma13.json").write_text(json.dumps(cert, indent=1) + "\n")


if __name__ == "__main__":
    main()
