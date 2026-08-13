#!/usr/bin/env python3
import json
import math
import sys
from fractions import Fraction as F
from math import comb
from pathlib import Path


def chebyshev_monomials(n):
    polys = [[F(1)], [F(0), F(1)]]
    for degree in range(2, n + 1):
        current = [F(0)] * (degree + 1)
        for i, coefficient in enumerate(polys[degree - 1]):
            current[i + 1] += 2 * coefficient
        for i, coefficient in enumerate(polys[degree - 2]):
            current[i] -= coefficient
        polys.append(current)
    return polys


def trim(polynomial):
    while polynomial and polynomial[-1] == 0:
        polynomial.pop()
    return polynomial


def polynomial_remainder(dividend, divisor):
    remainder = dividend[:]
    while remainder and len(remainder) >= len(divisor):
        factor = remainder[-1] / divisor[-1]
        shift = len(remainder) - len(divisor)
        for i, coefficient in enumerate(divisor):
            remainder[i + shift] -= factor * coefficient
        trim(remainder)
    return remainder


def sturm_sequence(polynomial):
    derivative = [i * polynomial[i] for i in range(1, len(polynomial))]
    sequence = [trim(polynomial[:]), trim(derivative)]
    while sequence[-1]:
        remainder = [-coefficient for coefficient in polynomial_remainder(sequence[-2], sequence[-1])]
        if not remainder:
            break
        # Positive rescaling preserves the Sturm sign variations and keeps the
        # exact rational coefficients small enough for a fast stdlib check.
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
    for k in range(len(work)):
        pivot = work[k][k]
        pivots.append(pivot)
        if pivot <= 0:
            return False, pivot
        for i in range(k + 1, len(work)):
            factor = work[i][k] / pivot
            for j in range(k, len(work)):
                work[i][j] -= factor * work[k][j]
    return True, min(pivots)


def bernstein_min(coefficients, lo, hi, degree):
    length = hi - lo
    reparameterized = [F(0)] * (degree + 1)
    for i, coefficient in enumerate(coefficients):
        for power in range(i + 1):
            reparameterized[power] += (
                coefficient * comb(i, power) * lo ** (i - power) * length ** power
            )
    return min(
        sum(
            F(comb(k, i), comb(degree, i)) * reparameterized[i]
            for i in range(k + 1)
        )
        for k in range(degree + 1)
    )


def interval_certificate_min(coefficients, lo, hi, subdivisions, degree):
    return min(
        bernstein_min(
            coefficients,
            lo + (hi - lo) * F(i, subdivisions),
            lo + (hi - lo) * F(i + 1, subdivisions),
            degree,
        )
        for i in range(subdivisions)
    )


def load_residues(path):
    return {
        int(line.split()[0]): int(line.split()[1])
        for line in path.read_text().splitlines()[1:]
    }


def main():
    arguments = sys.argv[1:]
    check_bernstein = "--check-bernstein" in arguments
    arguments = [argument for argument in arguments if argument != "--check-bernstein"]
    cert_path = Path(arguments[0]) if arguments else Path("cert_d128_w31_gamma13.json")
    cert = json.loads(cert_path.read_text())
    params = cert["params"]
    d = params["d"]
    w = params["w_nonzero"]
    b = params["b_mag2"]
    B = params["support_cap_B"]
    nominal_gamma = params["nominal_Gamma"]
    nfreq = params["num_distinct_freqs"]
    security_bits = params["lambda_bits"]
    nraw = comb(d, w) * comb(w, b) * 2**w
    moments = {int(k): F(v) for k, v in cert["moments_M0_to_M30"].items()}
    degree = cert["dual_poly"]["degree"]
    rational = {int(k): F(v) for k, v in cert["dual_poly"]["coeffs_r0_to_r30"].items()}
    ok = True

    def check(name, condition, detail=""):
        nonlocal ok
        ok &= condition
        print(f"[{'PASS' if condition else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    check("production dimensions", d == 128 and w == 31 and b == 0)
    check("B=l1^2", B == params["l1_norm"] ** 2 == 961)
    check("nominal dual threshold is 13", nominal_gamma == 13)
    check("64 conjugacy-distinct odd frequencies", nfreq == d // 2 == 64)
    check("raw support", str(nraw) == cert["N_raw"])
    check("M0=1 and M1=31", moments[0] == 1 and moments[1] == 31)

    containment = cert["fixed_point_containment"]
    fixed_point_q = containment["fractional_bits"]
    eps = containment["root_coordinate_error_units"]
    runtime_threshold = containment["runtime_strict_threshold"]
    r_error = params["l1_norm"] * eps
    squared_error = 8 * r_error * r_error
    margin = math.isqrt(squared_error)
    if margin * margin <= squared_error:
        margin += 1
    scale = 1 << fixed_point_q
    true_norm_upper = F(runtime_threshold * scale - margin, scale)
    tail_start_sq = true_norm_upper * true_norm_upper
    check("runtime table contract uses q=48 and root error at most 4",
          fixed_point_q == 48 and eps == 4)
    check("runtime threshold matches the nominal dual threshold",
          runtime_threshold == nominal_gamma)
    check("fixed-point margin is the least strict integer upper bound",
          containment["rounding_margin_units"] == margin
          and margin * margin > squared_error
          and (margin - 1) * (margin - 1) <= squared_error,
          f"h={margin}, 8r^2={squared_error}")
    check("certified true-norm upper bound matches fixed-point containment",
          F(containment["certified_true_norm_upper_bound"]) == true_norm_upper,
          str(true_norm_upper))
    check("certified spectral tail start is the square of that bound",
          F(containment["certified_tail_start_sq"]) == tail_start_sq,
          str(tail_start_sq))

    support_count = comb(d, w)
    residue_checks = True
    for prime in cert["moment_generation"]["reconstruction_primes"] + [cert["moment_generation"]["independent_check_prime"]]:
        path = cert_path.parent / f"moments_{prime}.txt"
        if not path.exists():
            residue_checks = False
            break
        residues = load_residues(path)
        for m in range(degree + 1):
            numerator_sum = moments[m] * support_count
            if numerator_sum.denominator != 1 or numerator_sum.numerator % prime != residues[m]:
                residue_checks = False
                break
    check("all moments match eight modular generator outputs", residue_checks)

    half = degree // 2
    matrices = [
        [[moments[i + j] for j in range(half + 1)] for i in range(half + 1)],
        [[moments[i + j + 1] for j in range(half)] for i in range(half)],
        [[B * moments[i + j] - moments[i + j + 1] for j in range(half)] for i in range(half)],
    ]
    for label, matrix in zip(["Hankel", "x localizer", "B-x localizer"], matrices):
        is_pd, minimum_pivot = positive_definite(matrix)
        check(f"moment admissibility {label}", is_pd, f"minimum LDL pivot {float(minimum_pivot):.3e}")

    chebyshev = chebyshev_monomials(degree)
    ez = [
        sum(comb(k, j) * F(2, B) ** j * (-1) ** (k - j) * moments[j] for j in range(k + 1))
        for k in range(degree + 1)
    ]
    expected_chebyshev = [
        sum(chebyshev[j][k] * ez[k] for k in range(len(chebyshev[j])))
        for j in range(degree + 1)
    ]
    q = sum(rational[j] * expected_chebyshev[j] for j in range(degree + 1))
    p0 = 1 - nfreq * q
    check("claimed marginal tail bound", q == F(cert["claimed"]["q1_star"]), f"q={float(q):.15f}")
    check("claimed acceptance floor", p0 == F(cert["claimed"]["p0"]), f"p0={float(p0):.15f}")

    z_coefficients = [F(0)] * (degree + 1)
    for j in range(degree + 1):
        for i, coefficient in enumerate(chebyshev[j]):
            z_coefficients[i] += rational[j] * coefficient
    positive, root_count = strictly_positive_on_interval(z_coefficients, F(-1), F(1))
    check("Q>0 exact Sturm certificate on [-1,1]", positive, f"roots={root_count}")
    z_minus_one = z_coefficients[:]
    z_minus_one[0] -= 1
    threshold_z = F(2, B) * tail_start_sq - 1
    positive, root_count = strictly_positive_on_interval(z_minus_one, threshold_z, F(1))
    check("Q-1>0 exact Sturm certificate on the tail", positive, f"roots={root_count}")

    if check_bernstein:
        x_coefficients = [F(0)] * (degree + 1)
        for j in range(degree + 1):
            for k, coefficient in enumerate(chebyshev[j]):
                for i in range(k + 1):
                    x_coefficients[i] += (
                        rational[j] * coefficient * comb(k, i) * F(2, B) ** i * (-1) ** (k - i)
                    )
        subdivisions = cert["bernstein_certificate"]["num_subintervals"]
        gap0 = interval_certificate_min(x_coefficients, F(0), F(B), subdivisions, degree)
        minus_one = x_coefficients[:]
        minus_one[0] -= 1
        gap1 = interval_certificate_min(minus_one, tail_start_sq, F(B), subdivisions, degree)
        check("Q>=0 exact Bernstein certificate", gap0 >= 0, f"minimum={float(gap0):.3e}")
        check("Q-1>=0 exact Bernstein certificate", gap1 >= 0, f"minimum={float(gap1):.3e}")
        check("claimed Bernstein minimum for Q",
              gap0 == F(cert["claimed"]["minimum_bernstein_Q"]))

    cutoff = F(1, nfreq) * (1 - F(2**security_bits, nraw))
    check("q below exact support-floor cutoff", q < cutoff, f"{float(q):.9f} < {float(cutoff):.9f}")
    check("accepted support is at least 2^128", F(nraw) * p0 >= 2**security_bits)
    accepted_bits = math.log2(nraw) + math.log2(p0.numerator) - math.log2(p0.denominator)
    print(f"RESULT: {'ALL CHECKS PASS' if ok else 'FAILURE'}")
    print(f"Pr[Gamma(c)<{true_norm_upper}] >= {float(p0):.12f}")
    print(f"log2(N_raw*p0) = {accepted_bits:.12f}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
