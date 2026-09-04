#!/usr/bin/env python3
import json
from fractions import Fraction as F
from math import comb, log2
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

D = 128
W = 40
B = W * W
GAMMA = 15
N = 30


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


def chebyshev_values(points, n):
    values = np.zeros((len(points), n + 1))
    values[:, 0] = 1
    values[:, 1] = points
    for degree in range(2, n + 1):
        values[:, degree] = 2 * points * values[:, degree - 1] - values[:, degree - 2]
    return values


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


def to_x_monomials(rational_coefficients, chebyshev):
    coefficients = [F(0)] * (N + 1)
    for j in range(N + 1):
        for k, coefficient in enumerate(chebyshev[j]):
            for i in range(k + 1):
                coefficients[i] += (
                    rational_coefficients[j]
                    * coefficient
                    * comb(k, i)
                    * F(2, B) ** i
                    * (-1) ** (k - i)
                )
    return coefficients


def certificate_gap(coefficients, lo, hi, subdivisions):
    return min(
        bernstein_min(
            coefficients,
            lo + (hi - lo) * F(i, subdivisions),
            lo + (hi - lo) * F(i + 1, subdivisions),
            N,
        )
        for i in range(subdivisions)
    )


def main():
    moments = {
        int(k): F(v)
        for k, v in json.loads(Path("moments_d128_w40.json").read_text()).items()
    }
    chebyshev = chebyshev_monomials(N)
    ez = [
        sum(comb(k, j) * F(2, B) ** j * (-1) ** (k - j) * moments[j] for j in range(k + 1))
        for k in range(N + 1)
    ]
    objective_exact = [
        sum(chebyshev[j][k] * ez[k] for k in range(len(chebyshev[j])))
        for j in range(N + 1)
    ]
    objective = np.array([float(value) for value in objective_exact])

    threshold = 2 * GAMMA * GAMMA / B - 1
    nonnegative_points = set(np.linspace(-1, 1, 600))
    tail_points = set(np.linspace(threshold, 1, 600))
    result = None
    for iteration in range(100):
        points0 = np.array(sorted(nonnegative_points))
        points1 = np.array(sorted(tail_points))
        matrix0 = chebyshev_values(points0, N)
        matrix1 = chebyshev_values(points1, N)
        result = linprog(
            objective,
            A_ub=np.vstack([-matrix0, -matrix1]),
            b_ub=np.concatenate([np.zeros(len(points0)), -np.ones(len(points1))]),
            bounds=[(-100.0, 100.0)] * (N + 1),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"iteration {iteration}: {result.message}")
        derivative_roots = np.polynomial.Chebyshev(result.x).deriv().roots()
        real_roots = [float(root.real) for root in derivative_roots if abs(root.imag) < 1e-8]
        candidates0 = [-1.0, 1.0] + [root for root in real_roots if -1 < root < 1]
        candidates1 = [threshold, 1.0] + [root for root in real_roots if threshold < root < 1]
        values0 = np.polynomial.chebyshev.chebval(candidates0, result.x)
        values1 = np.polynomial.chebyshev.chebval(candidates1, result.x) - 1
        min0 = float(np.min(values0))
        min1 = float(np.min(values1))
        print(f"iteration {iteration + 1}: q={result.fun:.12f}, minima={min0:.3e},{min1:.3e}")
        # The exact Bernstein replay below shifts either residual upward. A
        # looser numerical exit avoids HiGHS cycling near a degenerate optimum.
        if min0 >= -1e-4 and min1 >= -1e-4:
            print(f"exchange iterations: {iteration + 1}; minima {min0:.3e}, {min1:.3e}")
            break
        nonnegative_points.update(round(value, 12) for value in candidates0)
        tail_points.update(round(value, 12) for value in candidates1)
    else:
        raise RuntimeError("exchange algorithm did not converge")
    print(f"numerical optimum: {result.fun:.18f}")

    denominator = 2 ** 60
    rational = [F(round(value * denominator), denominator) for value in result.x]
    x_coefficients = to_x_monomials(rational, chebyshev)
    subdivisions = 4096
    gap0 = certificate_gap(x_coefficients, F(0), F(B), subdivisions)
    minus_one = x_coefficients[:]
    minus_one[0] -= 1
    gap1 = certificate_gap(minus_one, F(GAMMA * GAMMA), F(B), subdivisions)
    print(f"pre-shift Bernstein gaps: {float(gap0):.12e}, {float(gap1):.12e}")

    shift = max(F(0), -gap0, -gap1) + F(1, 10**10)
    rational[0] += shift
    x_coefficients[0] += shift
    minus_one[0] += shift
    gap0 = certificate_gap(x_coefficients, F(0), F(B), subdivisions)
    gap1 = certificate_gap(minus_one, F(GAMMA * GAMMA), F(B), subdivisions)
    assert gap0 > 0 and gap1 > 0

    q = sum(rational[j] * objective_exact[j] for j in range(N + 1))
    nraw = comb(D, W) * 2 ** W
    p0 = 1 - (D // 2) * q
    cutoff = F(1, D // 2) * (1 - F(2**145, nraw))
    print(f"shift: {float(shift):.12e}")
    print(f"post-shift Bernstein gaps: {float(gap0):.12e}, {float(gap1):.12e}")
    print(f"q: {float(q):.18f}")
    print(f"cutoff: {float(cutoff):.18f}")
    print(f"p0: {float(p0):.18f}")
    print(f"accepted bits: {log2(nraw) + log2(float(p0)):.12f}")

    Path("dual_coefficients.json").write_text(
        json.dumps({str(i): f"{v.numerator}/{v.denominator}" for i, v in enumerate(rational)}, indent=2) + "\n"
    )
    Path("dual_summary.json").write_text(
        json.dumps(
            {
                "degree": N,
                "subdivisions": subdivisions,
                "q": f"{q.numerator}/{q.denominator}",
                "p0": f"{p0.numerator}/{p0.denominator}",
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
