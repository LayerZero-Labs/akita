#!/usr/bin/env python3
"""Exact, floating-point-free checker for an operator-norm acceptance certificate.

Validates a certificate of the form produced for Appendix C of the Akita paper
(sec:opnorm-moment-method / sec:opnorm-worked): a lower bound on an exact
true-operator-norm subset that is contained in the runtime fixed-point
predicate.
for a d=64 exact-shell challenge family, via exact spectral moments, a
dual-polynomial marginal tail bound, and a union bound over the d/2 distinct
frequencies.

Every load-bearing check uses exact rational arithmetic (fractions.Fraction);
floating point appears only in the final log2 comparison, which clears its
threshold by a wide margin, and in human-readable printouts.

Usage:
    python3 check_cert.py [cert.json]
Exit code 0 iff all checks pass.
"""
import json
import math
import sys
from fractions import Fraction as F
from math import comb
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def frac(s):
    return F(s)


def load_residues(path):
    return {
        int(line.split()[0]): int(line.split()[1])
        for line in path.read_text().splitlines()[1:]
    }


def cheb_monomial_coeffs(n):
    """Integer monomial coefficients of T_0..T_n (Chebyshev, first kind)."""
    T = [[F(1)], [F(0), F(1)]]
    for k in range(2, n + 1):
        prev, prev2 = T[k - 1], T[k - 2]
        cur = [F(0)] * (k + 1)
        for i, c in enumerate(prev):
            cur[i + 1] += 2 * c
        for i, c in enumerate(prev2):
            cur[i] -= c
        T.append(cur)
    return T


def is_pos_def(mat):
    """Exact symmetric LDL; returns (all_pivots_positive, min_pivot)."""
    n = len(mat)
    A = [row[:] for row in mat]
    pivots = []
    for k in range(n):
        d = A[k][k]
        pivots.append(d)
        if d <= 0:
            return False, d
        for i in range(k + 1, n):
            f = A[i][k] / d
            for j in range(k, n):
                A[i][j] -= f * A[k][j]
    return True, min(pivots)


def bernstein_min_on(coeffs_x, lo, hi, n):
    """Minimum degree-n Bernstein coefficient of poly (x-monomial coeffs) on [lo,hi]."""
    L = hi - lo
    # reparameterize x = lo + L*t, get t-monomial coeffs d_i
    d = [F(0)] * (n + 1)
    for i in range(n + 1):
        ci = coeffs_x[i]
        if ci == 0:
            continue
        loi = [F(1)]
        # binomial expansion (lo + L t)^i
        for s in range(i + 1):
            d[s] += ci * comb(i, s) * (lo ** (i - s)) * (L ** s)
    mn = None
    for k in range(n + 1):
        bk = sum(comb(k, i) * d[i] / comb(n, i) for i in range(k + 1))
        if mn is None or bk < mn:
            mn = bk
    return mn


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        __file__.rsplit("/", 1)[0] + "/cert_d64_a31_b11_gamma18.json")
    cert = load(path)
    P = cert["params"]
    d = P["d"]
    a = P["a_mag1"]
    b = P["b_mag2"]
    w = P["w_nonzero"]
    B = P["support_cap_B"]
    nominal_gamma = P["nominal_Gamma"]
    lam = P["lambda_bits"]
    nfreq = P["num_distinct_freqs"]

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")

    print(f"Checking {path}")
    print(
        f"family d={d}, (a,b)=({a},{b}), B={B}, "
        f"runtime threshold={nominal_gamma}, lambda={lam}, freqs={nfreq}"
    )

    # ---- structural params ----
    check("production dimensions",
          d == 64 and a == 31 and b == 11 and w == 42)
    check("l1_norm = a + 2b", P["l1_norm"] == a + 2 * b, f"{P['l1_norm']}")
    check("support cap B = l1^2", B == P["l1_norm"] ** 2)
    Nraw = comb(d, w) * comb(w, b) * (2 ** w)
    check("N_raw = C(d,w) C(w,b) 2^w", str(Nraw) == cert["N_raw"], cert["N_raw"])

    # ---- fixed-point containment ----
    # If each tabulated real/imaginary root coordinate has error at most eps,
    # every frequency accumulator has coordinate error at most r=l1*eps. The
    # squared upper enclosure adds at most 2*sqrt(2)*r before squaring. Choose
    # the least integer h strictly above that irrational quantity. Then every
    # true norm <= runtime_threshold-h/2^q is accepted by the strict runtime
    # predicate at runtime_threshold.
    containment = cert["fixed_point_containment"]
    q = containment["fractional_bits"]
    eps = containment["root_coordinate_error_units"]
    runtime_threshold = containment["runtime_strict_threshold"]
    r_error = P["l1_norm"] * eps
    squared_error = 8 * r_error * r_error
    margin = math.isqrt(squared_error)
    if margin * margin <= squared_error:
        margin += 1
    scale = 1 << q
    true_norm_upper = F(runtime_threshold * scale - margin, scale)
    tail_start_sq = true_norm_upper * true_norm_upper
    check("runtime table contract uses q=48 and root error at most 4",
          q == 48 and eps == 4)
    check("runtime threshold matches the nominal dual threshold",
          runtime_threshold == nominal_gamma)
    check("fixed-point margin is the least strict integer upper bound",
          containment["rounding_margin_units"] == margin
          and margin * margin > squared_error
          and (margin - 1) * (margin - 1) <= squared_error,
          f"h={margin}, 8r^2={squared_error}")
    check("certified true-norm upper bound matches fixed-point containment",
          frac(containment["certified_true_norm_upper_bound"]) == true_norm_upper,
          str(true_norm_upper))
    check("certified spectral tail start is the square of that bound",
          frac(containment["certified_tail_start_sq"]) == tail_start_sq,
          str(tail_start_sq))

    # ---- moments ----
    Mr = cert["moments_M0_to_M30"]
    N = max(int(k) for k in Mr)
    M = {int(k): frac(v) for k, v in Mr.items()}
    check("M0 = 1 and M1 = l2_sq (spectral mean)",
          M[0] == 1 and M[1] == P["l2_sq"], f"M1={M[1]}")
    fourth_power_sum = a + 16 * b
    expected_M2 = (2 * P["l2_sq"] ** 2 - 2 * fourth_power_sum
                   + F(d * fourth_power_sum - P["l2_sq"] ** 2, d - 1))
    check("M2 matches independent mixed-shell closed form",
          M[2] == expected_M2, f"M2={M[2]}")

    # Bind the supplied moment vector to the exact mixed-shell generator. Eight
    # primes uniquely reconstruct every nonnegative numerator under the stated
    # family bound; a ninth prime is reserved for an independent residue check.
    generation = cert["moment_generation"]
    reconstruction_primes = generation["reconstruction_primes"]
    check_prime = generation["independent_check_prime"]
    support_count = comb(d, w) * comb(w, b)
    modulus_product = math.prod(reconstruction_primes)
    check("CRT modulus exceeds the degree-30 a priori numerator bound",
          modulus_product > support_count * (B ** N),
          f"{modulus_product.bit_length()} > {(support_count * B ** N).bit_length()} bits")
    residue_ok = True
    certificate_directory = Path(path).resolve().parent
    for prime in reconstruction_primes + [check_prime]:
        residue_path = certificate_directory / f"moments_{prime}.txt"
        if not residue_path.exists():
            residue_ok = False
            break
        residues = load_residues(residue_path)
        for moment in range(N + 1):
            numerator_sum = M[moment] * support_count
            if (numerator_sum.denominator != 1 or
                    numerator_sum.numerator % prime != residues.get(moment)):
                residue_ok = False
                break
    check("all moments match eight CRT generator outputs and the ninth check prime",
          residue_ok)

    # moment admissibility on [0,B]: Hankel + localizers PSD
    half = N // 2
    H = [[M[i + j] for j in range(half + 1)] for i in range(half + 1)]
    Hx = [[M[i + j + 1] for j in range(half)] for i in range(half)]
    HB = [[B * M[i + j] - M[i + j + 1] for j in range(half)] for i in range(half)]
    for nm, mat in [("Hankel [M_{i+j}] PSD", H),
                    ("x>=0 localizer PSD", Hx),
                    ("B-x>=0 localizer PSD", HB)]:
        pd, mn = is_pos_def(mat)
        check(f"moment admissibility: {nm}", pd, f"min pivot ~ {float(mn):.3e}")

    # ---- dual polynomial ----
    R = cert["dual_poly"]["coeffs_r0_to_r30"]
    r = {int(k): frac(v) for k, v in R.items()}
    Tc = cheb_monomial_coeffs(N)

    # E[Z^k], Z = 2X/B - 1
    EZ = [sum(comb(k, j) * F(2, B) ** j * F((-1) ** (k - j)) * M[j] for j in range(k + 1))
          for k in range(N + 1)]
    ETj = [sum(Tc[j][k] * EZ[k] for k in range(len(Tc[j]))) for j in range(N + 1)]
    q1 = sum(r[j] * ETj[j] for j in range(N + 1))
    p0 = 1 - nfreq * q1

    check("q1_star matches claimed (exact)", q1 == frac(cert["claimed"]["q1_star"]),
          f"{float(q1):.18f}")
    check("p0 matches claimed (exact)", p0 == frac(cert["claimed"]["p0"]),
          f"{float(p0):.18f}")
    cutoff = F(1, nfreq) * (1 - F(2 ** lam, Nraw))
    check("q1_star < per-frequency floor cutoff", q1 < cutoff,
          f"q1={float(q1):.6f} < cutoff={float(cutoff):.6f}")

    # ---- Bernstein nonnegativity (exact) ----
    # Q in x-monomial basis
    xc = [F(0)] * (N + 1)
    for j in range(N + 1):
        for k in range(len(Tc[j])):
            ck = Tc[j][k]
            if ck == 0:
                continue
            for i in range(k + 1):
                xc[i] += r[j] * ck * comb(k, i) * F(2, B) ** i * F((-1) ** (k - i))

    bc = cert["bernstein_certificate"]
    nsub = bc["num_subintervals"]

    lo0, hi0 = map(frac, bc["nonneg_interval_for_Q"])
    g0 = None
    for i in range(nsub):
        lo = (hi0 - lo0) * F(i, nsub) + lo0
        hi = (hi0 - lo0) * F(i + 1, nsub) + lo0
        mn = bernstein_min_on(xc, lo, hi, N)
        g0 = mn if g0 is None or mn < g0 else g0
    check(f"Q >= 0 on {bc['nonneg_interval_for_Q']} ({nsub} subintervals)",
          g0 >= 0, f"min Bernstein coeff = {float(g0):.6e}")

    lo1, hi1 = map(frac, bc["ge1_interval_for_Q_minus_1"])
    check("Q-1 tail interval starts at the certified fixed-point subset",
          lo1 == tail_start_sq)
    xc1 = xc[:]
    xc1[0] -= 1
    g1 = None
    for i in range(nsub):
        lo = (hi1 - lo1) * F(i, nsub) + lo1
        hi = (hi1 - lo1) * F(i + 1, nsub) + lo1
        mn = bernstein_min_on(xc1, lo, hi, N)
        g1 = mn if g1 is None or mn < g1 else g1
    check(f"Q-1 >= 0 on {bc['ge1_interval_for_Q_minus_1']} ({nsub} subintervals)",
          g1 >= 0, f"min Bernstein coeff = {float(g1):.6e}")

    # ---- support floor ----
    bits = math.log2(Nraw) + (math.log2(p0.numerator) - math.log2(p0.denominator))
    check(f"log2(N_raw * p0) >= {lam}", bits >= lam, f"{bits:.6f}")

    print()
    print(f"RESULT: {'ALL CHECKS PASS' if ok else 'FAILURE'}")
    print(f"  Pr[Gamma(c) < {true_norm_upper}] >= p0 = {float(p0):.10f}")
    print(f"  log2(N_raw * p0) = {bits:.6f}  (>= {lam} ? {'yes' if bits >= lam else 'no'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
