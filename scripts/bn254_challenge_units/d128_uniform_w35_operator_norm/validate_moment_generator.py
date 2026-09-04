#!/usr/bin/env python3
import itertools
import json
import subprocess
import tempfile
from fractions import Fraction as F
from math import comb
from pathlib import Path

PRIME = 2305843009213689601
PRIMITIVE = 11


def main():
    source_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="akita-opnorm-") as build_dir:
        executable = Path(build_dir) / "moments_mod"
        subprocess.run(
            [
                "clang++",
                "-O3",
                "-std=c++20",
                str(source_dir.parents[1] / "operator_norm" / "d128" / "moments_mod.cpp"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        generated_lines = subprocess.run(
            [str(executable), str(PRIME), str(PRIMITIVE), "4", "8", "3"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[1:]
    d, w, max_moment = 8, 3, 4
    generated = {int(line.split()[0]): int(line.split()[1]) for line in generated_lines}

    alpha = pow(PRIMITIVE, (PRIME - 1) // (2 * d), PRIME)
    alpha_inverse = pow(alpha, PRIME - 2, PRIME)
    inverse_sign_count = pow(2**w, PRIME - 2, PRIME)
    exhaustive = [0] * (max_moment + 1)
    for support in itertools.combinations(range(d), w):
        support_moments = [0] * (max_moment + 1)
        for signs in itertools.product((-1, 1), repeat=w):
            value = sum(sign * pow(alpha, position, PRIME) for sign, position in zip(signs, support)) % PRIME
            conjugate = sum(
                sign * pow(alpha_inverse, position, PRIME) for sign, position in zip(signs, support)
            ) % PRIME
            squared = value * conjugate % PRIME
            power = 1
            for m in range(max_moment + 1):
                support_moments[m] = (support_moments[m] + power) % PRIME
                power = power * squared % PRIME
        for m in range(max_moment + 1):
            exhaustive[m] = (exhaustive[m] + support_moments[m] * inverse_sign_count) % PRIME
    assert generated == {m: exhaustive[m] for m in range(max_moment + 1)}
    print("PASS: modular EGF generator matches exhaustive d=8,w=3 shell through moment 4")

    certificate = json.loads((source_dir / "cert_d128_uniform_w35_gamma14.json").read_text())
    moments = {
        int(k): F(v) for k, v in certificate["moments_M0_to_M30"].items()
    }
    expected_second = 2 * 35**2 - 2 * 35 + F(35 * (128 - 35), 128 - 1)
    assert moments[2] == expected_second
    print(f"PASS: D128 M2 matches independent closed form: {moments[2]}")


if __name__ == "__main__":
    main()
