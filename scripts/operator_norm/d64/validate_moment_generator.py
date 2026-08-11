#!/usr/bin/env python3
"""Compare the mixed-shell modular EGF generator with exhaustive enumeration."""

import itertools
import json
import subprocess
import tempfile
from fractions import Fraction as F
from pathlib import Path

PRIME = 2305843009213689601
PRIMITIVE = 11
SOURCE_DIRECTORY = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix="akita-d64-moments-") as temporary_directory:
        executable = str(Path(temporary_directory) / "moments_mod")
        subprocess.run(
            [
                "clang++",
                "-O3",
                "-march=native",
                "-std=c++20",
                str(SOURCE_DIRECTORY / "moments_mod.cpp"),
                "-o",
                executable,
            ],
            check=True,
        )
        dimension, mag1_count, mag2_count, max_moment = 8, 2, 1, 4
        generated_lines = subprocess.run(
            [
                executable,
                str(PRIME),
                str(PRIMITIVE),
                str(max_moment),
                str(dimension),
                str(mag1_count),
                str(mag2_count),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[1:]
    generated = {int(line.split()[0]): int(line.split()[1]) for line in generated_lines}

    root = pow(PRIMITIVE, (PRIME - 1) // (2 * dimension), PRIME)
    root_inverse = pow(root, PRIME - 2, PRIME)
    sign_count = 2 ** (mag1_count + mag2_count)
    inverse_sign_count = pow(sign_count, PRIME - 2, PRIME)
    exhaustive = [0] * (max_moment + 1)
    positions = range(dimension)
    for mag2_positions in itertools.combinations(positions, mag2_count):
        remaining = [position for position in positions if position not in mag2_positions]
        for mag1_positions in itertools.combinations(remaining, mag1_count):
            weighted_positions = [(position, 1) for position in mag1_positions]
            weighted_positions += [(position, 2) for position in mag2_positions]
            support_moments = [0] * (max_moment + 1)
            for signs in itertools.product((-1, 1), repeat=len(weighted_positions)):
                value = sum(
                    sign * magnitude * pow(root, position, PRIME)
                    for sign, (position, magnitude) in zip(signs, weighted_positions)
                ) % PRIME
                conjugate = sum(
                    sign * magnitude * pow(root_inverse, position, PRIME)
                    for sign, (position, magnitude) in zip(signs, weighted_positions)
                ) % PRIME
                squared = value * conjugate % PRIME
                power = 1
                for moment in range(max_moment + 1):
                    support_moments[moment] = (support_moments[moment] + power) % PRIME
                    power = power * squared % PRIME
            for moment in range(max_moment + 1):
                exhaustive[moment] = (
                    exhaustive[moment] + support_moments[moment] * inverse_sign_count
                ) % PRIME
    assert generated == {moment: exhaustive[moment] for moment in range(max_moment + 1)}
    print("PASS: modular EGF generator matches exhaustive d=8,(a,b)=(2,1) shell through moment 4")

    certificate = json.loads(
        (SOURCE_DIRECTORY / "cert_d64_a31_b11_gamma18.json").read_text()
    )
    moments = {int(k): F(v) for k, v in certificate["moments_M0_to_M30"].items()}
    spectral_mean = 31 + 4 * 11
    fourth_power_sum = 31 + 16 * 11
    expected_second = (
        2 * spectral_mean**2
        - 2 * fourth_power_sum
        + F(64 * fourth_power_sum - spectral_mean**2, 63)
    )
    assert moments[2] == expected_second
    print(f"PASS: D64 M2 matches independent closed form: {moments[2]}")


if __name__ == "__main__":
    main()
