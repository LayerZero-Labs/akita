#!/usr/bin/env python3
"""Reconstruct exact S96 spectral moments from modular residues."""

import json
from fractions import Fraction
from math import comb
from pathlib import Path

SELECTED_POSITIONS = 96
MAG1_COUNT = 35
MAG2_COUNT = 1
SUPPORT_CAP = 37**2
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


def load_residues(prime):
    lines = Path(f"moments_{prime}.txt").read_text().splitlines()[1:]
    return {int(line.split()[0]): int(line.split()[1]) for line in lines}


def crt_nonnegative(primes, residues):
    value = 0
    modulus = 1
    for prime, residue in zip(primes, residues):
        correction = (residue - value) * pow(modulus, -1, prime) % prime
        value += modulus * correction
        modulus *= prime
    return value, modulus


def main():
    residues = {prime: load_residues(prime) for prime in PRIMES}
    support_count = comb(SELECTED_POSITIONS, MAG1_COUNT + MAG2_COUNT) * comb(
        MAG1_COUNT + MAG2_COUNT, MAG2_COUNT
    )
    reconstruction_primes = PRIMES[:7]
    check_prime = PRIMES[7]
    moments = {}
    modulus_product = 1
    for prime in reconstruction_primes:
        modulus_product *= prime

    for moment in range(31):
        value, modulus = crt_nonnegative(
            reconstruction_primes,
            [residues[prime][moment] for prime in reconstruction_primes],
        )
        assert modulus == modulus_product
        assert value <= support_count * SUPPORT_CAP**moment
        assert value % check_prime == residues[check_prime][moment]
        exact_moment = Fraction(value, support_count)
        moments[str(moment)] = f"{exact_moment.numerator}/{exact_moment.denominator}"
        print(moment, moments[str(moment)])

    Path("moments_d128_s96_a35_b1.json").write_text(
        json.dumps(moments, indent=2) + "\n"
    )
    print(f"seven-prime CRT bits: {modulus_product.bit_length()}")
    print(
        "largest a priori bound bits: "
        f"{(support_count * SUPPORT_CAP**30).bit_length()}"
    )
    print("independent eighth-prime residue checks: PASS")


if __name__ == "__main__":
    main()
