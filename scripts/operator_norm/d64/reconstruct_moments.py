#!/usr/bin/env python3
"""Reconstruct exact D64 mixed-shell moments from modular generator outputs."""

import json
from fractions import Fraction
from math import comb
from pathlib import Path

from sympy.ntheory.modular import crt

D = 64
A = 31
B = 11
SUPPORT_CAP = (A + 2 * B) ** 2
RECONSTRUCTION_PRIMES = [
    2305843009213689601,
    2305843009213689089,
    2305843009213687297,
    2305843009213683713,
    2305843009213682689,
    2305843009213675777,
    2305843009213673729,
    2305843009213666049,
]
CHECK_PRIME = 2305843009213663489


def load_residues(prime):
    lines = Path(f"moments_{prime}.txt").read_text().splitlines()[1:]
    return {int(line.split()[0]): int(line.split()[1]) for line in lines}


def main():
    primes = RECONSTRUCTION_PRIMES + [CHECK_PRIME]
    residues = {prime: load_residues(prime) for prime in primes}
    support_count = comb(D, A + B) * comb(A + B, B)
    modulus_product = 1
    for prime in RECONSTRUCTION_PRIMES:
        modulus_product *= prime

    moments = {}
    for moment in range(31):
        value, modulus = crt(
            RECONSTRUCTION_PRIMES,
            [residues[prime][moment] for prime in RECONSTRUCTION_PRIMES],
            check=True,
        )
        value = int(value)
        assert int(modulus) == modulus_product
        assert value <= support_count * SUPPORT_CAP**moment
        assert value % CHECK_PRIME == residues[CHECK_PRIME][moment]
        exact = Fraction(value, support_count)
        moments[str(moment)] = f"{exact.numerator}/{exact.denominator}"

    Path("moments_d64_a31_b11.json").write_text(json.dumps(moments, indent=2) + "\n")
    print(f"eight-prime CRT bits: {modulus_product.bit_length()}")
    print(f"largest a priori bound bits: {(support_count * SUPPORT_CAP**30).bit_length()}")
    print("independent ninth-prime residue checks: PASS")


if __name__ == "__main__":
    main()
