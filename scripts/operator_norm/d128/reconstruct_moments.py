#!/usr/bin/env python3
import glob
import json
from fractions import Fraction
from math import comb
from pathlib import Path

from sympy.ntheory.modular import crt

D = 128
W = 31
B = W * W
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


def main():
    residues = {p: load_residues(p) for p in PRIMES}
    support_count = comb(D, W)
    moments = {}
    reconstruction_primes = PRIMES[:7]
    check_prime = PRIMES[7]
    modulus_product = 1
    for p in reconstruction_primes:
        modulus_product *= p

    for m in range(31):
        value, modulus = crt(
            reconstruction_primes,
            [residues[p][m] for p in reconstruction_primes],
            check=True,
        )
        value = int(value)
        assert int(modulus) == modulus_product
        assert value <= support_count * (B ** m), (m, value.bit_length())
        assert value % check_prime == residues[check_prime][m], m
        moment = Fraction(value, support_count)
        moments[str(m)] = f"{moment.numerator}/{moment.denominator}"
        print(m, moments[str(m)])

    Path("moments_d128_w31.json").write_text(json.dumps(moments, indent=2) + "\n")
    print(f"seven-prime CRT bits: {modulus_product.bit_length()}")
    print(f"largest a priori bound bits: {(support_count * B**30).bit_length()}")
    print(f"independent eighth-prime residue checks: PASS")


if __name__ == "__main__":
    main()
