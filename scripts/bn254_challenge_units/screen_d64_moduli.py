#!/usr/bin/env python3
"""Screen fully splitting D64 primes for short evaluation-kernel vectors.

This is an exploratory filter, not a lower-bound certificate. Every printed
vector norm is backed by an exactly checked modular relation. Failure to print
a vector below a target radius does not prove that no such vector exists.

The script requires fpylll, cysignals, and SymPy in the active Python
environment.
"""

from __future__ import annotations

import argparse

from fpylll import BKZ, LLL, IntegerMatrix
from sympy import isprime


DIMENSION = 64
ROOT_ORDER = 2 * DIMENSION


def largest_splitting_prime(bit_length: int) -> int:
    candidate = (1 << bit_length) - 1
    candidate -= (candidate - 1) % ROOT_ORDER
    while not isprime(candidate):
        candidate -= ROOT_ORDER
    assert candidate.bit_length() == bit_length
    return candidate


def primitive_root_of_unity(modulus: int) -> int:
    for base in range(2, 10_000):
        root = pow(base, (modulus - 1) // ROOT_ORDER, modulus)
        if pow(root, DIMENSION, modulus) == modulus - 1:
            return root
    raise RuntimeError("no projected primitive 128th root found")


def evaluation_kernel_basis(modulus: int, root: int) -> IntegerMatrix:
    basis = IntegerMatrix(DIMENSION, DIMENSION)
    basis[0, 0] = modulus
    power = root
    for row in range(1, DIMENSION):
        constant = (-power) % modulus
        if constant > modulus // 2:
            constant -= modulus
        basis[row, 0] = constant
        basis[row, row] = 1
        power = power * root % modulus
    return basis


def screened_norm(bit_length: int, block_size: int, loops: int) -> tuple[int, int]:
    modulus = largest_splitting_prime(bit_length)
    root = primitive_root_of_unity(modulus)
    basis = evaluation_kernel_basis(modulus, root)
    LLL.reduction(basis, delta=0.999)
    BKZ.reduction(
        basis,
        BKZ.Param(
            block_size=block_size,
            max_loops=loops,
            flags=BKZ.AUTO_ABORT,
        ),
    )
    vector = [int(basis[0, column]) for column in range(DIMENSION)]
    squared_norm = sum(coefficient * coefficient for coefficient in vector)
    assert (
        sum(
            coefficient * pow(root, exponent, modulus)
            for exponent, coefficient in enumerate(vector)
        )
        % modulus
        == 0
    )
    return modulus, squared_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, nargs="+")
    parser.add_argument("--start-bits", type=int, default=192)
    parser.add_argument("--end-bits", type=int, default=208)
    parser.add_argument("--block-size", type=int, default=38)
    parser.add_argument("--loops", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.bits is None and arguments.start_bits > arguments.end_bits:
        parser.error("--start-bits must not exceed --end-bits")

    print("bits,modulus,squared_norm")
    bit_lengths = arguments.bits or range(
        arguments.start_bits, arguments.end_bits + 1
    )
    for bit_length in bit_lengths:
        modulus, squared_norm = screened_norm(
            bit_length,
            block_size=arguments.block_size,
            loops=arguments.loops,
        )
        print(f"{bit_length},{modulus},{squared_norm}", flush=True)


if __name__ == "__main__":
    main()
