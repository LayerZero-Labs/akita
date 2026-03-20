#!/usr/bin/env python3
"""Compare direct, legacy, and native Hachi->Labrador handoff frontiers.

This script mirrors the size formulas used by the Rust proof objects:

- `PackedDigits`: `8 + 1 + ceil(num_elems * bits_per_elem / 8)`
- `FlatRingVec`: `4 + 8 + coeff_count * field_bytes`
- `PackedCoeffRow`: `4 + 8 + 1 + ceil(coeff_count * coeff_bits / 8)`
- `FlatLabradorWitness`: `4 + sum(packed_coeff_row_bytes(row_len, coeff_bits))`

It is intentionally a frontier-accounting tool, not a full recursive Labrador
planner. The runtime Rust prover now performs the actual `direct` vs `legacy`
vs `native` proof-estimate comparison before selecting a handoff path.

Example:
  python scripts/estimate_hachi_native_handoff.py \
    --ring-dim 64 \
    --field-bytes 16 \
    --witness-log-bits 4 \
    --current-w-digits 143680 \
    --n-a 1 \
    --n-d 1 \
    --num-blocks 528 \
    --block-len 1024 \
    --num-digits-open 1 \
    --num-digits-commit 1 \
    --num-digits-fold 1
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


U128_BYTES = 16


def packed_digits_bytes(num_elems: int, bits_per_elem: int) -> int:
    return 8 + 1 + math.ceil(num_elems * bits_per_elem / 8)


def flat_ring_vec_bytes(ring_len: int, ring_dim: int, field_bytes: int) -> int:
    coeff_count = ring_len * ring_dim
    return 4 + 8 + coeff_count * field_bytes


def packed_coeff_row_bytes(ring_len: int, ring_dim: int, coeff_bits: int) -> int:
    coeff_count = ring_len * ring_dim
    return 4 + 8 + 1 + math.ceil(coeff_count * coeff_bits / 8)


def packed_labrador_witness_bytes(
    row_lengths: list[int],
    row_bits: list[int],
    ring_dim: int,
) -> int:
    if len(row_lengths) != len(row_bits):
        raise ValueError("row_lengths and row_bits must have the same length")
    return 4 + sum(
        packed_coeff_row_bytes(row_len, ring_dim, coeff_bits)
        for row_len, coeff_bits in zip(row_lengths, row_bits)
    )


@dataclass
class FrontierReport:
    name: str
    witness_rows: list[int]
    witness_row_bits: list[int]
    witness_bytes: int
    public_bytes: int
    total_frontier_bytes: int


def build_reports(args: argparse.Namespace) -> tuple[int, FrontierReport, FrontierReport]:
    if args.current_w_digits % args.ring_dim != 0:
        raise ValueError("--current-w-digits must be divisible by --ring-dim")

    current_w_ring_len = args.current_w_digits // args.ring_dim
    t_hat_len = args.num_blocks * args.n_a * args.num_digits_open
    z_pre_len = args.block_len * args.num_digits_commit * args.num_digits_fold
    w_hat_len = args.num_blocks * args.num_digits_open

    direct_bytes = packed_digits_bytes(args.current_w_digits, args.witness_log_bits)

    legacy_rows = [w_hat_len, t_hat_len, z_pre_len]
    legacy_row_bits = [args.witness_log_bits] * len(legacy_rows)
    legacy_witness_bytes = packed_labrador_witness_bytes(
        legacy_rows,
        legacy_row_bits,
        args.ring_dim,
    )
    # The public tail payload matches `LabradorTail`: `v` has `n_d` ring
    # elements, `y_ring` is a single ring element, and the norm bound is u128.
    legacy_public_bytes = (
        1
        + flat_ring_vec_bytes(args.n_d, args.ring_dim, args.field_bytes)
        + flat_ring_vec_bytes(1, args.ring_dim, args.field_bytes)
        + U128_BYTES
    )
    legacy = FrontierReport(
        name="legacy_quad_eq",
        witness_rows=legacy_rows,
        witness_row_bits=legacy_row_bits,
        witness_bytes=legacy_witness_bytes,
        public_bytes=legacy_public_bytes,
        total_frontier_bytes=legacy_witness_bytes + legacy_public_bytes,
    )

    native_rows = [current_w_ring_len, t_hat_len] + list(args.native_extra_row_len)
    if args.native_extra_row_bits and len(args.native_extra_row_bits) != len(args.native_extra_row_len):
        raise ValueError("--native-extra-row-bits must match --native-extra-row-len arity")
    native_row_bits = [args.witness_log_bits, args.witness_log_bits] + (
        list(args.native_extra_row_bits)
        if args.native_extra_row_bits
        else [args.witness_log_bits] * len(args.native_extra_row_len)
    )
    native_witness_bytes = packed_labrador_witness_bytes(
        native_rows,
        native_row_bits,
        args.ring_dim,
    )
    native_public_bytes = (
        1
        + flat_ring_vec_bytes(0, args.ring_dim, args.field_bytes)
        + flat_ring_vec_bytes(1, args.ring_dim, args.field_bytes)
        + U128_BYTES
    )
    native = FrontierReport(
        name="native_opening",
        witness_rows=native_rows,
        witness_row_bits=native_row_bits,
        witness_bytes=native_witness_bytes,
        public_bytes=native_public_bytes,
        total_frontier_bytes=native_witness_bytes + native_public_bytes,
    )

    return direct_bytes, legacy, native


def print_report(args: argparse.Namespace, direct_bytes: int, legacy: FrontierReport, native: FrontierReport) -> None:
    if args.json:
        payload = {
            "direct_tail_bytes": direct_bytes,
            "legacy": asdict(legacy),
            "native": asdict(native),
            "legacy_vs_native_witness_delta": legacy.witness_bytes - native.witness_bytes,
            "legacy_vs_native_total_delta": legacy.total_frontier_bytes - native.total_frontier_bytes,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("## Hachi Native Handoff Estimate")
    print()
    print(f"ring_dim: {args.ring_dim}")
    print(f"field_bytes: {args.field_bytes}")
    print(f"current_w_digits: {args.current_w_digits}")
    print(f"current_w_ring_len: {args.current_w_digits // args.ring_dim}")
    print(f"witness_log_bits: {args.witness_log_bits}")
    print()
    print(f"direct_tail_bytes: {direct_bytes}")
    print()
    for report in (legacy, native):
        print(f"### {report.name}")
        print(f"witness_rows: {report.witness_rows}")
        print(f"witness_row_bits: {report.witness_row_bits}")
        print(f"witness_bytes: {report.witness_bytes}")
        print(f"public_bytes: {report.public_bytes}")
        print(f"total_frontier_bytes: {report.total_frontier_bytes}")
        print()
    print(f"legacy_minus_native_witness_bytes: {legacy.witness_bytes - native.witness_bytes}")
    print(f"legacy_minus_native_total_bytes: {legacy.total_frontier_bytes - native.total_frontier_bytes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-dim", type=int, required=True)
    parser.add_argument("--field-bytes", type=int, default=16)
    parser.add_argument("--witness-log-bits", type=int, required=True)
    parser.add_argument("--current-w-digits", type=int, required=True)
    parser.add_argument("--n-a", type=int, required=True)
    parser.add_argument("--n-d", type=int, required=True)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--block-len", type=int, required=True)
    parser.add_argument("--num-digits-open", type=int, required=True)
    parser.add_argument("--num-digits-commit", type=int, default=1)
    parser.add_argument("--num-digits-fold", type=int, required=True)
    parser.add_argument(
        "--native-extra-row-len",
        type=int,
        action="append",
        default=[],
        help="Optional extra native helper row lengths in ring elements.",
    )
    parser.add_argument(
        "--native-extra-row-bits",
        type=int,
        action="append",
        default=[],
        help="Optional per-helper-row coefficient bit widths; defaults to --witness-log-bits.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_bytes, legacy, native = build_reports(args)
    print_report(args, direct_bytes, legacy, native)


if __name__ == "__main__":
    main()
