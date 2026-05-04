#!/usr/bin/env python3
"""
Regenerate the SIS max-width table used by the Akita planner.

This script binary-searches for the maximum SIS width (in ring elements) that
provides >= 128-bit security for each (d, collision_inf, rank) triple. The
output is the Rust match arm body for `sis_max_widths` in
`src/planner/sis_security.rs`.

Requires SageMath and a checkout of the lattice-estimator repo
(https://github.com/malb/lattice-estimator).

Usage:
    sage -python scripts/gen_sis_table.py

Options:
    --estimator-path PATH   Path to lattice-estimator repo.
                            Falls back to LATTICE_ESTIMATOR_PATH env var,
                            then ../lattice-estimator (sibling checkout).
    --search-cap N          Override the per-D binary-search cap.
                            Defaults to 10^10 for D=32/64 and 5*10^10 for D=128.
    --target-bits BITS      Minimum security bits (default: 128).
    --d D                   Only run for ring dimension D (omit for all).
    --collision C           Only run for collision bound C (omit for all).

Modeling choices (matching the existing table):
    - Reduction model: BDGL16
    - Shape model: lgsa
    - Norm: Euclidean (l2)
    - Field modulus for estimation: q = 2^128 - 275
    - length_bound = sqrt(m) * collision_inf  (standard l2 conversion)

The runtime protocol prime may be a different 128-bit pseudo-Mersenne prime,
for example p = 2^128 - 2^32 + 22537. That distinction is immaterial for
these coarse SIS width estimates: lattice-estimator sees q through log(q), and
all supported 128-bit protocol primes differ from 2^128 by a tiny additive
offset. We therefore use one representative 128-bit q and state it explicitly.

The checked-in table uses a search cap of 10^10 for D=32/64 and 5*10^10 for
D=128. Entries that hit the cap are lower bounds, not tight cutoffs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

Q = (1 << 128) - 275
Q_LABEL = "2^128 - 275"

ALL_ENTRIES: list[tuple[int, int]] = [
    # D=32
    (32, 2), (32, 3), (32, 7), (32, 15), (32, 31),
    (32, 63), (32, 127), (32, 255), (32, 511), (32, 1023), (32, 2047),
    # D=64
    (64, 2), (64, 3), (64, 7), (64, 15), (64, 31),
    (64, 63), (64, 127), (64, 255), (64, 511), (64, 1023), (64, 2047),
    # D=128
    (128, 2), (128, 3), (128, 7), (128, 15), (128, 31), (128, 63),
]

MAX_RANK = 4
D128_SEARCH_CAP = 50_000_000_000
DEFAULT_SEARCH_CAP = 10_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the SIS max-width table for the Akita planner."
    )
    parser.add_argument("--estimator-path", help="Path to lattice-estimator repo.")
    parser.add_argument(
        "--search-cap",
        type=int,
        help=(
            "Override the binary-search cap for every D. By default, use "
            "10^10 for D=32/64 and 5*10^10 for D=128."
        ),
    )
    parser.add_argument(
        "--target-bits", type=float, default=128.0,
        help="Minimum security bits (default: 128)."
    )
    parser.add_argument("--d", type=int, help="Only run for this ring dimension.")
    parser.add_argument("--collision", type=int, help="Only run for this collision bound.")
    return parser.parse_args()


def default_search_cap(d: int) -> int:
    if d == 128:
        return D128_SEARCH_CAP
    return DEFAULT_SEARCH_CAP


def locate_estimator(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("LATTICE_ESTIMATOR_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    root = Path(__file__).resolve().parents[1]
    candidates.extend([
        root / "lattice-estimator",
        root.parent / "lattice-estimator",
    ])
    for c in candidates:
        if (c / "estimator" / "__init__.py").exists():
            return c.resolve()
    raise SystemExit(
        "Could not locate lattice-estimator. "
        "Set LATTICE_ESTIMATOR_PATH or pass --estimator-path."
    )


def load_estimator(path: Path):
    sys.path.insert(0, str(path))
    from estimator import SIS
    from estimator.reduction import RC
    from sage.all import log
    return SIS, RC, log


def estimate_bits(SIS, RC, log, d: int, rank: int, width: int, collision: int) -> float:
    n = rank * d
    m = width * d
    length_bound = (m ** 0.5) * collision
    out = SIS.lattice(
        SIS.Parameters(n=n, q=Q, m=m, length_bound=length_bound, norm=2, tag="sis_table"),
        red_cost_model=RC.BDGL16,
        red_shape_model="lgsa",
        log_level=0,
    )
    return float(log(out["rop"], 2))


def binary_search_max_width(
    SIS, RC, log,
    d: int, rank: int, collision: int,
    target_bits: float, search_cap: int,
) -> int:
    """Find the largest width in [1, search_cap] with security >= target_bits."""
    lo, hi = 1, search_cap

    if estimate_bits(SIS, RC, log, d, rank, 1, collision) < target_bits:
        return 0

    if estimate_bits(SIS, RC, log, d, rank, search_cap, collision) >= target_bits:
        return search_cap

    while lo < hi - 1:
        mid = (lo + hi) // 2
        bits = estimate_bits(SIS, RC, log, d, rank, mid, collision)
        if bits >= target_bits:
            lo = mid
        else:
            hi = mid

    return lo


def main() -> None:
    args = parse_args()
    estimator_path = locate_estimator(args.estimator_path)
    SIS, RC, log = load_estimator(estimator_path)

    entries = ALL_ENTRIES
    if args.d is not None:
        entries = [(d, c) for d, c in entries if d == args.d]
    if args.collision is not None:
        entries = [(d, c) for d, c in entries if c == args.collision]

    if not entries:
        print("No matching entries.", file=sys.stderr)
        return

    print(f"// Generated by: sage -python scripts/gen_sis_table.py")
    print(f"// Model: BDGL16 + lgsa, representative q = {Q_LABEL}")
    print(
        "// q note: exact 128-bit pseudo-Mersenne prime choice is negligible "
        "for these SIS estimates."
    )
    if args.search_cap is None:
        print(
            f"// Target: {args.target_bits} bits, search caps: "
            f"D=32/64 {DEFAULT_SEARCH_CAP:_}, D=128 {D128_SEARCH_CAP:_}"
        )
    else:
        print(f"// Target: {args.target_bits} bits, search cap override: {args.search_cap:_}")
    print(f"// Ranks: 1..={MAX_RANK}")
    print()

    current_d = None
    for d, collision in entries:
        if d != current_d:
            if current_d is not None:
                print()
            print(f"        // D={d}")
            current_d = d

        widths: list[int] = []
        search_cap = args.search_cap if args.search_cap is not None else default_search_cap(d)
        for rank in range(1, MAX_RANK + 1):
            t0 = time.time()
            w = binary_search_max_width(
                SIS, RC, log, d, rank, collision,
                args.target_bits, search_cap,
            )
            elapsed = time.time() - t0
            print(
                f"        // (d={d}, collision={collision}, rank={rank}): "
                f"max_width={w:_}, cap={search_cap:_} ({elapsed:.1f}s)",
                file=sys.stderr,
            )
            widths.append(w)

        ws = ", ".join(f"{w:_}" for w in widths)
        print(f"        ({d}, {collision}) => Some([{ws}]),")

    print()
    print("// Supported collision buckets per D:")
    for dim in sorted(set(d for d, _ in entries)):
        collisions = [c for d, c in entries if d == dim]
        cs = ", ".join(str(c) for c in collisions)
        print(f"//   D={dim}: &[{cs}]")


if __name__ == "__main__":
    main()
