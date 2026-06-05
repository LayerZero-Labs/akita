#!/usr/bin/env python3
"""
Regenerate the SIS max-width table used by the Akita planner.

This script binary-searches for the maximum SIS width (in ring elements) that
provides >= 128-bit security for each (d, collision_l2_sq, rank) triple. The
output is the Rust match arm body for `sis_max_widths` in
`crates/akita-types/src/sis/generated_sis_table.rs`.

Requires SageMath and a checkout of the lattice-estimator repo
(https://github.com/malb/lattice-estimator).

Usage:
    sage -python scripts/gen_sis_table.py

Options:
    --family FAMILY         Representative modulus family: q32, q64, or q128.
                            Defaults to q128.
    --q Q                   Explicit modulus integer. Overrides --family.
    --estimator-path PATH   Path to lattice-estimator repo.
                            Falls back to LATTICE_ESTIMATOR_PATH env var,
                            then ../lattice-estimator (sibling checkout).
    --search-cap N          Override the per-D binary-search cap.
                            Defaults to 10^10 for D=32/64 and 5*10^10 for D=128.
    --target-bits BITS      Minimum security bits (default: 128).
    --d D                   Only run for ring dimension D (omit for all).
    --collision C           Only run for collision bucket C (omit for all).

Key convention (L2 / operator-norm cutover):
    The bucket is the per-ring-row *squared* Euclidean collision bound
    `collision_l2_sq` (an exact integer), not an L-infinity coefficient bound.
    The SIS solution `z_A` spans `width` ring rows, each of squared Euclidean
    norm <= collision_l2_sq, so its whole-vector Euclidean length bound is
    `sqrt(width * collision_l2_sq)`. Buckets are exact powers of two (ratio 2 in
    the squared domain == sqrt(2) in the norm), so the Rust side rounds a raw
    key up via `next_power_of_two`.

    This reproduces the previous L-infinity table exactly for the B/D opening
    digits: their `collision_l2_sq = d * (2^lb - 1)^2` plugged into
    `sqrt(width * collision_l2_sq)` equals the old `sqrt(width*d) * (2^lb - 1)`.

Modeling choices (matching the existing table):
    - Reduction model: BDGL16
    - Shape model: lgsa
    - Norm: Euclidean (l2)
    - Field modulus for estimation: selected by --family / --q
    - length_bound = sqrt(width * collision_l2_sq)  (per-row squared key)

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
import signal
import sys
import time
from pathlib import Path

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# L2 collision buckets are exact powers of two in the *squared* per-row
# Euclidean domain: 2^MIN_LOG_BUCKET .. 2^MAX_LOG_BUCKET. The Rust side rounds a
# raw squared key up to the next power of two (`next_power_of_two`) and clamps to
# this range, so the two ladders must stay in lockstep with the
# `MIN_LOG_BUCKET` / `MAX_LOG_BUCKET` constants in `sis/ajtai_key.rs`.
MIN_LOG_BUCKET = 1
MAX_LOG_BUCKET = 84
SQUARED_BUCKETS = [1 << k for k in range(MIN_LOG_BUCKET, MAX_LOG_BUCKET + 1)]

FAMILIES: dict[str, tuple[int, str, list[int], list[int]]] = {
    "q32": (
        (1 << 32) - 99,
        "2^32 - 99",
        [32, 64, 128, 256],
        SQUARED_BUCKETS,
    ),
    "q64": (
        (1 << 64) - 59,
        "2^64 - 59",
        [32, 64, 128, 256],
        SQUARED_BUCKETS,
    ),
    "q128": (
        (1 << 128) - ((1 << 32) - 22537),
        "2^128 - (2^32 - 22537)",
        [32, 64, 128, 256],
        SQUARED_BUCKETS,
    ),
}

DEFAULT_RANK_SWEEP = 4
D128_SEARCH_CAP = 50_000_000_000
DEFAULT_SEARCH_CAP = 10_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the SIS max-width table for the Akita planner."
    )
    parser.add_argument(
        "--family",
        choices=sorted(FAMILIES),
        default="q128",
        help="Representative modulus family (default: q128).",
    )
    parser.add_argument(
        "--q",
        type=lambda x: int(x, 0),
        help="Explicit modulus integer. Overrides --family.",
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
    parser.add_argument(
        "--max-rank",
        type=int,
        default=DEFAULT_RANK_SWEEP,
        help=f"Maximum module rank to sweep (default: {DEFAULT_RANK_SWEEP}).",
    )
    parser.add_argument(
        "--format",
        choices=["rust", "csv"],
        default="rust",
        help="Output format for generated rows (default: rust).",
    )
    parser.add_argument("--d", type=int, help="Only run for this ring dimension.")
    parser.add_argument(
        "--dims",
        help="Comma-separated ring dimensions to run instead of the selected family list.",
    )
    parser.add_argument("--collision", type=int, help="Only run for this collision bound.")
    parser.add_argument(
        "--collisions",
        help="Comma-separated collision buckets to run instead of the selected family list.",
    )
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


def family_entries(family: str) -> list[tuple[int, int]]:
    _, _, dims, collisions = FAMILIES[family]
    return [(d, c) for d in dims for c in collisions]


def parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part, 0))
    if not values:
        raise SystemExit("empty comma-separated list")
    return values


# Per-call estimator wall-clock budget (seconds). Real `SIS.lattice` calls
# finish in well under a second; lattice-estimator nonetheless *diverges* on
# certain tiny degenerate instances (observed at d=32, rank=1, right at the
# `inf -> finite` security boundary, e.g. m=128, length_bound=256). A divergent
# call is treated as insecure (`-inf`), which is the conservative, security-safe
# choice: it can only *under*-report the secure width (forcing a slightly higher
# SIS rank), never over-report it, and the boundary answer is off by at most one
# width. The planner and the audit tests consume the same table, so the result
# stays self-consistent.
ESTIMATE_TIMEOUT_S = 5.0


class _EstimateTimeout(Exception):
    pass


def _on_timeout(_signum, _frame):
    raise _EstimateTimeout()


signal.signal(signal.SIGALRM, _on_timeout)


def estimate_bits(SIS, RC, log, q: int, d: int, rank: int, width: int, collision: int) -> float:
    # `collision` is the per-ring-row squared Euclidean bound (collision_l2_sq).
    # The SIS solution spans `width` ring rows, so the whole-vector Euclidean
    # length bound is sqrt(width * collision_l2_sq). The SIS matrix still has
    # m = width * d scalar columns and n = rank * d scalar rows.
    n = rank * d
    m = width * d
    length_bound = (width * collision) ** 0.5
    signal.setitimer(signal.ITIMER_REAL, ESTIMATE_TIMEOUT_S)
    try:
        out = SIS.lattice(
            SIS.Parameters(n=n, q=q, m=m, length_bound=length_bound, norm=2, tag="sis_table"),
            red_cost_model=RC.BDGL16,
            red_shape_model="lgsa",
            log_level=0,
        )
    except _EstimateTimeout:
        return float("-inf")
    except ValueError as exc:
        if "SIS trivially easy" in str(exc):
            return float("-inf")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return float(log(out["rop"], 2))


def binary_search_max_width(
    SIS, RC, log,
    q: int,
    d: int, rank: int, collision: int,
    target_bits: float, search_cap: int,
) -> int:
    """Find the largest width in [1, search_cap] with security >= target_bits.

    Security is monotone decreasing in width (more SIS columns => shorter
    solutions => fewer bits). Hybrid search, returning the same value as a plain
    binary search (same monotone predicate, only the probe order differs):

      1. width 1 insecure        -> 0.
      2. width search_cap secure -> search_cap (low-collision / near-cap answers
         resolve in a single huge-`m` probe).
      3. otherwise exponential-search from below (1, 2, 4, ...) to the first
         insecure power of two, then binary-search the last bracket. This keeps
         the common small-answer cells cheap (small `m = width*d`), instead of
         the cap-first binary search's ~log2(cap) huge-`m` probes per cell.
    """
    if estimate_bits(SIS, RC, log, q, d, rank, 1, collision) < target_bits:
        return 0

    if estimate_bits(SIS, RC, log, q, d, rank, search_cap, collision) >= target_bits:
        return search_cap

    hi = 1
    while True:
        nxt = min(hi * 2, search_cap)
        if estimate_bits(SIS, RC, log, q, d, rank, nxt, collision) >= target_bits:
            hi = nxt
            continue
        # Answer is in [hi, nxt): largest secure width strictly below `nxt`.
        lo, high = hi, nxt
        while lo < high - 1:
            mid = (lo + high) // 2
            if estimate_bits(SIS, RC, log, q, d, rank, mid, collision) >= target_bits:
                lo = mid
            else:
                high = mid
        return lo


def main() -> None:
    args = parse_args()
    if args.max_rank < 1:
        raise SystemExit("--max-rank must be at least 1")
    estimator_path = locate_estimator(args.estimator_path)
    SIS, RC, log = load_estimator(estimator_path)
    family_q, family_label, _, _ = FAMILIES[args.family]
    q = args.q if args.q is not None else family_q
    q_label = str(q) if args.q is not None else family_label

    custom_dims = parse_int_list(args.dims)
    custom_collisions = parse_int_list(args.collisions)
    if custom_dims is not None or custom_collisions is not None:
        _, _, family_dims, family_collisions = FAMILIES[args.family]
        dims = custom_dims if custom_dims is not None else family_dims
        collisions = custom_collisions if custom_collisions is not None else family_collisions
        entries = [(d, c) for d in dims for c in collisions]
    else:
        entries = family_entries(args.family)
    if args.d is not None:
        entries = [(d, c) for d, c in entries if d == args.d]
    if args.collision is not None:
        entries = [(d, c) for d, c in entries if c == args.collision]

    if not entries:
        print("No matching entries.", file=sys.stderr)
        return

    if args.format == "csv":
        print("q,d,collision,rank,max_width,target_bits,search_cap")
    else:
        print(f"// Generated by: sage -python scripts/gen_sis_table.py")
        print(f"// Family: {args.family.upper()}")
        print(f"// Model: BDGL16 + lgsa, representative q = {q_label}")
        print(f"// q = {q}")
        if args.search_cap is None:
            print(
                f"// Target: {args.target_bits} bits, search caps: "
                f"D=32/64 {DEFAULT_SEARCH_CAP:_}, D=128 {D128_SEARCH_CAP:_}"
            )
        else:
            print(f"// Target: {args.target_bits} bits, search cap override: {args.search_cap:_}")
        print(f"// Ranks: 1..={args.max_rank}")
        print()

    # Security is monotone in the bucket: once a (d) has no secure width at any
    # rank for some collision bucket, every larger bucket is also all-zero. We
    # truncate each (d) ladder at its first all-zero bucket: this keeps the table
    # compact (unreachable buckets become a `None` table miss, which the planner
    # already treats as "fold too large to price") and skips the dead searches.
    current_d = None
    dead_dims: set[int] = set()
    emitted: dict[int, list[int]] = {}
    for d, collision in entries:
        if d in dead_dims:
            continue

        widths: list[int] = []
        search_cap = args.search_cap if args.search_cap is not None else default_search_cap(d)
        for rank in range(1, args.max_rank + 1):
            t0 = time.time()
            w = binary_search_max_width(
                SIS, RC, log, q, d, rank, collision,
                args.target_bits, search_cap,
            )
            elapsed = time.time() - t0
            print(
                f"        // (d={d}, collision={collision}, rank={rank}): "
                f"max_width={w:_}, cap={search_cap:_} ({elapsed:.1f}s)",
                file=sys.stderr,
            )
            widths.append(w)
            if args.format == "csv":
                print(
                    f"{q},{d},{collision},{rank},{w},{args.target_bits},{search_cap}",
                    flush=True,
                )

        if all(w == 0 for w in widths):
            dead_dims.add(d)
            print(
                f"        // (d={d}): no secure width at collision_l2_sq>={collision}; "
                f"truncating ladder",
                file=sys.stderr,
            )
            continue

        emitted.setdefault(d, []).append(collision)
        if args.format == "rust":
            if d != current_d:
                if current_d is not None:
                    print()
                print(f"        // D={d}")
                current_d = d
            ws = ", ".join(f"{w:_}" for w in widths)
            print(f"        ({d}, {collision}) => Some(&[{ws}]),")

    if args.format == "rust":
        print()
        print("// Supported collision_l2_sq buckets per D:")
        for dim in sorted(emitted):
            cs = ", ".join(str(c) for c in emitted[dim])
            print(f"//   D={dim}: &[{cs}]")


if __name__ == "__main__":
    main()
