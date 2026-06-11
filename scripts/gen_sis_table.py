#!/usr/bin/env python3
"""
Regenerate the SIS max-width table used by the Akita planner.

This script binary-searches for the maximum SIS width (in ring elements) that
provides >= 128-bit security for each (d, collision_l2_sq, rank) triple. The
output is the Rust match arm body for `sis_max_widths` in the split
`crates/akita-types/src/sis/generated_sis_table/` modules.

Requires SageMath and the pinned lattice-estimator checkout under
`third_party/lattice-estimator` (or an explicit override).

Usage:
    sage -python scripts/gen_sis_table.py

Options:
    --family FAMILY         Representative modulus family: q32, q64, or q128.
                            Defaults to q128.
    --q Q                   Explicit modulus integer. Overrides --family.
    --estimator-path PATH   Path to lattice-estimator repo.
                            Falls back to LATTICE_ESTIMATOR_PATH env var,
                            then `third_party/lattice-estimator`.
    --search-cap N          Override the per-D binary-search cap.
                            Defaults to 10^10 for D=32/64 and 5*10^10 for D=128.
    --target-bits BITS      Minimum security bits (default: 128).
    --d D                   Only run for ring dimension D (omit for all).
    --collision C           Only run for collision bucket C (omit for all).
    --jobs N                Run N independent Sage subprocess shards (default: 1).

Collision bucket convention:
    Each bucket is the per-ring-row squared Euclidean collision bound
    `collision_l2_sq` (an exact integer). The SIS solution spans `width` ring
    rows, each of squared norm <= collision_l2_sq, so the whole-vector Euclidean
    length bound passed to lattice-estimator is `sqrt(width * collision_l2_sq)`.
    Buckets are exact powers of two; the Rust table rounds a raw key up via
    `next_power_of_two` and clamps to the same ladder as `MIN_LOG_BUCKET` /
    `MAX_LOG_BUCKET` in `sis/ajtai_key.rs`.

Modeling choices (matching the existing table):
    - Reduction model: BDGL16
    - Norm: Euclidean (l2); `red_shape_model` is ignored on this path
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
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# Squared collision buckets: 2^MIN_LOG_BUCKET .. 2^MAX_LOG_BUCKET. Keep this
# ladder in lockstep with `MIN_LOG_BUCKET` / `MAX_LOG_BUCKET` in `sis/ajtai_key.rs`.
MIN_LOG_BUCKET = 1
MAX_LOG_BUCKET = 84
SQUARED_BUCKETS = [1 << k for k in range(MIN_LOG_BUCKET, MAX_LOG_BUCKET + 1)]

# Coefficient-L∞ buckets for derived L2 keys K = d · B². Keep in lockstep with
# `COEFF_LINF_BUCKETS` in `crates/akita-types/src/sis/ajtai_key.rs`.
COEFF_LINF_BUCKETS = [
    2,
    3,
    7,
    15,
    31,
    63,
    127,
    255,
    511,
    1023,
    2047,
    4095,
    8191,
    16383,
    32767,
    65535,
    131071,
    262143,
    524287,
    1048575,
    2097151,
    4194303,
    8388607,
    16777215,
    33554431,
    67108863,
]

RING_DIMS = [32, 64, 128, 256]


def coeff_linf_bucket_sq(b: int) -> int:
    """Return `B²` for a coefficient-L∞ bucket `B` without float rounding."""
    if b <= 3:
        return b * b
    # Buckets after `3` are `2^k - 1`; square via bit shifts.
    k = (b + 1).bit_length() - 1
    return (1 << (2 * k)) - (1 << (k + 1)) + 1


def derived_l2_collision_keys() -> list[int]:
    keys: set[int] = set()
    for d in RING_DIMS:
        for b in COEFF_LINF_BUCKETS:
            keys.add(d * coeff_linf_bucket_sq(b))
    return sorted(keys)


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
DEFAULT_JOBS_CAP = 6
PINNED_LATTICE_ESTIMATOR_SHA = "27a581bb8e9d49f5e9e2db315bd48ac769d5f5f5"


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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Run independent Sage subprocess shards over (d, collision) work items. "
            f"Defaults to 1; capped at min({DEFAULT_JOBS_CAP}, num_cpus)."
        ),
    )
    return parser.parse_args()


def default_search_cap(d: int) -> int:
    if d == 128:
        return D128_SEARCH_CAP
    return DEFAULT_SEARCH_CAP


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def locate_estimator(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("LATTICE_ESTIMATOR_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    root = repo_root()
    candidates.extend([
        root / "third_party" / "lattice-estimator",
        root / "lattice-estimator",
        root.parent / "lattice-estimator",
    ])
    for c in candidates:
        if (c / "estimator" / "__init__.py").exists():
            return c.resolve()
    raise SystemExit(
        "Could not locate lattice-estimator. "
        "Initialize `third_party/lattice-estimator`, set LATTICE_ESTIMATOR_PATH, "
        "or pass --estimator-path."
    )


def estimator_git_sha(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def assert_pinned_estimator(path: Path) -> None:
    actual = estimator_git_sha(path)
    if actual != PINNED_LATTICE_ESTIMATOR_SHA:
        raise SystemExit(
            "lattice-estimator SHA mismatch: "
            f"expected {PINNED_LATTICE_ESTIMATOR_SHA}, got {actual} at {path}"
        )


def estimator_remote_url(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        return normalize_git_remote_url(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def normalize_git_remote_url(url: str) -> str:
    """Canonicalize common GitHub SSH remotes for reproducible metadata."""
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:")
    if url.startswith("ssh://git@github.com/"):
        return "https://github.com/" + url.removeprefix("ssh://git@github.com/")
    return url


def load_estimator(path: Path):
    sys.path.insert(0, str(path))
    from estimator import SIS
    from estimator.reduction import RC
    from sage.all import RealField, ZZ, log, oo
    return SIS, RC, log, oo, ZZ, RealField


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


def select_entries(args: argparse.Namespace) -> list[tuple[int, int]]:
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
    entries.sort()
    return entries


def sis_length_bound(ZZ, RealField, width: int, collision: int):
    """High-precision Euclidean SIS solution bound `sqrt(width * collision_l2_sq)`.

    Build the integer product exactly, then convert it to an upward-rounded Sage
    real with precision scaled to the product size. This avoids Python's f64
    `** 0.5` path while still giving lattice-estimator the numeric real it
    expects (rather than a symbolic `sqrt(...)` expression).
    """
    squared = ZZ(width) * ZZ(collision)
    precision = max(256, int(squared.nbits()) + 128)
    return RealField(precision, rnd="RNDU")(squared).sqrt()


def estimate_bits(
    SIS, RC, log, oo, ZZ, RealField,
    q: int, d: int, rank: int, width: int, collision: int,
) -> float:
    # `collision` is the per-ring-row squared Euclidean bound (collision_l2_sq).
    # The SIS solution spans `width` ring rows, so the whole-vector Euclidean
    # length bound is sqrt(width * collision_l2_sq). The SIS matrix still has
    # m = width * d scalar columns and n = rank * d scalar rows.
    n = rank * d
    m = width * d
    length_bound = sis_length_bound(ZZ, RealField, width, collision)
    try:
        out = SIS.lattice(
            SIS.Parameters(n=n, q=q, m=m, length_bound=length_bound, norm=2, tag="sis_table"),
            red_cost_model=RC.BDGL16,
            log_level=0,
        )
    except ValueError as exc:
        if "SIS trivially easy" in str(exc):
            return float("-inf")
        raise
    rop = out["rop"]
    if rop == oo:
        return float("inf")
    return float(log(rop, 2))


def binary_search_max_width(
    SIS, RC, log, oo, ZZ, RealField,
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
    if estimate_bits(SIS, RC, log, oo, ZZ, RealField, q, d, rank, 1, collision) < target_bits:
        return 0

    if estimate_bits(SIS, RC, log, oo, ZZ, RealField, q, d, rank, search_cap, collision) >= target_bits:
        return search_cap

    hi = 1
    while True:
        nxt = min(hi * 2, search_cap)
        if estimate_bits(SIS, RC, log, oo, ZZ, RealField, q, d, rank, nxt, collision) >= target_bits:
            hi = nxt
            continue
        lo, high = hi, nxt
        while lo < high - 1:
            mid = (lo + high) // 2
            if estimate_bits(SIS, RC, log, oo, ZZ, RealField, q, d, rank, mid, collision) >= target_bits:
                lo = mid
            else:
                high = mid
        return lo


def print_provenance_header(
    args: argparse.Namespace,
    estimator_path: Path,
    q: int,
    q_label: str,
) -> None:
    sha = estimator_git_sha(estimator_path)
    remote = estimator_remote_url(estimator_path)
    print(f"// Generated by: sage -python scripts/gen_sis_table.py")
    print(f"// Family: {args.family.upper()}")
    print(f"// lattice-estimator: {remote} @ {sha}")
    print(f"// Model: BDGL16, Euclidean (norm=2)")
    print(f"// Representative q = {q_label}")
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


def emit_rust_rows(
    rows: list[tuple[int, int, list[int]]],
) -> None:
    current_d = None
    emitted: dict[int, list[int]] = {}
    for d, collision, widths in rows:
        emitted.setdefault(d, []).append(collision)
        if d != current_d:
            if current_d is not None:
                print()
            print(f"        // D={d}")
            current_d = d
        ws = ", ".join(f"{w:_}" for w in widths)
        print(f"        ({d}, {collision}) => Some(&[{ws}]),")

    print()
    print("// Supported collision_l2_sq buckets per D:")
    for dim in sorted(emitted):
        cs = ", ".join(str(c) for c in emitted[dim])
        print(f"//   D={dim}: &[{cs}]")


def ladder_truncates_after_zero_width(args: argparse.Namespace) -> bool:
    """Only the monotonic power-of-two family sweep truncates a D after all-zero widths."""
    return args.dims is None and args.collisions is None


def run_entries(
    args: argparse.Namespace,
    entries: list[tuple[int, int]],
    estimator_path: Path,
) -> list[tuple[int, int, list[int]]]:
    SIS, RC, log, oo, ZZ, RealField = load_estimator(estimator_path)
    family_q, _, _, _ = FAMILIES[args.family]
    q = args.q if args.q is not None else family_q

    rows: list[tuple[int, int, list[int]]] = []
    dead_dims: set[int] = set()
    truncate_ladder = ladder_truncates_after_zero_width(args)
    for d, collision in entries:
        if truncate_ladder and d in dead_dims:
            continue

        widths: list[int] = []
        search_cap = args.search_cap if args.search_cap is not None else default_search_cap(d)
        for rank in range(1, args.max_rank + 1):
            t0 = time.time()
            w = binary_search_max_width(
                SIS, RC, log, oo, ZZ, RealField, q, d, rank, collision,
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
            if truncate_ladder:
                dead_dims.add(d)
                print(
                    f"        // (d={d}): no secure width at collision_l2_sq>={collision}; "
                    f"truncating ladder",
                    file=sys.stderr,
                )
            continue

        rows.append((d, collision, widths))
    return rows


def shard_jobs(requested: int) -> int:
    if requested < 1:
        raise SystemExit("--jobs must be at least 1")
    cpu_cap = os.cpu_count() or 1
    return min(requested, DEFAULT_JOBS_CAP, cpu_cap)


def sage_command() -> list[str]:
    return ["sage", "-python", str(Path(__file__).resolve())]


def build_child_argv(args: argparse.Namespace, d: int, collision: int) -> list[str]:
    argv = [
        *sage_command(),
        "--family", args.family,
        "--format", "csv",
        "--jobs", "1",
        "--max-rank", str(args.max_rank),
        "--target-bits", str(args.target_bits),
        "--dims", str(d),
        "--collisions", str(collision),
    ]
    if args.q is not None:
        argv.extend(["--q", str(args.q)])
    if args.search_cap is not None:
        argv.extend(["--search-cap", str(args.search_cap)])
    if args.estimator_path is not None:
        argv.extend(["--estimator-path", args.estimator_path])
    return argv


def parse_csv_rows(text: str) -> list[tuple[int, int, int, int, int, float, int]]:
    rows: list[tuple[int, int, int, int, int, float, int]] = []
    for line in text.splitlines():
        if not line or line.startswith("q,"):
            continue
        q_s, d_s, c_s, rank_s, width_s, bits_s, cap_s = line.split(",")
        rows.append((
            int(q_s), int(d_s), int(c_s), int(rank_s), int(width_s),
            float(bits_s), int(cap_s),
        ))
    return rows


def run_parallel_shards(
    args: argparse.Namespace,
    entries: list[tuple[int, int]],
    estimator_path: Path,
) -> list[tuple[int, int, list[int]]]:
    jobs = shard_jobs(args.jobs)
    if jobs == 1 or len(entries) <= 1:
        return run_entries(args, entries, estimator_path)

    print(
        f"// Sharding {len(entries)} (d, collision) cells across {jobs} Sage subprocesses",
        file=sys.stderr,
    )

    def run_one(work: tuple[int, int]) -> tuple[int, int, list[int]]:
        d, collision = work
        cmd = build_child_argv(args, d, collision)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise SystemExit(
                f"shard failed for d={d}, collision={collision} "
                f"(exit {proc.returncode}):\n{proc.stderr}"
            )
        parsed = parse_csv_rows(proc.stdout)
        if not parsed:
            return (d, collision, [0] * args.max_rank)
        widths = [width for *_rest, width, _bits, _cap in sorted(parsed, key=lambda r: r[3])]
        return (d, collision, widths)

    results: list[tuple[int, int, list[int]]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_one, work): work for work in entries}
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort()
    truncate_ladder = ladder_truncates_after_zero_width(args)
    dead_dims: set[int] = set()
    filtered: list[tuple[int, int, list[int]]] = []
    for d, collision, widths in results:
        if truncate_ladder and d in dead_dims:
            continue
        if all(w == 0 for w in widths):
            if truncate_ladder:
                dead_dims.add(d)
            continue
        filtered.append((d, collision, widths))
    return filtered


def main() -> None:
    args = parse_args()
    if args.max_rank < 1:
        raise SystemExit("--max-rank must be at least 1")

    entries = select_entries(args)
    if not entries:
        print("No matching entries.", file=sys.stderr)
        return

    estimator_path = locate_estimator(args.estimator_path)
    assert_pinned_estimator(estimator_path)
    family_q, family_label, _, _ = FAMILIES[args.family]
    q = args.q if args.q is not None else family_q
    q_label = str(q) if args.q is not None else family_label

    if args.format == "csv":
        print("q,d,collision,rank,max_width,target_bits,search_cap")
        run_entries(args, entries, estimator_path)
        return

    print_provenance_header(args, estimator_path, q, q_label)
    rows = run_parallel_shards(args, entries, estimator_path)
    emit_rust_rows(rows)


if __name__ == "__main__":
    main()
