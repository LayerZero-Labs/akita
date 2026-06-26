#!/usr/bin/env python3
"""
Regenerate the coefficient-L-infinity SIS max-width table for the Akita planner.

For each (family, d, collision_linf bucket B, rank), binary-searches the maximum
ring-element width with >= target_bits security under lattice-estimator
`norm=oo`, `red_cost_model=ADPS16`, `red_shape_model=lgsa`.

Usage:
    sage -python scripts/gen_sis_linf_table.py

See `scripts/stitch_generated_sis_linf_table.py` for the full stitch workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(__import__("signal"), "SIGPIPE"):
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

MIN_LOG_BUCKET = 1
MAX_LOG_BUCKET = 84

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

FAMILIES: dict[str, tuple[int, str, list[int], list[int]]] = {
    "q32": (
        (1 << 32) - 99,
        "2^32 - 99",
        RING_DIMS,
        COEFF_LINF_BUCKETS,
    ),
    "q64": (
        (1 << 64) - 59,
        "2^64 - 59",
        RING_DIMS,
        COEFF_LINF_BUCKETS,
    ),
    "q128": (
        (1 << 128) - ((1 << 32) - 22537),
        "2^128 - (2^32 - 22537)",
        RING_DIMS,
        COEFF_LINF_BUCKETS,
    ),
}

D128_SEARCH_CAP = 50_000_000_000
DEFAULT_SEARCH_CAP = 10_000_000_000
DEFAULT_JOBS_CAP = 6
DEFAULT_MAX_RANK = 20
PINNED_LATTICE_ESTIMATOR_SHA = "a31d06b9a614461fd007571880424919a71a8fda"
DEFAULT_ZETA_CANDIDATES = (0,)
DEFAULT_RED_COST_MODEL = "ADPS16"
DEFAULT_RED_SHAPE_MODEL = "lgsa"
# lattice-estimator becomes impractical beyond this column count; widths above
# `MAX_ESTIMATOR_M // d` are treated as securely above target.
MAX_ESTIMATOR_M = 8_388_608
DEFAULT_CACHE_DIR_NAME = ".sis_linf_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="q128")
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--estimator-path", default=None)
    parser.add_argument("--search-cap", type=int, default=None)
    parser.add_argument("--target-bits", type=float, default=128.0)
    parser.add_argument("--max-rank", type=int, default=DEFAULT_MAX_RANK)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--collision", type=int, default=None, help="Coefficient-Linf bucket B")
    parser.add_argument("--dims", help="Comma-separated ring dimensions")
    parser.add_argument("--collisions", help="Comma-separated coefficient-Linf buckets")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--format", choices=["rust", "csv"], default="rust")
    parser.add_argument(
        "--red-cost-model",
        choices=["ADPS16"],
        default=DEFAULT_RED_COST_MODEL,
    )
    parser.add_argument(
        "--red-shape-model",
        choices=["lgsa", "zgsa", "gsa", "cn11"],
        default=DEFAULT_RED_SHAPE_MODEL,
    )
    parser.add_argument(
        "--zeta-candidates",
        default="0",
        help="Comma-separated ignored-coordinate candidates (default: 0)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory for per-cell width caches (default: "
            f"<repo>/{DEFAULT_CACHE_DIR_NAME}). Enables resume and incremental stitch."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and do not write cell caches.",
    )
    parser.add_argument(
        "--cell-timeout",
        type=float,
        default=None,
        help=(
            "Per-cell wall-clock budget in seconds for --jobs sharding. A cell "
            "exceeding it is killed, left uncached, and reported as failed "
            "instead of stalling the run. Default: no timeout."
        ),
    )
    return parser.parse_args()


def default_cache_dir() -> Path:
    return repo_root() / DEFAULT_CACHE_DIR_NAME


def resolve_cache_dir(args: argparse.Namespace) -> Path | None:
    if args.no_cache:
        return None
    return Path(args.cache_dir).expanduser() if args.cache_dir else default_cache_dir()


def cell_cache_path(cache_dir: Path, family: str, d: int, collision: int) -> Path:
    return cache_dir / family / f"d{d}_c{collision}.json"


def read_cell_cache(path: Path, max_rank: int) -> list[int] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        widths = [int(w) for w in payload["widths"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if len(widths) != max_rank:
        return None
    return widths


def write_cell_cache(path: Path, widths: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"widths": widths}, indent=0) + "\n")


def effective_search_cap(d: int, search_cap: int) -> int:
    return min(search_cap, MAX_ESTIMATOR_M // d)


def default_search_cap(d: int) -> int:
    if d == 128:
        return effective_search_cap(d, D128_SEARCH_CAP)
    return effective_search_cap(d, DEFAULT_SEARCH_CAP)


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
        url = out.stdout.strip()
        if url.startswith("git@github.com:"):
            return "https://github.com/" + url.removeprefix("git@github.com:")
        if url.startswith("ssh://git@github.com/"):
            return "https://github.com/" + url.removeprefix("ssh://git@github.com/")
        return url
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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


def parse_zeta_candidates(raw: str) -> tuple[int, ...]:
    values = parse_int_list(raw)
    if values is None:
        return DEFAULT_ZETA_CANDIDATES
    return tuple(values)


def family_entries(family: str) -> list[tuple[int, int]]:
    _, _, dims, collisions = FAMILIES[family]
    return [(d, c) for d in dims for c in collisions]


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


def load_estimator(path: Path):
    sys.path.insert(0, str(path))
    from estimator import SIS
    from estimator.reduction import RC
    from sage.all import log, oo

    return SIS, RC, log, oo


def reduction_model(RC, model: str):
    if model == "ADPS16":
        return RC.ADPS16
    raise ValueError(f"unsupported reduction model {model}")


def estimate_bits(
    SIS,
    RC,
    log,
    oo,
    q: int,
    d: int,
    rank: int,
    width: int,
    collision_linf: int,
    red_cost_model: str,
    red_shape_model: str,
    zeta_candidates: tuple[int, ...],
) -> float:
    n = rank * d
    m = width * d
    try:
        out = SIS.lattice(
            SIS.Parameters(
                n=n,
                q=q,
                m=m,
                length_bound=collision_linf,
                norm=oo,
                tag="sis_linf_table",
            ),
            red_cost_model=reduction_model(RC, red_cost_model),
            red_shape_model=red_shape_model,
            zeta=zeta_candidates[0],
            log_level=0,
        )
    except ValueError as exc:
        msg = str(exc)
        if "SIS trivially easy" in msg or "Incorrect bounds" in msg:
            return float("-inf")
        raise
    except OverflowError:
        # Estimator overflow at extreme (m, beta); treat as above target security.
        return float("inf")
    except TypeError as exc:
        # The estimator's internal optimizer can return `None` and then crash
        # (e.g. `NoneType * int` in util.py `local_minimum.__next__`) at extreme,
        # heavily over-determined `m`. That regime is the same trivially-easy /
        # insecure one the "SIS trivially easy" ValueError maps to, so treat it
        # as below target (-inf) rather than aborting the whole run. Logged so a
        # genuinely new failure mode stays auditable instead of silent.
        print(
            f"        // WARN estimator TypeError (d={d}, rank={rank}, "
            f"width={width}, m={m}, collision_linf={collision_linf}): {exc}; "
            f"treating as insecure (-inf)",
            file=sys.stderr,
        )
        return float("-inf")
    rop = out["rop"]
    if rop == oo:
        return float("inf")
    return float(log(rop, 2))


def binary_search_max_width(
    SIS,
    RC,
    log,
    oo,
    q: int,
    d: int,
    rank: int,
    collision_linf: int,
    target_bits: float,
    search_cap: int,
    red_cost_model: str,
    red_shape_model: str,
    zeta_candidates: tuple[int, ...],
) -> int:
    def bits(width: int) -> float:
        return estimate_bits(
            SIS,
            RC,
            log,
            oo,
            q,
            d,
            rank,
            width,
            collision_linf,
            red_cost_model,
            red_shape_model,
            zeta_candidates,
        )

    if bits(1) < target_bits:
        return 0
    cap = min(search_cap, MAX_ESTIMATOR_M // d)
    if cap == 0:
        return 0
    if cap == 1:
        # width 1 already shown secure above; nothing larger to probe.
        return 1

    # Exponential bracketing from below. `bits(cap)` is a full max-`m` estimator
    # solve (m = cap * d ~ MAX_ESTIMATOR_M, ~100s); probing it on every cell
    # dominated runtime even though the secure boundary is almost always tiny
    # (single/double-digit widths). We now touch `cap` only if the doubling
    # search actually reaches it -- i.e. every width below cap is still secure --
    # so the common case never pays that cost. The `nxt == cap` guard also
    # prevents an infinite doubling loop once `hi * 2` saturates at the ceiling.
    hi = 1  # known secure: bits(1) >= target_bits
    while True:
        nxt = min(hi * 2, cap)
        if bits(nxt) >= target_bits:
            if nxt == cap:
                # Ceiling reached and still secure: cap is the answer.
                return cap
            hi = nxt
            continue
        # nxt is insecure; secure boundary lies in [hi, nxt).
        lo, high = hi, nxt
        while lo < high - 1:
            mid = (lo + high) // 2
            if bits(mid) >= target_bits:
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
    print(f"// Generated by: sage -python scripts/gen_sis_linf_table.py")
    print(f"// Family: {args.family.upper()}")
    print(f"// lattice-estimator: {remote} @ {sha}")
    print(
        f"// Model: {args.red_cost_model}, coefficient L-infinity (norm=oo), "
        f"shape={args.red_shape_model}"
    )
    print(f"// zeta_candidates={parse_zeta_candidates(args.zeta_candidates)}")
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


def emit_rust_rows(rows: list[tuple[int, int, list[int]]]) -> None:
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
    print("// Supported coefficient-Linf buckets per D:")
    for dim in sorted(emitted):
        cs = ", ".join(str(c) for c in emitted[dim])
        print(f"//   D={dim}: &[{cs}]")


def run_entries(
    args: argparse.Namespace,
    entries: list[tuple[int, int]],
    estimator_path: Path,
    cache_dir: Path | None = None,
) -> list[tuple[int, int, list[int]]]:
    SIS, RC, log, oo = load_estimator(estimator_path)
    family_q, _, _, _ = FAMILIES[args.family]
    q = args.q if args.q is not None else family_q
    zeta_candidates = parse_zeta_candidates(args.zeta_candidates)

    rows: list[tuple[int, int, list[int]]] = []
    dead_dims: set[int] = set()
    for d, collision in entries:
        if d in dead_dims:
            continue

        if cache_dir is not None:
            cached = read_cell_cache(
                cell_cache_path(cache_dir, args.family, d, collision),
                args.max_rank,
            )
            if cached is not None:
                print(
                    f"        // cache hit (d={d}, collision_linf={collision})",
                    file=sys.stderr,
                )
                if all(w == 0 for w in cached):
                    dead_dims.add(d)
                    continue
                rows.append((d, collision, cached))
                continue

        widths: list[int] = []
        search_cap = args.search_cap if args.search_cap is not None else default_search_cap(d)
        eff_cap = effective_search_cap(d, search_cap)
        saturated = False
        for rank in range(1, args.max_rank + 1):
            if saturated:
                # Max secure width is non-decreasing in rank (larger rank -> larger
                # n -> harder SIS). Once a rank saturates the estimator ceiling, every
                # higher rank does too, so skip the ~17s/rank ceiling probe entirely.
                w = eff_cap
                elapsed = 0.0
            else:
                t0 = time.time()
                w = binary_search_max_width(
                    SIS,
                    RC,
                    log,
                    oo,
                    q,
                    d,
                    rank,
                    collision,
                    args.target_bits,
                    search_cap,
                    args.red_cost_model,
                    args.red_shape_model,
                    zeta_candidates,
                )
                elapsed = time.time() - t0
                if w >= eff_cap:
                    saturated = True
            print(
                f"        // (d={d}, collision_linf={collision}, rank={rank}): "
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
                f"        // (d={d}): no secure width at collision_linf>={collision}; "
                f"truncating ladder",
                file=sys.stderr,
            )
            continue

        if cache_dir is not None:
            write_cell_cache(
                cell_cache_path(cache_dir, args.family, d, collision),
                widths,
            )

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
        "--family",
        args.family,
        "--format",
        "csv",
        "--jobs",
        "1",
        "--max-rank",
        str(args.max_rank),
        "--target-bits",
        str(args.target_bits),
        "--dims",
        str(d),
        "--collisions",
        str(collision),
        "--red-cost-model",
        args.red_cost_model,
        "--red-shape-model",
        args.red_shape_model,
        "--zeta-candidates",
        args.zeta_candidates,
    ]
    if args.q is not None:
        argv.extend(["--q", str(args.q)])
    if args.search_cap is not None:
        argv.extend(["--search-cap", str(args.search_cap)])
    if args.estimator_path is not None:
        argv.extend(["--estimator-path", args.estimator_path])
    if args.cache_dir is not None:
        argv.extend(["--cache-dir", args.cache_dir])
    return argv


def parse_csv_rows(text: str) -> list[tuple[int, int, int, int, int, float, int]]:
    rows: list[tuple[int, int, int, int, int, float, int]] = []
    for line in text.splitlines():
        if not line or line.startswith("q,"):
            continue
        q_s, d_s, c_s, rank_s, width_s, bits_s, cap_s = line.split(",")
        rows.append((
            int(q_s),
            int(d_s),
            int(c_s),
            int(rank_s),
            int(width_s),
            float(bits_s),
            int(cap_s),
        ))
    return rows


def run_parallel_shards(
    args: argparse.Namespace,
    entries: list[tuple[int, int]],
    estimator_path: Path,
    cache_dir: Path | None = None,
) -> list[tuple[int, int, list[int]]]:
    jobs = shard_jobs(args.jobs)
    if jobs == 1 or len(entries) <= 1:
        return run_entries(args, entries, estimator_path, cache_dir)

    pending: list[tuple[int, int]] = []
    results: list[tuple[int, int, list[int]]] = []
    for d, collision in entries:
        if cache_dir is not None:
            cached = read_cell_cache(
                cell_cache_path(cache_dir, args.family, d, collision),
                args.max_rank,
            )
            if cached is not None:
                results.append((d, collision, cached))
                continue
        pending.append((d, collision))

    if not pending:
        results.sort()
    else:
        print(
            f"// Sharding {len(pending)} (d, collision_linf) cells across "
            f"{jobs} Sage subprocesses ({len(results)} cached)",
            file=sys.stderr,
        )

        def run_one(work: tuple[int, int]) -> tuple[int, int, list[int] | None]:
            d, collision = work
            cmd = build_child_argv(args, d, collision)
            # Isolate per-cell failures: a crashed or timed-out cell returns a
            # `None` sentinel (left uncached for retry) rather than aborting the
            # whole sharded run. Lets an overnight regen finish the other cells
            # and report which ones need a rerun.
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=args.cell_timeout,
                )
            except subprocess.TimeoutExpired:
                print(
                    f"// FAIL d={d}, collision_linf={collision}: timed out after "
                    f"{args.cell_timeout}s; left uncached",
                    file=sys.stderr,
                )
                return (d, collision, None)
            if proc.returncode != 0:
                print(
                    f"// FAIL d={d}, collision_linf={collision} "
                    f"(exit {proc.returncode}); left uncached:\n{proc.stderr.strip()}",
                    file=sys.stderr,
                )
                return (d, collision, None)
            parsed = parse_csv_rows(proc.stdout)
            if not parsed:
                widths = [0] * args.max_rank
            else:
                widths = [
                    width for *_rest, width, _bits, _cap in sorted(parsed, key=lambda r: r[3])
                ]
            if cache_dir is not None:
                write_cell_cache(
                    cell_cache_path(cache_dir, args.family, d, collision),
                    widths,
                )
            return (d, collision, widths)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_one, work): work for work in pending}
            for fut in as_completed(futures):
                results.append(fut.result())

        results.sort()

    failures = [(d, c) for d, c, widths in results if widths is None]
    if failures:
        preview = ", ".join(f"(d={d}, c={c})" for d, c in failures[:10])
        more = "" if len(failures) <= 10 else f" (+{len(failures) - 10} more)"
        print(
            f"// {len(failures)} cell(s) FAILED and were left uncached; "
            f"rerun the same command to retry just these: {preview}{more}",
            file=sys.stderr,
        )

    dead_dims: set[int] = set()
    filtered: list[tuple[int, int, list[int]]] = []
    for d, collision, widths in results:
        if widths is None:
            continue
        if d in dead_dims:
            continue
        if all(w == 0 for w in widths):
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
    cache_dir = resolve_cache_dir(args)
    if cache_dir is not None:
        args.cache_dir = str(cache_dir)
    family_q, family_label, _, _ = FAMILIES[args.family]
    q = args.q if args.q is not None else family_q
    q_label = str(q) if args.q is not None else family_label

    if args.format == "csv":
        print("q,d,collision,rank,max_width,target_bits,search_cap")
        run_entries(args, entries, estimator_path, cache_dir)
        return

    print_provenance_header(args, estimator_path, q, q_label)
    if cache_dir is not None:
        print(f"// Cell cache: {cache_dir}", file=sys.stderr)
    rows = run_parallel_shards(args, entries, estimator_path, cache_dir)
    emit_rust_rows(rows)


if __name__ == "__main__":
    main()
