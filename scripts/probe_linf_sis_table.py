#!/usr/bin/env python3
"""
Probe direct L-infinity SIS estimates for Akita planner candidate cells.

This is a diagnostic companion to `gen_sis_table.py`, not a production table
generator.  It evaluates explicit `(d, B, width, rank)` cells where:

    n = rank * d
    m = width * d
    q = selected family modulus
    ||x||_infinity <= B

The intended first use is checking planner-derived A-role failures before
cutting the planner over from the legacy Euclidean/L2 surrogate table.

Example:

    sage -python scripts/probe_linf_sis_table.py \
      --family q32 --d 256 --buckets 34668544,138674176 \
      --widths 1024,4096 --ranks 4,5,6 \
      --estimator-path ../lattice-estimator-zeta-controls
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gen_sis_table import (  # noqa: E402
    DEFAULT_JOBS_CAP,
    FAMILIES,
    estimator_git_sha,
    estimator_remote_url,
    locate_estimator,
    parse_int_list,
)

DEFAULT_MAX_M = 2_000_000
DEFAULT_TARGET_BITS = 128.0

_SIS: Any = None
_RC: Any = None
_LOG: Any = None
_OO: Any = None
_Q: int | None = None
_RED_COST_MODEL: str = "ADPS16"
_RED_SHAPE_MODEL: str = "lgsa"
_ZETA_CANDIDATES: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class ProbePoint:
    d: int
    bucket: int
    width: int
    rank: int
    label: str = ""

    @property
    def n(self) -> int:
        return self.rank * self.d

    @property
    def m(self) -> int:
        return self.width * self.d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--require-estimator-sha",
        help="Abort unless the selected lattice-estimator checkout has this git SHA.",
    )
    parser.add_argument("--d", type=int, help="Ring dimension for grid mode.")
    parser.add_argument(
        "--buckets",
        help="Comma-separated coefficient-L∞ bounds B for grid mode.",
    )
    parser.add_argument("--widths", help="Comma-separated ring widths for grid mode.")
    parser.add_argument("--ranks", help="Comma-separated module ranks for grid mode.")
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help=(
            "Explicit point `d,B,width,rank[,label]`. May be repeated. "
            "Labels must not contain commas."
        ),
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        help="CSV with columns d,bucket,width,rank and optional label.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=f"Independent worker processes (default: 1, capped at {DEFAULT_JOBS_CAP}).",
    )
    parser.add_argument(
        "--max-m",
        type=int,
        default=DEFAULT_MAX_M,
        help=f"Skip points with m=width*d above this cap (default: {DEFAULT_MAX_M:_}).",
    )
    parser.add_argument(
        "--allow-large-m",
        action="store_true",
        help="Evaluate points even when m exceeds --max-m.",
    )
    parser.add_argument(
        "--target-bits",
        type=float,
        default=DEFAULT_TARGET_BITS,
        help=f"Status threshold for `secure`/`insecure` labels (default: {DEFAULT_TARGET_BITS}).",
    )
    parser.add_argument(
        "--red-cost-model",
        choices=["ADPS16"],
        default="ADPS16",
        help="Reduction cost model for L∞ probes (default: ADPS16).",
    )
    parser.add_argument(
        "--red-shape-model",
        choices=["lgsa", "zgsa", "gsa", "cn11"],
        default="lgsa",
        help="Reduction shape model for L∞ probes (default: lgsa).",
    )
    parser.add_argument(
        "--zeta-candidates",
        default="0",
        help="Comma-separated estimator ignored-coordinate candidates (default: 0).",
    )
    return parser.parse_args()


def worker_count(requested: int) -> int:
    if requested < 1:
        raise SystemExit("--jobs must be at least 1")
    return min(requested, DEFAULT_JOBS_CAP, os.cpu_count() or 1)


def parse_point(raw: str) -> ProbePoint:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in (4, 5) or any(not part for part in parts[:4]):
        raise SystemExit(f"invalid --point `{raw}`; expected d,B,width,rank[,label]")
    return ProbePoint(
        d=int(parts[0], 0),
        bucket=int(parts[1], 0),
        width=int(parts[2], 0),
        rank=int(parts[3], 0),
        label=parts[4] if len(parts) == 5 else "",
    )


def read_points_csv(path: Path) -> list[ProbePoint]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"d", "bucket", "width", "rank"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        points = []
        for row in reader:
            points.append(
                ProbePoint(
                    d=int(row["d"], 0),
                    bucket=int(row["bucket"], 0),
                    width=int(row["width"], 0),
                    rank=int(row["rank"], 0),
                    label=row.get("label", ""),
                )
            )
        return points


def grid_points(args: argparse.Namespace) -> list[ProbePoint]:
    buckets = parse_int_list(args.buckets)
    widths = parse_int_list(args.widths)
    ranks = parse_int_list(args.ranks)
    if args.d is None and not any([buckets, widths, ranks]):
        return []
    if args.d is None or buckets is None or widths is None or ranks is None:
        raise SystemExit("grid mode requires --d, --buckets, --widths, and --ranks")
    return [
        ProbePoint(args.d, bucket, width, rank)
        for bucket in buckets
        for width in widths
        for rank in ranks
    ]


def collect_points(args: argparse.Namespace) -> list[ProbePoint]:
    points = [parse_point(raw) for raw in args.point]
    if args.points_csv is not None:
        points.extend(read_points_csv(args.points_csv))
    points.extend(grid_points(args))
    if not points:
        raise SystemExit("no probe points; pass --point, --points-csv, or grid-mode arguments")
    for point in points:
        if point.d <= 0 or point.bucket <= 0 or point.width <= 0 or point.rank <= 0:
            raise SystemExit(f"probe point fields must be positive: {point}")
    return points


def init_worker(
    estimator_path: str,
    q: int,
    red_cost_model: str,
    red_shape_model: str,
    zeta_candidates: tuple[int, ...],
) -> None:
    global _SIS, _RC, _LOG, _OO, _Q, _RED_COST_MODEL, _RED_SHAPE_MODEL, _ZETA_CANDIDATES
    sys.path.insert(0, estimator_path)
    from estimator import SIS
    from estimator.reduction import RC
    from sage.all import log, oo

    _SIS = SIS
    _RC = RC
    _LOG = log
    _OO = oo
    _Q = q
    _RED_COST_MODEL = red_cost_model
    _RED_SHAPE_MODEL = red_shape_model
    _ZETA_CANDIDATES = zeta_candidates


def reduction_model():
    if _RED_COST_MODEL == "ADPS16":
        return _RC.ADPS16
    raise ValueError(f"unsupported reduction model {_RED_COST_MODEL}")


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def probe_one(point: ProbePoint, max_m: int, allow_large_m: bool, target_bits: float) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "label": point.label,
        "q": _Q,
        "d": point.d,
        "bucket": point.bucket,
        "width": point.width,
        "rank": point.rank,
        "n": point.n,
        "m": point.m,
        "target_bits": target_bits,
        "bits": "",
        "status": "",
        "linf_regime": "",
        "zeta": "",
        "d_eff": "",
        "beta": "",
        "eta": "",
        "log_success_probability": "",
        "seconds": "",
        "error": "",
    }
    if point.m > max_m and not allow_large_m:
        row["status"] = "skipped_m_cap"
        row["error"] = f"m={point.m} exceeds --max-m={max_m}"
        row["seconds"] = f"{time.perf_counter() - started:.3f}"
        return row
    try:
        cost = _SIS.lattice(
            _SIS.Parameters(
                n=point.n,
                q=_Q,
                m=point.m,
                length_bound=point.bucket,
                norm=_OO,
                tag="akita_linf_probe",
            ),
            red_cost_model=reduction_model(),
            red_shape_model=_RED_SHAPE_MODEL,
            zeta_candidates=list(_ZETA_CANDIDATES),
            diagnostics=True,
            log_level=0,
        )
        rop = cost["rop"]
        if rop == _OO:
            bits: float | str = "inf"
            row["status"] = "secure"
        else:
            bits = float(_LOG(rop, 2))
            row["status"] = "secure" if bits >= target_bits else "insecure"
        row.update(
            bits=f"{bits:.6f}" if isinstance(bits, float) else bits,
            linf_regime=csv_cell(cost.get("linf_regime", "")),
            zeta=csv_cell(cost.get("zeta", "")),
            d_eff=csv_cell(cost.get("d", "")),
            beta=csv_cell(cost.get("beta", "")),
            eta=csv_cell(cost.get("eta", "")),
            log_success_probability=csv_cell(cost.get("log_success_probability", "")),
        )
    except ValueError as exc:
        if "SIS trivially easy" in str(exc):
            row["status"] = "trivially_easy"
        else:
            row["status"] = "error"
        row["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - diagnostic script should report and continue.
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = f"{time.perf_counter() - started:.3f}"
    return row


def print_provenance(args: argparse.Namespace, estimator_path: Path, q: int, q_label: str) -> None:
    print(f"# Generated by: sage -python scripts/probe_linf_sis_table.py")
    print(f"# Family: {args.family.upper()}")
    print(f"# lattice-estimator: {estimator_remote_url(estimator_path)} @ {estimator_git_sha(estimator_path)}")
    print(f"# Model: {args.red_cost_model}, L-infinity (norm=oo), shape={args.red_shape_model}")
    print(f"# zeta_candidates={tuple(parse_int_list(args.zeta_candidates) or [])}")
    print(f"# Representative q = {q_label}")
    print(f"# q = {q}")
    print(f"# target_bits = {args.target_bits}")
    print(f"# max_m = {args.max_m} ({'ignored' if args.allow_large_m else 'enforced'})")


def main() -> None:
    args = parse_args()
    estimator_path = locate_estimator(args.estimator_path)
    if args.require_estimator_sha is not None:
        actual = estimator_git_sha(estimator_path)
        if actual != args.require_estimator_sha:
            raise SystemExit(
                "lattice-estimator SHA mismatch: "
                f"expected {args.require_estimator_sha}, got {actual} at {estimator_path}"
            )

    zeta_candidates = tuple(parse_int_list(args.zeta_candidates) or [])
    if not zeta_candidates:
        raise SystemExit("--zeta-candidates must not be empty")

    family_q, q_label, _, _ = FAMILIES[args.family]
    q = args.q if args.q is not None else family_q
    points = collect_points(args)
    jobs = worker_count(args.jobs)

    print_provenance(args, estimator_path, q, q_label)
    fieldnames = [
        "label",
        "q",
        "d",
        "bucket",
        "width",
        "rank",
        "n",
        "m",
        "target_bits",
        "bits",
        "status",
        "linf_regime",
        "zeta",
        "d_eff",
        "beta",
        "eta",
        "log_success_probability",
        "seconds",
        "error",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    if jobs == 1:
        init_worker(str(estimator_path), q, args.red_cost_model, args.red_shape_model, zeta_candidates)
        for point in points:
            writer.writerow(probe_one(point, args.max_m, args.allow_large_m, args.target_bits))
            sys.stdout.flush()
        return

    with ProcessPoolExecutor(
        max_workers=jobs,
        initializer=init_worker,
        initargs=(str(estimator_path), q, args.red_cost_model, args.red_shape_model, zeta_candidates),
    ) as pool:
        futures = {
            pool.submit(probe_one, point, args.max_m, args.allow_large_m, args.target_bits): idx
            for idx, point in enumerate(points)
        }
        rows: dict[int, dict[str, Any]] = {}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
        for idx in range(len(points)):
            writer.writerow(rows[idx])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
