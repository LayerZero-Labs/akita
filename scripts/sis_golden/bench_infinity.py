#!/usr/bin/env python3
"""Single-shot Rust-vs-Sage timings for SIS infinity estimator cells."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent
REPO_ROOT = GOLDEN_DIR.parents[1]

sys.path.insert(0, str(GOLDEN_DIR))

from infinity_core import (  # noqa: E402
    PR217_LATTICE_ESTIMATOR_SHA,
    assert_pr217_estimator,
    estimate_infinity_cell,
    estimator_git_sha,
    load_estimator,
    locate_estimator,
)

DEFAULT_CASES = [
    "small:q32:32:1:2:15",
    "medium:q32:128:1:8:2",
    "large:q64:256:1:8:2",
    "wide-large:q32:256:5:10:4095",
]


@dataclass(frozen=True)
class BenchCase:
    label: str
    family: str
    d: int
    rank: int
    width: int
    coeff_linf_bound: int

    @classmethod
    def parse(cls, raw: str) -> "BenchCase":
        parts = raw.split(":")
        if len(parts) != 6:
            raise SystemExit(
                "case must be label:family:d:rank:width:coeff_linf_bound"
            )
        label, family, d, rank, width, bound = parts
        return cls(label, family, int(d), int(rank), int(width), int(bound))

    @property
    def key(self) -> tuple[str, int, int, int, int]:
        return (self.family, self.d, self.rank, self.width, self.coeff_linf_bound)


@dataclass(frozen=True)
class BenchResult:
    case: BenchCase
    rust_seconds: float
    sage_seconds: float
    rust_rop_log2: str
    sage_rop_log2: str
    rust_beta: str
    sage_beta: str
    rust_zeta: str
    sage_zeta: str
    speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_DIR / "infinity_golden.csv",
        help="Golden CSV used to validate selected trusted cells.",
    )
    parser.add_argument("--estimator-path", help="Path to lattice-estimator PR217 checkout.")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Benchmark case as label:family:d:rank:width:coeff_linf_bound. May repeat.",
    )
    parser.add_argument(
        "--rust-iterations",
        type=int,
        default=1,
        help="Number of estimator calls inside the Rust binary timing loop.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Reuse target/release/examples/sis_estimator_once without rebuilding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rust_iterations <= 0:
        raise SystemExit("--rust-iterations must be positive")

    cases = [BenchCase.parse(raw) for raw in (args.cases or DEFAULT_CASES)]
    trusted_rows = load_trusted_rows(args.golden)
    for case in cases:
        if case.key not in trusted_rows:
            raise SystemExit(f"selected case is not a trusted golden row: {case}")

    rust_bin = build_rust_example(args.no_build)
    estimator_path = locate_estimator(args.estimator_path)
    assert_pr217_estimator(estimator_path)
    SIS, RC, log, oo, _RealField = load_estimator(estimator_path)
    quiet_lattice_estimator_logs()

    results = []
    for case in cases:
        rust_row = time_rust_case(rust_bin, case, args.rust_iterations)
        sage_row, sage_seconds = time_sage_case(SIS, RC, log, oo, case)
        results.append(
            BenchResult(
                case=case,
                rust_seconds=float(rust_row["seconds_per_iter"]),
                sage_seconds=sage_seconds,
                rust_rop_log2=rust_row["rop_log2"],
                sage_rop_log2=sage_row["rop_log2"],
                rust_beta=rust_row["beta"],
                sage_beta=sage_row["beta"],
                rust_zeta=rust_row["zeta"],
                sage_zeta=sage_row["zeta"],
                speedup=sage_seconds / float(rust_row["seconds_per_iter"]),
            )
        )

    print_results(results, estimator_git_sha(estimator_path), args.rust_iterations)
    return 0


def load_trusted_rows(path: Path) -> set[tuple[str, int, int, int, int]]:
    with path.open(newline="") as fh:
        rows = csv.DictReader(fh)
        return {
            (
                row["family"],
                int(row["d"]),
                int(row["rank"]),
                int(row["width"]),
                int(row["coeff_linf_bound"]),
            )
            for row in rows
            if row.get("trust") == "trusted"
        }


def build_rust_example(no_build: bool) -> Path:
    binary = REPO_ROOT / "target" / "release" / "examples" / "sis_estimator_once"
    if not no_build:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "akita-sis-estimator",
                "--example",
                "sis_estimator_once",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    if not binary.exists():
        raise SystemExit(f"Rust benchmark binary not found: {binary}")
    return binary


def time_rust_case(binary: Path, case: BenchCase, iterations: int) -> dict[str, str]:
    cmd = [
        str(binary),
        "--mode",
        "estimate",
        "--family",
        case.family,
        "--d",
        str(case.d),
        "--rank",
        str(case.rank),
        "--width",
        str(case.width),
        "--coeff-linf-bound",
        str(case.coeff_linf_bound),
        "--iterations",
        str(iterations),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return next(csv.DictReader(io.StringIO(out.stdout)))


def time_sage_case(SIS, RC, log, oo, case: BenchCase):
    t0 = time.perf_counter()
    row = estimate_infinity_cell(
        SIS,
        RC,
        log,
        oo,
        family=case.family,
        d=case.d,
        rank=case.rank,
        width=case.width,
        coeff_linf_bound=case.coeff_linf_bound,
        target_bits=138.0,
    )
    return row, time.perf_counter() - t0


def print_results(results: list[BenchResult], estimator_sha: str, rust_iterations: int) -> None:
    print(f"PR217 Sage SHA: {estimator_sha} (expected {PR217_LATTICE_ESTIMATOR_SHA})")
    print(f"Rust iterations per row: {rust_iterations}")
    print()
    print(
        "| case | row | Rust seconds | Sage seconds | Sage/Rust | beta | zeta | rop log2 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        case = result.case
        row = (
            f"{case.family} d={case.d} r={case.rank} "
            f"w={case.width} B={case.coeff_linf_bound}"
        )
        beta = checked_pair(result.rust_beta, result.sage_beta)
        zeta = checked_pair(result.rust_zeta, result.sage_zeta)
        rop = checked_pair(result.rust_rop_log2, result.sage_rop_log2)
        print(
            f"| {case.label} | {row} | {result.rust_seconds:.6f} | "
            f"{result.sage_seconds:.6f} | {result.speedup:.1f}x | "
            f"{beta} | {zeta} | {rop} |"
        )


def checked_pair(rust: str, sage: str) -> str:
    if rust == sage:
        return rust
    if close_log2(rust, sage):
        return sage
    return f"Rust {rust} / Sage {sage}"


def close_log2(lhs: str, rhs: str) -> bool:
    if lhs == rhs:
        return True
    if lhs == "inf" or rhs == "inf":
        return False
    try:
        return abs(float(lhs) - float(rhs)) <= 1e-6
    except ValueError:
        return False


def quiet_lattice_estimator_logs() -> None:
    from estimator.io import Logging  # noqa: WPS433

    Logging.set_level(Logging.CRITICAL)


if __name__ == "__main__":
    raise SystemExit(main())
