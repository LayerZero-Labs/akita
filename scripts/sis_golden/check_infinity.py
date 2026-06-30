#!/usr/bin/env python3
"""Replay committed SIS infinity-norm golden cells against PR217."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(GOLDEN_DIR))

from infinity_core import (  # noqa: E402
    FLOAT_FIELDS,
    FRAGILE,
    INT_FIELDS,
    TRUSTED,
    assert_pr217_estimator,
    estimate_infinity_cell,
    estimator_git_sha,
    load_estimator,
    locate_estimator,
    parse_float,
    row_key,
)

FLOAT_ABS_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_DIR / "infinity_golden.csv",
        help="Golden CSV path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=GOLDEN_DIR / "infinity_metadata.json",
        help="Golden metadata path.",
    )
    parser.add_argument("--estimator-path", help="Path to lattice-estimator PR217 checkout.")
    parser.add_argument(
        "--skip-monotonicity",
        action="store_true",
        help="Skip width and bound monotonicity checks.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def compare_float(field: str, expected: str, actual: str) -> str | None:
    if expected in {"inf", "-inf"} or actual in {"inf", "-inf"}:
        if expected != actual:
            return f"{field}: expected {expected}, got {actual}"
        return None
    if expected == "" or actual == "":
        if expected != actual:
            return f"{field}: expected {expected!r}, got {actual!r}"
        return None
    e = parse_float(expected)
    a = parse_float(actual)
    if math.isfinite(e) and math.isfinite(a) and abs(e - a) <= FLOAT_ABS_TOL:
        return None
    if e == a:
        return None
    return f"{field}: expected {expected}, got {actual}"


def check_monotonicity(rows: list[dict[str, str]]) -> list[str]:
    failures: list[str] = []
    by_width: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    by_bound: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("trust") != TRUSTED or row.get("rop_log2") in {"", "inf", "-inf"}:
            continue
        by_width[
            (
                row["family"],
                int(row["d"]),
                int(row["rank"]),
                int(row["coeff_linf_bound"]),
            )
        ].append(row)
        by_bound[
            (
                row["family"],
                int(row["d"]),
                int(row["rank"]),
                int(row["width"]),
            )
        ].append(row)

    for key, group in by_width.items():
        prior_width = None
        prior_bits = None
        for row in sorted(group, key=lambda r: int(r["width"])):
            width = int(row["width"])
            bits = parse_float(row["rop_log2"])
            if prior_bits is not None and bits > prior_bits + FLOAT_ABS_TOL:
                failures.append(
                    f"width monotonicity {key}: width {width} has {bits} bits "
                    f"> prior width {prior_width} bits {prior_bits}"
                )
            prior_width = width
            prior_bits = bits

    for key, group in by_bound.items():
        prior_bound = None
        prior_bits = None
        for row in sorted(group, key=lambda r: int(r["coeff_linf_bound"])):
            bound = int(row["coeff_linf_bound"])
            bits = parse_float(row["rop_log2"])
            if prior_bits is not None and bits > prior_bits + FLOAT_ABS_TOL:
                failures.append(
                    f"bound monotonicity {key}: bound {bound} has {bits} bits "
                    f"> prior bound {prior_bound} bits {prior_bits}"
                )
            prior_bound = bound
            prior_bits = bits
    return failures


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text())
    estimator_path = locate_estimator(args.estimator_path)
    assert_pr217_estimator(estimator_path)
    actual_sha = estimator_git_sha(estimator_path)
    expected_sha = metadata.get("lattice_estimator_sha")
    if expected_sha and actual_sha != expected_sha:
        print(
            f"estimator SHA mismatch: golden expects {expected_sha}, got {actual_sha}",
            file=sys.stderr,
        )
        return 1

    SIS, RC, log, oo, _RealField = load_estimator(estimator_path)
    rows = load_rows(args.golden)
    failures: list[str] = []
    checked = 0
    skipped_fragile = 0

    for expected in sorted(rows, key=row_key):
        trust = expected.get("trust", TRUSTED)
        if trust == FRAGILE:
            skipped_fragile += 1
            continue
        if trust != TRUSTED:
            failures.append(f"unknown trust value {trust!r} for row {row_key(expected)}")
            continue

        actual = estimate_infinity_cell(
            SIS,
            RC,
            log,
            oo,
            family=expected["family"],
            d=int(expected["d"]),
            rank=int(expected["rank"]),
            width=int(expected["width"]),
            coeff_linf_bound=int(expected["coeff_linf_bound"]),
            target_bits=float(expected["target_bits"]),
        )
        checked += 1

        for field in FLOAT_FIELDS:
            failure = compare_float(field, expected[field], actual[field])
            if failure is not None:
                failures.append(f"{row_key(expected)} {failure}")
        for field in INT_FIELDS:
            if expected[field] != actual[field]:
                failures.append(
                    f"{row_key(expected)} {field}: expected {expected[field]}, got {actual[field]}"
                )
        for field in ["security_met", "tiny_probability"]:
            if expected[field] != actual[field]:
                failures.append(
                    f"{row_key(expected)} {field}: expected {expected[field]}, got {actual[field]}"
                )

    if not args.skip_monotonicity:
        failures.extend(check_monotonicity(rows))

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        print(f"{len(failures)} infinity golden check(s) failed", file=sys.stderr)
        return 1

    print(
        f"OK: {checked} trusted infinity cell(s) match PR217 @ {actual_sha}; "
        f"skipped {skipped_fragile} fragile cell(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
