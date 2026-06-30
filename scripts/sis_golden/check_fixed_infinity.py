#!/usr/bin/env python3
"""Replay committed fixed-beta, fixed-zeta infinity golden cells against PR217."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(GOLDEN_DIR))

from infinity_core import (  # noqa: E402
    FLOAT_FIELDS,
    FRAGILE,
    INT_FIELDS,
    TRUSTED,
    assert_pr217_estimator,
    estimate_fixed_infinity_cell,
    estimator_git_sha,
    fixed_row_key,
    format_estimator_sha,
    load_estimator,
    locate_estimator,
    parse_float,
)

FLOAT_ABS_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_DIR / "fixed_infinity_golden.csv",
        help="Golden CSV path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=GOLDEN_DIR / "fixed_infinity_metadata.json",
        help="Golden metadata path.",
    )
    parser.add_argument("--estimator-path", help="Path to lattice-estimator PR217 checkout.")
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


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text())
    estimator_path = locate_estimator(args.estimator_path)
    assert_pr217_estimator(estimator_path)
    actual_sha = estimator_git_sha(estimator_path)
    expected_sha = metadata.get("lattice_estimator_sha")
    if expected_sha and actual_sha != expected_sha:
        print(
            "estimator SHA mismatch: "
            f"golden expects {format_estimator_sha(expected_sha)}, "
            f"got {format_estimator_sha(actual_sha)}",
            file=sys.stderr,
        )
        return 1

    SIS, RC, log, oo, _RealField = load_estimator(estimator_path)
    rows = load_rows(args.golden)
    if not any(row.get("trust") == TRUSTED for row in rows):
        print("FAIL no trusted rows in golden CSV", file=sys.stderr)
        return 1
    failures: list[str] = []
    checked = 0
    skipped_fragile = 0

    for row_index, expected in enumerate(sorted(rows, key=fixed_row_key), start=1):
        trust = expected.get("trust", TRUSTED)
        if trust == FRAGILE:
            skipped_fragile += 1
            continue
        if trust != TRUSTED:
            failures.append(f"row {row_index}: unknown trust value")
            continue

        actual = estimate_fixed_infinity_cell(
            SIS,
            RC,
            log,
            oo,
            family=expected["family"],
            d=int(expected["d"]),
            rank=int(expected["rank"]),
            width=int(expected["width"]),
            coeff_linf_bound=int(expected["coeff_linf_bound"]),
            beta=int(expected["beta_input"]),
            zeta=int(expected["zeta_input"]),
            target_bits=float(expected["target_bits"]),
        )
        checked += 1

        for field in FLOAT_FIELDS:
            failure = compare_float(field, expected[field], actual[field])
            if failure is not None:
                failures.append(f"row {row_index}: {field} mismatch")
        for field in INT_FIELDS:
            if expected[field] != actual[field]:
                failures.append(f"row {row_index}: {field} mismatch")

    if checked == 0:
        failures.append("no trusted fixed infinity cells checked")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        print(f"{len(failures)} fixed infinity golden check(s) failed", file=sys.stderr)
        return 1

    print(
        f"OK: {checked} trusted fixed infinity cell(s) match PR217 @ "
        f"{format_estimator_sha(actual_sha)}; "
        f"skipped {skipped_fragile} fragile cell(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
