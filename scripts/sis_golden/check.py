#!/usr/bin/env python3
"""Replay committed golden SIS width cells against the pinned lattice-estimator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
GOLDEN_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify scripts/sis_golden/golden.csv.")
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_DIR / "golden.csv",
        help="Golden CSV path (default: scripts/sis_golden/golden.csv).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=GOLDEN_DIR / "metadata.json",
        help="Golden metadata path.",
    )
    parser.add_argument("--estimator-path", help="Optional lattice-estimator override.")
    return parser.parse_args()


def load_golden(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(SCRIPTS))
    from gen_sis_table import (  # noqa: WPS433
        binary_search_max_width,
        estimator_git_sha,
        locate_estimator,
        load_estimator,
    )

    metadata = json.loads(args.metadata.read_text())
    estimator_path = locate_estimator(args.estimator_path)
    pinned_sha = metadata.get("lattice_estimator_sha")
    actual_sha = estimator_git_sha(estimator_path)
    if pinned_sha and actual_sha != pinned_sha:
        print(
            f"estimator SHA mismatch: golden expects {pinned_sha}, got {actual_sha}",
            file=sys.stderr,
        )
        return 1

    rows = load_golden(args.golden)
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (int(row["q"]), int(row["d"]), int(row["collision"]))
        grouped[key].append(row)

    SIS, RC, log, oo = load_estimator(estimator_path)
    failures = 0
    for (q, d, collision), group in sorted(grouped.items()):
        for row in sorted(group, key=lambda r: int(r["rank"])):
            rank = int(row["rank"])
            expected = int(row["max_width"])
            target_bits = float(row["target_bits"])
            search_cap = int(row["search_cap"])
            actual = binary_search_max_width(
                SIS, RC, log, oo,
                q, d, rank, collision,
                target_bits, search_cap,
            )
            if actual != expected:
                failures += 1
                print(
                    f"FAIL q={q} d={d} collision={collision} rank={rank}: "
                    f"expected max_width={expected}, got {actual}",
                    file=sys.stderr,
                )

    if failures:
        print(f"{failures} golden cell(s) failed", file=sys.stderr)
        return 1

    print(f"OK: {len(rows)} golden cell(s) match pinned estimator @ {actual_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
