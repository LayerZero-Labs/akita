#!/usr/bin/env python3
"""Regenerate scripts/sis_golden/golden.csv from the pinned lattice-estimator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
GOLDEN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(GOLDEN_DIR))

from gen_sis_table import (  # noqa: E402
    FAMILIES,
    binary_search_max_width,
    default_search_cap,
    estimator_git_sha,
    estimator_remote_url,
    locate_estimator,
    load_estimator,
)
from grid import GOLDEN_RANKS, GOLDEN_WORK_ITEMS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh scripts/sis_golden/golden.csv.")
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLDEN_DIR / "golden.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=GOLDEN_DIR / "metadata.json",
        help="Metadata JSON path to update.",
    )
    parser.add_argument("--estimator-path", help="Optional lattice-estimator override.")
    parser.add_argument(
        "--target-bits", type=float, default=128.0, help="Minimum security bits."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimator_path = locate_estimator(args.estimator_path)
    SIS, RC, log, oo = load_estimator(estimator_path)

    rows: list[dict[str, object]] = []
    t0 = time.time()
    for family, d, collision in GOLDEN_WORK_ITEMS:
        q, _, _, _ = FAMILIES[family]
        search_cap = default_search_cap(d)
        for rank in GOLDEN_RANKS:
            cell_t0 = time.time()
            max_width = binary_search_max_width(
                SIS, RC, log, oo,
                q, d, rank, collision,
                args.target_bits, search_cap,
            )
            elapsed = time.time() - cell_t0
            print(
                f"family={family} d={d} collision={collision} rank={rank} "
                f"max_width={max_width} ({elapsed:.1f}s)",
                file=sys.stderr,
            )
            rows.append({
                "q": q,
                "d": d,
                "collision": collision,
                "rank": rank,
                "max_width": max_width,
                "target_bits": args.target_bits,
                "search_cap": search_cap,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "q", "d", "collision", "rank", "max_width", "target_bits", "search_cap",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "description": (
            f"{len(GOLDEN_WORK_ITEMS)} work items × ranks {GOLDEN_RANKS} "
            f"= {len(rows)} golden cells across q32/q64/q128."
        ),
        "lattice_estimator_remote": estimator_remote_url(estimator_path),
        "lattice_estimator_sha": estimator_git_sha(estimator_path),
        "golden_ranks": GOLDEN_RANKS,
        "work_items": [
            {"family": family, "d": d, "collision": collision}
            for family, d, collision in GOLDEN_WORK_ITEMS
        ],
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        f"Wrote {len(rows)} cells to {args.output} in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
