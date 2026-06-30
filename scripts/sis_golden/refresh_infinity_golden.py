#!/usr/bin/env python3
"""Regenerate SIS infinity-norm golden cells from lattice-estimator PR217."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(GOLDEN_DIR))

from infinity_core import (  # noqa: E402
    FAMILIES,
    FLOAT_FIELDS,
    INT_FIELDS,
    PROFILE,
    PR217_LATTICE_ESTIMATOR_SHA,
    assert_pr217_estimator,
    estimate_infinity_cell,
    estimator_git_sha,
    estimator_remote_url,
    fragile_infinity_cell,
    load_estimator,
    locate_estimator,
    row_key,
)
from infinity_grid import (  # noqa: E402
    INFINITY_COEFF_LINF_BOUNDS,
    INFINITY_FAMILIES,
    INFINITY_RANKS,
    INFINITY_RING_DIMS,
    INFINITY_WIDTH_FACTORS,
    TARGET_BITS,
    infinity_work_items,
)

FIELDNAMES = [
    "family",
    "q",
    "d",
    "rank",
    "width",
    "coeff_linf_bound",
    "target_bits",
    *FLOAT_FIELDS,
    *INT_FIELDS,
    "security_met",
    "tiny_probability",
    "trust",
    "notes",
]


def parse_csv_list(raw: str | None, *, cast=str):
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("empty comma-separated filter")
    return {cast(value) for value in values}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator-path", help="Path to lattice-estimator PR217 checkout.")
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLDEN_DIR / "infinity_golden.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=GOLDEN_DIR / "infinity_metadata.json",
        help="Metadata JSON path.",
    )
    parser.add_argument("--target-bits", type=float, default=TARGET_BITS)
    parser.add_argument("--families", help="Comma-separated family filter.")
    parser.add_argument("--dims", help="Comma-separated ring-dimension filter.")
    parser.add_argument("--ranks", help="Comma-separated rank filter.")
    parser.add_argument("--bounds", help="Comma-separated coefficient-Linf bound filter.")
    parser.add_argument("--widths", help="Comma-separated width filter.")
    parser.add_argument("--limit", type=int, help="Limit rows after filtering, for smoke tests.")
    return parser.parse_args()


def select_work(args: argparse.Namespace) -> list[dict[str, int | str]]:
    families = parse_csv_list(args.families)
    dims = parse_csv_list(args.dims, cast=int)
    ranks = parse_csv_list(args.ranks, cast=int)
    bounds = parse_csv_list(args.bounds, cast=int)
    widths = parse_csv_list(args.widths, cast=int)

    work = []
    for item in infinity_work_items():
        if families is not None and item["family"] not in families:
            continue
        if dims is not None and item["d"] not in dims:
            continue
        if ranks is not None and item["rank"] not in ranks:
            continue
        if bounds is not None and item["coeff_linf_bound"] not in bounds:
            continue
        if widths is not None and item["width"] not in widths:
            continue
        work.append(item)
    work.sort(key=lambda row: (str(row["family"]), int(row["d"]), int(row["rank"]), int(row["width"]), int(row["coeff_linf_bound"])))
    if args.limit is not None:
        work = work[: args.limit]
    return work


def main() -> int:
    args = parse_args()
    estimator_path = locate_estimator(args.estimator_path)
    assert_pr217_estimator(estimator_path)
    SIS, RC, log, oo, _RealField = load_estimator(estimator_path)

    work = select_work(args)
    rows: list[dict[str, str]] = []
    t0 = time.time()
    for index, item in enumerate(work, start=1):
        cell_t0 = time.time()
        try:
            row = estimate_infinity_cell(
                SIS,
                RC,
                log,
                oo,
                family=str(item["family"]),
                d=int(item["d"]),
                rank=int(item["rank"]),
                width=int(item["width"]),
                coeff_linf_bound=int(item["coeff_linf_bound"]),
                target_bits=args.target_bits,
            )
        except Exception as exc:  # noqa: BLE001 - fragile goldens record estimator failures.
            row = fragile_infinity_cell(
                family=str(item["family"]),
                d=int(item["d"]),
                rank=int(item["rank"]),
                width=int(item["width"]),
                coeff_linf_bound=int(item["coeff_linf_bound"]),
                target_bits=args.target_bits,
                exc=exc,
            )
        elapsed = time.time() - cell_t0
        print(
            "infinity "
            f"{index}/{len(work)} family={row['family']} d={row['d']} "
            f"rank={row['rank']} width={row['width']} "
            f"linf={row['coeff_linf_bound']} rop_log2={row['rop_log2']} "
            f"prob_log2={row['prob_log2']} tiny={row['tiny_probability']} "
            f"trust={row['trust']} "
            f"({elapsed:.1f}s)",
            file=sys.stderr,
        )
        rows.append(row)

    rows.sort(key=row_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "description": (
            f"{len(rows)} SIS infinity golden cells using norm=oo, ADPS16, LGSA, "
            f"full zeta optimizer, and target_bits={args.target_bits}."
        ),
        "profile": PROFILE,
        "target_bits": args.target_bits,
        "lattice_estimator_remote": estimator_remote_url(estimator_path),
        "lattice_estimator_sha": estimator_git_sha(estimator_path),
        "expected_lattice_estimator_sha": PR217_LATTICE_ESTIMATOR_SHA,
        "families": {family: {"q": q, "label": label} for family, (q, label) in FAMILIES.items()},
        "grid": {
            "families": INFINITY_FAMILIES,
            "ring_dims": INFINITY_RING_DIMS,
            "ranks": INFINITY_RANKS,
            "coeff_linf_bounds": INFINITY_COEFF_LINF_BOUNDS,
            "width_factors": INFINITY_WIDTH_FACTORS,
        },
        "rows": [
            {
                "family": row["family"],
                "d": int(row["d"]),
                "rank": int(row["rank"]),
                "width": int(row["width"]),
                "coeff_linf_bound": int(row["coeff_linf_bound"]),
                "tiny_probability": row["tiny_probability"] == "true",
                "trust": row["trust"],
            }
            for row in rows
        ],
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        f"Wrote {len(rows)} infinity cells to {args.output} "
        f"in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
