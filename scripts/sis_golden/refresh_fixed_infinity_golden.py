#!/usr/bin/env python3
"""Regenerate fixed-beta, fixed-zeta SIS infinity golden cells."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(GOLDEN_DIR))

from fixed_infinity_grid import (  # noqa: E402
    TARGET_BITS,
    fixed_infinity_work_items,
)
from infinity_core import (  # noqa: E402
    FAMILIES,
    FLOAT_FIELDS,
    INT_FIELDS,
    PINNED_LATTICE_ESTIMATOR_SHA,
    assert_pinned_estimator,
    estimate_fixed_infinity_cell,
    estimator_git_sha,
    estimator_remote_url,
    fixed_row_key,
    fragile_fixed_infinity_cell,
    load_estimator,
    locate_estimator,
)
from infinity_profile import (  # noqa: E402
    add_profile_arguments,
    default_metadata_path,
    default_output_path,
    profile_from_args,
)

FIELDNAMES = [
    "family",
    "q",
    "d",
    "rank",
    "width",
    "coeff_linf_bound",
    "beta_input",
    "zeta_input",
    "target_bits",
    *FLOAT_FIELDS,
    *INT_FIELDS,
    "security_met",
    "tiny_probability",
    "trust",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator-path", help="Path to pinned lattice-estimator checkout.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: fixed_infinity_golden.csv or profile-specific name).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata JSON path (default: matches --output basename).",
    )
    parser.add_argument("--target-bits", type=float, default=TARGET_BITS)
    add_profile_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = profile_from_args(args, zeta="fixed")
    output = args.output or default_output_path(
        GOLDEN_DIR / "fixed_infinity_golden.csv",
        profile,
    )
    metadata_path = args.metadata or default_metadata_path(output)

    estimator_path = locate_estimator(args.estimator_path)
    assert_pinned_estimator(estimator_path)
    SIS, RC, log, oo, _RealField = load_estimator(estimator_path)

    work = fixed_infinity_work_items()
    if not work:
        print("error: no fixed infinity golden work items", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    t0 = time.time()
    for index, item in enumerate(work, start=1):
        cell_t0 = time.time()
        try:
            row = estimate_fixed_infinity_cell(
                SIS,
                RC,
                log,
                oo,
                family=str(item["family"]),
                d=int(item["d"]),
                rank=int(item["rank"]),
                width=int(item["width"]),
                coeff_linf_bound=int(item["coeff_linf_bound"]),
                beta=int(item["beta"]),
                zeta=int(item["zeta"]),
                target_bits=args.target_bits,
                profile=profile,
            )
        except Exception as exc:  # noqa: BLE001
            row = fragile_fixed_infinity_cell(
                family=str(item["family"]),
                d=int(item["d"]),
                rank=int(item["rank"]),
                width=int(item["width"]),
                coeff_linf_bound=int(item["coeff_linf_bound"]),
                beta=int(item["beta"]),
                zeta=int(item["zeta"]),
                target_bits=args.target_bits,
                exc=exc,
            )
        elapsed = time.time() - cell_t0
        print(
            f"fixed_infinity {index}/{len(work)} completed ({elapsed:.1f}s)",
            file=sys.stderr,
        )
        rows.append(row)

    rows.sort(key=fixed_row_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "description": (
            f"{len(rows)} fixed-beta, fixed-zeta SIS infinity golden cells using "
            f"{profile.description_suffix()} and target_bits={args.target_bits}."
        ),
        "profile": profile.to_metadata(),
        "target_bits": args.target_bits,
        "lattice_estimator_remote": estimator_remote_url(estimator_path),
        "lattice_estimator_sha": estimator_git_sha(estimator_path),
        "expected_lattice_estimator_sha": PINNED_LATTICE_ESTIMATOR_SHA,
        "families": {family: {"q": q, "label": label} for family, (q, label) in FAMILIES.items()},
        "grid": {"cells": work},
        "rows": [
            {
                "family": row["family"],
                "d": int(row["d"]),
                "rank": int(row["rank"]),
                "width": int(row["width"]),
                "coeff_linf_bound": int(row["coeff_linf_bound"]),
                "beta_input": int(row["beta_input"]),
                "zeta_input": int(row["zeta_input"]),
                "trust": row["trust"],
            }
            for row in rows
        ],
    }
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")

    print(
        f"Wrote {len(rows)} fixed infinity cells to {output} "
        f"in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
