#!/usr/bin/env python3
"""Run lattice-estimator on specific (D, rank, width, collision_inf) SIS
instances to quantify the actual MSIS hardness of the production schedules.

Usage:
    sage -python scripts/security_analysis/run_estimator.py [--quick]

Replays the lattice-estimator at the parameter quadruples that the planner's
schedule tables actually use. Produces a JSON document with one record per
quadruple, including:
  - n = rank * D
  - m = width * D
  - length_bound = sqrt(m) * collision_inf (Euclidean)
  - estimated MSIS hardness in bits

Each instance models one SIS query that an extractor adversary would mount
against the corresponding (A, B, D) commitment role at the schedule's chosen
rank. The bound matches the convention used by scripts/gen_sis_table.py and
sis_floor.rs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ESTIMATOR_DEFAULT = Path.home() / "GitHub" / "lattice-estimator"

Q = (1 << 128) - 275
Q_LABEL = "2^128 - 275"


def load_estimator(path: Path):
    if not (path / "estimator" / "__init__.py").exists():
        raise SystemExit(
            f"lattice-estimator not found at {path}; set LATTICE_ESTIMATOR_PATH "
            "or pass --estimator-path"
        )
    sys.path.insert(0, str(path))
    from estimator import SIS
    from estimator.reduction import RC
    from sage.all import log
    return SIS, RC, log


def estimate_bits(SIS, RC, log, d: int, rank: int, width: int, collision: int) -> dict:
    """Return a dict { 'bits', 'tag', 'n', 'm', 'length_bound' } for one SIS instance."""
    n = rank * d
    m = width * d
    length_bound = (m ** 0.5) * collision
    out = SIS.lattice(
        SIS.Parameters(
            n=n, q=Q, m=m, length_bound=length_bound, norm=2, tag=f"production_d{d}_r{rank}_w{width}"
        ),
        red_cost_model=RC.BDGL16,
        red_shape_model="lgsa",
        log_level=0,
    )
    bits = float(log(out["rop"], 2))
    return {
        "n": n,
        "m": m,
        "length_bound_log2": float(log(length_bound, 2)),
        "bits": bits,
    }


# These are the specific instances flagged by `summarize_quadruples` as
# "under-floor": planner-stored rank that falls below the SIS table cutoff at
# the production-shape collision bucket. The collision is the *exact* L_inf
# the extractor produces (not the ceiling bucket from the table), because
# the lattice estimator handles arbitrary bounds directly.
PRODUCTION_FLOOR_VIOLATIONS = [
    # (preset, role, D, rank_stored, width, collision_exact,
    #   "bucket_in_table", "rank_required_by_table")
    {
        "preset": "d32_full",
        "role": "A",
        "d": 32,
        "rank_stored": 2,
        "width": 20482,
        "collision_exact": 24,
        "bucket_in_table": 31,
        "rank_required_by_table": 3,
        "note": "Flat schedule; recursive layout-iteration bug, NOT a shape issue."
    },
    {
        "preset": "d128_full",
        "role": "A",
        "d": 128,
        "rank_stored": 1,
        "width": 335873,
        "collision_exact": 156,   # 3 * (4*omega=52) at tensor extraction
        "bucket_in_table": 255,
        "rank_required_by_table": 2,
        "note": "Tensor schedule; recursive derivation runs under shape=Flat."
    },
    {
        "preset": "d64_full",
        "role": "A",
        "d": 64,
        "rank_stored": 1,
        "width": 234,
        "collision_exact": 1080,  # 15 * (4*omega=72) at tensor extraction
        "bucket_in_table": 2047,
        "rank_required_by_table": 2,
        "note": "Tensor schedule; recursive derivation runs under shape=Flat."
    },
    # Also include the "fixed" rank (stored+1) for comparison.
    {
        "preset": "d32_full_FIXED",
        "role": "A",
        "d": 32,
        "rank_stored": 3,
        "width": 20482,
        "collision_exact": 24,
        "bucket_in_table": 31,
        "rank_required_by_table": 3,
        "note": "Same width but with rank bumped to satisfy SIS table."
    },
    {
        "preset": "d128_full_FIXED",
        "role": "A",
        "d": 128,
        "rank_stored": 2,
        "width": 335873,
        "collision_exact": 156,
        "bucket_in_table": 255,
        "rank_required_by_table": 2,
        "note": "Same width but with rank bumped to satisfy SIS table."
    },
    {
        "preset": "d64_full_FIXED",
        "role": "A",
        "d": 64,
        "rank_stored": 2,
        "width": 234,
        "collision_exact": 1080,
        "bucket_in_table": 2047,
        "rank_required_by_table": 2,
        "note": "Same width but with rank bumped to satisfy SIS table."
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator-path", default=os.environ.get("LATTICE_ESTIMATOR_PATH"))
    args = parser.parse_args()
    estimator_path = Path(args.estimator_path).expanduser() if args.estimator_path else ESTIMATOR_DEFAULT
    print(f"Loading lattice-estimator from {estimator_path}", file=sys.stderr)
    SIS, RC, log = load_estimator(estimator_path)

    results = []
    for case in PRODUCTION_FLOOR_VIOLATIONS:
        d = case["d"]
        rank = case["rank_stored"]
        width = case["width"]
        coll = case["collision_exact"]
        print(
            f"--> Estimating {case['preset']}/{case['role']}: "
            f"D={d}, rank={rank}, width={width}, exact_collision={coll}",
            file=sys.stderr,
        )
        t0 = time.time()
        try:
            bits_record = estimate_bits(SIS, RC, log, d, rank, width, coll)
            elapsed = time.time() - t0
            print(
                f"    bits={bits_record['bits']:.1f}  n={bits_record['n']}  m={bits_record['m']}  "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )
            case_out = dict(case)
            case_out.update(bits_record)
            case_out["elapsed_seconds"] = elapsed
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            case_out = dict(case)
            case_out["error"] = str(e)
        results.append(case_out)

    out = {
        "q": Q,
        "q_label": Q_LABEL,
        "model": {
            "red_cost": "BDGL16",
            "red_shape": "lgsa",
            "norm": "Euclidean (l2)",
            "length_bound_formula": "sqrt(m) * collision_inf",
        },
        "results": results,
    }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
