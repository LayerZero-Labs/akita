#!/usr/bin/env python3
"""Run lattice-estimator (SIS.lattice, BDGL16 + lgsa, q=2^128-275) at every
unique (D, bucket, rank, width) quadruple the regenerated production schedule
tables hit. Emit a JSON document mapping each quadruple to its MSIS bit count
and flag every entry below 128 bits.

Usage:
    sage -python scripts/security_analysis/run_estimator_all.py
        > scripts/security_analysis/estimator_all_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ESTIMATOR_PATH = Path(
    os.environ.get(
        "LATTICE_ESTIMATOR_PATH",
        str(Path.home() / "GitHub" / "lattice-estimator"),
    )
)

sys.path.insert(0, str(ESTIMATOR_PATH))
from estimator import SIS
from estimator.reduction import RC
from sage.all import log

Q = (1 << 128) - 275


def estimate_bits(d: int, rank: int, width: int, collision: int) -> float:
    n = rank * d
    m = width * d
    length_bound = (m ** 0.5) * collision
    out = SIS.lattice(
        SIS.Parameters(
            n=n, q=Q, m=m, length_bound=length_bound, norm=2,
            tag=f"d{d}_r{rank}_w{width}_c{collision}",
        ),
        red_cost_model=RC.BDGL16,
        red_shape_model="lgsa",
        log_level=0,
    )
    return float(log(out["rop"], 2))


def main() -> None:
    quadruples = json.load((HERE / "quadruples.json").open())
    print(f"Running estimator on {len(quadruples)} unique quadruples...", file=sys.stderr)
    results = []
    below_128 = []
    t_total = time.time()
    for i, (d, collision, rank, width) in enumerate(quadruples):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - t_total
            print(f"  {i}/{len(quadruples)} ({elapsed:.0f}s elapsed)", file=sys.stderr)
        try:
            bits = estimate_bits(d, rank, width, collision)
        except Exception as e:
            results.append({
                "d": d, "collision_bucket": collision, "rank": rank, "width": width,
                "error": str(e),
            })
            continue
        rec = {
            "d": d, "collision_bucket": collision, "rank": rank, "width": width,
            "n": rank * d, "m": width * d, "bits_msis_lattice": bits,
        }
        results.append(rec)
        if bits < 128:
            below_128.append(rec)

    total_elapsed = time.time() - t_total
    print(
        f"Estimator pass complete: {len(results)} quadruples in {total_elapsed:.1f}s. "
        f"Below 128 bits: {len(below_128)}",
        file=sys.stderr,
    )

    summary = {
        "q": Q,
        "q_label": "2^128 - 275",
        "model": {
            "estimator": "SIS.lattice",
            "red_cost": "BDGL16",
            "red_shape": "lgsa",
            "norm": "Euclidean (l2)",
            "length_bound_formula": "sqrt(m) * collision_inf",
        },
        "num_quadruples": len(results),
        "num_below_128_bits": len(below_128),
        "below_128_bits": below_128,
        "results": results,
    }
    json.dump(summary, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
