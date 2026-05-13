#!/usr/bin/env python3
"""Summarize unique (D, bucket, rank, width) SIS quadruples and stored-rank mismatches.

Reads `params.json` from extract_params.py and prints:
- The unique set of SIS quadruples actually hit across all six presets.
- The "tightest" quadruple per (D, bucket, rank) — the one closest to the
  next bucket/rank breakpoint, since that has the smallest MSIS margin.
- Every entry where the planner-stored rank disagrees with the recomputed
  minimum rank (would indicate a planner or table bug).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    with (HERE / "params.json").open() as f:
        data = json.load(f)

    quadruples = set()
    derivation_mismatches = []  # planner stored != recomputed-under-derivation-shape
    production_mismatches = []  # planner stored != recomputed-under-production-shape
    per_d = defaultdict(list)

    for preset, content in data.items():
        for fold in content["unique_folds"]:
            d = fold["key"]["d"]
            roles = [
                ("a", "a_bucket_production", "a_width",
                 "a_rank_required_production", "a_rank_required_derivation", "n_a"),
                ("b", "b_bucket", "b_width",
                 "b_rank_required", "b_rank_required", "n_b"),
                ("d", "d_bucket", "d_width",
                 "d_rank_required", "d_rank_required", "n_d"),
            ]
            for role, bucket_key, width_key, prod_rank_key, deriv_rank_key, stored_rank_key in roles:
                bucket = fold[bucket_key]
                width = fold[width_key]
                prod_rank = fold[prod_rank_key]
                deriv_rank = fold[deriv_rank_key]
                stored = fold["key"][stored_rank_key]
                if bucket is None or prod_rank is None:
                    print(f"UNCOVERED: preset={preset} role={role} fold={fold['key']} "
                          f"width={width} bucket={bucket}")
                    continue
                quadruples.add((d, bucket, prod_rank, width))
                per_d[(d, bucket, prod_rank)].append((preset, role, width, fold))
                if deriv_rank != stored:
                    derivation_mismatches.append(
                        (preset, role, fold, deriv_rank, stored))
                if prod_rank != stored:
                    production_mismatches.append(
                        (preset, role, fold, prod_rank, stored))

    print(f"Total unique (D, bucket, rank, width) quadruples: {len(quadruples)}")
    print(f"Unique (D, bucket, rank) triples: {len(per_d)}")
    print()
    print("Per-triple summary (width range, number of widths, max width):")
    for triple in sorted(per_d):
        widths = sorted({w for _, _, w, _ in per_d[triple]})
        d, bucket, rank = triple
        print(f"  D={d} bucket={bucket} rank={rank}: "
              f"widths min={min(widths)} max={max(widths)} count={len(widths)}")

    print()
    print(f"DERIVATION-SHAPE mismatches: {len(derivation_mismatches)} entries where "
          "stored rank disagrees with rank recomputed under the SHAPE THE PLANNER SAW")
    print(f"  (these would indicate a planner-vs-table inconsistency.)")
    if derivation_mismatches:
        for preset, role, fold, computed, stored in derivation_mismatches[:20]:
            sign = ">" if computed > stored else "<"
            print(f"  preset={preset} role={role} stored={stored} {sign} "
                  f"recomputed={computed}  step_index={fold['key']['step_index']} "
                  f"shape_seen={fold['derivation_shape']} width={fold[role+'_width']}")
    print()
    print(f"PRODUCTION-SHAPE mismatches: {len(production_mismatches)} entries where "
          "stored rank is BELOW the rank required at the PRODUCTION runtime shape")
    print("  (these indicate the planner picked a rank that is INSUFFICIENT for the "
          "actual tensor extraction the runtime applies. Each is a 128-bit floor "
          "violation candidate that must be confirmed with the lattice estimator.)")
    under = [m for m in production_mismatches if m[3] > m[4]]
    over = [m for m in production_mismatches if m[3] < m[4]]
    print(f"  under_floor (stored < required): {len(under)}")
    print(f"  over_provisioned (stored > required): {len(over)}")
    if under:
        print()
        print("  First 30 under_floor cases (sorted by preset, descending shape impact):")
        sorted_under = sorted(under, key=lambda m: (-m[3] + m[4], m[0]))
        for preset, role, fold, required, stored in sorted_under[:30]:
            print(f"    preset={preset} role={role} stored={stored} < required={required} "
                  f"step={fold['key']['step_index']} D={fold['key']['d']} "
                  f"production_shape={fold['production_shape']} "
                  f"width={fold[role + '_width']} "
                  f"derivation_bucket={fold.get(role + '_bucket_derivation', fold.get(role + '_bucket'))} "
                  f"production_bucket={fold.get(role + '_bucket_production', fold.get(role + '_bucket'))}")

    # Emit a flat list of (D, bucket, rank, width) for the estimator script.
    out = sorted(quadruples)
    with (HERE / "quadruples.json").open("w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Wrote {len(out)} quadruples to quadruples.json")


if __name__ == "__main__":
    main()
