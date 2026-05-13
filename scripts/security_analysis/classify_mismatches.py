#!/usr/bin/env python3
"""Classify rank mismatches between planner-stored and recomputed minimum ranks.

We separate:
  - `under_floor`: planner stored rank STRICTLY LESS than recomputed minimum.
    These are potential 128-bit floor violations.
  - `over_floor`:  planner stored rank STRICTLY GREATER than recomputed minimum.
    Overly conservative, not unsafe.

For under_floor cases we also report:
  - whether the preset is tensor or flat (the suspected bug is tensor-only),
  - the role (A vs B vs D),
  - the bucket and width involved,
  - the gap to the next rank's max width.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from extract_params import (
    load_sis_table, ceil_supported_collision, PRESET_FAMILY,
    extraction_linf,
)

HERE = Path(__file__).resolve().parent
TABLE = load_sis_table()

with (HERE / "params.json").open() as f:
    data = json.load(f)


under_floor = []
over_floor = []
per_role_under = defaultdict(int)
per_preset_under = defaultdict(int)

for preset, content in data.items():
    family_shape = content["family"]["shape"]
    for fold in content["unique_folds"]:
        d = fold["key"]["d"]
        for role, bucket_key, width_key, rank_key, stored_rank_key in [
            ("a", "a_bucket", "a_width", "a_rank_required", "n_a"),
            ("b", "b_bucket", "b_width", "b_rank_required", "n_b"),
            ("d", "d_bucket", "d_width", "d_rank_required", "n_d"),
        ]:
            stored = fold["key"][stored_rank_key]
            computed = fold[rank_key]
            if computed is None or stored == computed:
                continue
            record = {
                "preset": preset,
                "role": role,
                "stored": stored,
                "computed": computed,
                "shape": family_shape,
                "bucket": fold[bucket_key],
                "width": fold[width_key],
                "fold": fold["key"],
                "occurrences": fold["occurrences"],
                "max_num_vars_range": fold["max_num_vars_range"],
            }
            if computed > stored:
                under_floor.append(record)
                per_role_under[role] += 1
                per_preset_under[preset] += 1
            else:
                over_floor.append(record)


print(f"UNDER_FLOOR mismatches (computed > stored): {len(under_floor)}")
print(f"OVER_FLOOR  mismatches (computed < stored): {len(over_floor)}")
print()
print("UNDER_FLOOR by preset:")
for preset, count in sorted(per_preset_under.items()):
    print(f"  {preset}: {count}")
print()
print("UNDER_FLOOR by role:")
for role, count in sorted(per_role_under.items()):
    print(f"  {role}: {count}")
print()


print("UNDER_FLOOR by (shape, preset, role):")
buckets = defaultdict(list)
for r in under_floor:
    buckets[(r["shape"], r["preset"], r["role"])].append(r)
for key in sorted(buckets):
    print(f"  {key}: {len(buckets[key])}")

print()
print("All UNDER_FLOOR records (first 40):")
for r in under_floor[:40]:
    print(f"  preset={r['preset']} role={r['role']} "
          f"stored={r['stored']} computed={r['computed']} "
          f"bucket={r['bucket']} width={r['width']} "
          f"occ={r['occurrences']} mvars={r['fold']['m_vars']} rvars={r['fold']['r_vars']} "
          f"lb={r['fold']['log_basis']} dcommit={r['fold']['delta_commit']}")

print()
# For each under_floor record, compute the actual extraction-aware width
# threshold for stored rank vs the actual width. That tells us by how much
# the SIS floor is exceeded.
print("Width vs (stored_rank @ tensor bucket) max_width comparison:")
for r in under_floor[:20]:
    d = r["fold"]["d"]
    bucket = r["bucket"]
    if bucket is None:
        continue
    widths = TABLE.get((d, bucket))
    if widths is None:
        continue
    stored = r["stored"]
    if stored >= 1 and stored <= len(widths):
        stored_max_width = widths[stored - 1]
        ratio = r["width"] / stored_max_width if stored_max_width else float("inf")
        print(f"  D={d} bucket={bucket} stored_rank={stored} "
              f"max_width={stored_max_width} actual_width={r['width']} "
              f"OVER by {ratio:.2f}x ({r['preset']}/{r['role']})")
