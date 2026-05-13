#!/usr/bin/env python3
"""Aggregate per-preset MSIS bit statistics from estimator_all_results.json
and params.json. Emits a Markdown table for the security_analysis.md."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

with (HERE / "estimator_all_results.json").open() as f:
    est = json.load(f)
with (HERE / "params.json").open() as f:
    params = json.load(f)

# Build a quadruple → bits lookup.
bits_by_quad = {}
for r in est["results"]:
    if "bits_msis_lattice" in r:
        key = (r["d"], r["collision_bucket"], r["rank"], r["width"])
        bits_by_quad[key] = r["bits_msis_lattice"]


per_preset_role_bits: dict[tuple[str, str], list[float]] = defaultdict(list)
per_preset_min: dict[str, tuple[float, dict]] = {}

for preset, content in params.items():
    family = content["family"]
    print(f"\n## {preset}")
    print(f"  D={family['d']}, shape={family['shape']}, "
          f"omega={family['omega']}, infinity_norm={family['infinity_norm']}, "
          f"effective_l1_mass={family['effective_l1_mass']}, "
          f"msis_extraction_degradation={family['msis_extraction_degradation']}")
    role_min = {}
    for fold in content["unique_folds"]:
        d = fold["key"]["d"]
        for role, bucket_key, width_key, rank_key in [
            ("A", "a_bucket_production", "a_width", "n_a"),
            ("B", "b_bucket", "b_width", "n_b"),
            ("D", "d_bucket", "d_width", "n_d"),
        ]:
            bucket = fold[bucket_key]
            width = fold[width_key]
            rank = fold["key"][rank_key]
            if bucket is None or width == 0 or rank == 0:
                continue
            quad = (d, bucket, rank, width)
            bits = bits_by_quad.get(quad)
            if bits is None:
                continue
            per_preset_role_bits[(preset, role)].append(bits)
            cur_min = role_min.get(role)
            if cur_min is None or bits < cur_min[0]:
                role_min[role] = (
                    bits,
                    {
                        "rank": rank,
                        "width": width,
                        "bucket": bucket,
                        "step_index": fold["key"]["step_index"],
                        "log_basis": fold["key"]["log_basis"],
                        "m_vars": fold["key"]["m_vars"],
                        "r_vars": fold["key"]["r_vars"],
                        "delta_open": fold["key"]["delta_open"],
                        "delta_commit": fold["key"]["delta_commit"],
                    },
                )
    for role in ["A", "B", "D"]:
        if role in role_min:
            bits, info = role_min[role]
            print(f"  min({role}) = {bits:.1f} bits at {info}")
    overall_bits = [
        bits
        for (p, _), bs in per_preset_role_bits.items()
        for bits in bs
        if p == preset
    ]
    if overall_bits:
        per_preset_min[preset] = (min(overall_bits), max(overall_bits))

print()
print("## Summary table (min / median / max MSIS bits per preset)")
print()
print("| Preset | Min (A) | Min (B) | Min (D) | Min overall | Max overall |")
print("|---|---:|---:|---:|---:|---:|")
for preset in sorted(per_preset_role_bits, key=lambda x: x[0]):
    pass  # filter
for preset in sorted({p for (p, _) in per_preset_role_bits}):
    a_bits = per_preset_role_bits.get((preset, "A"), [])
    b_bits = per_preset_role_bits.get((preset, "B"), [])
    d_bits = per_preset_role_bits.get((preset, "D"), [])
    a_min = min(a_bits) if a_bits else float("nan")
    b_min = min(b_bits) if b_bits else float("nan")
    d_min = min(d_bits) if d_bits else float("nan")
    overall_min, overall_max = per_preset_min[preset]
    print(
        f"| `{preset}` | {a_min:.1f} | {b_min:.1f} | {d_min:.1f} | "
        f"**{overall_min:.1f}** | {overall_max:.1f} |"
    )
