#!/usr/bin/env python3
"""Aggregate Criterion field_arith baselines into bench-data CSV + markdown pivot."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Criterion 0.5 WallTime stores estimate point values in nanoseconds (see `WallTime::to_f64`).

# Criterion 0.5 flattens `/` in group and bench ids into `_`.
# group dir: field_arith_{family}_{latency_chain|throughput_stream}_{label}_w{width}
GROUP_RE = re.compile(
    r"^field_arith_(?P<family>[^_]+)_(?P<kind_path>latency_chain|throughput_stream)_(?P<label>.+)_w(?P<width>\d+)$"
)
# bench dir: scalar_add_chain_2048_ns_per_op or packed_mul_chain_512x4_ns_lane
BENCH_RE = re.compile(
    r"^(?P<kind>scalar|packed)_(?P<op>[a-z_]+)_(?:chain|stream)_"
)


def parse_label(label: str) -> tuple[str, str, str, str]:
    """Return (library, field, ext_degree, basis)."""
    library = "plonky3" if label.startswith("p3_") else "akita"
    if label.startswith("p3_"):
        field = label.removeprefix("p3_")
    else:
        field = label

    ext_degree = ""
    basis = ""
    if field.endswith("_ext4"):
        ext_degree = "4"
        field = field.removesuffix("_ext4")
    elif field.endswith("_ext5"):
        ext_degree = "5"
        field = field.removesuffix("_ext5")
    elif "_tower_fp4" in label:
        ext_degree = "4"
        basis = "tower"
        field = label.removesuffix("_tower_fp4")
    elif "_power_fp4" in label:
        ext_degree = "4"
        basis = "power"
        field = label.removesuffix("_power_fp4")
    elif "_ring_subfield_fp4" in label:
        ext_degree = "4"
        basis = "ring_subfield"
        field = label.removesuffix("_ring_subfield_fp4")

    return library, field, ext_degree, basis


def median_ns(estimates_path: Path) -> tuple[float, float, float]:
    data = json.loads(estimates_path.read_text())
    median = data["median"]
    ci = median["confidence_interval"]
    return (
        float(median["point_estimate"]),
        float(ci["lower_bound"]),
        float(ci["upper_bound"]),
    )


def collect_rows(criterion_root: Path, baseline: str, arch: str, simd: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = f"**/{baseline}/estimates.json"
    for est_path in criterion_root.glob(pattern):
        rel = est_path.relative_to(criterion_root)
        # {group_dir}/{bench_dir}/{baseline}/estimates.json
        parts = rel.parts
        if len(parts) < 4:
            continue
        bench_id = parts[-3]
        group_dir = parts[-4]
        gm = GROUP_RE.match(group_dir)
        if gm is None:
            continue
        bm = BENCH_RE.match(bench_id)
        if bm is None:
            continue

        label = gm.group("label")
        width = int(gm.group("width"))
        library, field, ext_degree, basis = parse_label(label)
        kind_path = gm.group("kind_path")
        op = bm.group("op")
        kind = bm.group("kind")

        # Criterion benches already normalize packed rows to ns/lane via
        # duration_per_logical_op(..., iters * WIDTH).
        mean_ns, lower_ns, upper_ns = median_ns(est_path)

        rows.append(
            {
                "library": library,
                "field": field,
                "ext_degree": ext_degree,
                "basis": basis,
                "op": op,
                "workload": kind_path,
                "vectorization": kind,
                "arch": arch,
                "simd": simd,
                "width": str(width),
                "family": gm.group("family"),
                "ns_per_op_or_lane": f"{mean_ns:.3f}",
                "lower": f"{lower_ns:.3f}",
                "upper": f"{upper_ns:.3f}",
                "group": group_dir,
                "bench_id": bench_id,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "library",
        "field",
        "ext_degree",
        "basis",
        "op",
        "workload",
        "vectorization",
        "arch",
        "simd",
        "width",
        "ns_per_op_or_lane",
        "lower",
        "upper",
        "family",
        "group",
        "bench_id",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda r: (
                    r["library"],
                    r["field"],
                    r["op"],
                    r["workload"],
                    r["vectorization"],
                ),
            )
        )


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headline_ops = ("mul", "square")
    filtered = [
        r
        for r in rows
        if r["vectorization"] == "packed"
        and r["op"] in headline_ops
        and r["family"] in ("ext4", "ext5")
    ]

    lines = [
        "# Field microbench (packed extension, headline ops)",
        "",
        "Highlighted rows: Akita degree-4 (`ext4`, `mersenne31_*_fp4`) vs Plonky3 degree-5 (`ext5`).",
        "",
        "`workload`: `latency_chain` is a dependent op chain (critical-path latency); "
        "`throughput_stream` is parallel streams with independent ops.",
        "",
        "| library | field | ext | basis | op | workload | arch | simd | w | ns/lane |",
        "|---------|-------|-----|-------|----|----------|------|------|---|--------:|",
    ]
    for r in sorted(
        filtered,
        key=lambda x: (
            x["simd"],
            x["workload"],
            x["library"],
            x["field"],
            x["op"],
        ),
    ):
        highlight = ""
        if r["library"] == "akita" and r["ext_degree"] == "4":
            highlight = " **"
        if r["library"] == "plonky3" and r["ext_degree"] == "5":
            highlight = " **"
        lines.append(
            f"| {r['library']} | {r['field']} | {r['ext_degree']} | {r['basis']} | {r['op']} | "
            f"{r['workload']} | {r['arch']} | {r['simd']} | {r['width']} | "
            f"{r['ns_per_op_or_lane']}{highlight} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path("target/criterion"),
        help="Criterion output root",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="NAME:ARCH:SIMD",
        help="Baseline tag, e.g. neon:aarch64:neon",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("bench-data/field-microbench.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("bench-data/field-microbench.md"),
    )
    args = parser.parse_args()

    if not args.baseline:
        print("error: pass at least one --baseline NAME:ARCH:SIMD", file=sys.stderr)
        return 1

    all_rows: list[dict[str, str]] = []
    for spec in args.baseline:
        name, arch, simd = spec.split(":")
        rows = collect_rows(args.criterion_dir, name, arch, simd)
        if not rows:
            print(f"warning: no rows for baseline {name}", file=sys.stderr)
        all_rows.extend(rows)

    write_csv(args.out_csv, all_rows)
    write_markdown(args.out_md, all_rows)
    print(f"wrote {len(all_rows)} rows to {args.out_csv} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
