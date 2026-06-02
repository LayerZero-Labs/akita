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
# Criterion truncates each path component to 64 chars (`MAX_DIRECTORY_NAME_LEN`).

# group dir: field_arith_{family}_{latency_chain|throughput_stream}_{label}_w{width}
GROUP_RE = re.compile(
    r"^field_arith_(?P<family>[^_]+)_(?P<kind_path>latency_chain|throughput_stream)_(?P<label>.+)_w(?P<width>\d+)$"
)
BENCH_RE = re.compile(
    r"^(?P<kind>scalar|packed)_(?P<op>[a-z_]+)_(?:chain|stream)_"
)

# Short ext4 labels (see crates/akita-pcs/benches/field_arith/ext4.rs). rs = ring_subfield (default).
AKITA_FP4_SHORT: dict[str, tuple[str, str, str]] = {
    "m31_rs_fp4": ("mersenne31", "4", "ring_subfield"),
    "m31_tw_fp4": ("mersenne31", "4", "tower"),
    "m31_pw_fp4": ("mersenne31", "4", "power"),
    "p31o19_rs_fp4": ("prime31_offset19", "4", "ring_subfield"),
    "p31o19_tw_fp4": ("prime31_offset19", "4", "tower"),
    "p31o19_pw_fp4": ("prime31_offset19", "4", "power"),
    "p32o99_rs_fp4": ("prime32_offset99", "4", "ring_subfield"),
    "p32o99_tw_fp4": ("prime32_offset99", "4", "tower"),
    "p32o99_pw_fp4": ("prime32_offset99", "4", "power"),
}

BASIS_RANK = {"ring_subfield": 0, "tower": 1, "power": 2, "": 3}


def parse_label(label: str) -> tuple[str, str, str, str]:
    """Return (library, field, ext_degree, basis)."""
    if label in AKITA_FP4_SHORT:
        field, ext_degree, basis = AKITA_FP4_SHORT[label]
        return "akita", field, ext_degree, basis

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
    elif label.endswith("_rs_fp4"):
        ext_degree = "4"
        basis = "ring_subfield"
        field = label.removesuffix("_rs_fp4")

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
    skipped_groups: list[str] = []
    pattern = f"**/{baseline}/estimates.json"
    for est_path in criterion_root.glob(pattern):
        rel = est_path.relative_to(criterion_root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        bench_id = parts[-3]
        group_dir = parts[-4]
        gm = GROUP_RE.match(group_dir)
        if gm is None:
            if group_dir.startswith("field_arith_"):
                skipped_groups.append(group_dir)
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

    if skipped_groups:
        unique = sorted(set(skipped_groups))
        print(
            f"warning: {len(unique)} group dirs did not match (often Criterion 64-char truncation). "
            f"Re-bench with short ext4 labels. Example: {unique[0]}",
            file=sys.stderr,
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
                    BASIS_RANK.get(r["basis"], 9),
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
        "Akita fp4 rows use **ring_subfield** as the default basis (tower/power are secondary).",
        "Highlighted: Akita ext4 ring_subfield vs Plonky3 ext5 (128-bit-equivalent over 31-bit base).",
        "",
        "`workload`: `latency_chain` = dependent critical path; `throughput_stream` = parallel streams.",
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
            BASIS_RANK.get(x["basis"], 9),
            x["op"],
        ),
    ):
        highlight = ""
        if r["library"] == "akita" and r["ext_degree"] == "4" and r["basis"] == "ring_subfield":
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
