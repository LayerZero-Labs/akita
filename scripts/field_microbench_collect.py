#!/usr/bin/env python3
"""Aggregate Criterion field_arith baselines into canonical CSV/markdown artifacts.

The collector intentionally treats the Criterion estimate tree as measurement data
only. Machine, toolchain, and build provenance are supplied per saved baseline so
copied remote runs do not accidentally inherit the local host's configuration.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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

# Per-row provenance columns. Each measurement carries the commit and capture
# time of the run that produced *that row*, so individual benches can be
# refreshed without re-running or misdating the rest of the table.
PROV_FIELDS = ("git_commit", "captured_at_utc")

BASIS_RANK = {"ring_subfield": 0, "tower": 1, "power": 2, "": 3}
HEADLINE_OPS = ("mul", "square")
RING_FOCUS_OPS = ("add", "sub", "mul", "mul_self", "square")
WORKLOAD_RANK = {"latency_chain": 0, "throughput_stream": 1}
DEFAULT_OUT_CSV = Path("bench-data/field-microbench.csv")
DEFAULT_OUT_MD = Path("bench-data/field-microbench.md")
DEFAULT_OUT_META = Path("bench-data/field-microbench-meta.json")


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    arch: str
    simd: str
    machine_config: str = ""

    @classmethod
    def parse(cls, spec: str) -> "BaselineSpec":
        parts = spec.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(
                f"baseline spec must be NAME:ARCH:SIMD[:MACHINE_CONFIG], got {spec!r}"
            )
        name, arch, simd = parts[:3]
        machine_config = parts[3] if len(parts) == 4 else ""
        if not name or not arch or not simd:
            raise ValueError(f"baseline spec has an empty component: {spec!r}")
        return cls(name=name, arch=arch, simd=simd, machine_config=machine_config)


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def first_cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        model = run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        hardware = run_text(["sysctl", "-n", "hw.model"])
        ncpu = run_text(["sysctl", "-n", "hw.ncpu"])
        pieces = [p for p in (model, hardware, f"{ncpu} logical CPUs" if ncpu else "") if p]
        return "; ".join(pieces)

    if system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(errors="ignore").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        lscpu = run_text(["lscpu"])
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def cpu_features() -> str:
    system = platform.system()
    if system == "Darwin":
        features = [
            run_text(["sysctl", "-n", "machdep.cpu.features"]),
            run_text(["sysctl", "-n", "machdep.cpu.leaf7_features"]),
        ]
        return " ".join(f for f in features if f).strip()

    if system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(errors="ignore").splitlines():
                if line.startswith(("flags", "Features")):
                    return line.split(":", 1)[1].strip()
    return ""


def parse_target_cpu(rustflags: str) -> str:
    match = re.search(r"(?:^|\s)-C\s*target-cpu=([^\s]+)", rustflags)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|\s)-Ctarget-cpu=([^\s]+)", rustflags)
    if match:
        return match.group(1)
    return ""


def kernel_config() -> str:
    info = platform.uname()
    pieces = [info.system, info.release, info.version, info.machine, info.processor]
    return " ".join(piece for piece in pieces if piece)


def capture_machine_metadata(args: argparse.Namespace) -> dict[str, str]:
    rustflags = args.rustflags if args.rustflags is not None else os.environ.get("RUSTFLAGS", "")
    target_cpu = args.target_cpu or parse_target_cpu(rustflags)
    cwd = Path.cwd()
    return {
        "baseline": args.baseline,
        "machine_config": args.machine_config,
        "arch": args.arch,
        "simd": args.simd,
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "kernel": kernel_config(),
        "cpu_model": first_cpu_model(),
        "cpu_features": cpu_features(),
        "rustc": run_text(["rustc", "--version"]),
        "rustc_verbose": run_text(["rustc", "--version", "--verbose"]),
        "cargo": run_text(["cargo", "--version"]),
        "rustflags": rustflags,
        "target_cpu": target_cpu,
        "criterion_baseline": args.baseline,
        "criterion_dir": str(args.criterion_dir),
        "bench_filter": args.bench_filter,
        "bench_command": args.bench_command,
        "git_commit": run_text(["git", "rev-parse", "HEAD"], cwd),
        "git_subject": run_text(["git", "log", "-1", "--format=%s"], cwd),
        "notes": args.notes,
    }


def parse_metadata_specs(specs: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"metadata spec must be BASELINE=PATH, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"metadata spec has an empty component: {spec!r}")
        paths[name] = Path(raw_path)
    return paths


def load_metadata(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(k): "" if v is None else str(v) for k, v in data.items()}


def baseline_metadata(
    specs: list[BaselineSpec],
    metadata_paths: dict[str, Path],
    warnings: list[str],
) -> dict[str, dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    for spec in specs:
        meta = {
            "baseline": spec.name,
            "machine_config": spec.machine_config,
            "arch": spec.arch,
            "simd": spec.simd,
            "target_cpu": "",
            "rustflags": "",
            "cpu_model": "",
            "cpu_features": "",
            "os": "",
            "kernel": "",
            "rustc": "",
            "cargo": "",
            "criterion_baseline": spec.name,
            "criterion_dir": "",
            "bench_filter": "",
            "bench_command": "",
            "git_commit": "",
            "git_subject": "",
            "captured_at_utc": "",
            "notes": "",
        }
        if spec.name in metadata_paths:
            loaded = load_metadata(metadata_paths[spec.name])
            meta.update(loaded)
            if loaded.get("baseline") not in (None, "", spec.name):
                warnings.append(
                    f"metadata for baseline {spec.name!r} says baseline "
                    f"{loaded.get('baseline')!r}"
                )
        else:
            warnings.append(
                f"baseline {spec.name!r} has no metadata JSON; machine/build "
                "columns are only the CLI spec"
            )
        meta["baseline"] = spec.name
        meta["arch"] = spec.arch
        meta["simd"] = spec.simd
        if spec.machine_config and not meta.get("machine_config"):
            meta["machine_config"] = spec.machine_config
        by_name[spec.name] = meta
    return by_name


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
    elif "_tower_fp4" in field:
        ext_degree = "4"
        basis = "tower"
        field = field.removesuffix("_tower_fp4")
    elif "_power_fp4" in field:
        ext_degree = "4"
        basis = "power"
        field = field.removesuffix("_power_fp4")
    elif "_ring_subfield_fp4" in field:
        ext_degree = "4"
        basis = "ring_subfield"
        field = field.removesuffix("_ring_subfield_fp4")
    elif field.endswith("_rs_fp4"):
        ext_degree = "4"
        basis = "ring_subfield"
        field = field.removesuffix("_rs_fp4")

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


def collect_rows(
    criterion_root: Path,
    spec: BaselineSpec,
    meta: dict[str, str],
    warnings: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    skipped_groups: list[str] = []
    pattern = f"**/{spec.name}/estimates.json"
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
                "baseline": spec.name,
                "machine_config": meta.get("machine_config", ""),
                "arch": spec.arch,
                "simd": spec.simd,
                "width": str(width),
                "unit": "ns/lane" if kind == "packed" else "ns/op",
                "family": gm.group("family"),
                "label": label,
                "ns_per_op_or_lane": f"{mean_ns:.3f}",
                "lower": f"{lower_ns:.3f}",
                "upper": f"{upper_ns:.3f}",
                "group": group_dir,
                "bench_id": bench_id,
            }
        )

    if skipped_groups:
        unique = sorted(set(skipped_groups))
        warnings.append(
            f"{len(unique)} group dirs did not match (often Criterion 64-char truncation). "
            f"Re-bench with short ext4 labels. Example: {unique[0]}"
        )
    return rows


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    """Identity of a single measurement, stable across re-captures."""
    return (
        row["baseline"],
        row["library"],
        row["field"],
        row["ext_degree"],
        row["basis"],
        row["op"],
        row["workload"],
        row["vectorization"],
        row["arch"],
        row["simd"],
        row["width"],
        row["family"],
    )


def measurement_value(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["ns_per_op_or_lane"], row["lower"], row["upper"])


def load_existing_csv(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row_key(row): row for row in csv.DictReader(f)}


def load_existing_meta_baselines(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    baselines = data.get("baselines") if isinstance(data, dict) else None
    return baselines if isinstance(baselines, dict) else {}


def stamp_provenance(
    fresh_rows: list[dict[str, str]],
    existing: dict[tuple[str, ...], dict[str, str]],
    current_commit: str,
    now: str,
) -> tuple[list[dict[str, str]], int]:
    """Assign per-row provenance and merge with carried-forward rows.

    A freshly collected row keeps the prior commit/timestamp when its measured
    value is byte-identical to the committed table (an un-rerun bench produces
    the same Criterion estimate), and is stamped with the current commit/time
    when the value changed or the key is new. Rows absent from this run
    (other baselines, un-collected benches) are carried forward verbatim.
    """
    fresh_keys: set[tuple[str, ...]] = set()
    restamped = 0
    for row in fresh_rows:
        key = row_key(row)
        fresh_keys.add(key)
        prev = existing.get(key)
        if prev and measurement_value(prev) == measurement_value(row):
            row["git_commit"] = prev.get("git_commit", "")
            row["captured_at_utc"] = prev.get("captured_at_utc", "")
        else:
            row["git_commit"] = current_commit
            row["captured_at_utc"] = now
            restamped += 1
    carried = [prev for key, prev in existing.items() if key not in fresh_keys]
    return [*fresh_rows, *carried], restamped


def dedupe_score(row: dict[str, str]) -> tuple[int, int]:
    label = row["label"]
    short_label = 1 if label in AKITA_FP4_SHORT else 0
    untruncated = 1 if len(row["group"]) < 64 else 0
    return short_label + untruncated, -len(row["group"])


def dedupe_rows(rows: list[dict[str, str]], warnings: list[str]) -> list[dict[str, str]]:
    keyed: dict[tuple[str, ...], dict[str, str]] = {}
    duplicate_count = 0
    for row in rows:
        key = row_key(row)
        previous = keyed.get(key)
        if previous is None:
            keyed[key] = row
            continue
        duplicate_count += 1
        if dedupe_score(row) > dedupe_score(previous):
            keyed[key] = row

    if duplicate_count:
        warnings.append(
            f"resolved {duplicate_count} duplicate rows after normalizing old long labels "
            "and new short labels"
        )
    return list(keyed.values())


def row_sort_key(r: dict[str, str]) -> tuple[object, ...]:
    return (
        r["baseline"],
        r["library"],
        r["field"],
        r["ext_degree"],
        BASIS_RANK.get(r["basis"], 9),
        r["op"],
        WORKLOAD_RANK.get(r["workload"], 9),
        r["vectorization"],
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "baseline",
        "machine_config",
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
        "unit",
        "ns_per_op_or_lane",
        "lower",
        "upper",
        "family",
        "label",
        "group",
        "bench_id",
        *PROV_FIELDS,
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=row_sort_key,
            )
        )


def fmt_ns(row: dict[str, str]) -> str:
    return f"{row['ns_per_op_or_lane']} [{row['lower']}, {row['upper']}]"


def markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    aligns = ["---" for _ in headers]
    for i, header in enumerate(headers):
        if header in {"w", "median [CI]", "rows"}:
            aligns[i] = "---:"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def coverage_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = (row["baseline"], row["family"], row["vectorization"], row["workload"])
        counts[key] = counts.get(key, 0) + 1
    return [[*key, str(count)] for key, count in sorted(counts.items())]


def missing_headline_warnings(rows: list[dict[str, str]]) -> list[str]:
    present = {
        (
            r["baseline"],
            r["library"],
            r["field"],
            r["ext_degree"],
            r["basis"],
            r["op"],
            r["workload"],
            r["vectorization"],
        )
        for r in rows
    }
    baselines = sorted({r["baseline"] for r in rows})
    expected_fields = [
        ("akita", "mersenne31", "4", "ring_subfield"),
        ("akita", "prime31_offset19", "4", "ring_subfield"),
        ("akita", "prime32_offset99", "4", "ring_subfield"),
        ("plonky3", "baby_bear", "5", ""),
        ("plonky3", "koala_bear", "5", ""),
        ("plonky3", "baby_bear", "", ""),
        ("plonky3", "koala_bear", "", ""),
        ("plonky3", "mersenne31", "", ""),
    ]
    missing: list[str] = []
    for baseline in baselines:
        for library, field, ext_degree, basis in expected_fields:
            for op in HEADLINE_OPS:
                for workload in WORKLOAD_RANK:
                    key = (
                        baseline,
                        library,
                        field,
                        ext_degree,
                        basis,
                        op,
                        workload,
                        "packed",
                    )
                    if key not in present:
                        missing.append(
                            f"{baseline}: missing packed {workload} {library} "
                            f"{field} ext{ext_degree or '-'} {basis or 'default'} {op}"
                        )
    return missing


def cross_baseline_parity_warnings(rows: list[dict[str, str]]) -> list[str]:
    """Warn when a measurement key is captured in some baselines but not others.

    Each baseline should cover the same (library, field, ext, basis, op, workload,
    vectorization, family) keys. A partial or stale capture shows up here as a set
    difference against the union of all baselines, which is how a missing cell is
    caught even when every expected-field allowlist still passes.
    """
    keysets: dict[str, set[tuple[str, ...]]] = {}
    for row in rows:
        key = (
            row["library"],
            row["field"],
            row["ext_degree"],
            row["basis"],
            row["op"],
            row["workload"],
            row["vectorization"],
            row["family"],
        )
        keysets.setdefault(row["baseline"], set()).add(key)

    baselines = sorted(keysets)
    if len(baselines) < 2:
        return []

    union: set[tuple[str, ...]] = set().union(*keysets.values())
    warnings: list[str] = []
    for baseline in baselines:
        missing = union - keysets[baseline]
        if not missing:
            continue
        by_family: dict[str, int] = {}
        for key in missing:
            family, library = key[7], key[0]
            by_family[f"{family}/{library}"] = by_family.get(f"{family}/{library}", 0) + 1
        breakdown = ", ".join(f"{count} {tag}" for tag, count in sorted(by_family.items()))
        warnings.append(
            f"baseline {baseline!r} is missing {len(missing)} measurement keys "
            f"present in other baselines ({breakdown})"
        )
    return warnings


def derived_warnings(rows: list[dict[str, str]]) -> list[str]:
    return [*missing_headline_warnings(rows), *cross_baseline_parity_warnings(rows)]


def write_markdown(
    path: Path,
    rows: list[dict[str, str]],
    metadata: dict[str, dict[str, str]],
    warnings: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    filtered = [
        r
        for r in rows
        if r["vectorization"] == "packed"
        and r["op"] in HEADLINE_OPS
        and r["family"] in ("ext4", "ext5")
    ]
    ring_focus = [
        r
        for r in rows
        if r["vectorization"] == "packed"
        and r["family"] == "ext4"
        and r["basis"] == "ring_subfield"
        and r["op"] in RING_FOCUS_OPS
        and r["field"] in ("mersenne31", "prime31_offset19", "prime32_offset99")
    ]

    quality_warnings = [*warnings, *derived_warnings(rows)]
    prov = provenance_summary(rows)
    baseline_body = []
    for name in sorted(metadata):
        meta = metadata[name]
        rust = meta.get("rustc", "").replace("|", "\\|")
        rustflags = meta.get("rustflags", "").replace("|", "\\|")
        target_cpu = meta.get("target_cpu", "").replace("|", "\\|")
        cpu_model = meta.get("cpu_model", "").replace("|", "\\|")
        commit_counts = prov.get(name, {}).get("commit_row_counts", {}) or {}
        if len(commit_counts) == 1:
            git_cell = next(iter(commit_counts))[:12]
        elif commit_counts:
            git_cell = f"{len(commit_counts)} commits"
        else:
            git_cell = meta.get("git_commit", "")[:12]
        baseline_body.append(
            [
                name,
                meta.get("machine_config", ""),
                meta.get("arch", ""),
                meta.get("simd", ""),
                target_cpu or rustflags or "(unspecified)",
                cpu_model or "(unrecorded)",
                rust or "(unrecorded)",
                git_cell,
            ]
        )

    lines = [
        "# Field Microbench Reference",
        "",
        f"Generated at `{generated_at}` by `scripts/field_microbench_collect.py` from Criterion saved baselines.",
        "",
        "This file is meant to be read as a benchmark reference, not just as a raw dump.",
        "The complete machine-readable table is `bench-data/field-microbench.csv`; this markdown highlights the rows most relevant to the 31-bit extension-field comparison.",
        "",
        "## What Is Measured",
        "",
        "- `latency_chain`: a dependent chain where each operation consumes the previous result; read this as critical-path latency.",
        "- `throughput_stream`: independent streams of the same operation; read this as reciprocal throughput under available instruction-level parallelism.",
        "- `scalar` rows are normalized as `ns/op`; `packed` rows are normalized as `ns/lane`, with `w` equal to the SIMD lane count.",
        "- `square` is the field's dedicated square operation. `mul_self` is the general multiplication path called as `x * x`, useful as the control when studying square-specific optimizations.",
        "- Values are Criterion medians in nanoseconds with the reported confidence interval shown as `median [lower, upper]`.",
        "",
        "## Baselines And Machine Configuration",
        "",
        *markdown_table(
            [
                "baseline",
                "machine_config",
                "arch",
                "simd",
                "target/RUSTFLAGS",
                "CPU",
                "rustc",
                "git",
            ],
            baseline_body,
        ),
        "",
        "## Data Quality Notes",
        "",
    ]
    if quality_warnings:
        lines.extend(f"- {warning}" for warning in sorted(set(quality_warnings)))
    else:
        lines.append("- No collector warnings.")

    lines.extend(
        [
            "",
            "## Coverage Summary",
            "",
            *markdown_table(
                ["baseline", "family", "vectorization", "workload", "rows"],
                coverage_rows(rows),
            ),
            "",
            "## Headline Packed Extension Rows",
            "",
            "Akita degree-4 fp4 rows are the Akita security-equivalent extension-field comparison. Plonky3 degree-5 rows are the security-equivalent 31-bit Plonky3 comparison; Plonky3 degree-4 rows are included as a lower-degree reference.",
            "",
            *markdown_table(
                [
                    "baseline",
                    "library",
                    "field",
                    "ext",
                    "basis",
                    "op",
                    "workload",
                    "simd",
                    "w",
                    "median [CI]",
                ],
                [
                    [
                        r["baseline"],
                        r["library"],
                        r["field"],
                        r["ext_degree"],
                        r["basis"] or "default",
                        r["op"],
                        r["workload"],
                        r["simd"],
                        r["width"],
                        fmt_ns(r),
                    ]
                    for r in sorted(
                        filtered,
                        key=lambda x: (
                            x["baseline"],
                            WORKLOAD_RANK.get(x["workload"], 9),
                            x["library"],
                            x["field"],
                            x["ext_degree"],
                            BASIS_RANK.get(x["basis"], 9),
                            x["op"],
                        ),
                    )
                ],
            ),
            "",
            "## Packed Ring-Subfield Focus",
            "",
            "These rows cover the Akita fp4 ring-subfield operations most relevant to the packed arithmetic optimization work. `mul_self` is shown only for latency chains when the bench emits it.",
            "",
            *markdown_table(
                [
                    "baseline",
                    "field",
                    "op",
                    "workload",
                    "simd",
                    "w",
                    "median [CI]",
                ],
                [
                    [
                        r["baseline"],
                        r["field"],
                        r["op"],
                        r["workload"],
                        r["simd"],
                        r["width"],
                        fmt_ns(r),
                    ]
                    for r in sorted(
                        ring_focus,
                        key=lambda x: (
                            x["baseline"],
                            x["field"],
                            WORKLOAD_RANK.get(x["workload"], 9),
                            RING_FOCUS_OPS.index(x["op"]),
                        ),
                    )
                ],
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def provenance_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    """Per-baseline summary of the per-row provenance actually present.

    Replaces the misleading single whole-machine commit: a baseline can now
    legitimately carry rows captured at several commits when only some benches
    were refreshed.
    """
    commits: dict[str, Counter[str]] = defaultdict(Counter)
    captured: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        baseline = row["baseline"]
        commits[baseline][row.get("git_commit", "")] += 1
        when = row.get("captured_at_utc", "")
        if when:
            captured[baseline].append(when)
    summary: dict[str, dict[str, object]] = {}
    for baseline, counts in commits.items():
        when = captured[baseline]
        summary[baseline] = {
            "row_count": sum(counts.values()),
            "commit_row_counts": dict(sorted(counts.items())),
            "captured_at_min": min(when) if when else "",
            "captured_at_max": max(when) if when else "",
        }
    return summary


def write_metadata_json(
    path: Path,
    metadata: dict[str, dict[str, str]],
    warnings: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "baselines": metadata,
        "row_provenance": provenance_summary(rows),
        "warnings": sorted(set([*warnings, *derived_warnings(rows)])),
        "row_count": len(rows),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def add_collect_args(parser: argparse.ArgumentParser) -> None:
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
        metavar="NAME:ARCH:SIMD[:MACHINE]",
        help="Saved Criterion baseline tag, e.g. neon:aarch64:neon:apple-m4-max",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="BASELINE=PATH",
        help="Per-baseline machine metadata JSON emitted by the machine-info command",
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-meta", type=Path, default=DEFAULT_OUT_META)
    parser.add_argument(
        "--git-commit",
        default="",
        help="Commit to stamp on freshly measured rows (default: current HEAD)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Rebuild the table from collected rows only, discarding carried-forward "
        "rows and their per-row provenance (default merges with the existing table)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if collector warnings are emitted",
    )


def add_machine_info_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--simd", required=True)
    parser.add_argument("--machine-config", default="")
    parser.add_argument("--criterion-dir", type=Path, default=Path("target/criterion"))
    parser.add_argument("--rustflags", default=None)
    parser.add_argument("--target-cpu", default="")
    parser.add_argument("--bench-filter", default="")
    parser.add_argument("--bench-command", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", type=Path, required=True)


def collect_main(args: argparse.Namespace) -> int:
    warnings: list[str] = []
    if not args.baseline:
        print("error: pass at least one --baseline NAME:ARCH:SIMD[:MACHINE]", file=sys.stderr)
        return 1

    try:
        specs = [BaselineSpec.parse(spec) for spec in args.baseline]
        metadata_paths = parse_metadata_specs(args.metadata)
        metadata = baseline_metadata(specs, metadata_paths, warnings)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    all_rows: list[dict[str, str]] = []
    for spec in specs:
        rows = collect_rows(args.criterion_dir, spec, metadata[spec.name], warnings)
        if not rows:
            warnings.append(f"no rows for baseline {spec.name!r}")
        all_rows.extend(rows)

    all_rows = dedupe_rows(all_rows, warnings)

    # Merge freshly collected rows over the committed table so a partial
    # re-capture only restamps the benches it actually measured.
    existing = {} if args.replace else load_existing_csv(args.out_csv)
    current_commit = args.git_commit or run_text(["git", "rev-parse", "HEAD"], Path.cwd())
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    merged_rows, restamped = stamp_provenance(all_rows, existing, current_commit, now)

    # Preserve machine metadata for baselines not passed on this run.
    if not args.replace:
        carried_meta = load_existing_meta_baselines(args.out_meta)
        metadata = {**carried_meta, **metadata}

    all_warnings = [*warnings, *derived_warnings(merged_rows)]
    write_csv(args.out_csv, merged_rows)
    write_markdown(args.out_md, merged_rows, metadata, warnings)
    write_metadata_json(args.out_meta, metadata, warnings, merged_rows)

    for warning in sorted(set(all_warnings)):
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"wrote {len(merged_rows)} rows ({restamped} freshly stamped at "
        f"{current_commit[:12] or '(no commit)'}) to {args.out_csv}, "
        f"{args.out_md}, and {args.out_meta}"
    )
    if args.strict and all_warnings:
        return 1
    return 0


def machine_info_main(args: argparse.Namespace) -> int:
    data = capture_machine_metadata(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote machine metadata to {args.out}")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] not in {"collect", "machine-info", "-h", "--help"}:
        argv = ["collect", *argv]

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect", help="collect Criterion estimates")
    add_collect_args(collect_parser)

    machine_parser = subparsers.add_parser(
        "machine-info",
        help="write machine/toolchain/build metadata for one saved baseline",
    )
    add_machine_info_args(machine_parser)

    args = parser.parse_args(argv)
    if args.command == "machine-info":
        return machine_info_main(args)
    if args.command in (None, "collect"):
        return collect_main(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
