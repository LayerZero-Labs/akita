#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

if __package__:
    from .profile_bench_fold_details import (
        PROOF_LEVEL_BYTE_FIELDS,
        challenge_line,
        detail_block,
        digit_count_phrase,
        exact_choice,
        format_witness_groups_inline,
        matrix_line,
        planned_group_key,
        planned_group_label,
        planned_group_planner_value,
        planned_group_work_value,
        planned_groups_for_render,
        planned_terminal_planner_value,
        planned_terminal_work_value,
        proof_component_group,
        proof_cost_summary,
        proof_field_present,
        proof_level_component_bytes,
        proof_step_label,
        render_fold_details,
        render_group_choices,
        response_bound_lines,
        sample_range,
    )
    from .profile_bench_format import (
        fmt_bytes,
        fmt_count,
        numeric_delta,
        value_with_baseline_delta,
    )
else:
    from profile_bench_fold_details import (
        PROOF_LEVEL_BYTE_FIELDS,
        challenge_line,
        detail_block,
        digit_count_phrase,
        exact_choice,
        format_witness_groups_inline,
        matrix_line,
        planned_group_key,
        planned_group_label,
        planned_group_planner_value,
        planned_group_work_value,
        planned_groups_for_render,
        planned_terminal_planner_value,
        planned_terminal_work_value,
        proof_component_group,
        proof_cost_summary,
        proof_field_present,
        proof_level_component_bytes,
        proof_step_label,
        render_fold_details,
        render_group_choices,
        response_bound_lines,
        sample_range,
    )
    from profile_bench_format import (
        fmt_bytes,
        fmt_count,
        numeric_delta,
        value_with_baseline_delta,
    )


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=(".*?"|\S+)')
RSS_PATTERNS = [
    re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)"),
    re.compile(r"^\s*(\d+)\s+maximum resident set size$", re.MULTILINE),
]
ONEHOT_ARITY = 256
ONEHOT_WORKLOAD_LABEL = f"1-of-{ONEHOT_ARITY} one-hot"
CASE_SCHEMA_VERSION = 9
VALID_RELATION_MODES = frozenset(("quotient_lift", "reduced_evaluation"))
REQUIRED_RUN_METRICS = (
    "setup_s",
    "commit_s",
    "prove_total_s",
    "verify_total_s",
    "proof_size_bytes",
    "accounted_bytes",
    "max_rss_kib",
    "crt_profile",
    "crt_num_primes",
    "crt_prime_modulus_bits",
    "crt_limb_bits",
    "balanced_digit_safe_width",
    "raw_i8_safe_width",
    "ext_degree",
    "akita_levels",
)
REQUIRED_RUN_SEQUENCES = ("planned_levels", "proof_levels")


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    mode: str
    num_vars: int
    num_polys: int
    setup_mode: str = "direct"

    @property
    def case_id(self) -> str:
        return case_id(self.mode, self.num_vars, self.num_polys, self.setup_mode)


@dataclass(frozen=True)
class CaseMetadata:
    field_family: str
    workload: str
    workload_label: str
    config: str
    opening_topology: str = "single_group"


# Securable families under honest committed-fold A-role pricing, i.e. the ones
# that ship a generated schedule table
# (`akita_config::generated_families::ALL_GENERATED_FAMILIES`). Modes outside
# this map still render via the `case_metadata` fallback below.
CASE_METADATA: dict[str, CaseMetadata] = {
    # Direct fp128 one-hot and dense use adaptive generated schedules.
    "onehot_fp128": CaseMetadata(
        "fp128", "onehot", ONEHOT_WORKLOAD_LABEL, "mixed D256 to D64"
    ),
    "dense_fp128": CaseMetadata("fp128", "dense", "dense", "adaptive D1024 to D64"),
    "onehot_fp128_multi_group": CaseMetadata(
        "fp128", "onehot", "multi-group one-hot", "multi-group", "multi_group"
    ),
    "onehot_fp128_multi_group_recursive": CaseMetadata(
        "fp128",
        "onehot",
        "multi-group one-hot",
        "adaptive recursive multi-group",
        "multi_group",
    ),
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2": CaseMetadata(
        "fp128",
        "onehot",
        "multi-group one-hot",
        "adaptive recursive multi-group W8R2",
        "multi_group",
    ),
    "onehot_fp128_multi_chunk_w8r2": CaseMetadata(
        "fp128", "onehot", ONEHOT_WORKLOAD_LABEL, "multi-chunk W8R2"
    ),
    "onehot_fp128_multi_chunk_w2r2": CaseMetadata(
        "fp128", "onehot", ONEHOT_WORKLOAD_LABEL, "multi-chunk W2R2"
    ),
    "onehot_fp128_multi_chunk_w4r2": CaseMetadata(
        "fp128", "onehot", ONEHOT_WORKLOAD_LABEL, "multi-chunk W4R2"
    ),
    # Small-field modes replay their catalog-selected adaptive dimensions.
    "dense_fp32": CaseMetadata("fp32", "dense", "dense", "adaptive"),
    "onehot_fp32": CaseMetadata("fp32", "onehot", ONEHOT_WORKLOAD_LABEL, "adaptive"),
    "dense_fp64": CaseMetadata("fp64", "dense", "dense", "adaptive"),
    "onehot_fp64": CaseMetadata("fp64", "onehot", ONEHOT_WORKLOAD_LABEL, "adaptive"),
}


def case_metadata(mode: str) -> CaseMetadata:
    if mode in CASE_METADATA:
        return CASE_METADATA[mode]
    field_family = "fp128"
    for family in ("fp32", "fp64", "fp128"):
        if family in mode:
            field_family = family
            break
    workload = "onehot" if "onehot" in mode else "dense"
    workload_label = ONEHOT_WORKLOAD_LABEL if workload == "onehot" else "dense"
    config_match = re.search(r"_d(\d+)$", mode)
    config = f"D{config_match.group(1)}" if config_match else "custom"
    return CaseMetadata(field_family, workload, workload_label, config)


def quotient_digit_count(metadata: CaseMetadata, log_basis: int) -> int:
    field_bits = int(metadata.field_family.removeprefix("fp"))
    return (field_bits + log_basis - 1) // log_basis


def parse_relation_mode(
    kvs: dict[str, str], event: str, *, allow_legacy: bool
) -> str:
    relation_mode = kvs.get("relation_mode")
    if relation_mode is None:
        if allow_legacy:
            return "quotient_lift"
        raise ValueError(f"{event} is missing relation_mode")
    if relation_mode not in VALID_RELATION_MODES:
        raise ValueError(f"{event} has invalid relation_mode {relation_mode!r}")
    return relation_mode


def validate_relation_quotient_digits(
    event: str, relation_mode: str, num_digits_quotient: int
) -> None:
    if relation_mode == "reduced_evaluation" and num_digits_quotient != 0:
        raise ValueError(
            f"{event} uses reduced_evaluation with "
            f"num_digits_quotient={num_digits_quotient}"
        )


def workload_slug(metadata: CaseMetadata, num_polys: int) -> str:
    if metadata.workload == "onehot" and num_polys > 1:
        return "onehot-batched"
    return metadata.workload


def slugify_config(config: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", config.lower()).strip("-") or "custom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and render the Akita profile benchmark report."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark and write summary files.")
    run_parser.add_argument("--binary", required=True, help="Path to the benchmark binary.")
    run_parser.add_argument(
        "--output-dir", required=True, help="Directory where logs and summary.json are written."
    )
    run_parser.add_argument(
        "--benchmark-shard",
        default="",
        help="Workflow matrix shard that owns every configured case.",
    )
    run_parser.add_argument("--mode", default="onehot_fp128", help="Benchmark mode.")
    run_parser.add_argument("--num-vars", type=int, default=32, help="Number of variables.")
    run_parser.add_argument(
        "--num-polys",
        type=int,
        default=1,
        help="Total number of polynomials in the mode-specific benchmark case.",
    )
    run_parser.add_argument(
        "--setup-mode",
        choices=VALID_SETUP_MODES,
        default="direct",
        help="SetupContributionMode to use for cases that do not specify one.",
    )
    run_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Benchmark case as NUM_VARS:NUM_POLYS, MODE:NUM_VARS:NUM_POLYS, "
            "or MODE:NUM_VARS:NUM_POLYS:SETUP_MODE. "
            "Can be repeated."
        ),
    )
    run_parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("AKITA_BENCH_RUNS", "1")),
        help="Number of samples to run for each benchmark case; reported timings use the median.",
    )
    run_parser.add_argument(
        "--warmups",
        type=int,
        default=int(os.environ.get("AKITA_BENCH_WARMUPS", "0")),
        help=(
            "Number of warm-up runs executed per case before the measured "
            "runs. Warm-ups prime CPU caches, the allocator, and any "
            "lazily-initialized statics (NTT roots, schedule tables) so the "
            "first measured run is not penalized. Their output is discarded "
            "and they do not contribute to the reported median."
        ),
    )
    run_parser.add_argument(
        "--baseline-binary",
        default="",
        help=(
            "Optional second binary (e.g. the PR merge-base build) benchmarked "
            "interleaved with --binary: every warm-up and measured run executes "
            "--binary immediately followed by the baseline, so machine-state "
            "drift lands on both sides of each pair instead of on one whole "
            "block."
        ),
    )
    run_parser.add_argument(
        "--baseline-output-dir",
        default="",
        help=(
            "Directory for the baseline side's logs and summary files (same "
            "layout as --output-dir). Required with --baseline-binary."
        ),
    )

    render_parser = subparsers.add_parser(
        "render", help="Render a markdown report from summary.json files."
    )
    render_parser.add_argument("summary", help="Path to the current summary.json file.")
    render_parser.add_argument(
        "--main-baseline-dir",
        default="",
        help="Optional artifact directory containing the main-baseline summary.json.",
    )
    render_parser.add_argument(
        "--previous-baseline-dir",
        default="",
        help="Optional artifact directory containing the previous-run summary.json.",
    )
    render_parser.add_argument(
        "--compact",
        action="store_true",
        help="Render only the matrix-first PR-comment summary.",
    )

    failure_parser = subparsers.add_parser(
        "failure-summary",
        help="Write a structured failure summary when the benchmark step produced none.",
    )
    failure_parser.add_argument(
        "--output-dir", required=True, help="Directory where summary files are written."
    )
    failure_parser.add_argument(
        "--benchmark-shard",
        default="",
        help="Workflow matrix shard that owns every configured case.",
    )
    failure_parser.add_argument("--mode", default="onehot_fp128", help="Benchmark mode.")
    failure_parser.add_argument("--num-vars", type=int, default=32, help="Number of variables.")
    failure_parser.add_argument(
        "--num-polys",
        type=int,
        default=1,
        help="Total number of polynomials in the mode-specific benchmark case.",
    )
    failure_parser.add_argument(
        "--setup-mode",
        choices=VALID_SETUP_MODES,
        default="direct",
        help="SetupContributionMode to use for cases that do not specify one.",
    )
    failure_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Benchmark case as NUM_VARS:NUM_POLYS, MODE:NUM_VARS:NUM_POLYS, "
            "or MODE:NUM_VARS:NUM_POLYS:SETUP_MODE. "
            "Can be repeated."
        ),
    )
    failure_parser.add_argument(
        "--failure-phase",
        default="benchmark workflow",
        help="Failure phase to show in the rendered report.",
    )
    failure_parser.add_argument(
        "--error",
        default="benchmark step failed before writing summary.json",
        help="Error message to show in the rendered report.",
    )

    return parser.parse_args()


def parse_kvs(line: str) -> dict[str, str]:
    line = ANSI_RE.sub("", line)
    out: dict[str, str] = {}
    for key, raw_value in KV_RE.findall(line):
        value = raw_value.rstrip(",")
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        out[key] = value
    return out


def parse_tracing_optional_int(value: str | None) -> int | None:
    if value is None or value == "None":
        return None
    match = re.fullmatch(r"(?:Some\((\d+)\)|(\d+))", value)
    if match is None:
        raise ValueError(f"invalid tracing optional integer: {value}")
    return int(match.group(1) or match.group(2))


def is_info_event(line: str, event: str) -> bool:
    """Match an INFO event with or without tracing span prefixes."""
    return (
        re.search(
            rf"\bINFO\s+(?:[^:]+:\s*)*{re.escape(event)}(?:\s|$)",
            line,
        )
        is not None
    )


def parse_witness_groups(value: str | None) -> list[dict[str, object]]:
    if not value:
        return []
    groups = []
    for item in value.split(";"):
        name, sep, raw_count = item.partition("=")
        if not sep or not name or not raw_count:
            continue
        groups.append({"group": name, "field_elements": int(raw_count)})
    return groups


def planned_current_w_len(kvs: dict[str, str]) -> list[dict[str, object]]:
    return parse_witness_groups(kvs.get("current_w_len")) or parse_witness_groups(
        kvs.get("current_w_groups")
    )


TAIL_SUMMARY_INT_FIELDS = (
    "tail_bytes",
    "final_w_num_elems",
    "final_w_bits_per_elem",
    "tail_log_basis_open",
    "tail_log_basis_inner",
    "tail_log_basis",
    "tail_z_prefix_bytes",
    "tail_z_golomb_bytes",
    "tail_z_bytes",
    "tail_z_field_elems",
    "tail_z_ring_elems",
    "tail_z_budget_bytes",
    "tail_z_slack_bytes",
    "tail_e_field_elems",
    "tail_e_ring_elems",
    "tail_t_field_elems",
    "tail_t_ring_elems",
    "tail_e_bytes",
    "tail_t_bytes",
    "z_rice_low_bits_wire",
    "z_coords",
    "z_packed_hypothetical_bytes",
    "z_golomb_savings_bytes",
)

TAIL_SUMMARY_FLOAT_FIELDS = (
    "z_bits_per_coord_golomb",
    "z_bits_per_coord_packed",
)

def ingest_tail_summary_fields(summary: dict[str, object], kvs: dict[str, str]) -> None:
    if "final_w_encoding" in kvs:
        summary["tail_encoding"] = kvs["final_w_encoding"]
    if "final_w_policy" in kvs:
        summary["tail_policy"] = kvs["final_w_policy"]
    if "final_w_num_elems" in kvs:
        summary["tail_num_elems"] = int(kvs["final_w_num_elems"])
        summary["terminal_w_len"] = int(kvs["final_w_num_elems"])
    bits_per_elem = kvs.get("final_w_bits_per_elem")
    if bits_per_elem is not None and bits_per_elem != "None":
        summary["tail_bits_per_elem"] = int(bits_per_elem)
    if kvs.get("final_w_encoding") == "packed_digits" and "final_w_bits_per_elem" in kvs:
        summary["terminal_log_basis"] = int(kvs["final_w_bits_per_elem"])
    for key in TAIL_SUMMARY_INT_FIELDS:
        if key in kvs:
            summary[key] = int(kvs[key])
    if "tail_z_coords" in kvs and "tail_z_field_elems" not in summary:
        summary["tail_z_field_elems"] = int(kvs["tail_z_coords"])
    for key in TAIL_SUMMARY_FLOAT_FIELDS:
        if key in kvs:
            summary[key] = float(kvs[key])
    if "z_linf_cap" in kvs:
        summary["z_linf_cap"] = parse_tracing_optional_int(kvs["z_linf_cap"])
    elif "z_witness_linf_cap" in kvs:
        summary["z_linf_cap"] = int(kvs["z_witness_linf_cap"])
    elif "z_beta_inf" in kvs:
        summary["z_linf_cap"] = int(kvs["z_beta_inf"])
    if "z_rice_low_bits_cap" in kvs:
        summary["z_rice_low_bits_cap"] = parse_tracing_optional_int(
            kvs["z_rice_low_bits_cap"]
        )
    terminal_log_basis = summary.get(
        "tail_log_basis_inner",
        summary.get("tail_log_basis_open", summary.get("tail_log_basis")),
    )
    if terminal_log_basis is not None:
        summary["terminal_log_basis"] = terminal_log_basis


def render_tail_encoding(current: dict[str, object]) -> None:
    encoding = current.get("tail_encoding")
    if encoding == "none" or (
        current.get("tail_bytes") == 0 and encoding in (None, "none")
    ):
        print(
            "- Tail encoding: `none` "
            "(root-direct zero-fold; profile bench has no cleartext tail witness)"
        )
        return
    if encoding is None:
        return

    print(f"- Tail encoding: `{encoding}`")

    if encoding == "packed_digits":
        if (
            current.get("tail_num_elems") is not None
            and current.get("tail_bits_per_elem") is not None
        ):
            print(
                f"  - Wire: `{fmt_count(float(current['tail_num_elems']))}` logical elements at "
                f"`{current['tail_bits_per_elem']}` bits for each element (uniform `PackedDigits`)"
            )
        return

    if encoding == "field_elements":
        if current.get("tail_num_elems") is not None:
            print(
                f"  - Wire: `{fmt_count(float(current['tail_num_elems']))}` raw field elements"
            )
        return

    if encoding not in ("segment_typed", "terminal_response"):
        return

    terminal_log_basis = current.get(
        "tail_log_basis_inner", current.get("tail_log_basis_open")
    )
    if current.get("tail_num_elems") is not None and terminal_log_basis is not None:
        print(
            "  - Clear response: "
            f"`{fmt_count(float(current['tail_num_elems']))}` coefficients across `z`, `e`, "
            f"and `t`. The incoming witness uses a basis width of `{terminal_log_basis}` bits."
        )

    z_prefix = current.get("tail_z_prefix_bytes")
    z_golomb = current.get("tail_z_golomb_bytes")
    segment_rows = []
    for label, bytes_key, field_key, ring_key, segment_encoding in (
        (
            "Folded response (`z`)",
            "tail_z_bytes",
            "tail_z_field_elems",
            "tail_z_ring_elems",
            "Golomb",
        ),
        (
            "Opening values (`e`)",
            "tail_e_bytes",
            "tail_e_field_elems",
            "tail_e_ring_elems",
            "raw field elements",
        ),
        (
            "Inner-commitment values (`t`)",
            "tail_t_bytes",
            "tail_t_field_elems",
            "tail_t_ring_elems",
            "raw field elements",
        ),
    ):
        if current.get(bytes_key) is None:
            continue
        if (
            bytes_key == "tail_z_bytes"
            and z_prefix is not None
            and z_golomb is not None
        ):
            segment_encoding = (
                f"{fmt_bytes(float(z_prefix))}-byte length prefix + "
                f"{fmt_bytes(float(z_golomb))}-byte Golomb payload"
            )
        field_coefficients = current.get(field_key)
        ring_elements = current.get(ring_key)
        segment_rows.append(
            (
                label,
                f"{fmt_bytes(float(current[bytes_key]))} bytes",
                (
                    fmt_count(float(field_coefficients))
                    if field_coefficients is not None
                    else "—"
                ),
                fmt_count(float(ring_elements)) if ring_elements is not None else "—",
                segment_encoding,
            )
        )
    if segment_rows:
        print()
        print("| Segment | Bytes | Field coefficients | Ring elements | Encoding |")
        print("| --- | ---: | ---: | ---: | --- |")
        for row in segment_rows:
            print("| " + " | ".join(row) + " |")

    z_budget = current.get("tail_z_budget_bytes")
    if z_budget is not None and z_golomb is not None:
        parameter = current.get("z_rice_low_bits_wire")
        scheduled_parameter = current.get("z_rice_low_bits_cap")
        cap = current.get("z_linf_cap", current.get("z_witness_linf_cap"))
        details = []
        if parameter is not None:
            details.append(f"Golomb parameter `{parameter}`")
        if (
            scheduled_parameter is not None
            and scheduled_parameter != parameter
        ):
            details.append(f"scheduled Golomb parameter `{scheduled_parameter}`")
        if cap is not None:
            details.append(f"coefficient limit `{cap}`")
        suffix = f" ({', '.join(details)})" if details else ""
        print(
            f"- Golomb payload: `{fmt_bytes(float(z_golomb))} bytes` out of the scheduled "
            f"`{fmt_bytes(float(z_budget))}-byte` budget{suffix}."
        )

    z_bits_golomb = current.get("z_bits_per_coord_golomb")
    z_bits_packed = current.get("z_bits_per_coord_packed")
    z_packed_hyp = current.get("z_packed_hypothetical_bytes")
    if z_bits_golomb is not None and z_bits_packed is not None:
        packed_size = (
            f" and `{fmt_bytes(float(z_packed_hyp))} bytes` total"
            if z_packed_hyp is not None
            else ""
        )
        print(
            f"- Folded response encoding: `{z_bits_golomb:.2f}` bits per coefficient with "
            f"Golomb coding, compared with `{z_bits_packed:.2f}` bits per coefficient"
            f"{packed_size} for uniform packed digits."
        )


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def time_command(binary: str) -> list[str]:
    if sys.platform == "darwin":
        return ["/usr/bin/time", "-l", binary]
    return ["/usr/bin/time", "-v", binary]


def require_float(summary: dict[str, object], key: str) -> float:
    value = summary.get(key)
    if value is None:
        raise ValueError(f"missing required metric: {key}")
    return float(value)


def require_int(summary: dict[str, object], key: str) -> int:
    value = summary.get(key)
    if value is None:
        raise ValueError(f"missing required metric: {key}")
    return int(value)


def missing_required_run_metrics(summary: dict[str, object]) -> list[str]:
    missing = [key for key in REQUIRED_RUN_METRICS if summary.get(key) is None]
    if (
        summary.get("verification_modes") == "multi_and_single"
        and summary.get("verify_single_total_s") is None
    ):
        missing.append("verify_single_total_s")
    for key in REQUIRED_RUN_SEQUENCES:
        value = summary.get(key)
        if not isinstance(value, list) or not value:
            missing.append(key)
    tail_bytes = summary.get("tail_bytes")
    tail_encoding = summary.get("tail_encoding")
    if tail_bytes not in (None, 0) and tail_encoding is None:
        missing.append("tail_encoding")
    if (
        tail_encoding not in ("none", None)
        and tail_bytes not in (None, 0)
        and summary.get("tail_num_elems") is None
    ):
        missing.append("tail_num_elems")
    if summary.get("tail_bits_per_elem") is None and tail_encoding == "packed_digits":
        missing.append("tail_bits_per_elem")
    proof_size = summary.get("proof_size_bytes")
    accounted = summary.get("accounted_bytes")
    if proof_size is not None and accounted is not None and int(proof_size) != int(accounted):
        missing.append("consistent_proof_accounting")
    return missing


TIMING_SAMPLE_METRICS = (
    "setup_s",
    "setup_expand_s",
    "backend_prepare_s",
    "commit_s",
    "prove_total_s",
    "verify_total_s",
    "verify_single_total_s",
    "prove_akita_s",
    "verify_akita_s",
    "verify_single_akita_s",
)
GRIND_SAMPLE_METRICS = (
    "grind_levels",
    "grind_nonce_max",
    "grind_attempts_sum",
)
SAMPLE_METRICS = TIMING_SAMPLE_METRICS + ("max_rss_kib",) + GRIND_SAMPLE_METRICS


VALID_SETUP_MODES = ("direct", "recursive")


def normalize_setup_mode(value: object) -> str:
    setup_mode = str(value).lower()
    if setup_mode not in VALID_SETUP_MODES:
        raise ValueError(
            f"invalid setup contribution mode {value!r}; expected one of "
            + ", ".join(VALID_SETUP_MODES)
        )
    return setup_mode


def setup_mode_case_suffix(setup_mode: str) -> str:
    setup_mode = normalize_setup_mode(setup_mode)
    if setup_mode == "direct":
        return ""
    return f"-setup-{setup_mode}"


def case_id(mode: str, num_vars: int, num_polys: int, setup_mode: str = "direct") -> str:
    metadata = case_metadata(mode)
    config = slugify_config(metadata.config)
    return (
        f"{metadata.field_family}-{workload_slug(metadata, num_polys)}"
        f"-nv{num_vars}-np{num_polys}-{config}{setup_mode_case_suffix(setup_mode)}"
    )


def benchmark_name(
    mode: str, num_vars: int, num_polys: int = 1, setup_mode: str = "direct"
) -> str:
    metadata = case_metadata(mode)
    setup_mode = normalize_setup_mode(setup_mode)
    setup_suffix = ""
    if setup_mode != "direct":
        setup_suffix = f" ({setup_mode} setup contribution)"
    if metadata.opening_topology == "multi_group":
        return (
            f"{metadata.field_family} multi-group opening, final nv{num_vars}, "
            f"{num_polys} polynomials total{setup_suffix}"
        )
    if metadata.workload == "onehot":
        if num_polys > 1:
            return (
                f"{metadata.field_family} {metadata.config} same-point "
                f"1-of-{ONEHOT_ARITY} one-hot x{num_polys} with {num_vars} variables"
                f"{setup_suffix}"
            )
        return (
            f"{metadata.field_family} {metadata.config} 1-of-{ONEHOT_ARITY} one-hot "
            f"with {num_vars} variables{setup_suffix}"
        )
    if num_polys > 1:
        return (
            f"{metadata.field_family} {metadata.config} dense x{num_polys} "
            f"with {num_vars} variables{setup_suffix}"
        )
    return f"{metadata.field_family} {metadata.config} dense with {num_vars} variables{setup_suffix}"


def parse_case_spec(
    spec: str, default_mode: str, default_setup_mode: str = "direct"
) -> BenchmarkCaseSpec:
    parts = spec.split(":")
    setup_mode = normalize_setup_mode(default_setup_mode)
    if len(parts) == 2:
        mode = default_mode
        num_vars_str, num_polys_str = parts
    elif len(parts) == 3:
        mode, num_vars_str, num_polys_str = parts
    elif len(parts) == 4:
        mode, num_vars_str, num_polys_str, setup_mode_str = parts
        setup_mode = normalize_setup_mode(setup_mode_str)
    else:
        raise ValueError(
            f"invalid case spec {spec!r}; expected NUM_VARS:NUM_POLYS, "
            "MODE:NUM_VARS:NUM_POLYS, or MODE:NUM_VARS:NUM_POLYS:SETUP_MODE"
        )
    num_vars = int(num_vars_str)
    num_polys = int(num_polys_str)
    if num_vars <= 0 or num_polys <= 0:
        raise ValueError(f"invalid case spec {spec!r}; NUM_VARS and NUM_POLYS must be positive")
    return BenchmarkCaseSpec(
        mode=mode, num_vars=num_vars, num_polys=num_polys, setup_mode=setup_mode
    )


def configured_cases(args: argparse.Namespace) -> list[BenchmarkCaseSpec]:
    setup_mode = normalize_setup_mode(getattr(args, "setup_mode", "direct"))
    if args.case:
        cases = [parse_case_spec(spec, args.mode, setup_mode) for spec in args.case]
    else:
        cases = [
            BenchmarkCaseSpec(
                mode=args.mode,
                num_vars=args.num_vars,
                num_polys=args.num_polys,
                setup_mode=setup_mode,
            )
        ]
    # case_id is the output dir name and the failure/aggregation key, so
    # duplicates would collide on disk and pool into one aggregate.
    case_ids = [case.case_id for case in cases]
    duplicates = sorted({cid for cid in case_ids if case_ids.count(cid) > 1})
    if duplicates:
        raise ValueError("duplicate benchmark case ids: " + ", ".join(duplicates))
    return cases


def extract_summary(
    log_text: str,
    mode: str,
    num_vars: int,
    num_polys: int,
    setup_mode: str = "direct",
    *,
    allow_legacy_relation_mode: bool = False,
) -> dict[str, object]:
    metadata = case_metadata(mode)
    setup_mode = normalize_setup_mode(setup_mode)
    summary: dict[str, object] = {
        "schema_version": CASE_SCHEMA_VERSION,
        "benchmark": benchmark_name(mode, num_vars, num_polys, setup_mode),
        "mode": mode,
        "setup_contribution_mode": setup_mode,
        "field_family": metadata.field_family,
        "workload": metadata.workload,
        "workload_label": metadata.workload_label,
        "config": metadata.config,
        "num_vars": num_vars,
        "num_polys": num_polys,
        "case_id": case_id(mode, num_vars, num_polys, setup_mode),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    planned_levels: dict[int, dict[str, object]] = {}
    planned_groups: dict[int, list[dict[str, object]]] = {}
    terminal_plan: dict[str, object] | None = None
    proof_levels: dict[int, dict[str, object]] = {}
    grinding_plan_summary: dict[str, object] | None = None
    grinding_plan_runs: list[dict[str, object]] = []
    onehot_commit_schedules: list[dict[str, object]] = []
    relation_phase_timings: list[dict[str, object]] = []
    active_verify_mode = "multi threaded"

    for line in log_text.splitlines():
        line = ANSI_RE.sub("", line)
        kvs = parse_kvs(line)
        if "profile thread pools" in line:
            summary["prove_threads"] = int(kvs["prove_threads"])
            summary["verify_multi_threads"] = int(
                kvs.get("verify_multi_threads", kvs.get("verify_threads", "1"))
            )
            summary["verify_single_threads"] = int(kvs.get("verify_single_threads", "1"))
        elif "profile verification start" in line and kvs.get("label") == mode:
            active_verify_mode = kvs["verify_mode"].replace("_", " ")
            summary["verification_modes"] = "multi_and_single"
        elif "verifier relation phase timing" in line and kvs.get("label") == mode:
            relation_phase_timings.append(
                {
                    "verify_mode": kvs["verify_mode"].replace("_", " "),
                    "relation_mode": kvs["relation_mode"],
                    "phase": kvs["phase"],
                    "calls": int(kvs["calls"]),
                    "mean_elapsed_nanos": int(kvs["mean_elapsed_nanos"]),
                    "total_elapsed_nanos": int(kvs["total_elapsed_nanos"]),
                }
            )
        elif is_info_event(line, "setup sizes") and kvs.get("label") == mode:
            setup_vector_bytes = int(kvs["setup_vector_bytes"])
            summary["setup_vector_bytes"] = setup_vector_bytes
            if "num_setup_field_elements" in kvs:
                num_setup_field_elements = int(kvs["num_setup_field_elements"])
            else:
                # Merge-base binaries before the flat-setup cutover report a
                # D-chunked count. Recover the comparable flat count from the
                # byte footprint instead of comparing incompatible units.
                field_bytes = {"fp32": 4, "fp64": 8, "fp128": 16}[
                    metadata.field_family
                ]
                if setup_vector_bytes % field_bytes != 0:
                    raise ValueError(
                        "setup vector byte count is not field-element aligned"
                    )
                num_setup_field_elements = setup_vector_bytes // field_bytes
            summary["num_setup_field_elements"] = num_setup_field_elements
            summary["setup_ntt_cache_bytes"] = int(kvs["setup_ntt_cache_bytes"])
        elif is_info_event(line, "verifier NTT cache size") and kvs.get("label") == mode:
            summary["verifier_ntt_cache_bytes"] = int(kvs["verifier_ntt_cache_bytes"])
        elif "CRT NTT profile" in line and kvs.get("label") == mode:
            summary["crt_profile"] = kvs["crt_profile"]
            summary["crt_num_primes"] = int(kvs["crt_num_primes"])
            summary["crt_prime_modulus_bits"] = int(
                kvs.get("crt_prime_modulus_bits", "30")
            )
            summary["crt_limb_bits"] = int(kvs["crt_limb_bits"])
            summary["max_i8_log_basis"] = int(kvs["max_i8_log_basis"])
            summary["balanced_digit_safe_width"] = int(kvs["balanced_digit_safe_width"])
            summary["raw_i8_safe_width"] = int(kvs["raw_i8_safe_width"])
        elif is_info_event(line, "setup_expand") and kvs.get("label") == mode:
            summary["setup_expand_s"] = float(kvs["elapsed_s"])
        elif is_info_event(line, "backend_prepare") and kvs.get("label") == mode:
            summary["backend_prepare_s"] = float(kvs["elapsed_s"])
        elif is_info_event(line, "setup") and kvs.get("label") == mode:
            summary["setup_s"] = float(kvs["elapsed_s"])
        elif is_info_event(line, "commit") and kvs.get("label") == mode:
            summary["commit_s"] = float(kvs["elapsed_s"])
        elif "one hot commit schedule" in line:
            onehot_commit_schedules.append(
                {
                    "sweep": kvs["sweep"],
                    "block_tile": int(kvs["block_tile"]),
                    "hot_terms": int(kvs["hot_terms"]),
                    "source_count": int(kvs["source_count"]),
                    "total_blocks": int(kvs["total_blocks"]),
                    "workers": int(kvs["workers"]),
                    "n_a": int(kvs["n_a"]),
                    "active_a_cols": int(kvs["active_a_cols"]),
                    "ring_dimension": int(kvs["ring_dimension"]),
                    "estimated_matrix_passes": int(kvs["estimated_matrix_passes"]),
                }
            )
        elif "akita prove complete" in line or "akita batched prove complete" in line:
            summary["prove_akita_s"] = float(kvs["elapsed_s"])
            if "levels" in kvs:
                summary["akita_levels"] = int(kvs["levels"])
        elif is_info_event(line, "prove") and kvs.get("label") == mode:
            summary["prove_total_s"] = float(kvs["elapsed_s"])
        elif "akita verify complete" in line or "akita batched verify complete" in line:
            key = (
                "verify_single_akita_s"
                if active_verify_mode == "single threaded"
                else "verify_akita_s"
            )
            summary[key] = float(kvs["elapsed_s"])
        elif "verify single threaded OK" in line and kvs.get("label") == mode:
            summary["verify_single_total_s"] = float(kvs["elapsed_s"])
        elif (
            "verify multi threaded OK" in line or "verify OK" in line
        ) and kvs.get("label") == mode:
            summary["verify_total_s"] = float(kvs["elapsed_s"])
        elif "proof summary" in line and kvs.get("label") == mode:
            summary["proof_size_bytes"] = int(kvs["proof_size_bytes"])
            summary["accounted_bytes"] = int(kvs["accounted_bytes"])
            summary["akita_fold_bytes"] = int(kvs["akita_fold_bytes"])
            summary["nonce_stream_bytes"] = int(kvs.get("nonce_stream_bytes", 0))
            summary["tail_bytes"] = int(kvs["tail_bytes"])
            if "levels" in kvs:
                summary["akita_levels"] = int(kvs["levels"])
        elif "grinding plan summary" in line and kvs.get("label") == mode:
            grinding_plan_summary = {
                "nominal_capacity_bits": int(kvs["nominal_capacity_bits"]),
                "total_nonce_bits": int(kvs["total_nonce_bits"]),
                "nonce_stream_bytes": int(kvs["nonce_stream_bytes"]),
                "padding_bits": int(kvs["padding_bits"]),
                "run_count": int(kvs["run_count"]),
                "expanded_query_count": int(kvs["expanded_query_count"]),
            }
            summary["nonce_stream_bits"] = int(kvs["total_nonce_bits"])
            summary["nonce_stream_padding_bits"] = int(kvs["padding_bits"])
        elif "grinding plan run" in line and kvs.get("label") == mode:
            grinding_plan_runs.append(
                {
                    "run_index": int(kvs["run_index"]),
                    "level": int(kvs["level"]),
                    "component": kvs["component"],
                    "query": kvs["query"],
                    "protocol": kvs["protocol"],
                    "stage": parse_tracing_optional_int(kvs.get("stage")),
                    "round": parse_tracing_optional_int(kvs.get("round")),
                    "group": parse_tracing_optional_int(kvs.get("group")),
                    "kind": kvs["kind"],
                    "loss_factor": int(kvs["loss_factor"]),
                    "grind_bits": int(kvs["grind_bits"]),
                    "nonce_bits": int(kvs["nonce_bits"]),
                    "multiplicity": int(kvs["multiplicity"]),
                    "run_nonce_bits": int(kvs["run_nonce_bits"]),
                }
            )
        elif "profile extension field" in line and kvs.get("label") == mode:
            summary["ext_degree"] = int(kvs["ext_degree"])
        elif "profile setup-contribution mode" in line and kvs.get("label") == mode:
            if "setup_contribution_mode" in kvs:
                summary["setup_contribution_mode"] = normalize_setup_mode(
                    kvs["setup_contribution_mode"]
                )
        elif "planned fold group" in line and kvs.get("label") == mode:
            level = int(kvs["level"])
            event = f"planned fold group at L{level}"
            relation_mode = parse_relation_mode(
                kvs, event, allow_legacy=allow_legacy_relation_mode
            )
            num_digits_quotient = int(
                kvs.get(
                    "num_digits_quotient",
                    str(quotient_digit_count(metadata, int(kvs["log_basis_open"]))),
                )
            )
            validate_relation_quotient_digits(
                event, relation_mode, num_digits_quotient
            )
            challenge_operator_norm_threshold = parse_tracing_optional_int(
                kvs.get("challenge_operator_norm_threshold")
            )
            planned_groups.setdefault(level, []).append(
                {
                    "group": kvs["group"],
                    "group_role": kvs["group_role"],
                    "consumer_level": int(kvs["consumer_level"]),
                    "witness_field_elements": int(kvs["witness_field_elements"]),
                    "public_num_vars": int(kvs.get("public_num_vars", "0")),
                    "public_num_polynomials": int(
                        kvs.get("public_num_polynomials", "0")
                    ),
                    "d_a": int(kvs["d_a"]),
                    "d_b": int(kvs["d_b"]),
                    "d_d": int(kvs["d_d"]),
                    **(
                        {"source_encoding": kvs["source_encoding"]}
                        if "source_encoding" in kvs
                        else {}
                    ),
                    **(
                        {"extension_degree": int(kvs["extension_degree"])}
                        if "extension_degree" in kvs
                        else {}
                    ),
                    **(
                        {"opening_method": kvs["opening_method"]}
                        if "opening_method" in kvs
                        else {}
                    ),
                    **{
                        key: parse_tracing_optional_int(kvs.get(key))
                        for key in (
                            "challenge_subring_dimension",
                            "packing_factor",
                            "packing_partial_width",
                            "packing_quotient_width",
                        )
                        if key in kvs
                    },
                    "a_width": int(kvs.get("a_width", "0")),
                    "b_width": int(kvs.get("b_width", "0")),
                    "d_width": int(kvs.get("d_width", "0")),
                    "n_a": int(kvs["n_a"]),
                    "n_b": int(kvs["n_b"]),
                    "n_d": int(kvs["n_d"]),
                    "security_route": (
                        "L2"
                        if parse_tracing_optional_int(kvs.get("response_l2_sq_cap"))
                        is not None
                        else "L-infinity"
                    ),
                    "response_l2_sq_cap": parse_tracing_optional_int(
                        kvs.get("response_l2_sq_cap")
                    ),
                    "b_slice_count": int(kvs.get("b_slice_count", "1")),
                    "physical_b_input_width": (
                        int(kvs["physical_b_input_width"])
                        if "physical_b_input_width" in kvs
                        else None
                    ),
                    "logical_b_rows": (
                        int(kvs["logical_b_rows"]) if "logical_b_rows" in kvs else None
                    ),
                    "complete_b_compression_bytes": parse_tracing_optional_int(
                        kvs.get("complete_b_compression_bytes")
                    ),
                    "log_basis_inner": int(kvs["log_basis_inner"]),
                    "log_basis_outer": int(kvs["log_basis_outer"]),
                    "log_basis_open": int(kvs["log_basis_open"]),
                    "num_digits_inner": int(kvs["num_digits_inner"]),
                    "num_digits_outer": int(kvs["num_digits_outer"]),
                    "num_digits_open": int(kvs["num_digits_open"]),
                    "num_digits_fold": int(kvs["num_digits_fold"]),
                    "relation_mode": relation_mode,
                    "num_digits_quotient": num_digits_quotient,
                    "challenge_l1_mass": int(kvs["challenge_l1_mass"]),
                    **{
                        key: int(kvs[key])
                        for key in ("challenge_count_pm1", "challenge_count_pm2")
                        if key in kvs
                    },
                    **(
                        {
                            "challenge_operator_norm_threshold": challenge_operator_norm_threshold
                        }
                        if challenge_operator_norm_threshold is not None
                        else {}
                    ),
                    "num_live_ring_elements_per_claim": int(
                        kvs["num_live_ring_elements_per_claim"]
                    ),
                    "num_live_blocks": int(kvs["num_live_blocks"]),
                    "num_positions_per_block": int(kvs["num_positions_per_block"]),
                    "block_index_domain_size": int(kvs["block_index_domain_size"]),
                    "setup_prefix_natural_field_elements": int(
                        kvs["setup_prefix_natural_field_elements"]
                    ),
                    "setup_prefix_padded_field_elements": int(
                        kvs["setup_prefix_padded_field_elements"]
                    ),
                }
            )
        elif "planned fold level" in line and kvs.get("label") == mode:
            level = int(kvs["level"])
            event = f"planned fold level L{level}"
            relation_mode = parse_relation_mode(
                kvs, event, allow_legacy=allow_legacy_relation_mode
            )
            # Benchmark runs parse both the PR binary and its merge-base binary.
            # Normalize the pre-cutover geometry names used by the merge base.
            position_index_bits = int(
                kvs.get("position_index_bits", kvs.get("position_bits", kvs.get("m_vars")))
            )
            block_index_bits = int(
                kvs.get("block_index_bits", kvs.get("block_bits", kvs.get("r_vars")))
            )
            legacy_d = int(kvs["d"])
            # The typed-schedule cutover renamed `current_w_len`/`next_w_len` to
            # `input_witness_len`/`output_witness_len` and dropped the planner
            # byte estimate (`level_bytes`) from the runtime log. Prefer the new
            # names and fall back to the merge-base names so both the PR binary
            # and its merge-base binary parse.
            input_witness_len = int(kvs.get("input_witness_len", kvs.get("current_w_len")))
            output_witness_len = int(kvs.get("output_witness_len", kvs.get("next_w_len")))
            num_live_ring_elements_per_claim = int(
                kvs.get(
                    "num_live_ring_elements_per_claim",
                    kvs.get(
                        "live_ring_elements_per_claim",
                        input_witness_len // legacy_d,
                    ),
                )
            )
            # Legacy traces exposed the Boolean-domain bit split plus
            # `block_len`/`num_blocks`; despite their names, those latter
            # values did not carry today's exact-live geometry. Reconstruct
            # the new semantics from the authoritative live source length and
            # domain bits so main/head deltas compare like with like.
            num_positions_per_block = int(
                kvs.get(
                    "num_positions_per_block",
                    kvs.get("positions_per_block", 1 << position_index_bits),
                )
            )
            num_live_blocks = int(
                kvs.get(
                    "num_live_blocks",
                    kvs.get(
                        "live_block_count",
                        (num_live_ring_elements_per_claim + num_positions_per_block - 1)
                        // num_positions_per_block,
                    ),
                )
            )
            block_index_domain_size = int(
                kvs.get("block_index_domain_size", 1 << block_index_bits)
            )
            challenge_operator_norm_threshold = parse_tracing_optional_int(
                kvs.get("challenge_operator_norm_threshold")
            )
            num_digits_quotient = int(
                kvs.get(
                    "num_digits_quotient",
                    str(
                        quotient_digit_count(
                            metadata,
                            int(kvs.get("log_basis_open") or kvs["log_basis"]),
                        )
                    ),
                )
            )
            validate_relation_quotient_digits(
                event, relation_mode, num_digits_quotient
            )
            planned_levels[level] = {
                "level": level,
                "d_a": int(kvs.get("d_a", legacy_d)),
                "d_b": int(kvs.get("d_b", legacy_d)),
                "d_d": int(kvs.get("d_d", legacy_d)),
                **(
                    {"source_encoding": kvs["source_encoding"]}
                    if "source_encoding" in kvs
                    else {}
                ),
                **(
                    {"witness_chunk_count": int(kvs["witness_chunk_count"])}
                    if "witness_chunk_count" in kvs
                    else {}
                ),
                **(
                    {
                        "witness_chunk_activated_levels": int(
                            kvs["witness_chunk_activated_levels"]
                        )
                    }
                    if "witness_chunk_activated_levels" in kvs
                    else {}
                ),
                **(
                    {"witness_chunk_active": kvs["witness_chunk_active"] == "true"}
                    if "witness_chunk_active" in kvs
                    else {}
                ),
                **(
                    {
                        "extension_opening_reduction_present": (
                            kvs["extension_opening_reduction_present"] == "true"
                        ),
                        "extension_opening_reduction_bytes": int(
                            kvs["extension_opening_reduction_bytes"]
                        ),
                    }
                    if "extension_opening_reduction_present" in kvs
                    and "extension_opening_reduction_bytes" in kvs
                    else {}
                ),
                **(
                    {"opening_method": kvs["opening_method"]}
                    if "opening_method" in kvs
                    else {}
                ),
                **{
                    key: parse_tracing_optional_int(kvs.get(key))
                    for key in (
                        "challenge_subring_dimension",
                        "packing_factor",
                        "packing_partial_width",
                        "packing_quotient_width",
                    )
                    if key in kvs
                },
                "a_width": int(kvs.get("a_width", "0")),
                "b_width": int(kvs.get("b_width", "0")),
                "d_width": int(kvs.get("d_width", "0")),
                "n_a": int(kvs["n_a"]),
                "n_b": int(kvs["n_b"]),
                "n_d": int(kvs["n_d"]),
                "security_route": (
                    "L2"
                    if parse_tracing_optional_int(kvs.get("response_l2_sq_cap")) is not None
                    else "L-infinity"
                ),
                "response_l2_sq_cap": parse_tracing_optional_int(
                    kvs.get("response_l2_sq_cap")
                ),
                "b_slice_count": int(kvs.get("b_slice_count", "1")),
                "physical_b_input_width": (
                    int(kvs["physical_b_input_width"])
                    if "physical_b_input_width" in kvs
                    else None
                ),
                "logical_b_rows": (
                    int(kvs["logical_b_rows"]) if "logical_b_rows" in kvs else None
                ),
                "complete_b_compression_bytes": parse_tracing_optional_int(
                    kvs.get("complete_b_compression_bytes")
                ),
                "challenge_l1_mass": int(kvs["challenge_l1_mass"]),
                **{
                    key: int(kvs[key])
                    for key in ("challenge_count_pm1", "challenge_count_pm2")
                    if key in kvs
                },
                **(
                    {
                        "challenge_operator_norm_threshold": challenge_operator_norm_threshold
                    }
                    if challenge_operator_norm_threshold is not None
                    else {}
                ),
                "log_basis_inner": int(kvs.get("log_basis_inner") or kvs["log_basis"]),
                "log_basis_outer": int(kvs.get("log_basis_outer") or kvs["log_basis"]),
                "log_basis_open": int(kvs.get("log_basis_open") or kvs["log_basis"]),
                "position_index_bits": position_index_bits,
                "block_index_bits": block_index_bits,
                "num_positions_per_block": num_positions_per_block,
                "num_live_blocks": num_live_blocks,
                "num_live_ring_elements_per_claim": num_live_ring_elements_per_claim,
                "block_index_domain_size": block_index_domain_size,
                "num_digits_inner": int(kvs.get("num_digits_inner") or kvs["delta_commit"]),
                "num_digits_outer": int(kvs.get("num_digits_outer") or kvs["delta_open"]),
                "num_digits_open": int(kvs.get("num_digits_open") or kvs["delta_open"]),
                "delta_fold": int(kvs["delta_fold"]),
                "relation_mode": relation_mode,
                "num_digits_quotient": num_digits_quotient,
                "input_witness_len": input_witness_len,
                "current_w_len": planned_current_w_len(kvs),
                "next_w_len": output_witness_len,
                "setup_prefix_natural_field_elements": int(
                    kvs.get("setup_prefix_natural_field_elements", "0")
                ),
                "setup_prefix_padded_field_elements": int(
                    kvs.get("setup_prefix_padded_field_elements", "0")
                ),
            }
            # `level_bytes` is only emitted by the pre-cutover merge-base binary
            # and is display-only (no correctness comparison), so keep it optional.
            if "level_bytes" in kvs:
                planned_levels[level]["level_bytes"] = int(kvs["level_bytes"])
        elif "planned terminal state" in line and kvs.get("label") == mode:
            terminal_plan = {
                "level": int(kvs["level"]) if "level" in kvs else None,
                "input_witness_len": int(
                    kvs.get("input_witness_len") or kvs["terminal_response_len"]
                ),
                "d_a": int(kvs.get("d_a") or kvs["final_inner_ring_dimension"]),
                "n_a": int(kvs.get("n_a") or kvs["final_inner_module_rank"]),
                "inner_width": int(kvs.get("inner_width", "0")),
                "log_basis_inner": int(
                    kvs.get("log_basis_inner") or kvs["final_inner_log_basis"]
                ),
                "num_digits_inner": int(kvs.get("num_digits_inner", "0")),
                "fold_log_basis": int(kvs.get("fold_log_basis", "0")),
                "fold_digit_count": int(kvs.get("fold_digit_count", "0")),
                "security_route": (
                    "L2"
                    if parse_tracing_optional_int(kvs.get("response_l2_sq_cap"))
                    is not None
                    else "L-infinity"
                ),
                "response_l2_sq_cap": parse_tracing_optional_int(
                    kvs.get("response_l2_sq_cap")
                ),
                "z_linf_cap": parse_tracing_optional_int(
                    kvs.get("z_linf_cap")
                ),
                **{
                    key: int(kvs[key])
                    for key in (
                        "challenge_l1_mass",
                        "challenge_count_pm1",
                        "challenge_count_pm2",
                        "num_live_ring_elements_per_claim",
                        "num_positions_per_block",
                        "num_live_blocks",
                        "block_index_domain_size",
                    )
                    if key in kvs
                },
                **(
                    {
                        "challenge_operator_norm_threshold": (
                            parse_tracing_optional_int(
                                kvs.get("challenge_operator_norm_threshold")
                            )
                        )
                    }
                    if parse_tracing_optional_int(
                        kvs.get("challenge_operator_norm_threshold")
                    )
                    is not None
                    else {}
                ),
                "complete": "fold_log_basis" in kvs,
            }
        elif "planned recursive setup edge" in line and kvs.get("label") == mode:
            producer_level = int(kvs["successor_level"]) - 1
            if producer_level in planned_levels:
                planned_levels[producer_level]["setup_prefix_natural_field_elements"] = int(
                    kvs["setup_prefix_natural_field_elements"]
                )
                planned_levels[producer_level]["setup_prefix_padded_field_elements"] = int(
                    kvs["setup_prefix_padded_field_elements"]
                )
        elif "proof fold level" in line and kvs.get("label") == mode:
            level = int(kvs["level"])
            present_byte_fields = [field for field in PROOF_LEVEL_BYTE_FIELDS if field in kvs]
            proof_levels[level] = {
                "level": level,
                "d": int(kvs["d"]),
                "total_bytes": int(kvs["total_bytes"]),
                "present_byte_fields": present_byte_fields,
                **{
                    field: int(kvs.get(field, "0"))
                    for field in PROOF_LEVEL_BYTE_FIELDS
                },
            }
            if "grind_nonce" in kvs:
                grind_nonce = int(kvs["grind_nonce"])
                grind_attempts = int(kvs.get("grind_attempts", grind_nonce + 1))
                if grind_attempts != grind_nonce + 1:
                    raise ValueError(
                        "fold grinding attempts must equal accepted nonce plus one: "
                        f"level={level}, nonce={grind_nonce}, attempts={grind_attempts}"
                    )
                proof_levels[level]["grind_nonce_val"] = grind_nonce
                proof_levels[level]["grind_attempts"] = grind_attempts
            response_l2_sq = parse_tracing_optional_int(kvs.get("response_l2_sq"))
            if response_l2_sq is not None:
                proof_levels[level]["response_l2_sq"] = response_l2_sq
            if "root_variant" in kvs:
                proof_levels[level]["root_variant"] = kvs["root_variant"]
        elif "proof tail summary" in line and kvs.get("label") == mode:
            ingest_tail_summary_fields(summary, kvs)
        elif "z fold encoding stats" in line and kvs.get("label") == mode:
            if summary.get("tail_encoding") != "segment_typed":
                summary["tail_encoding"] = "segment_typed"
            if "z_coords" in kvs:
                summary["z_coords"] = int(kvs["z_coords"])
            if "witness_linf_cap" in kvs:
                summary["z_witness_linf_cap"] = kvs["witness_linf_cap"]
            if "rice_low_bits_wire" in kvs:
                summary["z_rice_low_bits_wire"] = int(kvs["rice_low_bits_wire"])
            if "rice_low_bits_cap" in kvs:
                summary["z_rice_low_bits_cap"] = int(kvs["rice_low_bits_cap"])
            if "bits_per_coord_at_wire" in kvs:
                summary["z_bits_per_coord_golomb"] = float(kvs["bits_per_coord_at_wire"])
            if "bits_per_coord_packed" in kvs:
                summary["z_bits_per_coord_packed"] = float(kvs["bits_per_coord_packed"])
            if "z_payload_bytes" in kvs:
                summary["tail_z_golomb_bytes"] = int(kvs["z_payload_bytes"])
    for index, pattern in enumerate(RSS_PATTERNS):
        rss_match = pattern.search(log_text)
        if rss_match:
            rss_value = int(rss_match.group(1))
            if index == 1 and sys.platform == "darwin":
                rss_value //= 1024
            summary["max_rss_kib"] = rss_value
            break

    for level, groups in planned_groups.items():
        if level in planned_levels:
            level_mode = planned_levels[level]["relation_mode"]
            for group in groups:
                if group["relation_mode"] != level_mode:
                    raise ValueError(
                        f"planned fold group at L{level} has relation_mode "
                        f"{group['relation_mode']!r}, expected {level_mode!r}"
                    )
            planned_levels[level]["groups"] = groups
        else:
            summary.setdefault("warnings", []).append(
                f"planned fold groups for L{level} have no matching planned fold level"
            )
    if planned_levels:
        summary["planned_levels"] = [planned_levels[level] for level in sorted(planned_levels)]
        warning = public_opening_groups_warning(summary)
        if warning is not None:
            summary.setdefault("warnings", []).append(warning)
    if terminal_plan is not None:
        if terminal_plan["level"] is None:
            terminal_plan["level"] = (
                max(proof_levels)
                if proof_levels
                else max(planned_levels, default=-1) + 1
            )
        summary["terminal_plan"] = terminal_plan
    if proof_levels:
        summary["proof_levels"] = [proof_levels[level] for level in sorted(proof_levels)]
        grind_rows = [
            proof_levels[level]
            for level in sorted(proof_levels)
            if proof_levels[level].get("grind_nonce_val") is not None
        ]
        if grind_rows:
            summary["grind_levels"] = len(grind_rows)
            summary["grind_nonce_max"] = max(
                int(level["grind_nonce_val"]) for level in grind_rows
            )
            summary["grind_attempts_sum"] = sum(
                int(level["grind_attempts"]) for level in grind_rows
            )
            summary["grind_nonces"] = ",".join(
                str(level["grind_nonce_val"]) for level in grind_rows
            )
    if relation_phase_timings:
        summary["relation_phase_timings"] = relation_phase_timings
    if grinding_plan_summary is not None:
        grinding_plan_runs.sort(key=lambda run: int(run["run_index"]))
        expected_indices = list(range(int(grinding_plan_summary["run_count"])))
        actual_indices = [int(run["run_index"]) for run in grinding_plan_runs]
        if actual_indices != expected_indices:
            raise ValueError(
                "grinding plan run indices do not match the reported run count: "
                f"expected={expected_indices}, actual={actual_indices}"
            )
        total_run_bits = sum(int(run["run_nonce_bits"]) for run in grinding_plan_runs)
        if total_run_bits != int(grinding_plan_summary["total_nonce_bits"]):
            raise ValueError(
                "grinding plan run bits do not match the reported stream width: "
                f"runs={total_run_bits}, stream={grinding_plan_summary['total_nonce_bits']}"
            )
        expected_bytes = (total_run_bits + 7) // 8
        if expected_bytes != int(grinding_plan_summary["nonce_stream_bytes"]):
            raise ValueError(
                "grinding plan bit width does not match the reported stream bytes: "
                f"bits={total_run_bits}, bytes={grinding_plan_summary['nonce_stream_bytes']}"
            )
        expected_padding = expected_bytes * 8 - total_run_bits
        if expected_padding != int(grinding_plan_summary["padding_bits"]):
            raise ValueError(
                "grinding plan padding does not match the final byte: "
                f"expected={expected_padding}, actual={grinding_plan_summary['padding_bits']}"
            )
        grinding_plan_summary["runs"] = grinding_plan_runs
        summary["grinding_plan"] = grinding_plan_summary
    elif grinding_plan_runs:
        raise ValueError("grinding plan runs were emitted without a plan summary")
    if onehot_commit_schedules:
        summary["onehot_commit_schedules"] = onehot_commit_schedules

    return summary


def run_benchmark_case(
    binary: str,
    output_dir: pathlib.Path,
    case: BenchmarkCaseSpec,
    *,
    allow_legacy_relation_mode: bool = False,
) -> tuple[dict[str, object], int]:
    env = os.environ.copy()
    env["AKITA_MODE"] = case.mode
    env["AKITA_NUM_VARS"] = str(case.num_vars)
    env["AKITA_NUM_POLYS"] = str(case.num_polys)
    env["AKITA_SETUP_MODE"] = case.setup_mode
    env.setdefault("AKITA_PROFILE_TRACE", "0")
    env.setdefault("AKITA_PROFILE_SPAN_CLOSES", "0")
    env.setdefault("AKITA_PROFILE_LOG", "info")
    env.setdefault("AKITA_PROFILE_ANSI", "0")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = time_command(binary)
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    combined_log = completed.stdout + completed.stderr

    write_text(output_dir / "stdout.log", completed.stdout)
    write_text(output_dir / "stderr.log", completed.stderr)
    write_text(output_dir / "benchmark.log", combined_log)
    write_text(output_dir / "command.txt", " ".join(shlex.quote(part) for part in command) + "\n")

    summary = extract_summary(
        combined_log,
        mode=case.mode,
        num_vars=case.num_vars,
        num_polys=case.num_polys,
        setup_mode=case.setup_mode,
        allow_legacy_relation_mode=allow_legacy_relation_mode,
    )
    return_code = completed.returncode
    summary["command"] = command
    summary["binary"] = binary
    summary["exit_code"] = return_code
    summary["env"] = {
        "AKITA_MODE": env["AKITA_MODE"],
        "AKITA_NUM_VARS": env["AKITA_NUM_VARS"],
        "AKITA_NUM_POLYS": env["AKITA_NUM_POLYS"],
        "AKITA_SETUP_MODE": env["AKITA_SETUP_MODE"],
        "AKITA_PROFILE_TRACE": env["AKITA_PROFILE_TRACE"],
        "AKITA_PROFILE_SPAN_CLOSES": env["AKITA_PROFILE_SPAN_CLOSES"],
        "AKITA_PROFILE_LOG": env["AKITA_PROFILE_LOG"],
        "AKITA_PROFILE_ANSI": env["AKITA_PROFILE_ANSI"],
    }

    if return_code == 0:
        missing = missing_required_run_metrics(summary)
        if missing:
            summary["error"] = (
                "profile run exited successfully but did not emit required metrics: "
                + ", ".join(missing)
            )
            summary["failure_phase"] = infer_failure_phase(summary, missing[0])
            summary["exit_code"] = 1
            return_code = 1
    else:
        summary["error"] = f"profile run failed with exit code {return_code}"
        summary["failure_phase"] = infer_failure_phase(summary)

    write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary, return_code


def infer_failure_phase(summary: dict[str, object], first_missing: str | None = None) -> str:
    phase_by_metric = {
        "setup_s": "setup",
        "commit_s": "commit",
        "prove_total_s": "prove",
        "verify_total_s": "verify",
        "verify_single_total_s": "single-threaded verify",
        "proof_size_bytes": "proof summary",
        "accounted_bytes": "proof accounting",
        "consistent_proof_accounting": "proof accounting",
        "max_rss_kib": "memory",
        "crt_profile": "CRT profile",
        "crt_num_primes": "CRT profile",
        "crt_limb_bits": "CRT profile",
        "balanced_digit_safe_width": "CRT capacity",
        "raw_i8_safe_width": "CRT capacity",
        "ext_degree": "field role",
        "akita_levels": "proof levels",
        "planned_levels": "planned levels",
        "proof_levels": "proof levels",
        "tail_num_elems": "tail encoding",
        "tail_encoding": "tail encoding",
        "tail_bits_per_elem": "tail encoding",
    }
    if first_missing in phase_by_metric:
        return phase_by_metric[first_missing]
    for metric, phase in phase_by_metric.items():
        if metric == "consistent_proof_accounting":
            continue
        if summary.get(metric) is None:
            return phase
    return "unknown"


def compact_sample_summary(summary: dict[str, object]) -> dict[str, object]:
    sample = {
        "run_index": summary["run_index"],
        "exit_code": summary["exit_code"],
    }
    for key in SAMPLE_METRICS:
        if key in summary:
            sample[key] = summary[key]
    l2_observations = l2_grind_observations_for_run(summary)
    if l2_observations:
        sample["l2_grind_observations"] = l2_observations
    return sample


def l2_grind_observations_for_run(
    summary: dict[str, object],
) -> list[dict[str, object]]:
    planned_levels = summary.get("planned_levels")
    proof_levels = summary.get("proof_levels")
    if not isinstance(planned_levels, list):
        planned_levels = []
    l2_planned_levels = [
        planned
        for planned in planned_levels
        if isinstance(planned, dict) and planned.get("security_route") == "L2"
    ]
    terminal_plan = summary.get("terminal_plan")
    if isinstance(terminal_plan, dict) and terminal_plan.get("security_route") == "L2":
        l2_planned_levels.append(terminal_plan)
    run_failed = int(summary.get("exit_code", 0)) != 0
    if not isinstance(proof_levels, list):
        if l2_planned_levels and not run_failed:
            raise ValueError("successful L2 run is missing proof-level grinding diagnostics")
        return []
    proofs_by_level = {
        int(level["level"]): level
        for level in proof_levels
        if isinstance(level, dict) and level.get("level") is not None
    }
    observations = []
    for planned in l2_planned_levels:
        level = int(planned["level"])
        proof = proofs_by_level.get(level)
        if proof is None or proof.get("grind_attempts") is None:
            if run_failed:
                continue
            raise ValueError(f"L2 fold level {level} is missing grinding diagnostics")
        nonce = int(proof["grind_nonce_val"])
        attempts = int(proof["grind_attempts"])
        if attempts != nonce + 1:
            raise ValueError(f"L2 fold level {level} has inconsistent grinding attempts")
        response_l2_sq = proof.get("response_l2_sq")
        if response_l2_sq is None:
            if run_failed:
                continue
            raise ValueError(f"L2 fold level {level} is missing its measured response energy")
        response_l2_sq = int(response_l2_sq)
        response_l2_sq_cap = int(planned["response_l2_sq_cap"])
        if response_l2_sq > response_l2_sq_cap:
            raise ValueError(
                f"L2 fold level {level} exceeded its scheduled response cap: "
                f"{response_l2_sq} > {response_l2_sq_cap}"
            )
        observations.append(
            {
                "level": level,
                "response_l2_sq_cap": response_l2_sq_cap,
                "response_l2_sq": response_l2_sq,
                "cap_utilization": response_l2_sq / response_l2_sq_cap,
                "accepted_nonce": nonce,
                "attempts": attempts,
                "rejected_attempts": nonce,
            }
        )
    return observations


def combine_l2_grind_observations(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    combined: dict[tuple[int, int], dict[str, object]] = {}
    for summary in summaries:
        for observation in l2_grind_observations_for_run(summary):
            key = (int(observation["level"]), int(observation["response_l2_sq_cap"]))
            aggregate = combined.setdefault(
                key,
                {
                    "level": key[0],
                    "response_l2_sq_cap": key[1],
                    "samples": 0,
                    "attempts": 0,
                    "rejected_attempts": 0,
                    "accepted_nonces": [],
                    "response_l2_sq_values": [],
                },
            )
            aggregate["samples"] = int(aggregate["samples"]) + 1
            aggregate["attempts"] = int(aggregate["attempts"]) + int(
                observation["attempts"]
            )
            aggregate["rejected_attempts"] = int(aggregate["rejected_attempts"]) + int(
                observation["rejected_attempts"]
            )
            accepted_nonces = aggregate["accepted_nonces"]
            if not isinstance(accepted_nonces, list):
                raise TypeError("internal L2 grind nonce aggregate must be a list")
            accepted_nonces.append(int(observation["accepted_nonce"]))
            response_values = aggregate["response_l2_sq_values"]
            if not isinstance(response_values, list):
                raise TypeError("internal L2 response-energy aggregate must be a list")
            response_values.append(int(observation["response_l2_sq"]))
    result = []
    for key in sorted(combined):
        aggregate = combined[key]
        aggregate["observed_failure_rate"] = int(aggregate["rejected_attempts"]) / int(
            aggregate["attempts"]
        )
        response_values = aggregate["response_l2_sq_values"]
        if not isinstance(response_values, list) or not response_values:
            raise TypeError("internal L2 response-energy aggregate must be nonempty")
        aggregate["maximum_cap_utilization"] = max(response_values) / int(
            aggregate["response_l2_sq_cap"]
        )
        result.append(aggregate)
    return result


SUMMARY_CSV_COLUMNS = (
    "case_id",
    "benchmark_shard",
    "status",
    "failure_phase",
    "field_family",
    "workload",
    "config",
    "mode",
    "setup_contribution_mode",
    "num_vars",
    "num_polys",
    "runs",
    "setup_s",
    "setup_expand_s",
    "backend_prepare_s",
    "num_setup_field_elements",
    "setup_vector_bytes",
    "setup_ntt_cache_bytes",
    "verifier_ntt_cache_bytes",
    "crt_profile",
    "crt_num_primes",
    "crt_prime_modulus_bits",
    "crt_limb_bits",
    "balanced_digit_safe_width",
    "raw_i8_safe_width",
    "ext_degree",
    "commit_s",
    "prove_total_s",
    "verify_total_s",
    "verify_single_total_s",
    "prove_threads",
    "verify_multi_threads",
    "verify_single_threads",
    "max_rss_kib",
    "proof_size_bytes",
    "accounted_bytes",
    "akita_fold_bytes",
    "nonce_stream_bytes",
    "nonce_stream_bits",
    "nonce_stream_padding_bits",
    "tail_bytes",
    "akita_levels",
    "grind_levels",
    "grind_nonce_max",
    "grind_attempts_sum",
    "grind_nonces",
    "tail_num_elems",
    "tail_encoding",
    "tail_bits_per_elem",
    "exit_code",
    "error",
)


def write_summary_csv(path: pathlib.Path, cases: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for case in cases:
            row = {column: case.get(column, "") for column in SUMMARY_CSV_COLUMNS}
            row["status"] = case_status(case)
            writer.writerow(row)


def combine_grind_retry_observations(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    successful = [summary for summary in summaries if int(summary.get("exit_code", 0)) == 0]
    if not successful:
        return []
    retries_by_level: dict[int, list[int]] = {}
    for summary in successful:
        levels = grind_retries_by_level(summary)
        if levels is None:
            return []
        for level, retries in levels.items():
            if len(retries) != 1:
                raise ValueError("one benchmark run must emit one grinding nonce per fold")
            retries_by_level.setdefault(level, []).append(retries[0])
    if any(len(retries) != len(successful) for retries in retries_by_level.values()):
        raise ValueError("successful benchmark runs emitted different grinding fold levels")
    return [
        {"level": level, "retries": retries_by_level[level]}
        for level in sorted(retries_by_level)
    ]


def combine_case_run_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    combined = dict(summaries[0])
    combined["runs"] = len(summaries)
    combined["samples"] = [compact_sample_summary(summary) for summary in summaries]
    l2_grind_observations = combine_l2_grind_observations(summaries)
    if l2_grind_observations:
        combined["l2_grind_observations"] = l2_grind_observations
    grind_retry_observations = combine_grind_retry_observations(summaries)
    if grind_retry_observations:
        combined["grind_retry_observations"] = grind_retry_observations

    phase_samples: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for summary in summaries:
        timings = summary.get("relation_phase_timings")
        if not isinstance(timings, list):
            continue
        for timing in timings:
            if not isinstance(timing, dict):
                continue
            key = (
                str(timing["verify_mode"]),
                str(timing["relation_mode"]),
                str(timing["phase"]),
            )
            phase_samples.setdefault(key, []).append(timing)
    if phase_samples:
        combined["relation_phase_timings"] = [
            {
                "verify_mode": key[0],
                "relation_mode": key[1],
                "phase": key[2],
                "calls": int(statistics.median(int(value["calls"]) for value in values)),
                "mean_elapsed_nanos": int(
                    statistics.median(int(value["mean_elapsed_nanos"]) for value in values)
                ),
                "total_elapsed_nanos": int(
                    statistics.median(int(value["total_elapsed_nanos"]) for value in values)
                ),
            }
            for key, values in sorted(phase_samples.items())
        ]

    for key in TIMING_SAMPLE_METRICS:
        values = [float(summary[key]) for summary in summaries if summary.get(key) is not None]
        if values:
            combined[key] = statistics.median(values)

    for key in GRIND_SAMPLE_METRICS:
        values = [float(summary[key]) for summary in summaries if summary.get(key) is not None]
        if values:
            combined[key] = statistics.median(values)

    rss_values = [int(summary["max_rss_kib"]) for summary in summaries if summary.get("max_rss_kib")]
    if rss_values:
        combined["max_rss_kib"] = max(rss_values)

    failed = [summary for summary in summaries if int(summary.get("exit_code", 0)) != 0]
    if failed:
        latest_failure = failed[-1]
        combined["exit_code"] = latest_failure.get("exit_code", 1)
        combined["error"] = latest_failure.get("error", "profile run failed")
        combined["failure_phase"] = latest_failure.get("failure_phase", "unknown")

    return combined


@dataclass(frozen=True)
class ScheduledRun:
    """One planned execution of a benchmark binary."""

    binary: str
    summary_dir: pathlib.Path  # root whose summary.json this run's case feeds
    run_dir: pathlib.Path  # directory for this single execution's output
    case: BenchmarkCaseSpec
    kind: str  # "warmup" or "measured"
    run_index: int  # 0 for warm-ups, 1..runs for measured
    allow_legacy_relation_mode: bool = False


def plan_case_runs(
    binary: str,
    summary_dir: pathlib.Path,
    case: BenchmarkCaseSpec,
    runs: int,
    warmups: int,
    allow_legacy_relation_mode: bool = False,
) -> list[ScheduledRun]:
    """All executions of one case for one binary, in execution order."""
    case_dir = summary_dir / case.case_id
    schedule = [
        ScheduledRun(
            binary,
            summary_dir,
            case_dir / f"warmup-{warmup_index}",
            case,
            "warmup",
            0,
            allow_legacy_relation_mode,
        )
        for warmup_index in range(1, warmups + 1)
    ]
    for run_index in range(1, runs + 1):
        run_dir = case_dir if runs == 1 else case_dir / f"run-{run_index}"
        schedule.append(
            ScheduledRun(
                binary,
                summary_dir,
                run_dir,
                case,
                "measured",
                run_index,
                allow_legacy_relation_mode,
            )
        )
    return schedule


def execute_schedule(
    schedule: list[ScheduledRun],
) -> tuple[list[tuple[ScheduledRun, dict[str, object]]], int]:
    """Execute runs in order, recording the summaries that feed aggregation.

    Successful warm-up output is discarded. The first failure records its
    failure summary and cancels the case for every binary — rerunning the
    failing binary would repeat the same error, and a pairwise comparison
    is meaningless once one side fails. Remaining cases still run. Returns
    the recorded (run, summary) pairs and the first non-zero exit code,
    0 otherwise.
    """
    results: list[tuple[ScheduledRun, dict[str, object]]] = []
    failed_cases: set[str] = set()
    overall_return_code = 0
    for run in schedule:
        if run.case.case_id in failed_cases:
            continue
        summary, return_code = run_benchmark_case(
            run.binary,
            run.run_dir,
            run.case,
            allow_legacy_relation_mode=run.allow_legacy_relation_mode,
        )
        summary["run_index"] = run.run_index
        if return_code != 0:
            failed_cases.add(run.case.case_id)
            if overall_return_code == 0:
                overall_return_code = return_code
            results.append((run, summary))
        elif run.kind == "measured":
            results.append((run, summary))
    return results, overall_return_code


def failure_summaries_by_case(
    results: list[tuple[ScheduledRun, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    """Map case_id to the first recorded failure summary for that case."""
    failures: dict[str, dict[str, object]] = {}
    for run, summary in results:
        if int(summary.get("exit_code", 0)) != 0:
            failures.setdefault(run.case.case_id, summary)
    return failures


def propagate_sibling_case_failure(
    case_summaries: list[dict[str, object]],
    failure: dict[str, object],
) -> list[dict[str, object]]:
    """Mirror a paired-binary failure onto the sibling output root."""
    if any(int(summary.get("exit_code", 0)) != 0 for summary in case_summaries):
        return case_summaries
    propagated = dict(failure)
    propagated["error"] = (
        "case cancelled after the paired binary failed: "
        f"{failure.get('error', 'profile run failed')}"
    )
    propagated["exit_code"] = failure.get("exit_code", 1)
    propagated["failure_phase"] = failure.get("failure_phase", "unknown")
    return [*case_summaries, propagated]


def write_aggregate_summaries(
    summary_dirs: list[pathlib.Path],
    cases: list[BenchmarkCaseSpec],
    results: list[tuple[ScheduledRun, dict[str, object]]],
    warmups: int,
    benchmark_shard: str = "",
) -> None:
    """Aggregate recorded run summaries into summary.json/summary.csv per root."""
    generated_at = datetime.now(timezone.utc).isoformat()
    failures_by_case = failure_summaries_by_case(results)
    for summary_dir in summary_dirs:
        aggregate: dict[str, object] = {
            "schema_version": 3,
            "generated_at": generated_at,
            "warmups": warmups,
            "cases": [],
        }
        for case in cases:
            case_summaries = [
                summary
                for run, summary in results
                if run.summary_dir == summary_dir and run.case.case_id == case.case_id
            ]
            failure = failures_by_case.get(case.case_id)
            if failure is not None:
                case_summaries = propagate_sibling_case_failure(case_summaries, failure)
            if case_summaries:
                combined = combine_case_run_summaries(case_summaries)
                if benchmark_shard:
                    combined["benchmark_shard"] = benchmark_shard
                aggregate["cases"].append(combined)
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_text(
            summary_dir / "summary.json",
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        )
        write_summary_csv(summary_dir / "summary.csv", aggregate["cases"])


def run_benchmark(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")

    if bool(args.baseline_binary) != bool(args.baseline_output_dir):
        raise ValueError("--baseline-binary and --baseline-output-dir must be set together")
    binaries: list[tuple[str, pathlib.Path, bool]] = [(args.binary, output_dir, False)]
    if args.baseline_binary:
        baseline_dir = pathlib.Path(args.baseline_output_dir)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        binaries.append((args.baseline_binary, baseline_dir, True))

    cases = configured_cases(args)
    schedule: list[ScheduledRun] = []
    for case in cases:
        plans = [
            plan_case_runs(
                binary,
                summary_dir,
                case,
                args.runs,
                args.warmups,
                allow_legacy_relation_mode,
            )
            for binary, summary_dir, allow_legacy_relation_mode in binaries
        ]
        plan_lengths = {len(plan) for plan in plans}
        if len(plan_lengths) != 1:
            raise RuntimeError(f"internal benchmark schedule length mismatch: {sorted(plan_lengths)}")
        # Interleave the binaries' plans: each warm-up/measured slot runs
        # every binary back-to-back (PR, base, PR, base, ...), so
        # machine-state drift on shared runners lands on both sides of each
        # adjacent pair instead of on one whole block.
        schedule.extend(run for slot in zip(*plans) for run in slot)

    results, overall_return_code = execute_schedule(schedule)
    write_aggregate_summaries(
        [summary_dir for _, summary_dir, _ in binaries],
        cases,
        results,
        args.warmups,
        args.benchmark_shard,
    )
    return overall_return_code


def write_failure_summary(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc).isoformat()

    cases = []
    for case in configured_cases(args):
        metadata = case_metadata(case.mode)
        cases.append(
            {
                "schema_version": CASE_SCHEMA_VERSION,
                "benchmark": benchmark_name(
                    case.mode, case.num_vars, case.num_polys, case.setup_mode
                ),
                "mode": case.mode,
                "setup_contribution_mode": case.setup_mode,
                "field_family": metadata.field_family,
                "workload": metadata.workload,
                "workload_label": metadata.workload_label,
                "config": metadata.config,
                "num_vars": case.num_vars,
                "num_polys": case.num_polys,
                "case_id": case.case_id,
                "benchmark_shard": args.benchmark_shard,
                "collected_at": collected_at,
                "runs": 0,
                "samples": [],
                "exit_code": 1,
                "failure_phase": args.failure_phase,
                "error": args.error,
            }
        )

    aggregate_summary: dict[str, object] = {
        "schema_version": 3,
        "generated_at": collected_at,
        "cases": cases,
    }
    write_text(
        output_dir / "summary.json", json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n"
    )
    write_summary_csv(output_dir / "summary.csv", cases)
    return 0


def load_summary(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_case_summary(summary: dict[str, object]) -> dict[str, object]:
    normalized = dict(summary)
    allow_legacy_relation_mode = int(normalized.get("schema_version", 0)) < 9
    mode = str(normalized["mode"])
    num_vars = int(normalized["num_vars"])
    num_polys = int(normalized.get("num_polys", 1))
    setup_mode = normalize_setup_mode(normalized.get("setup_contribution_mode", "direct"))
    metadata = case_metadata(mode)
    normalized["num_polys"] = num_polys
    normalized["setup_contribution_mode"] = setup_mode
    normalized["case_id"] = case_id(mode, num_vars, num_polys, setup_mode)
    normalized["benchmark"] = benchmark_name(mode, num_vars, num_polys, setup_mode)
    normalized["field_family"] = metadata.field_family
    normalized["workload"] = metadata.workload
    normalized["workload_label"] = metadata.workload_label
    normalized["config"] = metadata.config
    planned_levels = normalized.get("planned_levels")
    if isinstance(planned_levels, list):
        normalized_levels = []
        for raw_level in planned_levels:
            level = dict(raw_level)
            legacy_d = int(level.get("d", level.get("d_a", 0)))
            level.setdefault("d_a", legacy_d)
            level.setdefault("d_b", legacy_d)
            level.setdefault("d_d", legacy_d)
            legacy_log_basis = level.get("log_basis")
            if legacy_log_basis is not None:
                level.setdefault("log_basis_inner", legacy_log_basis)
                level.setdefault("log_basis_outer", legacy_log_basis)
                level.setdefault("log_basis_open", legacy_log_basis)
            legacy_commit_digits = level.get("delta_commit")
            if legacy_commit_digits is not None:
                level.setdefault("num_digits_inner", legacy_commit_digits)
            legacy_open_digits = level.get("delta_open")
            if legacy_open_digits is not None:
                level.setdefault("num_digits_outer", legacy_open_digits)
                level.setdefault("num_digits_open", legacy_open_digits)
            current_w_len = level.get("current_w_len")
            if not isinstance(current_w_len, list):
                level["current_w_len"] = level.get("current_w_groups", [])
            level.setdefault("setup_prefix_natural_field_elements", 0)
            level.setdefault("setup_prefix_padded_field_elements", 0)
            relation_mode = level.get("relation_mode")
            if relation_mode is None:
                if not allow_legacy_relation_mode:
                    raise ValueError("current profile artifact is missing relation_mode")
                relation_mode = "quotient_lift"
                level["relation_mode"] = relation_mode
            if relation_mode not in VALID_RELATION_MODES:
                raise ValueError(
                    f"profile artifact has invalid relation_mode {relation_mode!r}"
                )
            if "num_digits_quotient" not in level:
                if not allow_legacy_relation_mode:
                    raise ValueError(
                        "current profile artifact is missing num_digits_quotient"
                    )
            else:
                validate_relation_quotient_digits(
                    "profile artifact level",
                    str(relation_mode),
                    int(level["num_digits_quotient"]),
                )
            groups = level.get("groups")
            if isinstance(groups, list):
                for group in groups:
                    if isinstance(group, dict):
                        group_mode = group.get("relation_mode")
                        if group_mode is None:
                            if not allow_legacy_relation_mode:
                                raise ValueError(
                                    "current profile artifact group is missing relation_mode"
                                )
                            group_mode = relation_mode
                            group["relation_mode"] = group_mode
                        if group_mode not in VALID_RELATION_MODES:
                            raise ValueError(
                                "profile artifact group has invalid relation_mode "
                                f"{group_mode!r}"
                            )
                        if group_mode != relation_mode:
                            raise ValueError(
                                "profile artifact group relation_mode disagrees with its level"
                            )
                        if "num_digits_quotient" not in group:
                            if not allow_legacy_relation_mode:
                                raise ValueError(
                                    "current profile artifact group is missing "
                                    "num_digits_quotient"
                                )
                        else:
                            validate_relation_quotient_digits(
                                "profile artifact group",
                                str(group_mode),
                                int(group["num_digits_quotient"]),
                            )
            normalized_levels.append(level)
        normalized["planned_levels"] = normalized_levels
        warning = public_opening_groups_warning(normalized)
        if warning is not None and warning not in normalized.get("warnings", []):
            normalized.setdefault("warnings", []).append(warning)
    # All production CRT profiles currently use moduli below 2^30 stored in
    # signed 32-bit limbs. Old baseline artifacts only recorded the storage
    # width, so normalize their missing modulus width here.
    if normalized.get("crt_limb_bits") == 32:
        normalized.setdefault("crt_prime_modulus_bits", 30)
    return normalized


def load_case_summaries(path: pathlib.Path) -> list[dict[str, object]]:
    raw = load_summary(path)
    cases = raw.get("cases")
    if isinstance(cases, list):
        return [normalize_case_summary(case) for case in cases]
    return [normalize_case_summary(raw)]


def load_optional_case_summaries(dir_path: str) -> dict[str, dict[str, object]] | None:
    if not dir_path:
        return None
    summary_path = pathlib.Path(dir_path) / "summary.json"
    if not summary_path.exists():
        return None
    cases = load_case_summaries(summary_path)
    return {str(case["case_id"]): case for case in cases}


def md_text(value: object) -> str:
    """Escape untrusted text before embedding it in Markdown/HTML output."""

    text = html.escape(str(value), quote=False).replace("\\", "\\\\")
    for char in "`*_{}[]()#+-.!|":
        text = text.replace(char, f"\\{char}")
    return text


def code_text(value: object) -> str:
    return f"<code>{html.escape(str(value), quote=False)}</code>"


def commit_ref(sha: str | None) -> str | None:
    if not sha:
        return None
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", sha) is None:
        return code_text(sha)
    short = sha[:7]
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        return f"[`{short}`](https://github.com/{repo}/commit/{sha})"
    return code_text(short)


def workflow_run_ref() -> str | None:
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        return None
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    label = f"run {run_id}"
    if run_attempt:
        label = f"{label} attempt {run_attempt}"
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        return f"[{label}]({server}/{repo}/actions/runs/{run_id})"
    return code_text(label)


def fmt_seconds(value: float) -> str:
    return f"{value:.3f}"


def fmt_milliseconds(value: float) -> str:
    return f"{value * 1_000.0:.1f}"


def fmt_mib(value_kib: float) -> str:
    return f"{value_kib / 1024.0:.1f}"


def fmt_mib_from_bytes(value_bytes: float) -> str:
    return f"{value_bytes / (1024.0 * 1024.0):.1f}"


def fmt_mib_with_exact_bytes(value_bytes: float) -> str:
    return (
        f"{fmt_mib_from_bytes(value_bytes)}<br>"
        f"<sub>{fmt_bytes(value_bytes)} bytes</sub>"
    )


def case_status(summary: dict[str, object]) -> str:
    return "ok" if int(summary.get("exit_code", 0)) == 0 else "fail"


def section_title(summary: dict[str, object]) -> str:
    return human_case_label(summary)


@dataclass(frozen=True)
class Metric:
    key: str
    name: str
    unit: str
    value_formatter: callable


MEASURED_METRICS = [
    Metric("setup_s", "Setup", "s", fmt_seconds),
    Metric("setup_expand_s", "↳ Setup expansion", "s", fmt_seconds),
    Metric("backend_prepare_s", "↳ Backend preparation", "s", fmt_seconds),
    Metric("commit_s", "Commit", "s", fmt_seconds),
    Metric("prove_total_s", "Prove", "s", fmt_seconds),
    Metric("verify_total_s", "Verify, multi-threaded", "ms", fmt_milliseconds),
    Metric("verify_single_total_s", "Verify, single-threaded", "ms", fmt_milliseconds),
    Metric("max_rss_kib", "Peak process RSS", "MiB", fmt_mib),
    Metric(
        "num_setup_field_elements",
        "Setup field elements",
        "field elements",
        fmt_count,
    ),
    Metric("setup_vector_bytes", "Setup vector", "MiB", fmt_mib_with_exact_bytes),
    Metric("setup_ntt_cache_bytes", "Prepared NTT cache", "MiB", fmt_mib_with_exact_bytes),
    Metric("verifier_ntt_cache_bytes", "Verifier NTT cache", "MiB", fmt_mib_with_exact_bytes),
    Metric("proof_size_bytes", "Proof size", "bytes", fmt_bytes),
    Metric("akita_fold_bytes", "Recursive fold payload", "bytes", fmt_bytes),
    Metric("tail_bytes", "Final-witness tail", "bytes", fmt_bytes),
]


def render_metric_row(
    metric: Metric,
    current: dict[str, object],
    baselines: list[tuple[str, dict[str, object] | None]],
    main_baseline: dict[str, object] | None,
) -> str:
    current_value = current.get(metric.key)
    if current_value is None:
        return ""

    columns = [metric.value_formatter(float(current_value))]
    for _, summary in baselines:
        if summary is None or summary.get(metric.key) is None:
            columns.append("n/a")
        else:
            columns.append(metric.value_formatter(float(summary[metric.key])))

    columns.append(numeric_delta(current, main_baseline, metric.key))
    return f"| {metric.name} | " + " | ".join(columns) + f" | {metric.unit} |"


def render_relation_phase_timings(summary: dict[str, object]) -> None:
    timings = summary.get("relation_phase_timings")
    if not isinstance(timings, list) or not timings:
        return
    print("#### Verifier relation phases")
    print()
    print("| Verifier | Relation mode | Phase | Calls | Replay total | Mean per call |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    phase_labels = {
        "coefficient_functional_preparation": "Functional preparation",
        "structured_groups": "Structured groups",
        "setup_scan": "Setup scan",
        "quotient_tail": "Quotient tail",
        "complete_stage2": "Complete Stage 2",
    }
    relation_labels = {"quotient": "Quotient lift", "reduced": "Reduced evaluation"}
    for timing in timings:
        if not isinstance(timing, dict):
            continue
        print(
            "| "
            + " | ".join(
                [
                    md_text(str(timing["verify_mode"])),
                    relation_labels.get(
                        str(timing["relation_mode"]), str(timing["relation_mode"])
                    ),
                    phase_labels.get(str(timing["phase"]), str(timing["phase"])),
                    fmt_count(float(timing["calls"])),
                    fmt_milliseconds(float(timing["total_elapsed_nanos"]) / 1_000_000_000),
                    fmt_milliseconds(float(timing["mean_elapsed_nanos"]) / 1_000_000_000),
                ]
            )
            + " |"
        )
    totals = []
    for key, label in [
        ("verify_total_s", "multi threaded"),
        ("verify_single_total_s", "single threaded"),
    ]:
        if summary.get(key) is not None:
            totals.append(f"{label} `{fmt_milliseconds(float(summary[key]))}ms`")
    print()
    if totals:
        print("- Complete verification: " + "; ".join(totals) + ".")
    print(
        "- Phase values come from one untimed honest replay after the measured verification; "
        "the complete-verification totals above remain the normal measured runs."
    )


def parameter_value(
    current: dict[str, object],
    baseline: dict[str, object] | None,
    keys: tuple[str, ...],
    render: callable,
) -> str | None:
    if any(current.get(key) is None for key in keys):
        return None
    current_values = tuple(current[key] for key in keys)
    rendered = render(*current_values)
    if baseline is None or any(baseline.get(key) is None for key in keys):
        return rendered
    baseline_values = tuple(baseline[key] for key in keys)
    if current_values == baseline_values:
        return rendered
    return f"{rendered} Merge base: {render(*baseline_values)}"


def render_execution_parameters(
    current: dict[str, object], baseline: dict[str, object] | None
) -> None:
    rows = [("Internal mode", code_text(current["mode"]))]

    if current.get("crt_profile") is not None:
        rows.append(("CRT profile", code_text(current["crt_profile"])))

    crt = parameter_value(
        current,
        baseline,
        ("crt_num_primes", "crt_prime_modulus_bits", "crt_limb_bits"),
        lambda primes, modulus_bits, limb_bits: (
            f"{code_text(fmt_count(float(primes)))} prime moduli of "
            f"{code_text(fmt_count(float(modulus_bits)))} bits in signed "
            f"{code_text(f'i{int(limb_bits)}')} lanes."
        ),
    )
    if crt is not None:
        rows.append(("CRT arithmetic", crt))

    safe_width = parameter_value(
        current,
        baseline,
        ("balanced_digit_safe_width", "raw_i8_safe_width"),
        lambda balanced, raw_i8: (
            f"{code_text(fmt_count(float(balanced)))} balanced digit terms and "
            f"{code_text(fmt_count(float(raw_i8)))} signed i8 terms."
        ),
    )
    if safe_width is not None:
        rows.append(("Safe accumulation limit", safe_width))

    extension_degree = parameter_value(
        current,
        baseline,
        ("ext_degree",),
        lambda degree: code_text(fmt_count(float(degree))),
    )
    if extension_degree is not None:
        rows.append(("Claim extension degree", extension_degree))

    verifier_threads = parameter_value(
        current,
        baseline,
        ("verify_multi_threads", "verify_single_threads"),
        lambda multi, single: (
            f"{code_text(fmt_count(float(multi)))} for the multi-threaded timing and "
            f"{code_text(fmt_count(float(single)))} for the single-threaded timing."
        ),
    )
    if verifier_threads is not None:
        rows.append(("Verifier threads", verifier_threads))

    print("#### Execution parameters")
    print()
    for label, value in rows:
        print(f"- {label}: {value}")


def optional_value_with_baseline_delta(
    current: dict[str, object],
    baseline: dict[str, object] | None,
    key: str,
    formatter: callable,
    unit: str = "",
    compare_to_baseline: bool = False,
    comparison_label: str = " vs base",
) -> str:
    value = current.get(key)
    if value is None:
        return "n/a"
    if compare_to_baseline and baseline is None:
        return f"{formatter(float(value))}{unit}<br><sub>no matching merge-base case</sub>"
    baseline_value = baseline.get(key) if baseline is not None else None
    return value_with_baseline_delta(
        value,
        baseline_value,
        formatter,
        unit,
        compare_to_baseline,
        comparison_label,
    )


def field_family_bits(field_family: object) -> int | None:
    match = re.fullmatch(r"fp(\d+)", str(field_family))
    return int(match.group(1)) if match else None


def field_family_sort_key(case: dict[str, object]) -> int:
    """Order report rows by field width so fp32/fp64 lead and every fp128 case
    groups together. Non-`fp<bits>` families sort last; ties keep input order
    because Python's sort is stable."""
    bits = field_family_bits(case.get("field_family", ""))
    return bits if bits is not None else 1 << 30


def report_case_sort_key(case: dict[str, object]) -> tuple[object, ...]:
    """Keep workflow shards together; use field order for legacy artifacts."""
    shard = str(case.get("benchmark_shard", ""))
    if shard:
        prefix = re.match(r"(\d+)-", shard)
        shard_index = int(prefix.group(1)) if prefix else 1 << 30
        return (0, shard_index, shard)
    return (1, field_family_sort_key(case))


def human_case_label(summary: dict[str, object]) -> str:
    """Render a short workload label without planner-selected dimensions."""
    field_family = str(summary.get("field_family", "field"))
    bits = field_family_bits(field_family)
    field_segment = f"Fp{bits}" if bits is not None else field_family
    workload = str(summary.get("workload", "dense"))
    metadata = case_metadata(str(summary.get("mode", "")))
    setup_mode = str(summary.get("setup_contribution_mode", "direct"))
    config = str(summary.get("config", ""))
    chunk_variant = re.search(r"W\d+R\d+", config, flags=re.IGNORECASE)

    if metadata.opening_topology == "multi_group":
        label = f"{field_segment} multi-group"
        if chunk_variant:
            label += f" {chunk_variant.group(0).upper()}"
        label += (
            f" (final nv{int(summary['num_vars'])}, "
            f"{int(summary.get('num_polys', 1))} polys total)"
        )
        return f"{label}, {setup_mode} setup check"

    workload_token = "one-hot" if workload == "onehot" else "dense"
    label = f"{field_segment} {workload_token} nv{int(summary['num_vars'])}"
    if chunk_variant:
        label += f" {chunk_variant.group(0).upper()}"
    num_polys = int(summary.get("num_polys", 1))
    if num_polys > 1:
        label += f", {num_polys} polynomials"
    return f"{label}, {setup_mode} setup check"


def public_opening_group_candidates(
    summary: dict[str, object],
) -> list[dict[str, object]]:
    levels = summary.get("planned_levels")
    if not isinstance(levels, list):
        return []
    root = next(
        (
            level
            for level in levels
            if isinstance(level, dict) and int(level.get("level", -1)) == 0
        ),
        None,
    )
    if root is None or not isinstance(root.get("groups"), list):
        return []
    return [
        group
        for group in root["groups"]
        if isinstance(group, dict)
        and group.get("group_role") in ("precommitted", "final")
        and int(group.get("public_num_polynomials", 0)) > 0
    ]


def public_opening_groups_warning(summary: dict[str, object]) -> str | None:
    groups = public_opening_group_candidates(summary)
    if not groups:
        return None
    described = sum(int(group["public_num_polynomials"]) for group in groups)
    expected = int(summary.get("num_polys", 1))
    if described == expected:
        return None
    return (
        f"public opening groups describe {described} of {expected} polynomials; "
        "using the generic opening statement"
    )


def public_opening_groups(summary: dict[str, object]) -> list[dict[str, object]]:
    groups = public_opening_group_candidates(summary)
    if public_opening_groups_warning(summary) is not None:
        return []
    return groups


def join_phrases(phrases: list[str]) -> str:
    if len(phrases) < 2:
        return "".join(phrases)
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def public_opening_statement(summary: dict[str, object]) -> str:
    """Describe the PCS statement independently of benchmark witness generation."""
    metadata = case_metadata(str(summary.get("mode", "")))
    bits = field_family_bits(metadata.field_family)
    field = f"Fp{bits}" if bits is not None else metadata.field_family
    if metadata.opening_topology == "multi_group":
        groups = public_opening_groups(summary)
        if groups:
            descriptions = []
            for group in groups:
                num_vars = int(group["public_num_vars"])
                num_polynomials = int(group["public_num_polynomials"])
                if num_polynomials == 1:
                    descriptions.append(
                        f"one {num_vars} variable polynomial at its own point"
                    )
                else:
                    descriptions.append(
                        f"{num_polynomials} {num_vars} variable polynomials at one shared point"
                    )
            total_polynomials = sum(
                int(group["public_num_polynomials"]) for group in groups
            )
            return (
                f"Over {field}, {total_polynomials} polynomials in {len(groups)} groups: "
                f"{join_phrases(descriptions)}."
            )
        return (
            f"Over {field}, {int(summary.get('num_polys', 1))} polynomials are split "
            "across independent opening groups."
        )

    num_vars = int(summary["num_vars"])
    num_polys = int(summary.get("num_polys", 1))
    if num_polys == 1:
        return (
            f"Over {field}, one committed {num_vars} variable multilinear polynomial "
            f"with 2^{num_vars} coefficients is opened at one {num_vars} coordinate point."
        )
    return (
        f"Over {field}, {num_polys} committed {num_vars} variable multilinear "
        f"polynomials are opened at one shared {num_vars} coordinate point."
    )


def render_profile_definitions(cases: list[dict[str, object]]) -> None:
    shards: dict[str, list[str]] = {}
    for case in cases:
        shard = str(case.get("benchmark_shard", "")) or "legacy artifact (shard not recorded)"
        label = human_case_label(case)
        labels = shards.setdefault(shard, [])
        if label not in labels:
            labels.append(label)

    print("### Benchmark shards")
    print()
    print("| CI shard | Profiles |")
    print("| --- | --- |")
    for shard, labels in shards.items():
        rendered_labels = "<br>".join(md_text(label) for label in labels)
        print(f"| {code_text(shard)} | {rendered_labels} |")

    grouped: dict[str, list[str]] = {}
    for case in cases:
        statement = public_opening_statement(case)
        label = human_case_label(case)
        labels = grouped.setdefault(statement, [])
        if label not in labels:
            labels.append(label)

    print()
    print("### Public opening statements")
    print()
    print("| Public opening statement | Profiles |")
    print("| --- | --- |")
    for statement, labels in grouped.items():
        rendered_labels = "<br>".join(md_text(label) for label in labels)
        print(f"| {md_text(statement)} | {rendered_labels} |")

    if any(case.get("workload") == "onehot" for case in cases):
        print()
        print(
            "One-hot profiles generate deterministic witnesses with one `1` in every "
            f"consecutive chunk of `{ONEHOT_ARITY}` coefficients. This witness shape is not "
            "a separate public claim."
        )
    if any(
        case_metadata(str(case.get("mode", ""))).opening_topology == "multi_group"
        for case in cases
    ):
        print()
        print(
            "Direct evaluates the public setup contribution during Stage 2. Recursive "
            "carries the same check through a Stage 3 setup-product sumcheck. Both modes "
            "execute the complete fold schedule and terminal verification."
        )
    chunk_variants = sorted(
        {
            match.group(0).upper()
            for case in cases
            if (match := re.search(r"W\d+R\d+", str(case.get("config", ""))))
        }
    )
    if chunk_variants:
        variants = ", ".join(f"`{variant}`" for variant in chunk_variants)
        print()
        print(
            f"The chunked profiles {variants} divide the witness relation into the stated "
            "number of exact chunks for the first two fold levels."
        )
    if any(
        "mixed" in str(case.get("config", "")).lower()
        or "adaptive" in str(case.get("config", "")).lower()
        for case in cases
    ):
        print()
        print(
            "Generated profiles may select different A, B, and D ring dimensions at "
            "different fold levels. The short profile names omit those dimensions."
        )


def fold_dimension_schedule(summary: dict[str, object]) -> str:
    """Render consecutive distinct A/B/D tuples from the resolved fold plan."""
    levels = summary.get("planned_levels")
    if not isinstance(levels, list):
        return "—"
    tuples: list[tuple[int, int, int]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        try:
            dims = (int(level["d_a"]), int(level["d_b"]), int(level["d_d"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not tuples or tuples[-1] != dims:
            tuples.append(dims)
    if not tuples:
        return "—"
    return " → ".join(f"{d_a}/{d_b}/{d_d}" for d_a, d_b, d_d in tuples)


def terminal_response_metric_value(
    current: dict[str, object],
    baseline: dict[str, object] | None,
    show_delta: bool,
) -> str:
    total = optional_value_with_baseline_delta(
        current,
        baseline,
        "tail_bytes",
        fmt_bytes,
        " bytes",
        show_delta,
        "",
    )
    if all(
        current.get(key) is not None
        for key in ("tail_z_bytes", "tail_e_bytes", "tail_t_bytes")
    ):
        total += (
            "<br><sub>"
            f"z {fmt_bytes(float(current['tail_z_bytes']))} · "
            f"e {fmt_bytes(float(current['tail_e_bytes']))} · "
            f"t {fmt_bytes(float(current['tail_t_bytes']))}"
            "</sub>"
        )
    return total


def grind_retries_by_level(summary: dict[str, object]) -> dict[int, list[int]] | None:
    observations = summary.get("grind_retry_observations")
    if isinstance(observations, list):
        result: dict[int, list[int]] = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            retries = observation.get("retries")
            if not isinstance(retries, list):
                continue
            result[int(observation["level"])] = [int(value) for value in retries]
        return result

    proof_levels = summary.get("proof_levels")
    if not isinstance(proof_levels, list):
        return None
    result = {}
    for level in proof_levels:
        if not isinstance(level, dict) or level.get("grind_nonce_val") is None:
            continue
        result[int(level["level"])] = [int(level["grind_nonce_val"])]
    return result or None


def grinding_retries_metric_value(
    current: dict[str, object],
    baseline: dict[str, object] | None,
) -> str:
    current_levels = grind_retries_by_level(current)
    if current_levels is None:
        return "n/a"
    baseline_levels = grind_retries_by_level(baseline) if baseline is not None else None
    visible_levels = sorted(
        level
        for level in set(current_levels) | set(baseline_levels or {})
        if any(current_levels.get(level, []))
        or any((baseline_levels or {}).get(level, []))
    )

    def render(levels: dict[int, list[int]] | None) -> str:
        if levels is None:
            return "n/a"
        if not visible_levels:
            return "None"
        rows = []
        for level in visible_levels:
            retries = levels.get(level, [])
            values = " / ".join(fmt_count(float(value)) for value in retries) or "n/a"
            rows.append(f"L{level}: {values}")
        return "<br>".join(rows)

    current_value = render(current_levels)
    if baseline is None:
        return f"{current_value}<br><sub>no matching merge-base case</sub>"
    return exact_choice(current_value, render(baseline_levels))


def render_matrix_summary(
    current_cases: list[dict[str, object]],
    main_baseline: dict[str, dict[str, object]] | None,
) -> None:
    tables = [
        (
            "Phase time",
            [
                Metric("setup_s", "Setup", " s", fmt_seconds),
                Metric("commit_s", "Commit", " s", fmt_seconds),
                Metric("prove_total_s", "Prove", " s", fmt_seconds),
                Metric(
                    "verify_total_s",
                    "Verify, multi-threaded",
                    " ms",
                    fmt_milliseconds,
                ),
                Metric(
                    "verify_single_total_s",
                    "Verify, single-threaded",
                    " ms",
                    fmt_milliseconds,
                ),
            ],
            False,
        ),
        (
            "Proof size and protocol shape",
            [
                Metric("proof_size_bytes", "Total proof", " bytes", fmt_bytes),
                Metric("akita_fold_bytes", "Fold payload", " bytes", fmt_bytes),
                Metric("tail_bytes", "Terminal response", " bytes", fmt_bytes),
                Metric("akita_levels", "Fold levels", "", fmt_count),
            ],
            True,
        ),
        (
            "Memory and setup size",
            [
                Metric("setup_vector_bytes", "Setup vector", " MiB", fmt_mib_from_bytes),
                Metric(
                    "setup_ntt_cache_bytes",
                    "Prepared NTT cache",
                    " MiB",
                    fmt_mib_from_bytes,
                ),
                Metric(
                    "verifier_ntt_cache_bytes",
                    "Verifier NTT cache",
                    " MiB",
                    fmt_mib_from_bytes,
                ),
                Metric("max_rss_kib", "Peak RSS", " MiB", fmt_mib),
            ],
            False,
        ),
    ]

    for table_index, (title, metrics, include_fold_schedule) in enumerate(tables):
        if table_index:
            print()
        print(f"### {title}")
        print()
        shape_headers = ["Fold A/B/D schedule"] if include_fold_schedule else []
        grind_headers = ["Grinding retries"] if include_fold_schedule else []
        headers = [
            "Profile",
            *shape_headers,
            *(metric.name for metric in metrics),
            *grind_headers,
        ]
        print("| " + " | ".join(headers) + " |")
        print(
            "| "
            + " | ".join(
                [
                    "---",
                    *("---" for _ in shape_headers),
                    *("---:" for _ in metrics),
                    *("---" for _ in grind_headers),
                ]
            )
            + " |"
        )

        for current in current_cases:
            baseline = main_baseline.get(str(current["case_id"])) if main_baseline else None
            row = [md_text(human_case_label(current))]
            if include_fold_schedule:
                row.append(fold_dimension_schedule(current))
            for metric in metrics:
                if metric.key == "tail_bytes":
                    row.append(
                        terminal_response_metric_value(
                            current,
                            baseline,
                            main_baseline is not None,
                        )
                    )
                else:
                    row.append(
                        optional_value_with_baseline_delta(
                            current,
                            baseline,
                            metric.key,
                            metric.value_formatter,
                            metric.unit,
                            main_baseline is not None,
                            "",
                        )
                    )
            if include_fold_schedule:
                row.append(grinding_retries_metric_value(current, baseline))
            print("| " + " | ".join(row) + " |")
        if include_fold_schedule:
            print()
            print(
                "Grinding retries are rejected attempts at each fold, listed in "
                "measured-run order. Zero means the first sampled nonce was accepted."
            )
            if any(
                case_metadata(str(case.get("mode", ""))).opening_topology
                == "multi_group"
                for case in current_cases
            ):
                print()
                print(
                    "Each multi-group profile has two precommitted nv16 singleton "
                    "polynomials and two polynomials in the displayed final group."
                )

    if main_baseline is not None:
        print()
        print(
            "Deltas are shown only for profiles with a matching merge-base case. "
            "Negative is smaller or faster."
        )

    failing_cases = [case for case in current_cases if case_status(case) != "ok"]
    if failing_cases:
        print()
        print("Failed cases:")
        for case in failing_cases:
            print(
                f"- {code_text(case['case_id'])}: phase "
                f"{code_text(case.get('failure_phase', 'unknown'))}; "
                f"{md_text(case.get('error', 'profile run failed'))}."
            )

    warnings = [
        (case, warning)
        for case in current_cases
        for warning in case.get("warnings", [])
        if isinstance(warning, str)
    ]
    if warnings:
        print()
        print("Report warnings:")
        for case, warning in warnings:
            print(f"- {code_text(case['case_id'])}: {md_text(warning)}.")




def validate_case_consistency(summary: dict[str, object]) -> None:
    proof_size = summary.get("proof_size_bytes")
    accounted = summary.get("accounted_bytes")
    if proof_size is not None and accounted is not None and int(proof_size) != int(accounted):
        raise ValueError(
            "proof accounting mismatch: "
            f"proof_size_bytes={proof_size}, accounted_bytes={accounted}"
        )

    tail_component_keys = ("tail_z_bytes", "tail_e_bytes", "tail_t_bytes")
    if summary.get("tail_bytes") is not None and all(
        summary.get(key) is not None for key in tail_component_keys
    ):
        component_total = sum(int(summary[key]) for key in tail_component_keys)
        if component_total != int(summary["tail_bytes"]):
            raise ValueError(
                "terminal response component mismatch: "
                f"tail_bytes={summary['tail_bytes']}, z_e_t_sum={component_total}"
            )

    planned_levels = summary.get("planned_levels")
    proof_levels = summary.get("proof_levels")
    if not isinstance(planned_levels, list) or not isinstance(proof_levels, list):
        return
    terminal_plan = summary.get("terminal_plan")
    if isinstance(terminal_plan, dict) and planned_levels:
        terminal_level = int(terminal_plan["level"])
        expected_terminal_level = int(planned_levels[-1]["level"]) + 1
        if terminal_level != expected_terminal_level:
            raise ValueError(
                "planned terminal level mismatch: "
                f"planned={terminal_level}, expected={expected_terminal_level}"
            )
        predecessor_output = int(planned_levels[-1]["next_w_len"])
        terminal_input = int(terminal_plan["input_witness_len"])
        if terminal_input != predecessor_output:
            raise ValueError(
                "planned terminal input mismatch: "
                f"terminal={terminal_input}, predecessor_output={predecessor_output}"
            )
        matching_proof = next(
            (
                proof
                for proof in proof_levels
                if int(proof["level"]) == terminal_level
            ),
            None,
        )
        if matching_proof is not None and int(matching_proof["d"]) != int(
            terminal_plan["d_a"]
        ):
            raise ValueError(
                f"planned/proof terminal A ring dimension mismatch at L{terminal_level}: "
                f"planned={terminal_plan['d_a']}, proof={matching_proof['d']}"
            )
    # The prover emits the direct terminal as an extra "proof fold level"
    # (`print_terminal_level_breakdown`), whereas the planner reports the
    # terminal separately as "planned terminal state" rather than a "planned
    # fold level". So the proof carries exactly the planned non-terminal folds,
    # optionally plus one trailing terminal level. Tolerate that single extra
    # level; the per-level checks below still cover every planned fold.
    if len(proof_levels) not in (len(planned_levels), len(planned_levels) + 1):
        raise ValueError(
            "planned/proof level count mismatch: "
            f"planned={len(planned_levels)}, proof={len(proof_levels)}"
        )

    for planned, proof in zip(planned_levels, proof_levels):
        planned_level = int(planned["level"])
        proof_level = int(proof["level"])
        if planned_level != proof_level:
            raise ValueError(
                "planned/proof level index mismatch: "
                f"planned={planned_level}, proof={proof_level}"
            )
        planned_d = int(planned["d_a"])
        proof_d = int(proof["d"])
        if planned_d != proof_d:
            raise ValueError(
                f"planned/proof A ring dimension mismatch at L{planned_level}: "
                f"planned={planned_d}, proof={proof_d}"
            )
        component_bytes = proof_level_component_bytes(proof)
        total_bytes = int(proof["total_bytes"])
        if component_bytes != total_bytes:
            raise ValueError(
                f"proof level component sum mismatch at L{proof_level}: "
                f"total_bytes={total_bytes}, component_sum={component_bytes}"
            )
        # Intentionally no per-level `level_bytes` vs `total_bytes` comparison.
        # The header-stripped planner estimate is only a conservative upper bound
        # in *aggregate*: it can over- or under-attribute bytes to any individual
        # level (e.g. dense_fp128_d64 nv24 has levels where the runtime proof
        # exceeds the per-level estimate while the total stays under it). The
        # total-overcount invariant is asserted in the profile binary itself
        # (`ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES` in
        # `crates/akita-pcs/examples/profile/workload.rs`). Proof-size deltas vs
        # baselines are reported in the PR comment but are not CI gates. Here we
        # only enforce the structural level shape (count / index / D) above.


def render_report(args: argparse.Namespace) -> int:
    summary_path = pathlib.Path(args.summary)
    current_cases = load_case_summaries(summary_path)
    current_cases.sort(key=report_case_sort_key)
    raw_summary = load_summary(summary_path)
    warmups = int(raw_summary.get("warmups", 0) or 0)

    baselines: list[tuple[str, dict[str, dict[str, object]] | None]] = [
        ("Merge base", load_optional_case_summaries(args.main_baseline_dir)),
        ("Prior PR run", load_optional_case_summaries(args.previous_baseline_dir)),
    ]
    visible_baselines = [(label, summary) for label, summary in baselines if summary is not None]

    source_sha = os.environ.get("AKITA_BENCH_SOURCE_SHA")
    source_subject = os.environ.get("AKITA_BENCH_SOURCE_SUBJECT")
    source_branch = os.environ.get("AKITA_BENCH_SOURCE_BRANCH") or os.environ.get("GITHUB_REF_NAME")
    base_ref = os.environ.get("AKITA_BENCH_BASE_REF")
    main_baseline_sha = os.environ.get("AKITA_BENCH_MAIN_BASELINE_SHA")
    main_baseline_label = os.environ.get("AKITA_BENCH_MAIN_BASELINE_LABEL")
    previous_baseline_sha = os.environ.get("AKITA_BENCH_PREVIOUS_BASELINE_SHA")
    previous_baseline_label = os.environ.get("AKITA_BENCH_PREVIOUS_BASELINE_LABEL")

    if len(current_cases) == 1:
        only_case = current_cases[0]
        print("## " f"{md_text(human_case_label(only_case))} " "Profile Benchmark")
    else:
        print("## PCS Profile Benchmark")
    print()
    ref = commit_ref(source_sha)
    if ref:
        print(f"- Head: {ref}")
    if source_subject and not args.compact:
        print(f"- Message: {md_text(source_subject)}")
    if source_branch and not args.compact:
        print(f"- Ref: {code_text(source_branch)}")
    run_ref = workflow_run_ref()
    if run_ref:
        print(f"- Workflow run: {run_ref}")
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if not args.compact:
        print(f"- Report generated: `{generated_at}`.")
    if visible_baselines:
        main_ref = commit_ref(main_baseline_sha)
        if baselines[0][1] is not None:
            if main_ref and main_baseline_label:
                print(f"- Merge base: {main_ref} from {md_text(main_baseline_label)}.")
            elif main_ref:
                print(f"- Merge base: {main_ref}.")
            elif main_baseline_label:
                print(f"- Merge base: {md_text(main_baseline_label)}.")

        previous_ref = commit_ref(previous_baseline_sha)
        if baselines[1][1] is not None and not args.compact:
            if previous_ref and previous_baseline_label:
                print(f"- Prior PR run: {previous_ref} from {md_text(previous_baseline_label)}.")
            elif previous_ref:
                print(f"- Prior PR run: {previous_ref}.")
            elif previous_baseline_label:
                print(f"- Prior PR run: {md_text(previous_baseline_label)}.")
    if base_ref and baselines[0][1] is None:
        print(f"- Merge base: no reusable benchmark artifact found for `{base_ref}`.")
    if not args.compact:
        print("- Binary: `target/release/examples/profile`.")
        print("- Memory: maximum resident set size from `/usr/bin/time` on the benchmark process.")
    print()

    for current in current_cases:
        if case_status(current) == "ok":
            validate_case_consistency(current)

    passed = sum(case_status(case) == "ok" for case in current_cases)
    print(f"{passed} of {len(current_cases)} profiles passed.")
    print()
    run_counts = sorted({int(case.get("runs", 1)) for case in current_cases})
    if passed > 0 and len(run_counts) == 1:
        warmup_label = "run" if warmups == 1 else "runs"
        print(
            f"Times are medians of `{run_counts[0]}` measured runs after `{warmups}` "
            f"discarded warmup {warmup_label}. Peak RSS is the largest measured value."
        )
        print()
        if any(case.get("verify_single_total_s") is not None for case in current_cases):
            print(
                "Each sample verifies the same proof first with the configured multi-threaded "
                "pool and then with one thread. Both timings reuse the same verifier setup."
            )
            print()
    if baselines[0][1] is not None:
        matching_base_cases = sum(
            str(case["case_id"]) in baselines[0][1] for case in current_cases
        )
        print(
            f"Merge-base comparisons are available for `{matching_base_cases}` of "
            f"`{len(current_cases)}` profiles. For matching profiles, the head and merge-base "
            "binaries ran interleaved on the same runner."
        )
        if matching_base_cases != len(current_cases):
            print()
            print(
                "Profiles without a matching merge-base mode are measured at the head only "
                "and are marked `no matching merge-base case` instead of showing a delta."
            )
        print()
    render_profile_definitions(current_cases)
    print()
    print(
        "Each sample generates deterministic witnesses and opening points, prepares setup, "
        "commits, proves, serializes the proof, checks its size, prepares verifier setup, "
        "and verifies the claimed openings. It does not test malformed proofs."
    )
    print()
    render_matrix_summary(current_cases, baselines[0][1])
    if args.compact:
        print()
        print(
            "The uploaded `report.md` benchmark artifact contains the detailed fold schedule, "
            "proof-size breakdown, terminal response segments, and Golomb diagnostics."
        )
        return 0

    print()

    for index, current in enumerate(current_cases):
        if len(current_cases) > 1:
            print("<details>")
            print(f"<summary>{html.escape(section_title(current), quote=False)} details</summary>")
            print()
        print(f"- Profile: {md_text(human_case_label(current))}")
        print(f"- Public statement: {md_text(public_opening_statement(current))}")
        print(f"- Status: `{case_status(current)}`.")
        if current.get("error"):
            print(
                f"- Failure: phase `{current.get('failure_phase', 'unknown')}`; "
                f"{md_text(current['error'])}."
            )
        for warning in current.get("warnings", []):
            print(f"- Report warning: {md_text(warning)}.")
        if current.get("workload") == "onehot":
            print(
                f"- Benchmark witness: one `1` in every consecutive chunk of "
                f"`{ONEHOT_ARITY}` coefficients in each generated polynomial."
            )
        env = current.get("env", {})
        command_env = [
            code_text(f"AKITA_MODE={env.get('AKITA_MODE', current['mode'])}"),
            code_text(f"AKITA_NUM_VARS={env.get('AKITA_NUM_VARS', current['num_vars'])}"),
            code_text(f"AKITA_NUM_POLYS={env.get('AKITA_NUM_POLYS', current.get('num_polys', 1))}"),
            code_text(
                "AKITA_SETUP_MODE="
                f"{env.get('AKITA_SETUP_MODE', current.get('setup_contribution_mode', 'direct'))}"
            ),
        ]
        print(
            "- Command: `target/release/examples/profile` with "
            f"{' '.join(command_env)} "
            "`AKITA_PROFILE_TRACE=0` `AKITA_PROFILE_SPAN_CLOSES=0` "
            "`AKITA_PROFILE_LOG=info` `AKITA_PROFILE_ANSI=0`."
        )
        case_runs = int(current.get("runs", 1))
        if case_runs > 1 or warmups > 0:
            warmup_clause = (
                f" after `{warmups}` discarded warmup "
                f"{'run' if warmups == 1 else 'runs'}"
                if warmups > 0
                else ""
            )
            print(
                f"- Samples: metrics are the median of `{case_runs}` runs{warmup_clause}; "
                "Peak process RSS is the maximum sample."
            )
        print()

        case_baselines = [
            (label, summary.get(str(current["case_id"])) if summary is not None else None)
            for label, summary in visible_baselines
        ]
        main_case = (
            baselines[0][1].get(str(current["case_id"]))
            if baselines[0][1] is not None
            else None
        )
        print("#### Measured result")
        print()
        column_labels = ["Head"] + [md_text(label) for label, _ in case_baselines]
        print("| Metric | " + " | ".join(column_labels) + " | Delta versus merge base | Unit |")
        print(
            "| --- | "
            + " | ".join("---:" for _ in column_labels)
            + " | ---: | --- |"
        )

        for metric in MEASURED_METRICS:
            row = render_metric_row(metric, current, case_baselines, main_case)
            if row:
                print(row)

        if case_runs > 1:
            ranges = []
            for key, label in [
                ("setup_s", "setup"),
                ("commit_s", "commit"),
                ("prove_total_s", "prove"),
                ("verify_total_s", "multi-threaded verify"),
                ("verify_single_total_s", "single-threaded verify"),
            ]:
                observed_range = sample_range(current, key)
                if observed_range is not None:
                    is_verify = key in ("verify_total_s", "verify_single_total_s")
                    formatter = fmt_milliseconds if is_verify else fmt_seconds
                    unit = "ms" if is_verify else "s"
                    ranges.append(
                        f"{label} `{formatter(observed_range[0])}-{formatter(observed_range[1])}{unit}`"
                    )
            if ranges:
                print()
                print(f"- Sample ranges: {', '.join(ranges)}.")

        print()
        render_relation_phase_timings(current)
        print()
        render_execution_parameters(current, main_case)
        onehot_schedules = current.get("onehot_commit_schedules")
        if isinstance(onehot_schedules, list) and onehot_schedules:
            routes = []
            for schedule in onehot_schedules:
                routes.append(
                    f"`{schedule['sweep']}` sweep, tile `{schedule['block_tile']}`, "
                    f"D`{schedule['ring_dimension']}`, `{schedule['source_count']}` source(s), "
                    f"`{schedule['total_blocks']}` blocks, "
                    f"`{schedule['estimated_matrix_passes']}` estimated matrix pass(es)"
                )
            print("- One hot commit routes: " + "; ".join(routes) + ".")
        print()
        print("#### Terminal response")
        print()
        render_tail_encoding(current)
        if (
            current.get("terminal_w_len") is not None
            and current.get("terminal_log_basis") is not None
            and current.get("tail_encoding")
            not in ("segment_typed", "terminal_response", "none", None)
        ):
            print(
                "- Observed terminal state: "
                f"`{fmt_count(float(current['terminal_w_len']))}` field elements with a "
                f"gadget basis width of `{current['terminal_log_basis']}` bits"
            )
        elif (
            current.get("terminal_w_len") is not None
            and current.get("tail_encoding") == "field_elements"
        ):
            print(
                "- Observed terminal state: "
                f"`{fmt_count(float(current['terminal_w_len']))}` field elements with "
                "field-element encoding"
            )

        planned_levels_value = current.get("planned_levels")
        proof_levels_value = current.get("proof_levels")
        planned_levels = planned_levels_value if isinstance(planned_levels_value, list) else []
        proof_levels = proof_levels_value if isinstance(proof_levels_value, list) else []
        if planned_levels or proof_levels:
            print()
            baseline_planned_levels = (
                main_case.get("planned_levels") if main_case is not None else None
            )
            baseline_proof_levels = main_case.get("proof_levels") if main_case is not None else None
            render_fold_details(
                planned_levels,
                proof_levels,
                current.get("terminal_plan"),
                baseline_planned_levels,
                baseline_proof_levels,
                main_case.get("terminal_plan") if main_case is not None else None,
                current.get("grinding_plan"),
                main_case.get("grinding_plan") if main_case is not None else None,
            )
        if len(current_cases) > 1:
            print()
            print("</details>")
        if index + 1 < len(current_cases):
            print()

    return 0


def main() -> int:
    args = parse_args()
    if args.command == "run":
        return run_benchmark(args)
    if args.command == "render":
        return render_report(args)
    if args.command == "failure-summary":
        return write_failure_summary(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
