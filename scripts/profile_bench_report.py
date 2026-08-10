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


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=(".*?"|\S+)')
RSS_PATTERNS = [
    re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)"),
    re.compile(r"^\s*(\d+)\s+maximum resident set size$", re.MULTILINE),
]
ONEHOT_ARITY = 256
ONEHOT_WORKLOAD_LABEL = f"1-of-{ONEHOT_ARITY} one-hot"
CASE_SCHEMA_VERSION = 7
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

# Byte columns emitted by `crates/akita-pcs/examples/profile/report.rs` for each
# fold level. Their sum must match `total_bytes`. The parser separately retains
# field presence so structurally absent proof components render as an em dash,
# rather than a misleading zero-byte component.
PROOF_LEVEL_BYTE_FIELDS = (
    "extension_opening_partials_bytes",
    "extension_opening_sumcheck_bytes",
    "fold_grind_nonce_bytes",
    "opening_payload_bytes",
    "stage1_sumcheck_bytes",
    "stage1_interstage_claims_bytes",
    "stage1_range_image_evaluation_bytes",
    "stage2_sumcheck_bytes",
    "stage3_sumcheck_bytes",
    "next_w_payload_bytes",
    "next_w_eval_bytes",
)


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
    "onehot_fp128_recursive": CaseMetadata(
        "fp128", "onehot", ONEHOT_WORKLOAD_LABEL, "mixed D256 to D64"
    ),
    "dense_fp128": CaseMetadata("fp128", "dense", "dense", "mixed D256 to D64"),
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
    "z_rice_low_bits_cap",
    "z_coords",
    "z_packed_hypothetical_bytes",
    "z_golomb_savings_bytes",
)

TAIL_SUMMARY_FLOAT_FIELDS = (
    "z_bits_per_coord_golomb",
    "z_bits_per_coord_packed",
)

TAIL_ENCODING_POLICIES = {
    "segment_typed": "non-zk folded terminal (default in profile bench)",
    "terminal_response": "non-zk quotient-free terminal response (default in profile bench)",
    "packed_digits": "zk-feature folded terminal fallback",
    "field_elements": "root-direct cleartext witness",
    "none": "root-direct zero-fold (no cleartext tail)",
}


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
    if "z_witness_linf_cap" in kvs:
        summary["z_witness_linf_cap"] = kvs["z_witness_linf_cap"]
    elif "z_beta_inf" in kvs:
        summary["z_witness_linf_cap"] = kvs["z_beta_inf"]
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

    policy = current.get("tail_policy")
    policy_hint = TAIL_ENCODING_POLICIES.get(str(encoding), str(policy or encoding))
    print(f"- Tail encoding: `{encoding}` ({policy_hint})")

    if encoding == "packed_digits":
        if current.get("tail_num_elems") is not None and current.get("tail_bits_per_elem") is not None:
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
        basis_role = "inner" if encoding == "terminal_response" else "D/open"
        print(
            f"  - Logical witness: `{fmt_count(float(current['tail_num_elems']))}` elements, "
            f"{basis_role} gadget basis width `{terminal_log_basis}` bits, "
            "folded-witness (`z`) segment first on the wire"
        )

    z_prefix = current.get("tail_z_prefix_bytes")
    z_golomb = current.get("tail_z_golomb_bytes")
    z_wire = current.get("tail_z_bytes")
    z_field = current.get("tail_z_field_elems")
    z_ring = current.get("tail_z_ring_elems")
    if z_wire is not None and z_field is not None and z_ring is not None:
        prefix_golomb = ""
        if z_prefix is not None and z_golomb is not None:
            prefix_golomb = (
                f" (length prefix `{fmt_bytes(float(z_prefix))} bytes` + Golomb "
                f"`{fmt_bytes(float(z_golomb))} bytes`)"
            )
        print(
            f"  - Folded-witness (`z`) segment: `{fmt_bytes(float(z_wire))} bytes`{prefix_golomb}, "
            f"`{fmt_count(float(z_field))}` field coefficients, "
            f"`{fmt_count(float(z_ring))}` ring elements"
        )

    for segment_label, bytes_key, field_key, ring_key in (
        ("Opening-digit (`e`)", "tail_e_bytes", "tail_e_field_elems", "tail_e_ring_elems"),
        (
            "Inner-commitment (`t`)",
            "tail_t_bytes",
            "tail_t_field_elems",
            "tail_t_ring_elems",
        ),
    ):
        seg_bytes = current.get(bytes_key)
        field_coeffs = current.get(field_key)
        ring_elems = current.get(ring_key)
        if seg_bytes is None:
            continue
        detail = f"`{fmt_bytes(float(seg_bytes))} bytes`"
        if field_coeffs is not None:
            detail += f", `{fmt_count(float(field_coeffs))}` field coefficients"
        if ring_elems is not None:
            detail += f", `{fmt_count(float(ring_elems))}` ring elements"
        print(f"  - {segment_label} segment: {detail}")

    if all(
        current.get(key) is not None
        for key in ("tail_z_bytes", "tail_e_bytes", "tail_t_bytes")
    ):
        wire_total = (
            int(current["tail_z_bytes"])
            + int(current["tail_e_bytes"])
            + int(current["tail_t_bytes"])
        )
        print(f"  - Wire total (z+e+t): `{fmt_bytes(float(wire_total))} bytes`")

    z_budget = current.get("tail_z_budget_bytes")
    z_slack = current.get("tail_z_slack_bytes")
    if z_budget is not None and z_golomb is not None:
        slack_note = (
            f", slack `{fmt_bytes(float(z_slack))} bytes` under planner upper bound"
            if z_slack is not None
            else ""
        )
        print(
            f"  - Folded-witness Golomb budget: realized `{fmt_bytes(float(z_golomb))} bytes` out of "
            f"a scheduled upper bound of `{fmt_bytes(float(z_budget))} bytes`{slack_note}"
        )

    z_witness_linf_cap = current.get("z_witness_linf_cap")
    z_rice_low_bits_wire = current.get("z_rice_low_bits_wire")
    z_rice_low_bits_cap = current.get("z_rice_low_bits_cap")
    z_field_coeffs = current.get("tail_z_field_elems") or current.get("z_coords")
    z_ring_elems = current.get("tail_z_ring_elems")
    z_bits_golomb = current.get("z_bits_per_coord_golomb")
    z_bits_packed = current.get("z_bits_per_coord_packed")
    z_packed_hyp = current.get("z_packed_hypothetical_bytes")
    z_savings = current.get("z_golomb_savings_bytes")
    if z_witness_linf_cap is not None and z_rice_low_bits_wire is not None and z_field_coeffs is not None:
        comparison = ""
        if z_bits_golomb is not None and z_bits_packed is not None:
            k_note = f"wire Golomb parameter=`{z_rice_low_bits_wire}`"
            if z_rice_low_bits_cap is not None:
                k_note += f", planner-cap Golomb parameter=`{z_rice_low_bits_cap}`"
            comparison = (
                f", `{z_bits_golomb:.2f}` bits for each field coefficient "
                f"({k_note}, derived from folded-witness infinity-norm cap "
                f"`{z_witness_linf_cap}`) vs "
                f"`{z_bits_packed:.2f}` bits for each field coefficient "
                "(legacy uniform `PackedDigits` z planes)"
            )
        savings_note = ""
        if z_packed_hyp is not None and z_golomb is not None and z_savings is not None:
            savings_note = (
                f"; hypothetical packed z `{fmt_bytes(float(z_packed_hyp))} bytes`, "
                f"savings `{fmt_bytes(float(z_savings))} bytes`"
            )
        ring_note = (
            f"`{fmt_count(float(z_ring_elems))}` ring elements, "
            if z_ring_elems is not None
            else ""
        )
        print(
            f"  - Folded-witness Golomb model: {ring_note}"
            f"`{fmt_count(float(z_field_coeffs))}` field coefficients{comparison}{savings_note}"
        )


def render_terminal_response_components(
    cases: list[dict[str, object]], include_heading: bool = True
) -> None:
    rows = [
        case
        for case in cases
        if case_status(case) == "ok"
        and case.get("tail_encoding") in ("segment_typed", "terminal_response")
        and all(
            case.get(key) is not None
            for key in ("tail_z_bytes", "tail_e_bytes", "tail_t_bytes", "tail_bytes")
        )
    ]
    if not rows:
        return

    if include_heading:
        print("### Terminal response component breakdown")
        print()
    print(
        "| Workload | Folded response (`z`) | Opening values (`e`) | "
        "Inner-commitment values (`t`) | Total terminal response |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for case in rows:
        print(
            f"| {md_text(human_case_label(case))} | "
            f"{fmt_bytes(float(case['tail_z_bytes']))} bytes | "
            f"{fmt_bytes(float(case['tail_e_bytes']))} bytes | "
            f"{fmt_bytes(float(case['tail_t_bytes']))} bytes | "
            f"{fmt_bytes(float(case['tail_bytes']))} bytes |"
        )
    print()
    print(
        "The `z` column includes its per-segment length prefixes and Golomb payload; `e` and `t` "
        "are raw field bytes. These three columns sum exactly to the serialized terminal response."
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
        return f"{metadata.field_family} multi-group opening with {num_polys} polynomials{setup_suffix}"
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
    log_text: str, mode: str, num_vars: int, num_polys: int, setup_mode: str = "direct"
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
    proof_levels: dict[int, dict[str, object]] = {}
    onehot_commit_schedules: list[dict[str, object]] = []
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
        elif " INFO setup sizes" in line and kvs.get("label") == mode:
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
        elif " INFO verifier NTT cache size" in line and kvs.get("label") == mode:
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
        elif " INFO setup_expand" in line and kvs.get("label") == mode:
            summary["setup_expand_s"] = float(kvs["elapsed_s"])
        elif " INFO backend_prepare" in line and kvs.get("label") == mode:
            summary["backend_prepare_s"] = float(kvs["elapsed_s"])
        elif " INFO setup" in line and kvs.get("label") == mode:
            summary["setup_s"] = float(kvs["elapsed_s"])
        elif " INFO commit" in line and kvs.get("label") == mode:
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
        elif " INFO prove" in line and kvs.get("label") == mode:
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
            summary["tail_bytes"] = int(kvs["tail_bytes"])
            if "levels" in kvs:
                summary["akita_levels"] = int(kvs["levels"])
        elif "profile extension field" in line and kvs.get("label") == mode:
            summary["ext_degree"] = int(kvs["ext_degree"])
        elif "profile setup-contribution mode" in line and kvs.get("label") == mode:
            if "setup_contribution_mode" in kvs:
                summary["setup_contribution_mode"] = normalize_setup_mode(
                    kvs["setup_contribution_mode"]
                )
        elif "extension opening used root-direct fallback" in line and kvs.get("label") == mode:
            summary["extension_root_direct_fallback"] = True
        elif "planned fold group" in line and kvs.get("label") == mode:
            level = int(kvs["level"])
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
                    "n_a": int(kvs["n_a"]),
                    "n_b": int(kvs["n_b"]),
                    "n_d": int(kvs["n_d"]),
                    "log_basis_inner": int(kvs["log_basis_inner"]),
                    "log_basis_outer": int(kvs["log_basis_outer"]),
                    "log_basis_open": int(kvs["log_basis_open"]),
                    "num_digits_inner": int(kvs["num_digits_inner"]),
                    "num_digits_outer": int(kvs["num_digits_outer"]),
                    "num_digits_open": int(kvs["num_digits_open"]),
                    "num_digits_fold": int(kvs["num_digits_fold"]),
                    "challenge_l1_mass": int(kvs["challenge_l1_mass"]),
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
            planned_levels[level] = {
                "level": level,
                "d_a": int(kvs.get("d_a", legacy_d)),
                "d_b": int(kvs.get("d_b", legacy_d)),
                "d_d": int(kvs.get("d_d", legacy_d)),
                "n_a": int(kvs["n_a"]),
                "n_b": int(kvs["n_b"]),
                "n_d": int(kvs["n_d"]),
                "challenge_l1_mass": int(kvs["challenge_l1_mass"]),
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
                proof_levels[level]["grind_nonce_val"] = int(kvs["grind_nonce"])
            if "grind_attempts" in kvs:
                proof_levels[level]["grind_attempts"] = int(kvs["grind_attempts"])
            if "root_variant" in kvs:
                proof_levels[level]["root_variant"] = kvs["root_variant"]
        elif "fold grind summary" in line and kvs.get("label") == mode:
            summary["grind_levels"] = int(kvs["grind_levels"])
            if int(kvs["grind_levels"]) > 0:
                summary["grind_nonce_max"] = int(kvs["grind_nonce_max"])
                summary["grind_attempts_sum"] = int(kvs["grind_attempts_sum"])
                summary["grind_nonces"] = kvs["grind_nonces"]
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
    if proof_levels:
        summary["proof_levels"] = [proof_levels[level] for level in sorted(proof_levels)]
    if onehot_commit_schedules:
        summary["onehot_commit_schedules"] = onehot_commit_schedules

    return summary


def run_benchmark_case(
    binary: str, output_dir: pathlib.Path, case: BenchmarkCaseSpec
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
    return sample


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


def combine_case_run_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    combined = dict(summaries[0])
    combined["runs"] = len(summaries)
    combined["samples"] = [compact_sample_summary(summary) for summary in summaries]

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


def plan_case_runs(
    binary: str,
    summary_dir: pathlib.Path,
    case: BenchmarkCaseSpec,
    runs: int,
    warmups: int,
) -> list[ScheduledRun]:
    """All executions of one case for one binary, in execution order."""
    case_dir = summary_dir / case.case_id
    schedule = [
        ScheduledRun(
            binary, summary_dir, case_dir / f"warmup-{warmup_index}", case, "warmup", 0
        )
        for warmup_index in range(1, warmups + 1)
    ]
    for run_index in range(1, runs + 1):
        run_dir = case_dir if runs == 1 else case_dir / f"run-{run_index}"
        schedule.append(ScheduledRun(binary, summary_dir, run_dir, case, "measured", run_index))
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
        summary, return_code = run_benchmark_case(run.binary, run.run_dir, run.case)
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
    binaries: list[tuple[str, pathlib.Path]] = [(args.binary, output_dir)]
    if args.baseline_binary:
        baseline_dir = pathlib.Path(args.baseline_output_dir)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        binaries.append((args.baseline_binary, baseline_dir))

    cases = configured_cases(args)
    schedule: list[ScheduledRun] = []
    for case in cases:
        plans = [
            plan_case_runs(binary, summary_dir, case, args.runs, args.warmups)
            for binary, summary_dir in binaries
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
        [summary_dir for _, summary_dir in binaries],
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


def fmt_bytes(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_mib_with_exact_bytes(value_bytes: float) -> str:
    return (
        f"{fmt_mib_from_bytes(value_bytes)}<br>"
        f"<sub>{fmt_bytes(value_bytes)} bytes</sub>"
    )


def fmt_count(value: float) -> str:
    return f"{int(round(value)):,}"


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
    Metric("setup_s", "Setup and preparation", "s", fmt_seconds),
    Metric("setup_expand_s", "Setup expansion", "s", fmt_seconds),
    Metric("backend_prepare_s", "Backend preparation", "s", fmt_seconds),
    Metric("commit_s", "Commit", "s", fmt_seconds),
    Metric("prove_total_s", "Prove", "s", fmt_seconds),
    Metric("verify_total_s", "Verify, multi-threaded", "ms", fmt_milliseconds),
    Metric("verify_single_total_s", "Verify, single-threaded", "ms", fmt_milliseconds),
    Metric("verify_akita_s", "Verifier core, multi-threaded", "ms", fmt_milliseconds),
    Metric(
        "verify_single_akita_s",
        "Verifier core, single-threaded",
        "ms",
        fmt_milliseconds,
    ),
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


def numeric_delta(
    current: dict[str, object],
    baseline: dict[str, object] | None,
    key: str,
) -> str:
    """Format a percentage delta of `current[key]` against `baseline[key]`.

    Returns `"n/a"` when either side is missing. A zero baseline is reported as
    unchanged when both values are zero, or explicitly as a new nonzero value;
    other comparisons render as e.g. `"+5.2%"` or `"-1.2%"`. All report
    comparisons use this formatter so proof size, prover wall-time, and other
    numeric metrics have consistent deltas.
    """
    if baseline is None:
        return "n/a"
    current_value = current.get(key)
    baseline_value = baseline.get(key)
    if current_value is None or baseline_value is None:
        return "n/a"
    if float(baseline_value) == 0.0:
        return "unchanged" if float(current_value) == 0.0 else "new; merge base is zero"
    delta = (float(current_value) / float(baseline_value) - 1.0) * 100.0
    sign = "+" if delta >= 0.0 else ""
    return f"{sign}{delta:.1f}%"


def value_with_baseline_delta(
    current_value: object,
    baseline_value: object | None,
    formatter: callable,
    unit: str = "",
    compare_to_baseline: bool = False,
    comparison_label: str = " vs merge base",
) -> str:
    value = f"{formatter(float(current_value))}{unit}"
    if baseline_value is None:
        if compare_to_baseline:
            return f"{value}<br><sub>n/a{comparison_label}</sub>"
        return value
    delta = numeric_delta({"value": current_value}, {"value": baseline_value}, "value")
    return f"{value}<br><sub>{delta}{comparison_label}</sub>"


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
    ]

    for table_index, (title, metrics, include_fold_schedule) in enumerate(tables):
        if table_index:
            print()
        print(f"### {title}")
        print()
        shape_headers = ["Fold A/B/D schedule"] if include_fold_schedule else []
        headers = ["Profile", *shape_headers, *(metric.name for metric in metrics)]
        print("| " + " | ".join(headers) + " |")
        print(
            "| "
            + " | ".join(
                ["---", *("---" for _ in shape_headers), *("---:" for _ in metrics)]
            )
            + " |"
        )

        for current in current_cases:
            baseline = main_baseline.get(str(current["case_id"])) if main_baseline else None
            row = [md_text(human_case_label(current))]
            if include_fold_schedule:
                row.append(fold_dimension_schedule(current))
            for metric in metrics:
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
            print("| " + " | ".join(row) + " |")

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


def sample_range(summary: dict[str, object], key: str) -> tuple[float, float] | None:
    samples = summary.get("samples")
    if not isinstance(samples, list):
        return None
    values = [float(sample[key]) for sample in samples if isinstance(sample, dict) and key in sample]
    if len(values) <= 1:
        return None
    return min(values), max(values)


def proof_level_component_bytes(level: dict[str, object]) -> int:
    return sum(int(level.get(field, 0)) for field in PROOF_LEVEL_BYTE_FIELDS)


def proof_field_present(level: dict[str, object], field: str) -> bool:
    present = level.get("present_byte_fields")
    if isinstance(present, list):
        return field in present
    return level.get("root_variant") != "direct"


def proof_step_label(level: dict[str, object]) -> str:
    variant = level.get("root_variant")
    level_index = int(level["level"])
    if variant == "direct":
        return "direct root"
    if variant == "terminal":
        return "terminal root"
    if variant == "fold":
        return "fold root" if level_index == 0 else "terminal fold"
    return "intermediate fold"


def exact_choice(current: str, baseline: str | None) -> str:
    if baseline is None or current == baseline:
        return current
    return f"{current}<br><sub>Merge base</sub><br>{baseline}"


def detail_block(title: str, rows: list[str]) -> str:
    return f"<strong>{title}</strong><br>" + "<br>".join(rows)


def format_witness_groups_inline(groups: object) -> str:
    if not isinstance(groups, list) or not groups:
        return "n/a"
    values = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        name = group.get("group")
        field_elements = group.get("field_elements")
        if name is None or field_elements is None:
            continue
        values.append(f"{name} {fmt_count(float(field_elements))}")
    return "; ".join(values) if values else "n/a"


def planned_group_label(group: dict[str, object]) -> str:
    role = str(group["group_role"])
    name = str(group["group"])
    if role == "final":
        return "Final group"
    if role == "folded":
        return "Folded witness"
    if role == "precommitted" and name.startswith("pre"):
        index = name.removeprefix("pre")
        return f"Precommit {int(index) + 1}" if index.isdigit() else name
    if role == "setup_offload":
        return f"Setup offload → L{int(group['consumer_level'])}"
    return name


def planned_groups_for_render(level: dict[str, object]) -> list[dict[str, object]]:
    groups = level.get("groups")
    typed_groups = (
        [group for group in groups if isinstance(group, dict)]
        if isinstance(groups, list)
        else []
    )
    if typed_groups:
        return typed_groups

    level_index = int(level["level"])
    role = "final" if level_index == 0 else "folded"
    witness_groups = level.get("current_w_len")
    witness_field_elements = (
        sum(
            int(group.get("field_elements", 0))
            for group in witness_groups
            if isinstance(group, dict)
        )
        if isinstance(witness_groups, list)
        else 0
    )
    return [
        {
            **level,
            "group": role,
            "group_role": role,
            "consumer_level": level_index,
            "witness_field_elements": witness_field_elements
            or int(level.get("input_witness_len", 0)),
            "num_digits_fold": int(level["delta_fold"]),
            "legacy_level": True,
        }
    ]


def planned_group_key(group: dict[str, object]) -> tuple[str, str, int]:
    return (
        str(group["group_role"]),
        str(group["group"]),
        int(group["consumer_level"]),
    )


def planned_group_planner_value(group: dict[str, object]) -> str:
    matrix = (
        f"Rings A/B/D: {fmt_count(float(group['d_a']))} / "
        f"{fmt_count(float(group['d_b']))} / {fmt_count(float(group['d_d']))}<br>"
        f"Rows A/B/D: {fmt_count(float(group['n_a']))} / "
        f"{fmt_count(float(group['n_b']))} / {fmt_count(float(group['n_d']))}"
    )
    decomposition = (
        f"Basis bits A/B/D: {fmt_count(float(group['log_basis_inner']))} / "
        f"{fmt_count(float(group['log_basis_outer']))} / "
        f"{fmt_count(float(group['log_basis_open']))}<br>"
        f"Digits A/B/D/W: {fmt_count(float(group['num_digits_inner']))} / "
        f"{fmt_count(float(group['num_digits_outer']))} / "
        f"{fmt_count(float(group['num_digits_open']))} / "
        f"{fmt_count(float(group['num_digits_fold']))}"
    )
    return detail_block(
        planned_group_label(group),
        [
            f"<em>Matrix geometry</em><br>{matrix}",
            f"<br><em>Decomposition</em><br>{decomposition}",
            f"<br><em>Challenge</em><br>L1 mass: {fmt_count(float(group['challenge_l1_mass']))}",
        ],
    )


def planned_group_work_value(group: dict[str, object]) -> str:
    role = str(group["group_role"])
    label = planned_group_label(group)
    relation = (
        f"Live per claim: {fmt_count(float(group['num_live_ring_elements_per_claim']))}<br>"
        f"Blocks × positions: {fmt_count(float(group['num_live_blocks']))} × "
        f"{fmt_count(float(group['num_positions_per_block']))}<br>"
        f"Domain slots: {fmt_count(float(group['block_index_domain_size']))}"
    )
    if group.get("legacy_level"):
        source = (
            f"Input → output: {format_witness_groups_inline(group.get('current_w_len'))} → "
            f"{fmt_count(float(group['next_w_len']))}"
        )
    elif role == "setup_offload":
        source = (
            f"Natural → padded: "
            f"{fmt_count(float(group['setup_prefix_natural_field_elements']))} → "
            f"{fmt_count(float(group['setup_prefix_padded_field_elements']))}"
        )
    else:
        source = f"Field elements: {fmt_count(float(group['witness_field_elements']))}"
    parts = [
        f"<em>{'Setup prefix' if role == 'setup_offload' else 'Witness'}</em><br>{source}",
        f"<br><em>Relation geometry</em><br>{relation}",
    ]
    if group.get("legacy_level") and (
        int(group.get("setup_prefix_natural_field_elements", 0)) != 0
        or int(group.get("setup_prefix_padded_field_elements", 0)) != 0
    ):
        parts.append(
            "<br><em>Setup prefix</em><br>Natural → padded: "
            f"{fmt_count(float(group['setup_prefix_natural_field_elements']))} → "
            f"{fmt_count(float(group['setup_prefix_padded_field_elements']))}"
        )
    return detail_block(
        label,
        parts,
    )


def render_group_choices(
    groups: list[dict[str, object]],
    baseline_groups: list[dict[str, object]],
    value: callable,
) -> str:
    current = {planned_group_key(group): group for group in groups}
    baseline = {planned_group_key(group): group for group in baseline_groups}
    keys = [*current, *(key for key in baseline if key not in current)]
    rows = []
    for key in keys:
        current_group = current.get(key)
        baseline_group = baseline.get(key)
        label_source = current_group or baseline_group
        if label_source is None:
            continue
        current_text = (
            value(current_group)
            if current_group is not None
            else detail_block(planned_group_label(label_source), ["absent"])
        )
        baseline_text = (
            value(baseline_group)
            if baseline_group is not None
            else (
                detail_block(planned_group_label(label_source), ["absent"])
                if baseline_groups
                else None
            )
        )
        rows.append(exact_choice(current_text, baseline_text))
    return "<br><br>".join(rows)


def proof_component_group(
    level: dict[str, object],
    baseline: dict[str, object] | None,
    group_label: str,
    components: tuple[tuple[str, str], ...],
) -> str | None:
    def group_value(source: dict[str, object] | None) -> tuple[int, list[str]]:
        if source is None:
            return 0, []
        values = []
        total = 0
        for field, label in components:
            if not proof_field_present(source, field):
                continue
            value = int(source.get(field, 0))
            total += value
            if value != 0:
                values.append(f"{label} {fmt_bytes(float(value))}")
        return total, values

    def render_value(total: int, values: list[str]) -> str:
        detail = (
            f"<br><sub>{' · '.join(values)}</sub>"
            if len(components) > 1 and values
            else ""
        )
        return f"<strong>{group_label}</strong><br>{fmt_bytes(float(total))} bytes{detail}"

    current_total, current_values = group_value(level)
    baseline_total, baseline_values = group_value(baseline)
    if current_total == 0 and (baseline is None or baseline_total == 0):
        return None
    baseline_text = (
        render_value(baseline_total, baseline_values) if baseline is not None else None
    )
    return exact_choice(render_value(current_total, current_values), baseline_text)


def proof_cost_summary(
    level: dict[str, object], baseline: dict[str, object] | None
) -> str:
    total = value_with_baseline_delta(
        level["total_bytes"],
        baseline.get("total_bytes") if baseline else None,
        fmt_bytes,
        " bytes",
        baseline is not None,
    )
    rows = [f"<strong>Total</strong><br>{total}"]
    groups = (
        (
            "Opening",
            (
                ("extension_opening_partials_bytes", "partials"),
                ("extension_opening_sumcheck_bytes", "sumcheck"),
                ("opening_payload_bytes", "p_H"),
            ),
        ),
        (
            "Stage 1",
            (
                ("stage1_sumcheck_bytes", "sumcheck"),
                ("stage1_interstage_claims_bytes", "claims"),
                ("stage1_range_image_evaluation_bytes", "range image"),
            ),
        ),
        ("Stage 2", (("stage2_sumcheck_bytes", "sumcheck"),)),
        ("Stage 3", (("stage3_sumcheck_bytes", "sumcheck"),)),
        (
            "Next witness",
            (
                ("next_w_payload_bytes", "payload"),
                ("next_w_eval_bytes", "evaluation"),
            ),
        ),
        ("Grinding nonce", (("fold_grind_nonce_bytes", "nonce"),)),
    )
    for group_label, components in groups:
        rendered = proof_component_group(
            level, baseline, group_label, components
        )
        if rendered is not None:
            rows.append(rendered)
    return "<br><br>".join(rows)


def render_fold_details(
    planned_levels: list[dict[str, object]],
    proof_levels: list[dict[str, object]],
    baseline_planned_levels: list[dict[str, object]] | None,
    baseline_proof_levels: list[dict[str, object]] | None,
) -> None:
    planned = {int(level["level"]): level for level in planned_levels}
    proof = {int(level["level"]): level for level in proof_levels}
    baseline_planned = {
        int(level["level"]): level for level in (baseline_planned_levels or [])
    }
    baseline_proof = {
        int(level["level"]): level for level in (baseline_proof_levels or [])
    }
    level_indices = sorted(set(planned) | set(proof))
    print("<details>")
    print("<summary>Fold schedule and proof cost</summary>")
    print()
    print("#### Fold by fold")
    print()
    headers = ["Fold", "Step", "Planner choice", "Work at this fold", "Proof bytes"]
    print("| " + " | ".join(headers) + " |")
    print("| --- | --- | --- | --- | --- |")

    for level_index in level_indices:
        schedule = planned.get(level_index)
        proof_level = proof.get(level_index)
        baseline_schedule = baseline_planned.get(level_index)
        baseline_proof_level = baseline_proof.get(level_index)
        step = proof_step_label(proof_level) if proof_level is not None else "scheduled fold"
        if schedule is None:
            schedule_choice = "—"
            work = "—"
        else:
            current_groups = planned_groups_for_render(schedule)
            baseline_groups = (
                planned_groups_for_render(baseline_schedule)
                if baseline_schedule is not None
                else []
            )
            schedule_choice = render_group_choices(
                current_groups, baseline_groups, planned_group_planner_value
            )
            work = render_group_choices(
                current_groups, baseline_groups, planned_group_work_value
            )
            next_w = f"Field elements: {fmt_count(float(schedule['next_w_len']))}"
            baseline_next_w = None
            if (
                baseline_schedule is not None
                and baseline_schedule.get("next_w_len") is not None
            ):
                baseline_next_w = (
                    f"Field elements: "
                    f"{fmt_count(float(baseline_schedule['next_w_len']))}"
                )
            work = (
                f"{work}<br><br>"
                f"{detail_block('Folded output', [exact_choice(next_w, baseline_next_w)])}"
            )

        proof_bytes = "n/a"
        if proof_level is not None:
            proof_bytes = proof_cost_summary(proof_level, baseline_proof_level)
        row = [f"L{level_index}", step, schedule_choice, work, proof_bytes]
        print("| " + " | ".join(row) + " |")

    print()
    print(
        "Role tuples use A / B / D order. The digit tuple adds folded witness W as "
        "its fourth value. Proof groups with zero bytes are omitted. Component details "
        "appear below the group total when a group contains multiple fields. "
        "Unchanged choices omit merge base text. Exact proof component comparisons "
        "show merge base bytes without a percentage. The terminal response is "
        "reported separately and is not part of the terminal fold byte total."
    )
    grind_rows = [
        level
        for level in proof_levels
        if int(level.get("grind_nonce_val", 0)) != 0
        or int(level.get("grind_attempts", 0)) != 0
    ]
    if grind_rows:
        print()
        print("#### Grinding retries")
        print()
        print("| Fold | Accepted nonce | Attempts |")
        print("| --- | ---: | ---: |")
        for level in grind_rows:
            baseline = baseline_proof.get(int(level["level"]))
            nonce = exact_choice(
                fmt_count(float(level.get("grind_nonce_val", 0))),
                fmt_count(float(baseline.get("grind_nonce_val", 0))) if baseline else None,
            )
            attempts = exact_choice(
                fmt_count(float(level.get("grind_attempts", 0))),
                fmt_count(float(baseline.get("grind_attempts", 0))) if baseline else None,
            )
            print(f"| L{level['level']} | {nonce} | {attempts} |")
    elif proof_levels:
        print()
        print("No fold needed a grinding retry.")
    else:
        print()
        print("Grinding was not measured because no proof fold data was emitted.")
    print()
    print("</details>")


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
        print("<details>")
        print("<summary>Terminal response components</summary>")
        print()
        render_terminal_response_components(current_cases, include_heading=False)
        print()
        print("</details>")
        print()
        print(
            "Detailed schedule and proof-size breakdowns by fold level are available in "
            "the uploaded `report.md` benchmark artifact."
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
        render_execution_parameters(current, main_case)
        if current.get("extension_root_direct_fallback"):
            print()
            print(
                "- Extension opening fallback: root-direct proof; folded planner byte estimates "
                "do not apply until the Frobenius optimization is wired."
            )
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
                baseline_planned_levels,
                baseline_proof_levels,
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
