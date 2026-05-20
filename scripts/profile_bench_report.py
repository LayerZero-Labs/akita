#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
REQUIRED_RUN_METRICS = (
    "setup_s",
    "commit_s",
    "prove_total_s",
    "verify_total_s",
    "proof_size_bytes",
    "accounted_bytes",
)


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    mode: str
    num_vars: int
    num_polys: int

    @property
    def case_id(self) -> str:
        return case_id(self.mode, self.num_vars, self.num_polys)


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
    run_parser.add_argument("--mode", default="onehot", help="Benchmark mode.")
    run_parser.add_argument("--num-vars", type=int, default=32, help="Number of variables.")
    run_parser.add_argument(
        "--num-polys",
        type=int,
        default=1,
        help="Number of same-point polynomials in the benchmark case.",
    )
    run_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Benchmark case as NUM_VARS:NUM_POLYS or MODE:NUM_VARS:NUM_POLYS. "
            "Can be repeated."
        ),
    )
    run_parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("AKITA_BENCH_RUNS", "1")),
        help="Number of samples to run for each benchmark case; reported timings use the median.",
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
    proof_size = summary.get("proof_size_bytes")
    accounted = summary.get("accounted_bytes")
    if proof_size is not None and accounted is not None and int(proof_size) != int(accounted):
        missing.append("consistent_proof_accounting")
    return missing


TIMING_SAMPLE_METRICS = (
    "setup_s",
    "commit_s",
    "prove_total_s",
    "verify_total_s",
    "prove_akita_s",
    "verify_akita_s",
)
SAMPLE_METRICS = TIMING_SAMPLE_METRICS + ("max_rss_kib",)


def case_id(mode: str, num_vars: int, num_polys: int) -> str:
    return f"{mode}-nv{num_vars}-np{num_polys}"


def benchmark_name(mode: str, num_vars: int, num_polys: int = 1) -> str:
    if mode == "onehot":
        if num_polys > 1:
            return (
                f"{num_polys} same-point 1-of-{ONEHOT_ARITY} one-hot polynomials "
                f"with {num_vars} variables each"
            )
        return f"1-of-{ONEHOT_ARITY} one-hot with {num_vars} variables"
    if num_polys > 1:
        return f"{num_polys} same-point {mode} polynomials with {num_vars} variables each"
    return f"{mode} with {num_vars} variables"


def parse_case_spec(spec: str, default_mode: str) -> BenchmarkCaseSpec:
    parts = spec.split(":")
    if len(parts) == 2:
        mode = default_mode
        num_vars_str, num_polys_str = parts
    elif len(parts) == 3:
        mode, num_vars_str, num_polys_str = parts
    else:
        raise ValueError(
            f"invalid case spec {spec!r}; expected NUM_VARS:NUM_POLYS or MODE:NUM_VARS:NUM_POLYS"
        )
    num_vars = int(num_vars_str)
    num_polys = int(num_polys_str)
    if num_vars <= 0 or num_polys <= 0:
        raise ValueError(f"invalid case spec {spec!r}; NUM_VARS and NUM_POLYS must be positive")
    return BenchmarkCaseSpec(mode=mode, num_vars=num_vars, num_polys=num_polys)


def configured_cases(args: argparse.Namespace) -> list[BenchmarkCaseSpec]:
    if args.case:
        return [parse_case_spec(spec, args.mode) for spec in args.case]
    return [BenchmarkCaseSpec(mode=args.mode, num_vars=args.num_vars, num_polys=args.num_polys)]


def extract_summary(log_text: str, mode: str, num_vars: int, num_polys: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema_version": 3,
        "benchmark": benchmark_name(mode, num_vars, num_polys),
        "mode": mode,
        "num_vars": num_vars,
        "num_polys": num_polys,
        "case_id": case_id(mode, num_vars, num_polys),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    planned_levels: dict[int, dict[str, int]] = {}
    proof_levels: dict[int, dict[str, int]] = {}

    for line in log_text.splitlines():
        line = ANSI_RE.sub("", line)
        kvs = parse_kvs(line)
        if " INFO setup" in line and kvs.get("label") == mode:
            summary["setup_s"] = float(kvs["elapsed_s"])
        elif " INFO commit" in line and kvs.get("label") == mode:
            summary["commit_s"] = float(kvs["elapsed_s"])
        elif "akita prove complete" in line or "akita batched prove complete" in line:
            summary["prove_akita_s"] = float(kvs["elapsed_s"])
            if "levels" in kvs:
                summary["akita_levels"] = int(kvs["levels"])
        elif " INFO prove" in line and kvs.get("label") == mode:
            summary["prove_total_s"] = float(kvs["elapsed_s"])
        elif "akita verify complete" in line or "akita batched verify complete" in line:
            summary["verify_akita_s"] = float(kvs["elapsed_s"])
        elif "verify OK" in line and kvs.get("label") == mode:
            summary["verify_total_s"] = float(kvs["elapsed_s"])
        elif "proof summary" in line and kvs.get("label") == mode:
            summary["proof_size_bytes"] = int(kvs["proof_size_bytes"])
            summary["accounted_bytes"] = int(kvs["accounted_bytes"])
            summary["akita_fold_bytes"] = int(kvs["akita_fold_bytes"])
            summary["tail_bytes"] = int(kvs["tail_bytes"])
            if "proof_framing_bytes" in kvs:
                summary["proof_framing_bytes"] = int(kvs["proof_framing_bytes"])
            if "levels" in kvs and "akita_levels" not in summary:
                summary["akita_levels"] = int(kvs["levels"])
        elif "profile field roles" in line and kvs.get("label") == mode:
            summary["claim_ext_degree"] = int(kvs["claim_ext_degree"])
            summary["challenge_ext_degree"] = int(kvs["challenge_ext_degree"])
        elif "extension opening used root-direct fallback" in line and kvs.get("label") == mode:
            summary["extension_root_direct_fallback"] = True
        elif "planned fold level" in line and kvs.get("label") == mode:
            level = int(kvs["level"])
            planned_levels[level] = {
                "level": level,
                "d": int(kvs["d"]),
                "n_a": int(kvs["n_a"]),
                "n_b": int(kvs["n_b"]),
                "n_d": int(kvs["n_d"]),
                "challenge_l1_mass": int(kvs["challenge_l1_mass"]),
                "log_basis": int(kvs["log_basis"]),
                "m_vars": int(kvs["m_vars"]),
                "r_vars": int(kvs["r_vars"]),
                "num_blocks": int(kvs["num_blocks"]),
                "block_len": int(kvs["block_len"]),
                "delta_commit": int(kvs["delta_commit"]),
                "delta_open": int(kvs["delta_open"]),
                "delta_fold": int(kvs["delta_fold"]),
                "current_w_len": int(kvs["current_w_len"]),
                "next_w_ring": int(kvs["next_w_ring"]),
                "next_w_len": int(kvs["next_w_len"]),
                "level_bytes": int(kvs["level_bytes"]),
            }
        elif "proof fold level" in line and kvs.get("label") == mode:
            level = int(kvs["level"])
            # The emitter omits keys for components that don't exist in
            # the current proof-step variant (e.g. terminal levels have
            # no `v`, `stage1_*`, or `next_w_*`; root-direct has none of
            # the per-component fields). Default to "0" for missing keys
            # so the table column for that step renders as 0.
            proof_levels[level] = {
                "level": level,
                "d": int(kvs["d"]),
                "total_bytes": int(kvs["total_bytes"]),
                "y_ring_bytes": int(kvs.get("y_ring_bytes", "0")),
                "v_bytes": int(kvs.get("v_bytes", "0")),
                "stage1_sumcheck_bytes": int(kvs.get("stage1_sumcheck_bytes", "0")),
                "stage1_interstage_claims_bytes": int(
                    kvs.get("stage1_interstage_claims_bytes", "0")
                ),
                "stage1_s_claim_bytes": int(kvs.get("stage1_s_claim_bytes", "0")),
                "stage2_sumcheck_bytes": int(kvs.get("stage2_sumcheck_bytes", "0")),
                "next_w_commitment_bytes": int(kvs.get("next_w_commitment_bytes", "0")),
                "next_w_eval_bytes": int(kvs.get("next_w_eval_bytes", "0")),
            }
            if "root_variant" in kvs:
                proof_levels[level]["root_variant"] = kvs["root_variant"]
        elif "proof tail summary" in line and kvs.get("label") == mode:
            summary["tail_num_elems"] = int(kvs["final_w_num_elems"])
            if "final_w_encoding" in kvs:
                summary["tail_encoding"] = kvs["final_w_encoding"]
            bits_per_elem = kvs.get("final_w_bits_per_elem")
            summary["terminal_w_len"] = int(kvs["final_w_num_elems"])
            if bits_per_elem is not None and bits_per_elem != "None":
                summary["tail_bits_per_elem"] = int(bits_per_elem)
                summary["terminal_log_basis"] = int(bits_per_elem)
    for index, pattern in enumerate(RSS_PATTERNS):
        rss_match = pattern.search(log_text)
        if rss_match:
            rss_value = int(rss_match.group(1))
            if index == 1 and sys.platform == "darwin":
                rss_value //= 1024
            summary["max_rss_kib"] = rss_value
            break

    if planned_levels:
        summary["planned_levels"] = [planned_levels[level] for level in sorted(planned_levels)]
    if proof_levels:
        summary["proof_levels"] = [proof_levels[level] for level in sorted(proof_levels)]

    return summary


def run_benchmark_case(
    binary: str, output_dir: pathlib.Path, case: BenchmarkCaseSpec
) -> tuple[dict[str, object], int]:
    env = os.environ.copy()
    env["AKITA_MODE"] = case.mode
    env["AKITA_NUM_VARS"] = str(case.num_vars)
    env["AKITA_NUM_POLYS"] = str(case.num_polys)
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
        combined_log, mode=case.mode, num_vars=case.num_vars, num_polys=case.num_polys
    )
    return_code = completed.returncode
    summary["command"] = command
    summary["binary"] = binary
    summary["exit_code"] = return_code
    summary["env"] = {
        "AKITA_MODE": env["AKITA_MODE"],
        "AKITA_NUM_VARS": env["AKITA_NUM_VARS"],
        "AKITA_NUM_POLYS": env["AKITA_NUM_POLYS"],
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
            summary["exit_code"] = 1
            return_code = 1

    write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary, return_code


def compact_sample_summary(summary: dict[str, object]) -> dict[str, object]:
    sample = {
        "run_index": summary["run_index"],
        "exit_code": summary["exit_code"],
    }
    for key in SAMPLE_METRICS:
        if key in summary:
            sample[key] = summary[key]
    return sample


def combine_case_run_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    combined = dict(summaries[0])
    combined["runs"] = len(summaries)
    combined["samples"] = [compact_sample_summary(summary) for summary in summaries]

    for key in TIMING_SAMPLE_METRICS:
        values = [float(summary[key]) for summary in summaries if summary.get(key) is not None]
        if values:
            combined[key] = statistics.median(values)

    rss_values = [int(summary["max_rss_kib"]) for summary in summaries if summary.get("max_rss_kib")]
    if rss_values:
        combined["max_rss_kib"] = max(rss_values)

    return combined


def run_benchmark(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    cases = configured_cases(args)
    aggregate_summary: dict[str, object] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": [],
    }

    for case in cases:
        case_dir = output_dir / case.case_id
        run_summaries = []
        for run_index in range(1, args.runs + 1):
            run_dir = case_dir if args.runs == 1 else case_dir / f"run-{run_index}"
            summary, return_code = run_benchmark_case(args.binary, run_dir, case)
            summary["run_index"] = run_index
            run_summaries.append(summary)
            if return_code != 0:
                aggregate_summary["cases"].append(combine_case_run_summaries(run_summaries))
                write_text(
                    output_dir / "summary.json",
                    json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n",
                )
                return return_code
        aggregate_summary["cases"].append(combine_case_run_summaries(run_summaries))

    write_text(
        output_dir / "summary.json", json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


def load_summary(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_case_summary(summary: dict[str, object]) -> dict[str, object]:
    normalized = dict(summary)
    mode = str(normalized["mode"])
    num_vars = int(normalized["num_vars"])
    num_polys = int(normalized.get("num_polys", 1))
    normalized["num_polys"] = num_polys
    normalized["case_id"] = str(normalized.get("case_id", case_id(mode, num_vars, num_polys)))
    normalized["benchmark"] = benchmark_name(mode, num_vars, num_polys)
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


def fmt_mib(value_kib: float) -> str:
    return f"{value_kib / 1024.0:.1f}"


def fmt_bytes(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_count(value: float) -> str:
    return f"{int(round(value)):,}"


def section_title(summary: dict[str, object]) -> str:
    num_polys = int(summary.get("num_polys", 1))
    num_vars = int(summary["num_vars"])
    if num_polys == 1:
        return f"Single Polynomial x {num_vars} Variables"
    return f"{num_polys} Polynomials x {num_vars} Variables"


@dataclass(frozen=True)
class Metric:
    key: str
    name: str
    unit: str
    value_formatter: callable


TIME_METRICS = [
    Metric("setup_s", "Setup", "s", fmt_seconds),
    Metric("commit_s", "Commit", "s", fmt_seconds),
    Metric("prove_total_s", "Prove", "s", fmt_seconds),
    Metric("verify_total_s", "Verify", "s", fmt_seconds),
    Metric("max_rss_kib", "Max RSS", "MiB", fmt_mib),
]


def render_metric_row(
    metric: Metric,
    current: dict[str, object],
    baselines: list[tuple[str, dict[str, object] | None]],
) -> str:
    current_value = current.get(metric.key)
    if current_value is None:
        return ""

    columns: list[str] = []
    for _, summary in baselines:
        if summary is None or summary.get(metric.key) is None:
            columns.append("n/a")
        else:
            columns.append(metric.value_formatter(float(summary[metric.key])))

    columns.append(metric.value_formatter(float(current_value)))
    return f"| {metric.name} | " + " | ".join(columns) + f" | {metric.unit} |"


def sample_range(summary: dict[str, object], key: str) -> tuple[float, float] | None:
    samples = summary.get("samples")
    if not isinstance(samples, list):
        return None
    values = [float(sample[key]) for sample in samples if isinstance(sample, dict) and key in sample]
    if len(values) <= 1:
        return None
    return min(values), max(values)


def render_planned_levels(levels: list[dict[str, object]]) -> None:
    print("<details>")
    print("<summary>Per-level parameters</summary>")
    print()
    print(
        "| L | Config | D | nA | nB | nD | lb | l1 | m | r | "
        "δcommit | δopen | δfold | next w (ring) | next w (field) | planned bytes |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for level in levels:
        print(
            f"| L{level['level']} | `D{level['d']}-na{level['n_a']}` | "
            f"{level['d']} | {level['n_a']} | {level['n_b']} | {level['n_d']} | "
            f"{level['log_basis']} | {level['challenge_l1_mass']} | {level['m_vars']} | {level['r_vars']} | "
            f"{level['delta_commit']} | {level['delta_open']} | {level['delta_fold']} | "
            f"{fmt_count(float(level['next_w_ring']))} | {fmt_count(float(level['next_w_len']))} | "
            f"{fmt_bytes(float(level['level_bytes']))} B |"
        )
    print()
    print("</details>")


def render_proof_levels(levels: list[dict[str, object]]) -> None:
    print("<details>")
    print("<summary>Per-level proof-size breakdown</summary>")
    print()
    print(
        "| L | total | y_ring | v | stage1 sc | interstage | s_claim | "
        "stage2 sc | next_w_commit | next_w_eval |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for level in levels:
        print(
            f"| L{level['level']} | {fmt_bytes(float(level['total_bytes']))} B | "
            f"{fmt_bytes(float(level['y_ring_bytes']))} | {fmt_bytes(float(level['v_bytes']))} | "
            f"{fmt_bytes(float(level['stage1_sumcheck_bytes']))} | "
            f"{fmt_bytes(float(level['stage1_interstage_claims_bytes']))} | "
            f"{fmt_bytes(float(level['stage1_s_claim_bytes']))} | "
            f"{fmt_bytes(float(level['stage2_sumcheck_bytes']))} | "
            f"{fmt_bytes(float(level['next_w_commitment_bytes']))} | "
            f"{fmt_bytes(float(level['next_w_eval_bytes']))} |"
        )
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

    planned_levels = summary.get("planned_levels")
    proof_levels = summary.get("proof_levels")
    if not isinstance(planned_levels, list) or not isinstance(proof_levels, list):
        return
    if len(planned_levels) != len(proof_levels):
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
        planned_d = int(planned["d"])
        proof_d = int(proof["d"])
        if planned_d != proof_d:
            raise ValueError(
                f"planned/proof D mismatch at L{planned_level}: "
                f"planned={planned_d}, proof={proof_d}"
            )
        planned_bytes = int(planned["level_bytes"])
        proof_bytes = int(proof["total_bytes"])
        if planned_bytes != proof_bytes:
            raise ValueError(
                f"planned/proof byte mismatch at L{planned_level}: "
                f"planned={planned_bytes}, proof={proof_bytes}"
            )


def render_report(args: argparse.Namespace) -> int:
    summary_path = pathlib.Path(args.summary)
    current_cases = load_case_summaries(summary_path)

    baselines: list[tuple[str, dict[str, dict[str, object]] | None]] = [
        ("Main baseline", load_optional_case_summaries(args.main_baseline_dir)),
        ("Previous run", load_optional_case_summaries(args.previous_baseline_dir)),
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
        print(
            "## "
            f"{md_text(benchmark_name(only_case['mode'], int(only_case['num_vars']), int(only_case.get('num_polys', 1))))} "
            "Benchmark Report"
        )
    else:
        print("## Benchmark Report")
    print()
    ref = commit_ref(source_sha)
    if ref:
        print(f"- Latest run: {ref}")
    if source_subject:
        print(f"- Message: {md_text(source_subject)}")
    if source_branch:
        print(f"- Ref: {code_text(source_branch)}")
    run_ref = workflow_run_ref()
    if run_ref:
        print(f"- Workflow run: {run_ref}")
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    print(f"- Report generated: `{generated_at}`.")
    if visible_baselines:
        main_ref = commit_ref(main_baseline_sha)
        if baselines[0][1] is not None:
            if main_ref and main_baseline_label:
                print(f"- Main baseline: {main_ref} from {md_text(main_baseline_label)}.")
            elif main_ref:
                print(f"- Main baseline: {main_ref}.")
            elif main_baseline_label:
                print(f"- Main baseline: {md_text(main_baseline_label)}.")

        previous_ref = commit_ref(previous_baseline_sha)
        if baselines[1][1] is not None:
            if previous_ref and previous_baseline_label:
                print(f"- Previous run: {previous_ref} from {md_text(previous_baseline_label)}.")
            elif previous_ref:
                print(f"- Previous run: {previous_ref}.")
            elif previous_baseline_label:
                print(f"- Previous run: {md_text(previous_baseline_label)}.")
    if base_ref and baselines[0][1] is None:
        print(f"- Main baseline: no reusable benchmark artifact found for `{base_ref}`.")
    print("- Binary: `target/release/examples/profile`.")
    print("- Memory: maximum resident set size from `/usr/bin/time` on the benchmark process.")
    print()

    for index, current in enumerate(current_cases):
        validate_case_consistency(current)
        if len(current_cases) > 1:
            print(f"### {section_title(current)}")
            print()
        print(
            "- Benchmark: "
            f"{code_text(benchmark_name(current['mode'], int(current['num_vars']), int(current.get('num_polys', 1))))}"
        )
        if current["mode"] == "onehot":
            num_polys = int(current.get("num_polys", 1))
            if num_polys > 1:
                print(
                    f"- Batch: same-point opening of `{num_polys}` polynomials, "
                    f"each with `{current['num_vars']}` variables."
                )
            print(
                f"- Sparsity: each polynomial is `1-of-{ONEHOT_ARITY}` one-hot "
                f"(equivalently, `1`-sparse over `{ONEHOT_ARITY}` slots, density `{100.0 / ONEHOT_ARITY:.2f}%`)."
            )
        env = current.get("env", {})
        command_env = [
            code_text(f"AKITA_MODE={env.get('AKITA_MODE', current['mode'])}"),
            code_text(f"AKITA_NUM_VARS={env.get('AKITA_NUM_VARS', current['num_vars'])}"),
            code_text(f"AKITA_NUM_POLYS={env.get('AKITA_NUM_POLYS', current.get('num_polys', 1))}"),
        ]
        print(
            "- Command: `target/release/examples/profile` with "
            f"{' '.join(command_env)} "
            "`AKITA_PROFILE_TRACE=0` `AKITA_PROFILE_SPAN_CLOSES=0` "
            "`AKITA_PROFILE_LOG=info` `AKITA_PROFILE_ANSI=0`."
        )
        runs = int(current.get("runs", 1))
        if runs > 1:
            print(f"- Samples: metrics are the median of `{runs}` runs; Max RSS is the maximum sample.")
        print()

        case_baselines = [
            (label, summary.get(str(current["case_id"])) if summary is not None else None)
            for label, summary in visible_baselines
        ]
        column_labels = [md_text(label) for label, _ in case_baselines] + ["Latest run"]
        print("| Metric | " + " | ".join(column_labels) + " | Unit |")
        print("| --- | " + " | ".join("---:" for _ in column_labels) + " | --- |")

        for metric in TIME_METRICS:
            row = render_metric_row(metric, current, case_baselines)
            if row:
                print(row)

        if runs > 1:
            ranges = []
            for key, label in [
                ("setup_s", "setup"),
                ("commit_s", "commit"),
                ("prove_total_s", "prove"),
                ("verify_total_s", "verify"),
            ]:
                observed_range = sample_range(current, key)
                if observed_range is not None:
                    ranges.append(
                        f"{label} `{fmt_seconds(observed_range[0])}-{fmt_seconds(observed_range[1])}s`"
                    )
            if ranges:
                print()
                print(f"- Sample ranges: {', '.join(ranges)}.")

        print()
        if current.get("proof_size_bytes") is not None:
            print(f"- Proof size: `{fmt_bytes(float(current['proof_size_bytes']))} B`")
        if current.get("akita_fold_bytes") is not None:
            print(f"- Akita fold bytes: `{fmt_bytes(float(current['akita_fold_bytes']))} B`")
        if current.get("tail_bytes") is not None:
            print(f"- Tail bytes: `{fmt_bytes(float(current['tail_bytes']))} B`")
        if (
            current.get("proof_framing_bytes") is not None
            or (
                current.get("proof_size_bytes") is not None
                and current.get("akita_fold_bytes") is not None
                and current.get("tail_bytes") is not None
            )
        ):
            framing_bytes = int(current.get("proof_framing_bytes", 0))
            if "proof_framing_bytes" not in current:
                framing_bytes = int(current["proof_size_bytes"]) - int(current["akita_fold_bytes"]) - int(
                    current["tail_bytes"]
                )
            print(f"- Proof framing bytes: `{fmt_bytes(float(framing_bytes))} B`")
        if current.get("akita_levels") is not None:
            print(f"- Akita levels: `{current['akita_levels']}`")
        if current.get("claim_ext_degree") is not None or current.get("challenge_ext_degree") is not None:
            print(
                f"- Field roles: `claim_ext_degree={current.get('claim_ext_degree', 'n/a')}`, "
                f"`challenge_ext_degree={current.get('challenge_ext_degree', 'n/a')}`"
            )
        if current.get("extension_root_direct_fallback"):
            print(
                "- Extension opening fallback: root-direct proof; folded planner byte estimates "
                "do not apply until the Frobenius/multipoint optimization is wired."
            )
        if current.get("tail_num_elems") is not None and current.get("tail_bits_per_elem") is not None:
            print(
                f"- Tail shape: `{fmt_count(float(current['tail_num_elems']))}` elems at "
                f"`{current['tail_bits_per_elem']}` bits/elem"
            )
        elif current.get("tail_num_elems") is not None and current.get("tail_encoding") == "field_elements":
            print(f"- Tail shape: `{fmt_count(float(current['tail_num_elems']))}` field elements")
        if current.get("terminal_w_len") is not None and current.get("terminal_log_basis") is not None:
            print(
                f"- Observed terminal state: `w_len={fmt_count(float(current['terminal_w_len']))}` "
                f"with `log_basis={current['terminal_log_basis']}`"
            )
        elif current.get("terminal_w_len") is not None and current.get("tail_encoding") == "field_elements":
            print(
                f"- Observed terminal state: `w_len={fmt_count(float(current['terminal_w_len']))}` "
                f"with field-element encoding"
            )

        planned_levels = current.get("planned_levels")
        if isinstance(planned_levels, list) and planned_levels:
            print()
            render_planned_levels(planned_levels)

        proof_levels = current.get("proof_levels")
        if isinstance(proof_levels, list) and proof_levels:
            print()
            render_proof_levels(proof_levels)
        if index + 1 < len(current_cases):
            print()

    return 0


def main() -> int:
    args = parse_args()
    if args.command == "run":
        return run_benchmark(args)
    if args.command == "render":
        return render_report(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
