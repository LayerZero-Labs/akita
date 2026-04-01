#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(r'([A-Za-z_]+)=(".*?"|\S+)')
RSS_PATTERNS = [
    re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)"),
    re.compile(r"^\s*(\d+)\s+maximum resident set size$", re.MULTILINE),
]
ONEHOT_ARITY = 256


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
        description="Run and render the Hachi onehot benchmark report."
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
        "schema_version": 1,
        "benchmark": benchmark_name(mode, num_vars, num_polys),
        "mode": mode,
        "num_vars": num_vars,
        "num_polys": num_polys,
        "case_id": case_id(mode, num_vars, num_polys),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }

    for line in log_text.splitlines():
        line = ANSI_RE.sub("", line)
        kvs = parse_kvs(line)
        if " INFO setup" in line and kvs.get("label") == mode:
            summary["setup_s"] = float(kvs["elapsed_s"])
        elif " INFO commit" in line and kvs.get("label") == mode:
            summary["commit_s"] = float(kvs["elapsed_s"])
        elif "hachi prove complete" in line or "hachi batched prove complete" in line:
            summary["prove_hachi_s"] = float(kvs["elapsed_s"])
            if "levels" in kvs:
                summary["hachi_levels"] = int(kvs["levels"])
        elif " INFO prove" in line and kvs.get("label") == mode:
            summary["prove_total_s"] = float(kvs["elapsed_s"])
        elif "hachi verify complete" in line or "hachi batched verify complete" in line:
            summary["verify_hachi_s"] = float(kvs["elapsed_s"])
        elif "verify OK" in line and kvs.get("label") == mode:
            summary["verify_total_s"] = float(kvs["elapsed_s"])
        elif "proof summary" in line and kvs.get("label") == mode:
            summary["proof_size_bytes"] = int(kvs["proof_size_bytes"])
            summary["hachi_fold_bytes"] = int(kvs["hachi_fold_bytes"])
            summary["tail_bytes"] = int(kvs["tail_bytes"])
            if "levels" in kvs and "hachi_levels" not in summary:
                summary["hachi_levels"] = int(kvs["levels"])
    for index, pattern in enumerate(RSS_PATTERNS):
        rss_match = pattern.search(log_text)
        if rss_match:
            rss_value = int(rss_match.group(1))
            if index == 1 and sys.platform == "darwin":
                rss_value //= 1024
            summary["max_rss_kib"] = rss_value
            break

    return summary


def run_benchmark_case(
    binary: str, output_dir: pathlib.Path, case: BenchmarkCaseSpec
) -> tuple[dict[str, object], int]:
    env = os.environ.copy()
    env["HACHI_MODE"] = case.mode
    env["HACHI_NUM_VARS"] = str(case.num_vars)
    env["HACHI_NUM_POLYS"] = str(case.num_polys)
    env.setdefault("HACHI_PROFILE_TRACE", "0")
    env.setdefault("HACHI_PROFILE_SPAN_CLOSES", "0")
    env.setdefault("HACHI_PROFILE_LOG", "info")
    env.setdefault("HACHI_PROFILE_ANSI", "0")

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
    summary["command"] = command
    summary["binary"] = binary
    summary["exit_code"] = completed.returncode
    summary["env"] = {
        "HACHI_MODE": env["HACHI_MODE"],
        "HACHI_NUM_VARS": env["HACHI_NUM_VARS"],
        "HACHI_NUM_POLYS": env["HACHI_NUM_POLYS"],
        "HACHI_PROFILE_TRACE": env["HACHI_PROFILE_TRACE"],
        "HACHI_PROFILE_SPAN_CLOSES": env["HACHI_PROFILE_SPAN_CLOSES"],
        "HACHI_PROFILE_LOG": env["HACHI_PROFILE_LOG"],
        "HACHI_PROFILE_ANSI": env["HACHI_PROFILE_ANSI"],
    }

    write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary, completed.returncode


def run_benchmark(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = configured_cases(args)
    aggregate_summary: dict[str, object] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": [],
    }

    for case in cases:
        case_dir = output_dir / case.case_id
        summary, return_code = run_benchmark_case(args.binary, case_dir, case)
        aggregate_summary["cases"].append(summary)
        if return_code != 0:
            write_text(
                output_dir / "summary.json",
                json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n",
            )
            return return_code

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


def commit_ref(sha: str | None) -> str | None:
    if not sha:
        return None
    short = sha[:7]
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return f"[`{short}`](https://github.com/{repo}/commit/{sha})"
    return f"`{short}`"


def fmt_seconds(value: float) -> str:
    return f"{value:.3f}"


def fmt_mib(value_kib: float) -> str:
    return f"{value_kib / 1024.0:.1f}"


def fmt_bytes(value: float) -> str:
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
    Metric("prove_hachi_s", "Prove (Hachi)", "s", fmt_seconds),
    Metric("prove_total_s", "Prove (Total)", "s", fmt_seconds),
    Metric("verify_hachi_s", "Verify (Hachi)", "s", fmt_seconds),
    Metric("verify_total_s", "Verify (Total)", "s", fmt_seconds),
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


def render_report(args: argparse.Namespace) -> int:
    summary_path = pathlib.Path(args.summary)
    current_cases = load_case_summaries(summary_path)

    baselines: list[tuple[str, dict[str, dict[str, object]] | None]] = [
        ("Main baseline", load_optional_case_summaries(args.main_baseline_dir)),
        ("Previous run", load_optional_case_summaries(args.previous_baseline_dir)),
    ]
    visible_baselines = [(label, summary) for label, summary in baselines if summary is not None]

    source_sha = os.environ.get("HACHI_BENCH_SOURCE_SHA")
    source_subject = os.environ.get("HACHI_BENCH_SOURCE_SUBJECT")
    source_branch = os.environ.get("HACHI_BENCH_SOURCE_BRANCH") or os.environ.get("GITHUB_REF_NAME")
    main_baseline_sha = os.environ.get("HACHI_BENCH_MAIN_BASELINE_SHA")
    main_baseline_label = os.environ.get("HACHI_BENCH_MAIN_BASELINE_LABEL")
    previous_baseline_sha = os.environ.get("HACHI_BENCH_PREVIOUS_BASELINE_SHA")
    previous_baseline_label = os.environ.get("HACHI_BENCH_PREVIOUS_BASELINE_LABEL")

    print("## One-hot Benchmark Report")
    print()
    ref = commit_ref(source_sha)
    if ref:
        print(f"- Latest run: {ref}")
    if source_subject:
        print(f"- Message: {source_subject}")
    if source_branch:
        print(f"- Ref: `{source_branch}`")
    if visible_baselines:
        main_ref = commit_ref(main_baseline_sha)
        if baselines[0][1] is not None:
            if main_ref and main_baseline_label:
                print(f"- Main baseline: {main_ref} from {main_baseline_label}.")
            elif main_ref:
                print(f"- Main baseline: {main_ref}.")
            elif main_baseline_label:
                print(f"- Main baseline: {main_baseline_label}.")

        previous_ref = commit_ref(previous_baseline_sha)
        if baselines[1][1] is not None:
            if previous_ref and previous_baseline_label:
                print(f"- Previous run: {previous_ref} from {previous_baseline_label}.")
            elif previous_ref:
                print(f"- Previous run: {previous_ref}.")
            elif previous_baseline_label:
                print(f"- Previous run: {previous_baseline_label}.")
    print("- Binary: `target/release/examples/profile`.")
    print("- Memory: maximum resident set size from `/usr/bin/time` on the benchmark process.")
    print()

    for index, current in enumerate(current_cases):
        if len(current_cases) > 1:
            print(f"### {section_title(current)}")
            print()
        print(f"- Benchmark: `{benchmark_name(current['mode'], int(current['num_vars']), int(current.get('num_polys', 1)))}`")
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
        print(
            "- Command: `target/release/examples/profile` with "
            f"`HACHI_MODE={env.get('HACHI_MODE', current['mode'])}` "
            f"`HACHI_NUM_VARS={env.get('HACHI_NUM_VARS', current['num_vars'])}` "
            f"`HACHI_NUM_POLYS={env.get('HACHI_NUM_POLYS', current.get('num_polys', 1))}` "
            "`HACHI_PROFILE_TRACE=0` `HACHI_PROFILE_SPAN_CLOSES=0` "
            "`HACHI_PROFILE_LOG=info` `HACHI_PROFILE_ANSI=0`."
        )
        print()

        case_baselines = [
            (label, summary.get(str(current["case_id"])) if summary is not None else None)
            for label, summary in visible_baselines
        ]
        column_labels = [label for label, _ in case_baselines] + ["Latest run"]
        print("| Metric | " + " | ".join(column_labels) + " | Unit |")
        print("| --- | " + " | ".join("---:" for _ in column_labels) + " | --- |")

        for metric in TIME_METRICS:
            row = render_metric_row(metric, current, case_baselines)
            if row:
                print(row)

        print()
        if current.get("proof_size_bytes") is not None:
            print(f"- Proof size: `{fmt_bytes(float(current['proof_size_bytes']))} B`")
        if current.get("hachi_levels") is not None:
            print(f"- Hachi levels: `{current['hachi_levels']}`")
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
