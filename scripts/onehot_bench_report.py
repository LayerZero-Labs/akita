#!/usr/bin/env python3
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

    render_parser = subparsers.add_parser(
        "render", help="Render a markdown report from summary.json files."
    )
    render_parser.add_argument("summary", help="Path to the current summary.json file.")
    render_parser.add_argument(
        "baseline_dir",
        nargs="?",
        default="",
        help="Optional artifact directory containing baseline summary.json.",
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


def benchmark_name(mode: str, num_vars: int) -> str:
    if mode == "onehot":
        return f"1-of-{ONEHOT_ARITY} one-hot with {num_vars} variables"
    return f"{mode} with {num_vars} variables"


def extract_summary(log_text: str, mode: str, num_vars: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema_version": 1,
        "benchmark": benchmark_name(mode, num_vars),
        "mode": mode,
        "num_vars": num_vars,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }

    for line in log_text.splitlines():
        line = ANSI_RE.sub("", line)
        kvs = parse_kvs(line)
        if " INFO setup" in line and kvs.get("label") == mode:
            summary["setup_s"] = float(kvs["elapsed_s"])
        elif " INFO commit" in line and kvs.get("label") == mode:
            summary["commit_s"] = float(kvs["elapsed_s"])
        elif "hachi prove complete" in line:
            summary["prove_hachi_s"] = float(kvs["elapsed_s"])
            if "levels" in kvs:
                summary["hachi_levels"] = int(kvs["levels"])
        elif "labrador prove complete" in line:
            summary["prove_labrador_s"] = float(kvs["elapsed_s"])
            if "levels" in kvs:
                summary["labrador_levels"] = int(kvs["levels"])
        elif " INFO prove" in line and kvs.get("label") == mode:
            summary["prove_total_s"] = float(kvs["elapsed_s"])
        elif "hachi verify complete" in line:
            summary["verify_hachi_s"] = float(kvs["elapsed_s"])
        elif "labrador verify complete" in line:
            summary["verify_labrador_s"] = float(kvs["elapsed_s"])
            if "levels" in kvs and "labrador_levels" not in summary:
                summary["labrador_levels"] = int(kvs["levels"])
        elif "verify OK" in line and kvs.get("label") == mode:
            summary["verify_total_s"] = float(kvs["elapsed_s"])
        elif "proof summary" in line and kvs.get("label") == mode:
            summary["proof_size_bytes"] = int(kvs["proof_size_bytes"])
            summary["hachi_fold_bytes"] = int(kvs["hachi_fold_bytes"])
            summary["tail_bytes"] = int(kvs["tail_bytes"])
            if "levels" in kvs and "hachi_levels" not in summary:
                summary["hachi_levels"] = int(kvs["levels"])
        elif "estimated tail comparison" in line:
            if "selected_tail" in kvs:
                summary["selected_tail"] = kvs["selected_tail"]
            if "packed_direct_bytes" in kvs:
                summary["packed_direct_bytes"] = int(kvs["packed_direct_bytes"])
            if "estimated_labrador_tail_bytes" in kvs:
                summary["estimated_labrador_tail_bytes"] = int(
                    kvs["estimated_labrador_tail_bytes"]
                )

    for index, pattern in enumerate(RSS_PATTERNS):
        rss_match = pattern.search(log_text)
        if rss_match:
            rss_value = int(rss_match.group(1))
            if index == 1 and sys.platform == "darwin":
                rss_value //= 1024
            summary["max_rss_kib"] = rss_value
            break

    prove_total = require_float(summary, "prove_total_s")
    prove_hachi = float(summary.get("prove_hachi_s", 0.0))
    prove_labrador = float(summary.get("prove_labrador_s", 0.0))
    if prove_hachi == 0.0 and prove_labrador != 0.0:
        summary["prove_hachi_s"] = max(prove_total - prove_labrador, 0.0)
    elif prove_labrador == 0.0:
        summary["prove_labrador_s"] = max(prove_total - prove_hachi, 0.0)

    verify_total = require_float(summary, "verify_total_s")
    verify_hachi = float(summary.get("verify_hachi_s", 0.0))
    verify_labrador = float(summary.get("verify_labrador_s", 0.0))
    if verify_hachi == 0.0 and verify_labrador != 0.0:
        summary["verify_hachi_s"] = max(verify_total - verify_labrador, 0.0)
    elif verify_labrador == 0.0:
        summary["verify_labrador_s"] = max(verify_total - verify_hachi, 0.0)

    summary.setdefault("selected_tail", "unknown")
    return summary


def run_benchmark(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HACHI_MODE"] = args.mode
    env["HACHI_NUM_VARS"] = str(args.num_vars)
    env.setdefault("HACHI_PROFILE_TRACE", "0")
    env.setdefault("HACHI_PROFILE_SPAN_CLOSES", "0")
    env.setdefault("HACHI_PROFILE_LOG", "info")
    env.setdefault("HACHI_PROFILE_ANSI", "0")

    command = time_command(args.binary)
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    combined_log = completed.stdout + completed.stderr

    write_text(output_dir / "stdout.log", completed.stdout)
    write_text(output_dir / "stderr.log", completed.stderr)
    write_text(output_dir / "benchmark.log", combined_log)
    write_text(output_dir / "command.txt", " ".join(shlex.quote(part) for part in command) + "\n")

    if completed.returncode != 0:
        return completed.returncode

    summary = extract_summary(combined_log, mode=args.mode, num_vars=args.num_vars)
    summary["command"] = command
    summary["binary"] = args.binary
    summary["exit_code"] = completed.returncode
    summary["env"] = {
        "HACHI_MODE": env["HACHI_MODE"],
        "HACHI_NUM_VARS": env["HACHI_NUM_VARS"],
        "HACHI_PROFILE_TRACE": env["HACHI_PROFILE_TRACE"],
        "HACHI_PROFILE_SPAN_CLOSES": env["HACHI_PROFILE_SPAN_CLOSES"],
        "HACHI_PROFILE_LOG": env["HACHI_PROFILE_LOG"],
        "HACHI_PROFILE_ANSI": env["HACHI_PROFILE_ANSI"],
    }

    write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def load_summary(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def fmt_delta_seconds(value: float) -> str:
    return f"{value:+.3f}"


def fmt_mib(value_kib: float) -> str:
    return f"{value_kib / 1024.0:.1f}"


def fmt_delta_mib(delta_kib: float) -> str:
    return f"{delta_kib / 1024.0:+.1f}"


def fmt_bytes(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_delta_bytes(value: float) -> str:
    return f"{int(round(value)):+,}"


@dataclass(frozen=True)
class Metric:
    key: str
    name: str
    unit: str
    value_formatter: callable
    delta_formatter: callable


TIME_METRICS = [
    Metric("setup_s", "Setup", "s", fmt_seconds, fmt_delta_seconds),
    Metric("commit_s", "Commit", "s", fmt_seconds, fmt_delta_seconds),
    Metric("prove_hachi_s", "Prove (Hachi)", "s", fmt_seconds, fmt_delta_seconds),
    Metric("prove_labrador_s", "Prove (Labrador)", "s", fmt_seconds, fmt_delta_seconds),
    Metric("prove_total_s", "Prove (Total)", "s", fmt_seconds, fmt_delta_seconds),
    Metric("verify_hachi_s", "Verify (Hachi)", "s", fmt_seconds, fmt_delta_seconds),
    Metric("verify_labrador_s", "Verify (Labrador)", "s", fmt_seconds, fmt_delta_seconds),
    Metric("verify_total_s", "Verify (Total)", "s", fmt_seconds, fmt_delta_seconds),
    Metric("max_rss_kib", "Max RSS", "MiB", fmt_mib, fmt_delta_mib),
]


def render_metric_row(metric: Metric, current: dict[str, object], baseline: dict[str, object] | None) -> str:
    current_value = current.get(metric.key)
    if current_value is None:
        return ""

    current_rendered = metric.value_formatter(float(current_value))
    if baseline is None:
        return f"| {metric.name} | {current_rendered} | {metric.unit} |"

    if baseline.get(metric.key) is not None:
        baseline_value = baseline[metric.key]
        delta = float(current_value) - float(baseline_value)
        return (
            f"| {metric.name} | {metric.value_formatter(float(baseline_value))} | "
            f"{current_rendered} | {metric.delta_formatter(delta)} | {metric.unit} |"
        )

    return f"| {metric.name} | n/a | {current_rendered} | n/a | {metric.unit} |"


def render_report(args: argparse.Namespace) -> int:
    summary_path = pathlib.Path(args.summary)
    current = load_summary(summary_path)

    baseline_dir = pathlib.Path(args.baseline_dir) if args.baseline_dir else None
    baseline_summary_path = baseline_dir / "summary.json" if baseline_dir else None
    baseline = load_summary(baseline_summary_path) if baseline_summary_path and baseline_summary_path.exists() else None

    source_sha = os.environ.get("HACHI_BENCH_SOURCE_SHA")
    source_subject = os.environ.get("HACHI_BENCH_SOURCE_SUBJECT")
    source_branch = os.environ.get("HACHI_BENCH_SOURCE_BRANCH") or os.environ.get("GITHUB_REF_NAME")
    baseline_sha = os.environ.get("HACHI_BENCH_BASELINE_SHA")
    baseline_label = os.environ.get("HACHI_BENCH_BASELINE_LABEL")

    print("## One-hot 32 Variables Benchmark Report")
    print()
    print(f"- Benchmark: `{benchmark_name(current['mode'], int(current['num_vars']))}`")
    if current["mode"] == "onehot":
        print(
            f"- Sparsity: `1-of-{ONEHOT_ARITY}` one-hot "
            f"(equivalently, `1`-sparse over `{ONEHOT_ARITY}` slots, density `{100.0 / ONEHOT_ARITY:.2f}%`)."
        )
    ref = commit_ref(source_sha)
    if ref:
        print(f"- Commit: {ref}")
    if source_subject:
        print(f"- Message: {source_subject}")
    if source_branch:
        print(f"- Ref: `{source_branch}`")
    if baseline:
        baseline_ref = commit_ref(baseline_sha)
        if baseline_ref and baseline_label:
            print(f"- Comparison baseline: {baseline_ref} from {baseline_label}.")
        elif baseline_ref:
            print(f"- Comparison baseline: {baseline_ref}.")
        elif baseline_label:
            print(f"- Comparison baseline: {baseline_label}.")
    print(
        "- Command: `target/release/examples/profile` with "
        f"`HACHI_MODE={current['mode']}` `HACHI_NUM_VARS={current['num_vars']}` "
        "`HACHI_PROFILE_TRACE=0` `HACHI_PROFILE_SPAN_CLOSES=0` "
        "`HACHI_PROFILE_LOG=info` `HACHI_PROFILE_ANSI=0`."
    )
    print("- Memory: maximum resident set size from `/usr/bin/time` on the benchmark process.")
    print()

    if baseline:
        print("| Metric | Baseline | Current | Delta | Unit |")
        print("| --- | ---: | ---: | ---: | --- |")
    else:
        print("| Metric | Current | Unit |")
        print("| --- | ---: | --- |")

    for metric in TIME_METRICS:
        row = render_metric_row(metric, current, baseline)
        if row:
            print(row)

    print()
    print(f"- Tail: `{current.get('selected_tail', 'unknown')}`")
    if current.get("proof_size_bytes") is not None:
        print(f"- Proof size: `{fmt_bytes(float(current['proof_size_bytes']))} B`")
    if current.get("hachi_levels") is not None:
        print(f"- Hachi levels: `{current['hachi_levels']}`")
    if current.get("labrador_levels") is not None:
        print(f"- Labrador levels: `{current['labrador_levels']}`")

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
