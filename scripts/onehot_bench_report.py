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
KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=(".*?"|\S+)')
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



def benchmark_name(mode: str, num_vars: int) -> str:
    if mode == "onehot":
        return f"1-of-{ONEHOT_ARITY} one-hot with {num_vars} variables"
    return f"{mode} with {num_vars} variables"


def extract_summary(log_text: str, mode: str, num_vars: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema_version": 2,
        "benchmark": benchmark_name(mode, num_vars),
        "mode": mode,
        "num_vars": num_vars,
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
        elif "hachi prove complete" in line:
            summary["prove_hachi_s"] = float(kvs["elapsed_s"])
            if "levels" in kvs:
                summary["hachi_levels"] = int(kvs["levels"])
        elif " INFO prove" in line and kvs.get("label") == mode:
            summary["prove_total_s"] = float(kvs["elapsed_s"])
        elif "hachi verify complete" in line:
            summary["verify_hachi_s"] = float(kvs["elapsed_s"])
        elif "verify OK" in line and kvs.get("label") == mode:
            summary["verify_total_s"] = float(kvs["elapsed_s"])
        elif "proof summary" in line and kvs.get("label") == mode:
            summary["proof_size_bytes"] = int(kvs["proof_size_bytes"])
            summary["hachi_fold_bytes"] = int(kvs["hachi_fold_bytes"])
            summary["tail_bytes"] = int(kvs["tail_bytes"])
            if "levels" in kvs and "hachi_levels" not in summary:
                summary["hachi_levels"] = int(kvs["levels"])
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
        elif "planned terminal state" in line and kvs.get("label") == mode:
            summary["terminal_w_len"] = int(kvs["final_w_len"])
            summary["terminal_log_basis"] = int(kvs["final_log_basis"])
        elif "proof fold level" in line and kvs.get("label") == mode:
            level = int(kvs["level"])
            proof_levels[level] = {
                "level": level,
                "d": int(kvs["d"]),
                "total_bytes": int(kvs["total_bytes"]),
                "y_ring_bytes": int(kvs["y_ring_bytes"]),
                "v_bytes": int(kvs["v_bytes"]),
                "stage1_sumcheck_bytes": int(kvs["stage1_sumcheck_bytes"]),
                "stage1_interstage_claims_bytes": int(kvs["stage1_interstage_claims_bytes"]),
                "stage1_s_claim_bytes": int(kvs["stage1_s_claim_bytes"]),
                "stage2_sumcheck_bytes": int(kvs["stage2_sumcheck_bytes"]),
                "next_w_commitment_bytes": int(kvs["next_w_commitment_bytes"]),
                "next_w_eval_bytes": int(kvs["next_w_eval_bytes"]),
            }
        elif "proof tail summary" in line and kvs.get("label") == mode:
            summary["tail_num_elems"] = int(kvs["final_w_num_elems"])
            summary["tail_bits_per_elem"] = int(kvs["final_w_bits_per_elem"])
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


def load_optional_summary(dir_path: str) -> dict[str, object] | None:
    if not dir_path:
        return None
    summary_path = pathlib.Path(dir_path) / "summary.json"
    if not summary_path.exists():
        return None
    return load_summary(summary_path)


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
            f"{fmt_bytes(float(level['next_w_ring']))} | {fmt_bytes(float(level['next_w_len']))} | "
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


def render_report(args: argparse.Namespace) -> int:
    summary_path = pathlib.Path(args.summary)
    current = load_summary(summary_path)

    baselines: list[tuple[str, dict[str, object] | None]] = [
        ("Main baseline", load_optional_summary(args.main_baseline_dir)),
        ("Previous run", load_optional_summary(args.previous_baseline_dir)),
    ]
    visible_baselines = [(label, summary) for label, summary in baselines if summary is not None]

    source_sha = os.environ.get("HACHI_BENCH_SOURCE_SHA")
    source_subject = os.environ.get("HACHI_BENCH_SOURCE_SUBJECT")
    source_branch = os.environ.get("HACHI_BENCH_SOURCE_BRANCH") or os.environ.get("GITHUB_REF_NAME")
    main_baseline_sha = os.environ.get("HACHI_BENCH_MAIN_BASELINE_SHA")
    main_baseline_label = os.environ.get("HACHI_BENCH_MAIN_BASELINE_LABEL")
    previous_baseline_sha = os.environ.get("HACHI_BENCH_PREVIOUS_BASELINE_SHA")
    previous_baseline_label = os.environ.get("HACHI_BENCH_PREVIOUS_BASELINE_LABEL")

    print(f"## {benchmark_name(current['mode'], int(current['num_vars']))} Benchmark Report")
    print()
    print(f"- Benchmark: `{benchmark_name(current['mode'], int(current['num_vars']))}`")
    if current["mode"] == "onehot":
        print(
            f"- Sparsity: `1-of-{ONEHOT_ARITY}` one-hot "
            f"(equivalently, `1`-sparse over `{ONEHOT_ARITY}` slots, density `{100.0 / ONEHOT_ARITY:.2f}%`)."
        )
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
    print(
        "- Command: `target/release/examples/profile` with "
        f"`HACHI_MODE={current['mode']}` `HACHI_NUM_VARS={current['num_vars']}` "
        "`HACHI_PROFILE_TRACE=0` `HACHI_PROFILE_SPAN_CLOSES=0` "
        "`HACHI_PROFILE_LOG=info` `HACHI_PROFILE_ANSI=0`."
    )
    print("- Memory: maximum resident set size from `/usr/bin/time` on the benchmark process.")
    print()

    column_labels = [label for label, _ in visible_baselines] + ["Latest run"]
    print("| Metric | " + " | ".join(column_labels) + " | Unit |")
    print("| --- | " + " | ".join("---:" for _ in column_labels) + " | --- |")

    for metric in TIME_METRICS:
        row = render_metric_row(metric, current, visible_baselines)
        if row:
            print(row)

    print()
    if current.get("proof_size_bytes") is not None:
        print(f"- Proof size: `{fmt_bytes(float(current['proof_size_bytes']))} B`")
    if current.get("hachi_fold_bytes") is not None:
        print(f"- Hachi fold bytes: `{fmt_bytes(float(current['hachi_fold_bytes']))} B`")
    if current.get("tail_bytes") is not None:
        print(f"- Tail bytes: `{fmt_bytes(float(current['tail_bytes']))} B`")
    if (
        current.get("proof_size_bytes") is not None
        and current.get("hachi_fold_bytes") is not None
        and current.get("tail_bytes") is not None
    ):
        framing_bytes = int(current["proof_size_bytes"]) - int(current["hachi_fold_bytes"]) - int(
            current["tail_bytes"]
        )
        print(f"- Proof framing bytes: `{fmt_bytes(float(framing_bytes))} B`")
    if current.get("hachi_levels") is not None:
        print(f"- Hachi levels: `{current['hachi_levels']}`")
    if current.get("tail_num_elems") is not None and current.get("tail_bits_per_elem") is not None:
        print(
            f"- Tail shape: `{fmt_bytes(float(current['tail_num_elems']))}` elems at "
            f"`{current['tail_bits_per_elem']}` bits/elem"
        )
    if current.get("terminal_w_len") is not None and current.get("terminal_log_basis") is not None:
        print(
            f"- Terminal state: `w_len={fmt_bytes(float(current['terminal_w_len']))}` "
            f"with `log_basis={current['terminal_log_basis']}`"
        )

    planned_levels = current.get("planned_levels")
    if isinstance(planned_levels, list) and planned_levels:
        print()
        render_planned_levels(planned_levels)

    proof_levels = current.get("proof_levels")
    if isinstance(proof_levels, list) and proof_levels:
        print()
        render_proof_levels(proof_levels)

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
