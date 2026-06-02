#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass


MARKER = "<!-- akita-ci-test-timing -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and render CI test timing reports from cargo-nextest JUnit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge", help="Merge JUnit XML into summary.json.")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--source-sha", required=True)
    merge.add_argument("--source-branch", required=True)
    merge.add_argument("--workflow-run-id", type=int, default=0)
    merge.add_argument("--pass", dest="passes", action="append", default=[])
    merge.add_argument("--junit", dest="junits", action="append", default=[])
    merge.add_argument("--started-at", dest="started_ats", action="append", default=[])
    merge.add_argument("--finished-at", dest="finished_ats", action="append", default=[])
    merge.add_argument("--exit-code", dest="exit_codes", action="append", default=[])

    render = subparsers.add_parser("render", help="Render comment.md/report.md from summary.json.")
    render.add_argument("summary", help="Path to current summary.json")
    render.add_argument("--output-dir", required=True)
    render.add_argument("--main-baseline-dir", default="")
    render.add_argument("--previous-baseline-dir", default="")
    render.add_argument(
        "--compact",
        action="store_true",
        help="Write a shorter report (intended for CI $GITHUB_STEP_SUMMARY).",
    )

    failure = subparsers.add_parser(
        "failure-summary", help="Write a failure-only summary.json and comment/report."
    )
    failure.add_argument("--output-dir", required=True)
    failure.add_argument("--source-sha", required=True)
    failure.add_argument("--source-branch", required=True)
    failure.add_argument("--workflow-run-id", type=int, default=0)
    failure.add_argument("--error", required=True)
    failure.add_argument("--passes", nargs="+", default=["non-zk", "all-features"])

    return parser.parse_args()


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: pathlib.Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def md_text(value: object) -> str:
    text = html.escape(str(value), quote=False).replace("\\", "\\\\")
    for char in "`*_{}[]()#+-.!|":
        text = text.replace(char, f"\\{char}")
    return text


def code_text(value: object) -> str:
    return f"<code>{html.escape(str(value), quote=False)}</code>"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TestCase:
    binary: str
    test: str
    classname: str
    duration_s: float
    status: str  # ok|skipped|failed
    id: str

    @property
    def match_key(self) -> tuple[str, str, str]:
        return (self.binary, self.test, self.classname)


def parse_junit(junit_path: pathlib.Path) -> list[TestCase]:
    tree = ET.parse(junit_path)
    root = tree.getroot()

    testsuites: list[ET.Element] = []
    if root.tag == "testsuites":
        testsuites = list(root.findall("testsuite"))
    elif root.tag == "testsuite":
        testsuites = [root]
    else:
        testsuites = list(root.findall(".//testsuite"))

    out: list[TestCase] = []
    seen: dict[str, int] = {}

    for suite in testsuites:
        suite_name = suite.attrib.get("name", "") or "unknown"
        for case in suite.findall(".//testcase"):
            classname = case.attrib.get("classname", "") or suite_name
            binary = classname
            test_name = case.attrib.get("name", "") or "unknown"
            duration_s = safe_float(case.attrib.get("time", ""), 0.0)

            status = "ok"
            if case.find("skipped") is not None:
                status = "skipped"
            if case.find("failure") is not None or case.find("error") is not None:
                status = "failed"

            base_id = f"{binary}::{test_name}"
            count = seen.get(base_id, 0) + 1
            seen[base_id] = count
            test_id = base_id if count == 1 else f"{base_id}#{count}"

            out.append(
                TestCase(
                    binary=binary,
                    test=test_name,
                    classname=classname,
                    duration_s=duration_s,
                    status=status,
                    id=test_id,
                )
            )

    out.sort(key=lambda t: t.duration_s, reverse=True)
    return out


def load_summary(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_summary(dir_path: str) -> dict[str, object] | None:
    if not dir_path:
        return None
    path = pathlib.Path(dir_path) / "summary.json"
    if not path.exists():
        return None
    return load_summary(path)


def normalize_pass(summary: dict[str, object], pass_name: str) -> dict[str, object] | None:
    passes = summary.get("passes")
    if not isinstance(passes, dict):
        return None
    value = passes.get(pass_name)
    if not isinstance(value, dict):
        return None
    return value


def cases_by_key(pass_obj: dict[str, object] | None) -> dict[tuple[str, str, str], dict[str, object]]:
    if pass_obj is None:
        return {}
    tests = pass_obj.get("tests")
    if not isinstance(tests, list):
        return {}
    out: dict[tuple[str, str, str], dict[str, object]] = {}
    for raw in tests:
        if not isinstance(raw, dict):
            continue
        binary = str(raw.get("binary", ""))
        test = str(raw.get("test", ""))
        classname = str(raw.get("classname", ""))
        out[(binary, test, classname)] = raw
    return out


def percent_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline == 0.0:
        return None
    return (current / baseline - 1.0) * 100.0


def render_report(
    current: dict[str, object],
    main: dict[str, object] | None,
    previous: dict[str, object] | None,
    compact: bool,
) -> tuple[str, str]:
    generated_at = now_utc_iso()

    current_sha = str(current.get("source_sha", ""))
    current_branch = str(current.get("source_branch", ""))
    workflow_run_id = safe_int(str(current.get("workflow_run_id", "0")), 0)

    main_sha = str(main.get("source_sha", "")) if isinstance(main, dict) else ""
    prev_sha = str(previous.get("source_sha", "")) if isinstance(previous, dict) else ""

    lines: list[str] = []
    lines.append(MARKER)
    lines.append("")
    lines.append("## CI test timing")
    lines.append("")
    lines.append(f"- Report generated: `{generated_at}`.")
    if current_sha:
        lines.append(f"- Source: {code_text(current_sha[:7])} on {code_text(current_branch)}.")
    if workflow_run_id:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        if repo:
            lines.append(f"- Workflow run: [{workflow_run_id}]({server}/{repo}/actions/runs/{workflow_run_id}).")
    if main_sha:
        lines.append(f"- Main baseline: {code_text(main_sha[:7])}.")
    if prev_sha:
        lines.append(f"- Previous run: {code_text(prev_sha[:7])}.")
    lines.append("")

    pass_order = ["non-zk", "all-features"]
    lines.append("### Pass summary")
    lines.append("")
    lines.append("| Pass | Wall s | Main wall s | Main Δ | Ratio | Tests | Skipped | Failed | Status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for pass_name in pass_order:
        cur_pass = normalize_pass(current, pass_name) or {}
        main_pass = normalize_pass(main, pass_name) if main is not None else None

        cur_wall = cur_pass.get("wall_s")
        cur_wall_f = float(cur_wall) if cur_wall is not None else None
        main_wall_f = float(main_pass["wall_s"]) if main_pass and main_pass.get("wall_s") is not None else None
        delta_pct = percent_delta(cur_wall_f, main_wall_f)
        ratio = (cur_wall_f / main_wall_f) if (cur_wall_f is not None and main_wall_f not in (None, 0.0)) else None

        status = "ok" if safe_int(str(cur_pass.get("exit_code", "0")), 0) == 0 else "fail"
        tests = safe_int(str(cur_pass.get("test_count", "0")), 0)
        skipped = safe_int(str(cur_pass.get("skipped", "0")), 0)
        failed = safe_int(str(cur_pass.get("failed", "0")), 0)

        ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    md_text(pass_name),
                    fmt_seconds(cur_wall_f),
                    fmt_seconds(main_wall_f),
                    fmt_pct(delta_pct),
                    ratio_str,
                    str(tests),
                    str(skipped),
                    str(failed),
                    status,
                ]
            )
            + " |"
        )
    lines.append("")

    def render_slowest(pass_name: str) -> None:
        cur_pass = normalize_pass(current, pass_name) or {}
        tests_raw = cur_pass.get("tests")
        if not isinstance(tests_raw, list) or not tests_raw:
            lines.append(f"### Slowest tests ({md_text(pass_name)})")
            lines.append("")
            lines.append("_No JUnit data available for this pass._")
            lines.append("")
            return
        rows: list[dict[str, object]] = [t for t in tests_raw if isinstance(t, dict)]
        rows.sort(key=lambda r: float(r.get("duration_s", 0.0) or 0.0), reverse=True)
        top_n = 10 if compact else 20
        lines.append(f"### Slowest tests ({md_text(pass_name)})")
        lines.append("")
        lines.append("| Rank | Duration s | Test |")
        lines.append("| ---: | ---: | --- |")
        for i, row in enumerate(rows[:top_n], start=1):
            duration = safe_float(str(row.get("duration_s", "0")), 0.0)
            test_id = str(row.get("id", ""))
            lines.append(f"| {i} | {fmt_seconds(duration)} | {code_text(test_id)} |")
        lines.append("")

    for pass_name in pass_order:
        render_slowest(pass_name)

    if not compact and main is not None:
        lines.append("### Regressions vs main")
        lines.append("")
        regression_rows: list[tuple[str, float, float, float]] = []
        for pass_name in pass_order:
            cur_pass = normalize_pass(current, pass_name) or {}
            main_pass = normalize_pass(main, pass_name) or {}
            cur_map = cases_by_key(cur_pass)
            main_map = cases_by_key(main_pass)
            for key, cur_row in cur_map.items():
                base_row = main_map.get(key)
                if base_row is None:
                    continue
                cur_d = safe_float(str(cur_row.get("duration_s", "0")), 0.0)
                base_d = safe_float(str(base_row.get("duration_s", "0")), 0.0)
                if base_d <= 0.0:
                    continue
                delta = cur_d - base_d
                if delta <= 0.0:
                    continue
                threshold = max(5.0, 0.10 * base_d)
                if delta >= threshold:
                    regression_rows.append((pass_name, delta, base_d, cur_d, str(cur_row.get("id", ""))))
        regression_rows.sort(key=lambda r: r[1], reverse=True)
        if not regression_rows:
            lines.append("_No per-test regressions above the threshold._")
            lines.append("")
        else:
            lines.append("| Pass | Δ s | Baseline s | Current s | Test |")
            lines.append("| --- | ---: | ---: | ---: | --- |")
            for pass_name, delta, base_d, cur_d, test_id in regression_rows[:15]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            md_text(pass_name),
                            fmt_seconds(delta),
                            fmt_seconds(base_d),
                            fmt_seconds(cur_d),
                            code_text(test_id),
                        ]
                    )
                    + " |"
                )
            lines.append("")

        lines.append("### New slow tests")
        lines.append("")
        new_rows: list[tuple[str, float, str]] = []
        for pass_name in pass_order:
            cur_pass = normalize_pass(current, pass_name) or {}
            main_pass = normalize_pass(main, pass_name) or {}
            cur_map = cases_by_key(cur_pass)
            main_map = cases_by_key(main_pass)
            for key, cur_row in cur_map.items():
                if key in main_map:
                    continue
                cur_d = safe_float(str(cur_row.get("duration_s", "0")), 0.0)
                if cur_d >= 30.0:
                    new_rows.append((pass_name, cur_d, str(cur_row.get("id", ""))))
        new_rows.sort(key=lambda r: r[1], reverse=True)
        if not new_rows:
            lines.append("_No new tests ≥30s vs main baseline._")
            lines.append("")
        else:
            lines.append("| Pass | Duration s | Test |")
            lines.append("| --- | ---: | --- |")
            for pass_name, cur_d, test_id in new_rows[:15]:
                lines.append(f"| {md_text(pass_name)} | {fmt_seconds(cur_d)} | {code_text(test_id)} |")
            lines.append("")

    comment_md = "\n".join(lines).rstrip() + "\n"
    report_md = comment_md
    return comment_md, report_md


def merge_command(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    passes = list(args.passes)
    junits = list(args.junits)
    started_ats = list(args.started_ats)
    finished_ats = list(args.finished_ats)
    exit_codes = list(args.exit_codes)

    n = len(passes)
    if not (len(junits) == len(started_ats) == len(finished_ats) == len(exit_codes) == n):
        raise ValueError("pass metadata arrays must have equal length")

    merged: dict[str, object] = {
        "schema_version": 1,
        "generated_at": now_utc_iso(),
        "source_sha": args.source_sha,
        "source_branch": args.source_branch,
        "workflow_run_id": int(args.workflow_run_id),
        "passes": {},
    }

    passes_out: dict[str, object] = {}
    for pass_name, junit_raw, started_raw, finished_raw, exit_raw in zip(
        passes, junits, started_ats, finished_ats, exit_codes
    ):
        junit_path = pathlib.Path(junit_raw)
        started = safe_int(started_raw, 0)
        finished = safe_int(finished_raw, 0)
        exit_code = safe_int(exit_raw, 1)
        wall_s = float(finished - started) if started and finished and finished >= started else None

        pass_obj: dict[str, object] = {
            "profile": "",
            "started_at_epoch": started or None,
            "finished_at_epoch": finished or None,
            "wall_s": wall_s,
            "exit_code": exit_code,
            "test_count": 0,
            "skipped": 0,
            "failed": 0,
            "missing_junit": False,
            "tests": [],
        }

        tests: list[TestCase] = []
        if junit_path.exists():
            try:
                tests = parse_junit(junit_path)
            except Exception as e:
                pass_obj["missing_junit"] = True
                pass_obj["junit_error"] = str(e)
        else:
            pass_obj["missing_junit"] = True

        pass_obj["profile"] = "ci-non-zk" if pass_name == "non-zk" else "ci-all-features"
        pass_obj["test_count"] = len(tests)
        pass_obj["skipped"] = sum(1 for t in tests if t.status == "skipped")
        pass_obj["failed"] = sum(1 for t in tests if t.status == "failed")
        pass_obj["tests"] = [
            {
                "id": t.id,
                "package": "",
                "crate": "",
                "binary": t.binary,
                "test": t.test,
                "classname": t.classname,
                "duration_s": t.duration_s,
                "status": t.status,
            }
            for t in tests
        ]

        passes_out[pass_name] = pass_obj

    merged["passes"] = passes_out
    write_json(output_dir / "summary.json", merged)
    return 0


def failure_summary_command(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = now_utc_iso()

    merged: dict[str, object] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source_sha": args.source_sha,
        "source_branch": args.source_branch,
        "workflow_run_id": int(args.workflow_run_id),
        "error": args.error,
        "passes": {},
    }
    passes_out: dict[str, object] = {}
    for pass_name in args.passes:
        passes_out[str(pass_name)] = {
            "profile": "ci-non-zk" if pass_name == "non-zk" else "ci-all-features",
            "started_at_epoch": None,
            "finished_at_epoch": None,
            "wall_s": None,
            "exit_code": 1,
            "test_count": 0,
            "skipped": 0,
            "failed": 0,
            "missing_junit": True,
            "tests": [],
        }
    merged["passes"] = passes_out
    write_json(output_dir / "summary.json", merged)

    comment_md, report_md = render_report(merged, None, None, compact=False)
    write_text(output_dir / "comment.md", comment_md)
    write_text(output_dir / "report.md", report_md)
    return 0


def render_command(args: argparse.Namespace) -> int:
    current = load_summary(pathlib.Path(args.summary))
    main = load_optional_summary(args.main_baseline_dir)
    previous = load_optional_summary(args.previous_baseline_dir)

    comment_md, report_md = render_report(current, main, previous, compact=bool(args.compact))
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "comment.md", comment_md)
    write_text(output_dir / "report.md", report_md)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "merge":
        return merge_command(args)
    if args.command == "render":
        return render_command(args)
    if args.command == "failure-summary":
        return failure_summary_command(args)
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())

