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
DEFAULT_PASS = "ci"
BASELINE_PASS_ALIASES = {"ci": ("non-zk", "ci"), "non-zk": ("non-zk",)}


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
    merge.add_argument("--profile", dest="profiles", action="append", default=[])
    merge.add_argument("--junit", dest="junits", action="append", default=[])
    merge.add_argument("--timing", dest="timings", action="append", default=[])
    merge.add_argument("--started-at", dest="started_ats", action="append", default=[])
    merge.add_argument("--finished-at", dest="finished_ats", action="append", default=[])
    merge.add_argument("--exit-code", dest="exit_codes", action="append", default=[])
    merge.add_argument(
        "--passes-sharded",
        action="store_true",
        help="The run was split across parallel nextest slice shards.",
    )
    merge.add_argument(
        "--shard-count",
        type=int,
        default=0,
        help="Number of nextest shards (for summary metadata).",
    )

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
    failure.add_argument("--passes", nargs="+", default=[DEFAULT_PASS])

    combine = subparsers.add_parser(
        "combine-junit", help="Merge multiple nextest JUnit XML files into one."
    )
    combine.add_argument("--output", required=True)
    combine.add_argument("--junit", dest="junits", action="append", default=[])

    prepare = subparsers.add_parser(
        "prepare-shards",
        help="Combine sharded JUnit/timing files for one CI run.",
    )
    prepare.add_argument("--input-dir", required=True)
    prepare.add_argument("--junit-glob", default="junit-shard-*.xml")
    prepare.add_argument("--timing-glob", default="timing-shard-*.json")
    prepare.add_argument("--output-junit", required=True)
    prepare.add_argument("--output-timing", required=True)
    prepare.add_argument(
        "--expected-shard-count",
        type=int,
        default=0,
        help="Require this many shard JUnit and timing files (0 disables).",
    )

    prepare_legacy = subparsers.add_parser(
        "prepare-pass",
        help="Deprecated alias for prepare-shards.",
    )
    prepare_legacy.add_argument("--input-dir", required=True)
    prepare_legacy.add_argument("--junit-glob", default="junit-shard-*.xml")
    prepare_legacy.add_argument("--timing-glob", default="timing-shard-*.json")
    prepare_legacy.add_argument("--output-junit", required=True)
    prepare_legacy.add_argument("--output-timing", required=True)
    prepare_legacy.add_argument("--expected-shard-count", type=int, default=0)

    read_timing = subparsers.add_parser(
        "read-timing", help="Print started_at finished_at exit_code from timing JSON."
    )
    read_timing.add_argument("timing", help="Path to timing JSON")

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


def summary_schema_version(summary: dict[str, object]) -> int:
    return safe_int(str(summary.get("schema_version", 1)), 1)


def summary_pass_keys(summary: dict[str, object] | None) -> list[str]:
    if summary is None:
        return []
    passes = summary.get("passes")
    if not isinstance(passes, dict):
        return []
    order = summary.get("pass_order")
    if isinstance(order, list) and order:
        keys = [str(k) for k in order if str(k) in passes]
        if keys:
            return keys
    return sorted(str(k) for k in passes.keys())


def baseline_pass_for(current_pass: str, baseline: dict[str, object] | None) -> str | None:
    if baseline is None:
        return None
    baseline_keys = summary_pass_keys(baseline)
    for candidate in BASELINE_PASS_ALIASES.get(current_pass, (current_pass,)):
        if candidate in baseline_keys:
            return candidate
    return None


def baseline_layout_mismatch(current: dict[str, object], main: dict[str, object] | None) -> bool:
    if main is None:
        return False
    cur_keys = set(summary_pass_keys(current))
    main_keys = set(summary_pass_keys(main))
    if cur_keys == main_keys:
        return False
    if cur_keys == {DEFAULT_PASS} and "non-zk" in main_keys:
        return True
    return False


def pass_status(pass_obj: dict[str, object]) -> str:
    if pass_obj.get("missing_junit") and safe_int(str(pass_obj.get("test_count", "0")), 0) == 0:
        return "n/a"
    return "ok" if safe_int(str(pass_obj.get("exit_code", "1")), 1) == 0 else "fail"


def timing_fields_from_path(timing_path: pathlib.Path) -> tuple[int, int, int]:
    if not timing_path.exists():
        return 0, 0, 1
    data = json.loads(timing_path.read_text(encoding="utf-8"))
    started = safe_int(str(data.get("started_at_epoch") or 0), 0)
    finished = safe_int(str(data.get("finished_at_epoch") or 0), 0)
    exit_code = data.get("exit_code")
    if exit_code is None:
        exit_code = 1
    return started, finished, safe_int(str(exit_code), 1)


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


def junit_testsuites(path: pathlib.Path) -> list[ET.Element]:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag == "testsuites":
        return list(root.findall("testsuite"))
    if root.tag == "testsuite":
        return [root]
    return list(root.findall(".//testsuite"))


def combine_junit_files(junit_paths: list[pathlib.Path], output_path: pathlib.Path) -> None:
    existing = [path for path in junit_paths if path.exists()]
    if not existing:
        raise FileNotFoundError("no JUnit inputs exist")

    combined = ET.Element("testsuites")
    for junit_path in existing:
        for suite in junit_testsuites(junit_path):
            combined.append(suite)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(combined).write(output_path, encoding="utf-8", xml_declaration=True)


def shard_totals_from_timing(timing_paths: list[pathlib.Path]) -> tuple[int, set[int]]:
    totals: set[int] = set()
    indices: set[int] = set()
    for path in timing_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        totals.add(safe_int(str(data.get("shard_total", 0)), 0))
        indices.add(safe_int(str(data.get("shard_index", 0)), 0))
    totals.discard(0)
    if len(totals) != 1:
        return 0, indices
    return next(iter(totals)), indices


def aggregate_timing_files(timing_paths: list[pathlib.Path]) -> tuple[int, int, int]:
    starts: list[int] = []
    ends: list[int] = []
    exit_codes: list[int] = []
    for path in timing_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        starts.append(safe_int(str(data.get("started_at_epoch", 0)), 0))
        ends.append(safe_int(str(data.get("finished_at_epoch", 0)), 0))
        exit_codes.append(safe_int(str(data.get("exit_code", 1)), 1))

    if not starts:
        return 0, 0, 1
    return min(starts), max(ends), (1 if any(code != 0 for code in exit_codes) else 0)


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
    if isinstance(value, dict):
        return value
    for alias in BASELINE_PASS_ALIASES.get(pass_name, ()):
        alt = passes.get(alias)
        if isinstance(alt, dict):
            return alt
    return None


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

    pass_order = summary_pass_keys(current)
    if not pass_order:
        pass_order = [DEFAULT_PASS]
    multi_pass = len(pass_order) > 1
    layout_mismatch = baseline_layout_mismatch(current, main)

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

    if layout_mismatch:
        lines.append(
            "_Baseline layout mismatch: current run uses single-pass schema; main baseline "
            "predates the cutover. Wall and per-test comparisons use the `non-zk` baseline pass only._"
        )
        lines.append("")

    summary_title = "### Pass summary" if multi_pass else "### Run summary"
    lines.append(summary_title)
    lines.append("")
    if multi_pass:
        lines.append(
            "| Pass | Wall s | Main wall s | Main Δ | Ratio | Tests | Skipped | Failed | Status |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    else:
        lines.append("| Wall s | Main wall s | Main Δ | Ratio | Tests | Skipped | Failed | Status |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    for pass_name in pass_order:
        cur_pass = normalize_pass(current, pass_name) or {}
        baseline_name = baseline_pass_for(pass_name, main) if main is not None else None
        main_pass = normalize_pass(main, baseline_name) if baseline_name else None

        cur_wall = cur_pass.get("wall_s")
        cur_wall_f = float(cur_wall) if cur_wall is not None else None
        main_wall_f = float(main_pass["wall_s"]) if main_pass and main_pass.get("wall_s") is not None else None
        delta_pct = percent_delta(cur_wall_f, main_wall_f)
        ratio = (cur_wall_f / main_wall_f) if (cur_wall_f is not None and main_wall_f not in (None, 0.0)) else None

        status = pass_status(cur_pass)
        tests = safe_int(str(cur_pass.get("test_count", "0")), 0)
        skipped = safe_int(str(cur_pass.get("skipped", "0")), 0)
        failed = safe_int(str(cur_pass.get("failed", "0")), 0)
        ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"

        row = [
            fmt_seconds(cur_wall_f),
            fmt_seconds(main_wall_f),
            fmt_pct(delta_pct),
            ratio_str,
            str(tests),
            str(skipped),
            str(failed),
            status,
        ]
        if multi_pass:
            row = [md_text(pass_name)] + row
        lines.append("| " + " | ".join(row) + " |")

    shard_count = current.get("shard_count")
    if current.get("passes_sharded") and shard_count:
        lines.append("")
        lines.append(f"_Wall time spans {shard_count} parallel nextest slice shards._")
    lines.append("")

    def render_slowest(pass_name: str) -> None:
        cur_pass = normalize_pass(current, pass_name) or {}
        tests_raw = cur_pass.get("tests")
        if not isinstance(tests_raw, list) or not tests_raw:
            title = f"### Slowest tests ({md_text(pass_name)})" if multi_pass else "### Slowest tests"
            lines.append(title)
            lines.append("")
            lines.append("_No JUnit data available for this run._")
            lines.append("")
            return
        rows: list[dict[str, object]] = [t for t in tests_raw if isinstance(t, dict)]
        rows.sort(key=lambda r: float(r.get("duration_s", 0.0) or 0.0), reverse=True)
        top_n = 10 if compact else 20
        title = f"### Slowest tests ({md_text(pass_name)})" if multi_pass else "### Slowest tests"
        lines.append(title)
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
        regression_rows: list[tuple[str, float, float, float, str]] = []
        for pass_name in pass_order:
            cur_pass = normalize_pass(current, pass_name) or {}
            baseline_name = baseline_pass_for(pass_name, main)
            main_pass = normalize_pass(main, baseline_name) if baseline_name else {}
            cur_map = cases_by_key(cur_pass)
            main_map = cases_by_key(main_pass)
            label = baseline_name or pass_name
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
                    regression_rows.append((label, delta, base_d, cur_d, str(cur_row.get("id", ""))))
        regression_rows.sort(key=lambda r: r[1], reverse=True)
        if not regression_rows:
            lines.append("_No per-test regressions above the threshold._")
            lines.append("")
        else:
            if multi_pass:
                lines.append("| Pass | Δ s | Baseline s | Current s | Test |")
                lines.append("| --- | ---: | ---: | ---: | --- |")
            else:
                lines.append("| Δ s | Baseline s | Current s | Test |")
                lines.append("| ---: | ---: | ---: | --- |")
            for label, delta, base_d, cur_d, test_id in regression_rows[:15]:
                cells = [fmt_seconds(delta), fmt_seconds(base_d), fmt_seconds(cur_d), code_text(test_id)]
                if multi_pass:
                    cells = [md_text(label)] + cells
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

        lines.append("### New slow tests")
        lines.append("")
        new_rows: list[tuple[str, float, str]] = []
        for pass_name in pass_order:
            cur_pass = normalize_pass(current, pass_name) or {}
            baseline_name = baseline_pass_for(pass_name, main)
            main_pass = normalize_pass(main, baseline_name) if baseline_name else {}
            cur_map = cases_by_key(cur_pass)
            main_map = cases_by_key(main_pass)
            label = baseline_name or pass_name
            for key, cur_row in cur_map.items():
                if key in main_map:
                    continue
                cur_d = safe_float(str(cur_row.get("duration_s", "0")), 0.0)
                if cur_d >= 30.0:
                    new_rows.append((label, cur_d, str(cur_row.get("id", ""))))
        new_rows.sort(key=lambda r: r[1], reverse=True)
        if not new_rows:
            lines.append("_No new tests ≥30s vs main baseline._")
            lines.append("")
        else:
            if multi_pass:
                lines.append("| Pass | Duration s | Test |")
                lines.append("| --- | ---: | --- |")
            else:
                lines.append("| Duration s | Test |")
                lines.append("| ---: | --- |")
            for label, cur_d, test_id in new_rows[:15]:
                if multi_pass:
                    lines.append(f"| {md_text(label)} | {fmt_seconds(cur_d)} | {code_text(test_id)} |")
                else:
                    lines.append(f"| {fmt_seconds(cur_d)} | {code_text(test_id)} |")
            lines.append("")

    comment_md = "\n".join(lines).rstrip() + "\n"
    report_md = comment_md
    return comment_md, report_md


def merge_pass_records(args: argparse.Namespace) -> list[tuple[str, str, pathlib.Path, int, int, int]]:
    passes = list(args.passes)
    profiles = list(args.profiles)
    junits = list(args.junits)
    timings = list(args.timings)
    started_ats = list(args.started_ats)
    finished_ats = list(args.finished_ats)
    exit_codes = list(args.exit_codes)

    if not passes and junits:
        passes = [DEFAULT_PASS]
        profiles = profiles or [DEFAULT_PASS]

    if timings and not started_ats:
        for timing_raw in timings:
            started, finished, exit_code = timing_fields_from_path(pathlib.Path(timing_raw))
            started_ats.append(str(started))
            finished_ats.append(str(finished))
            exit_codes.append(str(exit_code))

    n = len(passes)
    if not n:
        raise ValueError("at least one --pass or --junit is required")
    if not (len(junits) == len(started_ats) == len(finished_ats) == len(exit_codes) == n):
        raise ValueError("pass metadata arrays must have equal length")
    if profiles and len(profiles) not in (1, n):
        raise ValueError("--profile count must be 1 or match --pass count")

    records: list[tuple[str, str, pathlib.Path, int, int, int]] = []
    for i, (pass_name, junit_raw, started_raw, finished_raw, exit_raw) in enumerate(
        zip(passes, junits, started_ats, finished_ats, exit_codes)
    ):
        profile = profiles[i] if len(profiles) == n else (profiles[0] if profiles else pass_name)
        records.append(
            (
                pass_name,
                profile,
                pathlib.Path(junit_raw),
                safe_int(started_raw, 0),
                safe_int(finished_raw, 0),
                safe_int(exit_raw, 1),
            )
        )
    return records


def merge_command(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = merge_pass_records(args)
    pass_names = [record[0] for record in records]
    schema_version = 2 if len(pass_names) == 1 else 1

    merged: dict[str, object] = {
        "schema_version": schema_version,
        "generated_at": now_utc_iso(),
        "source_sha": args.source_sha,
        "source_branch": args.source_branch,
        "workflow_run_id": int(args.workflow_run_id),
        "pass_layout": "single" if len(pass_names) == 1 else "dual",
        "pass_order": pass_names,
        "passes_sharded": bool(args.passes_sharded),
        "shard_count": int(args.shard_count) if int(args.shard_count) > 0 else None,
        "passes": {},
    }

    passes_out: dict[str, object] = {}
    for pass_name, profile, junit_path, started, finished, exit_code in records:
        wall_s = float(finished - started) if started and finished and finished >= started else None

        pass_obj: dict[str, object] = {
            "profile": profile,
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
    pass_names = list(args.passes) or [DEFAULT_PASS]

    merged: dict[str, object] = {
        "schema_version": 2 if len(pass_names) == 1 else 1,
        "generated_at": generated_at,
        "source_sha": args.source_sha,
        "source_branch": args.source_branch,
        "workflow_run_id": int(args.workflow_run_id),
        "error": args.error,
        "pass_layout": "single" if len(pass_names) == 1 else "dual",
        "pass_order": pass_names,
        "passes": {},
    }
    passes_out: dict[str, object] = {}
    for pass_name in pass_names:
        passes_out[str(pass_name)] = {
            "profile": pass_name,
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


def combine_junit_command(args: argparse.Namespace) -> int:
    junit_paths = [pathlib.Path(raw) for raw in args.junits]
    combine_junit_files(junit_paths, pathlib.Path(args.output))
    return 0


def read_timing_command(args: argparse.Namespace) -> int:
    started, finished, exit_code = timing_fields_from_path(pathlib.Path(args.timing))
    print(f"{started} {finished} {exit_code}")
    return 0


def prepare_shards_command(args: argparse.Namespace) -> int:
    input_dir = pathlib.Path(args.input_dir)
    junit_paths = sorted(input_dir.rglob(args.junit_glob))
    timing_paths = sorted(input_dir.rglob(args.timing_glob))
    output_junit = pathlib.Path(args.output_junit)
    output_timing = pathlib.Path(args.output_timing)
    expected = int(args.expected_shard_count)
    shard_total, shard_indices = shard_totals_from_timing(timing_paths)
    if expected <= 0:
        expected = shard_total

    missing_shards = expected <= 0 or len(junit_paths) != expected or len(timing_paths) != expected
    if not missing_shards:
        missing_shards = shard_indices != set(range(1, expected + 1))
    if missing_shards:
        write_json(
            output_timing,
            {
                "started_at_epoch": None,
                "finished_at_epoch": None,
                "exit_code": 1,
                "missing_shards": True,
                "expected_shard_count": expected or 0,
                "shard_total": shard_total or 0,
                "junit_shard_count": len(junit_paths),
                "timing_shard_count": len(timing_paths),
            },
        )
        return 1

    combine_junit_files(junit_paths, output_junit)
    started_at, finished_at, exit_code = aggregate_timing_files(timing_paths)
    write_json(
        output_timing,
        {
            "started_at_epoch": started_at or None,
            "finished_at_epoch": finished_at or None,
            "exit_code": exit_code,
            "shard_total": expected,
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "merge":
        return merge_command(args)
    if args.command == "render":
        return render_command(args)
    if args.command == "failure-summary":
        return failure_summary_command(args)
    if args.command == "combine-junit":
        return combine_junit_command(args)
    if args.command in ("prepare-shards", "prepare-pass"):
        return prepare_shards_command(args)
    if args.command == "read-timing":
        return read_timing_command(args)
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
