#!/usr/bin/env python3
"""Collect the exact requested matrix, retaining explicit missing-shard failures."""
import argparse
import json
import sys
from pathlib import Path

if __package__:
    from .profile_bench_merge_base_policy import read_matrix_cases
    from .profile_bench_report import (
        SAMPLE_METRICS,
        failure_case,
        normalize_case_summary,
        parse_case_spec,
        write_text,
    )
else:
    from profile_bench_merge_base_policy import read_matrix_cases
    from profile_bench_report import (
        SAMPLE_METRICS,
        failure_case,
        normalize_case_summary,
        parse_case_spec,
        write_text,
    )


def read_shard_summary(path: Path, expected_ids: set[str]) -> tuple[int, dict[str, dict]]:
    """Validate the shared head/baseline envelope before consuming its records."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("shard summary must be an object with a cases list")
    warmups = payload.get("warmups", 0)
    if type(warmups) is not int or warmups < 0:
        raise ValueError("invalid shard warmup count")
    if "schema_version" in payload and (
        type(payload["schema_version"]) is not int or payload["schema_version"] < 0
    ):
        raise ValueError("invalid shard schema version")
    records = {}
    for raw in payload["cases"]:
        if not isinstance(raw, dict):
            raise ValueError("case result must be an object")
        if not isinstance(raw.get("mode"), str) or not raw["mode"]:
            raise ValueError("case mode must be a nonempty string")
        for key, default in (
            ("num_vars", None), ("num_polys", 1), ("exit_code", None), ("schema_version", 0)
        ):
            value = raw.get(key, default)
            minimum = 0 if key == "schema_version" else 1
            if type(value) is not int or (key != "exit_code" and value < minimum):
                raise ValueError(f"invalid case {key}")
        for key in (
            "setup_contribution_mode", "error", "failure_phase", "workload_label",
            "baseline_status", "baseline_reason", "case_id",
        ):
            if key in raw and not isinstance(raw[key], str):
                raise ValueError(f"invalid case {key}")
        if "baseline_status" in raw and raw["baseline_status"] not in ("available", "unsupported"):
            raise ValueError("invalid baseline status")
        if raw.get("baseline_status") == "unsupported" and not raw.get("baseline_reason"):
            raise ValueError("unsupported baseline requires a reason")
        if "env" in raw and not isinstance(raw["env"], dict):
            raise ValueError("invalid case env")
        for key in ("terminal_plan", "grinding_plan"):
            if raw.get(key) is not None and not isinstance(raw[key], dict):
                raise ValueError(f"invalid case {key}")
        for key in SAMPLE_METRICS:
            value = raw.get(key)
            if value is not None and (
                type(value) not in (int, float)
                or not (-sys.float_info.max <= value <= sys.float_info.max)
            ):
                raise ValueError(f"invalid case {key}")
        for key in ("planned_levels", "proof_levels", "samples"):
            if key in raw and (not isinstance(raw[key], list)
                               or any(not isinstance(row, dict) for row in raw[key])):
                raise ValueError(f"invalid case {key}")
        if "warnings" in raw and (not isinstance(raw["warnings"], list)
                                  or any(not isinstance(w, str) for w in raw["warnings"])):
            raise ValueError("invalid case warnings")
        try:
            case = normalize_case_summary(raw)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"invalid case result: {error}") from error
        case_id = case["case_id"]
        if case_id not in expected_ids:
            raise ValueError("unexpected case result in shard")
        if case_id in records:
            raise ValueError("duplicate case result")
        records[case_id] = case
    return warmups, records


def collect(parts: Path, output: Path, baseline: Path, workflow: Path) -> None:
    expected = read_matrix_cases(workflow)
    head_cases = []
    base_cases = []
    warmups = 0
    for group in dict.fromkeys(group for group, _, _ in expected):
        requested = [parse_case_spec(spec, "onehot_fp128")
                     for owner, _, spec in expected if owner == group]
        expected_ids = {case.case_id for case in requested}
        directory = parts / f"profile-bench-data-{group}"
        path = directory / "pr/summary.json"
        error = "benchmark shard produced no case summary"
        try:
            group_warmups, records = read_shard_summary(path, expected_ids)
            warmups = max(warmups, group_warmups)
            for case in records.values():
                case["benchmark_shard"] = group
        except (OSError, ValueError) as failure:
            error = f"invalid or missing shard summary: {failure}"
            records = {}
        for case in requested:
            head_cases.append(records.get(case.case_id) or failure_case(
                case, "benchmark orchestration", error, group
            ))
        base_path = directory / "base/summary.json"
        paired = [case for case in head_cases if case.get("benchmark_shard") == group
                  and case.get("baseline_status") == "available"]
        if base_path.exists() or paired:
            try:
                _, base_records = read_shard_summary(base_path, expected_ids)
                if {case["case_id"] for case in paired} - base_records.keys():
                    raise ValueError("missing supported baseline case result")
                for failed in base_records.values():
                    if failed["exit_code"] != 0:
                        for case in head_cases:
                            if case["case_id"] == failed["case_id"]:
                                case.update(exit_code=failed["exit_code"],
                                            failure_phase="baseline aggregation",
                                            error=failed.get("error", "baseline case failed"))
                base_cases.extend(base_records.values())
            except (OSError, ValueError) as failure:
                affected = paired or [case for case in head_cases
                                      if case.get("benchmark_shard") == group]
                for case in affected:
                    case.update(exit_code=1, failure_phase="baseline aggregation", error=str(failure))
    if len({case["case_id"] for case in head_cases}) != len(expected):
        raise ValueError("matrix case identities are not unique")
    write_text(output / "summary.json", json.dumps(
        {"schema_version": 3, "warmups": warmups, "cases": head_cases}, indent=2
    ) + "\n")
    if base_cases:
        write_text(baseline / "summary.json", json.dumps(
            {"schema_version": 3, "warmups": warmups, "cases": base_cases}, indent=2
        ) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-output-dir", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, default=Path(".github/workflows/profile-bench.yml"))
    args = parser.parse_args()
    collect(args.parts_dir, args.output_dir, args.baseline_output_dir, args.workflow)
