#!/usr/bin/env python3
"""Collect the exact requested matrix, retaining explicit missing-shard failures."""
import argparse
import json
from pathlib import Path
if __package__:
    from .profile_bench_merge_base_policy import read_matrix_cases
    from .profile_bench_report import failure_case, normalize_case_summary, parse_case_spec, write_text
else:
    from profile_bench_merge_base_policy import read_matrix_cases
    from profile_bench_report import failure_case, normalize_case_summary, parse_case_spec, write_text


def collect(parts: Path, output: Path, baseline: Path, workflow: Path) -> None:
    expected = read_matrix_cases(workflow)
    head_cases = []
    base_cases = []
    warmups = 0
    for group in dict.fromkeys(group for group, _, _ in expected):
        requested = [parse_case_spec(spec, "onehot_fp128")
                     for owner, _, spec in expected if owner == group]
        directory = parts / f"profile-bench-data-{group}"
        path = directory / "pr/summary.json"
        error = "benchmark shard produced no case summary"
        records = {}
        try:
            payload = json.loads(path.read_text())
            warmups = max(warmups, payload.get("warmups", 0))
            for raw in payload["cases"]:
                case = normalize_case_summary(raw)
                if type(case.get("exit_code")) is not int:
                    raise ValueError("invalid case exit status")
                case["benchmark_shard"] = group
                if case["case_id"] in records:
                    raise ValueError("duplicate case result")
                records[case["case_id"]] = case
            if records.keys() - {case.case_id for case in requested}:
                raise ValueError("unexpected case result in shard")
        except (OSError, ValueError, KeyError, TypeError) as failure:
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
                group_base = [normalize_case_summary(raw) for raw in
                              json.loads(base_path.read_text())["cases"]]
                if any(type(case.get("exit_code")) is not int for case in group_base):
                    raise ValueError("invalid baseline exit status")
                base_ids = [case["case_id"] for case in group_base]
                if len(set(base_ids)) != len(base_ids):
                    raise ValueError("duplicate baseline case result")
                if set(base_ids) - {case.case_id for case in requested}:
                    raise ValueError("unexpected baseline case result")
                if {case["case_id"] for case in paired} - set(base_ids):
                    raise ValueError("missing supported baseline case result")
                for failed in group_base:
                    if failed["exit_code"] != 0:
                        for case in head_cases:
                            if case["case_id"] == failed["case_id"]:
                                case.update(exit_code=failed["exit_code"],
                                            failure_phase="baseline aggregation",
                                            error=failed.get("error", "baseline case failed"))
                base_cases.extend(group_base)
            except (OSError, ValueError, KeyError, TypeError) as failure:
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
