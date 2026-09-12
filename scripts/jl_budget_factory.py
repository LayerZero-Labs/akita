#!/usr/bin/env python3
"""Audit and search exact JL failure allocations over a finite endpoint frontier.

This is a planning tool.  It does not certify endpoint geometry; provenance and
trust status come from the input file and are preserved in every report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from fractions import Fraction
from typing import Any


class PlanError(ValueError):
    """Raised when a planning input violates the auditable schema."""


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise PlanError(f"{where} must be an integer or an integer string")
    try:
        parsed = int(value)
    except ValueError as error:
        raise PlanError(f"{where} must be an integer or an integer string") from error
    if isinstance(value, str) and str(parsed) != value:
        raise PlanError(f"{where} must use canonical integer syntax")
    return parsed


def _positive_integer(value: Any, where: str) -> int:
    parsed = _integer(value, where)
    if parsed <= 0:
        raise PlanError(f"{where} must be positive")
    return parsed


def _rational(value: Any, where: str) -> Fraction:
    if not isinstance(value, dict):
        raise PlanError(f"{where} must be a rational object")
    if set(value) != {"numerator", "denominator"}:
        raise PlanError(f"{where} must contain exactly numerator and denominator")
    numerator = _integer(value["numerator"], f"{where}.numerator")
    denominator = _positive_integer(value["denominator"], f"{where}.denominator")
    return Fraction(numerator, denominator)


def _failure_cap(value: Any, where: str) -> tuple[str, Fraction]:
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        raise PlanError(f"{where} must have a non-empty name")
    name = value["name"]
    if not name:
        raise PlanError(f"{where}.name must be non-empty")
    has_dyadic = "dyadicExponent" in value
    has_rational = "numerator" in value or "denominator" in value
    if has_dyadic == has_rational:
        raise PlanError(
            f"{where} must specify exactly one of dyadicExponent or numerator/denominator"
        )
    if has_dyadic:
        if set(value) != {"name", "dyadicExponent"}:
            raise PlanError(f"{where} has unknown fields")
        exponent = _integer(value["dyadicExponent"], f"{where}.dyadicExponent")
        if exponent < 0:
            raise PlanError(f"{where}.dyadicExponent must be non-negative")
        return name, Fraction(1, 1 << exponent)
    if "numerator" not in value or "denominator" not in value:
        raise PlanError(f"{where} rational cap needs numerator and denominator")
    if set(value) != {"name", "numerator", "denominator"}:
        raise PlanError(f"{where} has unknown fields")
    cap = _rational(
        {"numerator": value["numerator"], "denominator": value["denominator"]},
        where,
    )
    if cap <= 0:
        raise PlanError(f"{where} must be positive")
    if cap > 1:
        raise PlanError(f"{where} cannot exceed one")
    return name, cap


def _fraction_json(value: Fraction) -> dict[str, str]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
    }


def _require_string(record: dict[str, Any], field: str, where: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise PlanError(f"{where}.{field} must be a non-empty string")
    return value


def _load_endpoints(plan: dict[str, Any], tail: str) -> dict[str, dict[str, Any]]:
    frontier = plan.get("frontier")
    if not isinstance(frontier, dict):
        raise PlanError("frontier must be an object")
    records = frontier.get(tail)
    if not isinstance(records, list) or not records:
        raise PlanError(f"frontier.{tail} must be a non-empty list")
    endpoints: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        where = f"frontier.{tail}[{index}]"
        if not isinstance(record, dict):
            raise PlanError(f"{where} must be an object")
        endpoint_id = _require_string(record, "id", where)
        if endpoint_id in endpoints:
            raise PlanError(f"duplicate {tail} endpoint id {endpoint_id!r}")
        status = _require_string(record, "status", where)
        if status not in {"certified", "provisional"}:
            raise PlanError(f"{where}.status must be certified or provisional")
        threshold = _rational(record.get("threshold"), f"{where}.threshold")
        if threshold <= 0:
            raise PlanError(f"{where}.threshold must be positive")
        cap_name, cap = _failure_cap(record.get("failureCap"), f"{where}.failureCap")
        hypotheses = record.get("modulusHypotheses")
        if (
            not isinstance(hypotheses, list)
            or not hypotheses
            or not all(isinstance(item, str) and item for item in hypotheses)
        ):
            raise PlanError(f"{where}.modulusHypotheses must be non-empty strings")
        endpoints[endpoint_id] = {
            "id": endpoint_id,
            "tail": tail,
            "status": status,
            "rowLaw": _require_string(record, "rowLaw", where),
            "rows": _positive_integer(record.get("rows"), f"{where}.rows"),
            "threshold": threshold,
            "capName": cap_name,
            "cap": cap,
            "theorem": _require_string(record, "theorem", where),
            "sourceRevision": _require_string(record, "sourceRevision", where),
            "modulusHypotheses": tuple(hypotheses),
        }
    return endpoints


def _load_schedule(
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    folds = plan.get("folds")
    if not isinstance(folds, list) or not folds:
        raise PlanError("folds must be a non-empty list")
    uses: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    envelope_locations: dict[str, tuple[int, int]] = {}
    levels: set[int] = set()
    candidates: dict[int, int] = {}
    for fold_index, fold in enumerate(folds):
        where = f"folds[{fold_index}]"
        if not isinstance(fold, dict):
            raise PlanError(f"{where} must be an object")
        level = _integer(fold.get("level"), f"{where}.level")
        if level in levels:
            raise PlanError(f"duplicate fold level {level}")
        levels.add(level)
        candidates[level] = _positive_integer(fold.get("candidates"), f"{where}.candidates")
        families = fold.get("useFamilies")
        if not isinstance(families, list) or not families:
            raise PlanError(f"{where}.useFamilies must be a non-empty list")
        by_role: dict[str, list[int]] = {}
        extensions: list[tuple[list[str], int]] = []
        for family_index, family in enumerate(families):
            family_where = f"{where}.useFamilies[{family_index}]"
            if not isinstance(family, dict):
                raise PlanError(f"{family_where} must be an object")
            role = _require_string(family, "role", family_where)
            blocks = family.get("blocks")
            depths = family.get("depths")
            envelopes = family.get("matrixEnvelopes")
            if not all(
                isinstance(value, list) and value for value in (blocks, depths, envelopes)
            ):
                raise PlanError(f"{family_where} block/depth/envelope lists must be non-empty")
            if not (len(blocks) == len(depths) == len(envelopes)):
                raise PlanError(f"{family_where} block/depth/envelope list lengths differ")
            extends = family.get("extendsPaths", [])
            if not isinstance(extends, list) or not all(
                isinstance(item, str) and item for item in extends
            ):
                raise PlanError(f"{family_where}.extendsPaths must be a string list")
            for item_index, (block_value, depth_value, envelope) in enumerate(
                zip(blocks, depths, envelopes, strict=True)
            ):
                item_where = f"{family_where}[{item_index}]"
                block_count = _positive_integer(block_value, f"{item_where}.blocks")
                depth = _integer(depth_value, f"{item_where}.depth")
                if depth < 0:
                    raise PlanError(f"{item_where}.depth must be non-negative")
                if not isinstance(envelope, str) or not envelope:
                    raise PlanError(f"{item_where}.matrixEnvelope must be non-empty")
                location = (level, depth)
                old_location = envelope_locations.setdefault(envelope, location)
                if old_location != location:
                    raise PlanError(
                        f"matrix envelope {envelope!r} is reused across fold/depth locations"
                    )
                use_id = f"{level}:{role}:{depth}"
                if any(use["id"] == use_id for use in uses):
                    raise PlanError(f"duplicate projection use {use_id!r}")
                use_index = len(uses)
                uses.append(
                    {
                        "id": use_id,
                        "level": level,
                        "role": role,
                        "depth": depth,
                        "blocks": block_count,
                        "envelope": envelope,
                    }
                )
                if extends:
                    extensions.append((extends, use_index))
                else:
                    by_role.setdefault(role, []).append(use_index)
        for role, indices in by_role.items():
            depths = [uses[index]["depth"] for index in indices]
            if len(depths) != len(set(depths)) or depths != sorted(depths):
                raise PlanError(
                    f"fold {level} path {role!r} is not declared in strict depth order"
                )
        for roles, use_index in extensions:
            for role in roles:
                if role not in by_role:
                    raise PlanError(f"fold {level} extends missing path role {role!r}")
                parent_depth = max(uses[index]["depth"] for index in by_role[role])
                if uses[use_index]["depth"] <= parent_depth:
                    raise PlanError(
                        f"fold {level} extension for path {role!r} must be later than "
                        f"depth {parent_depth}"
                    )
                by_role[role].append(use_index)
        for role, indices in sorted(by_role.items()):
            depths = [uses[index]["depth"] for index in indices]
            if len(depths) != len(set(depths)) or depths != sorted(depths):
                raise PlanError(f"fold {level} path {role!r} is not strictly fresh by depth")
            path_envelopes = [uses[index]["envelope"] for index in indices]
            if len(path_envelopes) != len(set(path_envelopes)):
                raise PlanError(f"fold {level} path {role!r} reuses a matrix envelope")
            paths.append({"id": f"{level}:{role}", "uses": tuple(indices)})

    census = {
        "chargedBlocks": sum(use["blocks"] for use in uses),
        "rawMatrixEnvelopes": len(uses),
        "distinctMatrixEnvelopes": len(envelope_locations),
    }
    expected = plan.get("expectedCensus")
    if not isinstance(expected, dict):
        raise PlanError("expectedCensus must be an object")
    for field, actual in census.items():
        wanted = _positive_integer(expected.get(field), f"expectedCensus.{field}")
        if wanted != actual:
            raise PlanError(f"census mismatch for {field}: expected {wanted}, computed {actual}")
    return uses, paths, {str(level): count for level, count in candidates.items()}


def load_plan(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot load {path}: {error}") from error
    if not isinstance(raw, dict):
        raise PlanError("plan root must be an object")
    if raw.get("schema") != "akita-jl-budget-planning-v1":
        raise PlanError("unsupported or missing schema")
    planning_status = _require_string(raw, "planningStatus", "plan")
    if planning_status not in {"certified", "untrusted-exploration"}:
        raise PlanError("planningStatus must be certified or untrusted-exploration")
    lower = _load_endpoints(raw, "lower")
    upper = _load_endpoints(raw, "upper")
    cap_names = [entry["capName"] for entry in (*lower.values(), *upper.values())]
    if len(cap_names) != len(set(cap_names)):
        raise PlanError("every lower and upper failure cap must have an independent name")
    uses, paths, candidates = _load_schedule(raw)
    budget = _rational(raw.get("failureBudget"), "failureBudget")
    if budget <= 0:
        raise PlanError("failureBudget must be positive")
    return {
        "raw": raw,
        "planningStatus": planning_status,
        "lower": lower,
        "upper": upper,
        "uses": uses,
        "paths": paths,
        "candidates": candidates,
        "budget": budget,
    }


def _pair(
    plan: dict[str, Any], selection: Any, where: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(selection, dict) or set(selection) != {"lower", "upper"}:
        raise PlanError(f"{where} must contain exactly lower and upper endpoint ids")
    lower = plan["lower"].get(selection["lower"])
    upper = plan["upper"].get(selection["upper"])
    if lower is None or upper is None:
        raise PlanError(f"{where} references an unknown endpoint")
    for field in ("rowLaw", "rows"):
        if lower[field] != upper[field]:
            raise PlanError(f"{where} lower/upper endpoints disagree on {field}")
    if lower["threshold"] >= upper["threshold"]:
        raise PlanError(f"{where} must have lower threshold strictly below upper threshold")
    return lower, upper


def audit_scenario(
    plan: dict[str, Any], scenario: dict[str, Any], budget: Fraction | None = None
) -> dict[str, Any]:
    name = _require_string(scenario, "name", "scenario")
    scope = scenario.get("scope")
    if scope not in {"level", "use"}:
        raise PlanError(f"scenario {name!r} scope must be level or use")
    selections = scenario.get("selections")
    if not isinstance(selections, dict):
        raise PlanError(f"scenario {name!r} selections must be an object")
    expected_keys = (
        set(plan["candidates"])
        if scope == "level"
        else {use["id"] for use in plan["uses"]}
    )
    if set(selections) != expected_keys:
        missing = sorted(expected_keys - set(selections))
        extra = sorted(set(selections) - expected_keys)
        raise PlanError(
            f"scenario {name!r} selection keys differ; missing={missing}, extra={extra}"
        )
    chosen: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        key: _pair(plan, selections[key], f"scenario {name!r} selection {key!r}")
        for key in sorted(selections)
    }
    total = Fraction(0)
    level_costs: dict[str, Fraction] = {level: Fraction(0) for level in plan["candidates"]}
    endpoint_statuses: set[str] = set()
    use_ratios: list[Fraction] = []
    for use in plan["uses"]:
        key = str(use["level"]) if scope == "level" else use["id"]
        lower, upper = chosen[key]
        endpoint_statuses.update((lower["status"], upper["status"]))
        candidate_count = plan["candidates"][str(use["level"])]
        charge = candidate_count * use["blocks"] * (lower["cap"] + upper["cap"])
        total += charge
        level_costs[str(use["level"])] += charge
        use_ratios.append(upper["threshold"] / lower["threshold"])
    path_distortions = {
        path["id"]: _product(use_ratios[index] for index in path["uses"])
        for path in plan["paths"]
    }
    worst_distortion = max(path_distortions.values(), default=Fraction(1))
    effective_budget = plan["budget"] if budget is None else budget
    admitted = total <= effective_budget
    return {
        "name": name,
        "scope": scope,
        "budgetSatisfied": admitted,
        "productionAdmissible": False,
        "planningStatus": plan["planningStatus"],
        "selectedEndpointStatuses": sorted(endpoint_statuses),
        "failureCost": _fraction_json(total),
        "failureBudget": _fraction_json(effective_budget),
        "budgetUtilization": _fraction_json(total / effective_budget),
        "budgetMargin": _fraction_json(effective_budget - total),
        "perLevelFailureCost": {
            level: _fraction_json(value) for level, value in sorted(level_costs.items())
        },
        "pathDistortion": {
            path: _fraction_json(value) for path, value in sorted(path_distortions.items())
        },
        "worstPathDistortion": _fraction_json(worst_distortion),
        "selectedModulusHypotheses": {
            key: {
                "lower": list(pair[0]["modulusHypotheses"]),
                "upper": list(pair[1]["modulusHypotheses"]),
            }
            for key, pair in sorted(chosen.items())
        },
        "selectedEndpoints": {
            key: {
                tail: {
                    "id": endpoint["id"],
                    "status": endpoint["status"],
                    "threshold": _fraction_json(endpoint["threshold"]),
                    "failureCapName": endpoint["capName"],
                    "failureCap": _fraction_json(endpoint["cap"]),
                }
                for tail, endpoint in (("lower", pair[0]), ("upper", pair[1]))
            }
            for key, pair in sorted(chosen.items())
        },
        "selections": selections,
    }


def _product(values: Any) -> Fraction:
    result = Fraction(1)
    for value in values:
        result *= value
    return result


def _census_report(plan: dict[str, Any]) -> dict[str, Any]:
    blocks_by_level = {
        level: sum(use["blocks"] for use in plan["uses"] if str(use["level"]) == level)
        for level in plan["candidates"]
    }
    candidate_weighted_blocks = sum(
        plan["candidates"][str(use["level"])] * use["blocks"] for use in plan["uses"]
    )
    return {
        "chargedBlocks": sum(blocks_by_level.values()),
        "rawMatrixEnvelopes": len(plan["uses"]),
        "distinctMatrixEnvelopes": len({use["envelope"] for use in plan["uses"]}),
        "candidateWeightedBlockUses": candidate_weighted_blocks,
        "lowerUpperTailOpportunities": 2 * candidate_weighted_blocks,
        "blockCountsByLevel": blocks_by_level,
        "candidatesByLevel": plan["candidates"],
    }


def audit_report(plan: dict[str, Any]) -> dict[str, Any]:
    scenarios_raw = plan["raw"].get("scenarios", [])
    if not isinstance(scenarios_raw, list):
        raise PlanError("scenarios must be a list")
    scenarios = [audit_scenario(plan, scenario) for scenario in scenarios_raw]
    by_name = {scenario["name"]: scenario for scenario in scenarios}
    if len(by_name) != len(scenarios):
        raise PlanError("scenario names must be unique")
    comparisons_raw = plan["raw"].get("comparisons", [])
    if not isinstance(comparisons_raw, list):
        raise PlanError("comparisons must be a list")
    comparisons = []
    for index, comparison in enumerate(comparisons_raw):
        where = f"comparisons[{index}]"
        if not isinstance(comparison, dict):
            raise PlanError(f"{where} must be an object")
        baseline_name = _require_string(comparison, "baseline", where)
        candidate_name = _require_string(comparison, "candidate", where)
        if baseline_name not in by_name or candidate_name not in by_name:
            raise PlanError(f"{where} references an unknown scenario")
        baseline = by_name[baseline_name]
        candidate = by_name[candidate_name]
        baseline_cost = _rational(baseline["failureCost"], f"{where}.baselineCost")
        candidate_cost = _rational(candidate["failureCost"], f"{where}.candidateCost")
        baseline_distortion = _rational(
            baseline["worstPathDistortion"], f"{where}.baselineDistortion"
        )
        candidate_distortion = _rational(
            candidate["worstPathDistortion"], f"{where}.candidateDistortion"
        )
        path_ratios = {}
        for path, baseline_value in baseline["pathDistortion"].items():
            path_ratios[path] = _fraction_json(
                _rational(candidate["pathDistortion"][path], f"{where}.candidatePath")
                / _rational(baseline_value, f"{where}.baselinePath")
            )
        comparisons.append(
            {
                "baseline": baseline_name,
                "candidate": candidate_name,
                "failureCostRatio": _fraction_json(candidate_cost / baseline_cost),
                "worstPathDistortionRatio": _fraction_json(
                    candidate_distortion / baseline_distortion
                ),
                "pathDistortionRatio": path_ratios,
                "optimalityClaim": False,
            }
        )
    return {
        "schema": "akita-jl-budget-report-v1",
        "planningStatus": plan["planningStatus"],
        "productionWarning": (
            "UNTRUSTED EXPLORATION: endpoint theorem provenance is not production-ready"
            if plan["planningStatus"] != "certified"
            else None
        ),
        "census": _census_report(plan),
        "scenarios": scenarios,
        "comparisons": comparisons,
    }


def _search_pairs(plan: dict[str, Any]) -> list[dict[str, str]]:
    configured = plan["raw"].get("searchPairs")
    if configured is None:
        configured = [
            {"lower": lower, "upper": upper}
            for lower in sorted(plan["lower"])
            for upper in sorted(plan["upper"])
        ]
    if not isinstance(configured, list) or not configured:
        raise PlanError("searchPairs must be a non-empty list")
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, selection in enumerate(configured):
        lower, upper = _pair(plan, selection, f"searchPairs[{index}]")
        key = (lower["id"], upper["id"])
        if key not in seen:
            seen.add(key)
            pairs.append({"lower": key[0], "upper": key[1]})
    return pairs


def search_report(
    plan: dict[str, Any],
    scope: str,
    budget: Fraction,
    max_combinations: int,
    max_results: int,
) -> dict[str, Any]:
    pairs = _search_pairs(plan)
    input_pair_count = len(pairs)
    if scope == "level":
        local_frontier: list[tuple[Fraction, Fraction, dict[str, str]]] = []
        for selection in pairs:
            lower, upper = _pair(plan, selection, "search pair")
            metrics = (
                lower["cap"] + upper["cap"],
                upper["threshold"] / lower["threshold"],
            )
            if any(
                entry[:2] == metrics or _dominates(entry[:2], metrics)
                for entry in local_frontier
            ):
                continue
            local_frontier = [
                entry for entry in local_frontier if not _dominates(metrics, entry[:2])
            ]
            local_frontier.append((*metrics, selection))
        pairs = [entry[2] for entry in sorted(local_frontier, key=lambda entry: entry[:2])]
    units = (
        sorted(plan["candidates"], key=int)
        if scope == "level"
        else [use["id"] for use in plan["uses"]]
    )
    combination_count = len(pairs) ** len(units)
    if combination_count > max_combinations:
        raise PlanError(
            f"finite search has {combination_count} combinations, above --max-combinations "
            f"{max_combinations}; narrow searchPairs or raise the explicit limit"
        )
    frontier: list[tuple[Any, ...]] = []
    admitted_count = 0
    for pair_indices in itertools.product(range(len(pairs)), repeat=len(units)):
        selections = {unit: pairs[pair_index] for unit, pair_index in zip(units, pair_indices)}
        scenario = audit_scenario(
            plan,
            {"name": "search-candidate", "scope": scope, "selections": selections},
            budget,
        )
        if not scenario["budgetSatisfied"]:
            continue
        admitted_count += 1
        metrics = (
            _rational(scenario["failureCost"], "search.failureCost"),
            *(
                _rational(value, f"search.pathDistortion.{path}")
                for path, value in sorted(scenario["pathDistortion"].items())
            ),
        )
        if scope == "level":
            # Each level affects at least one disjoint path.  After local
            # (failure, U/L) pruning, improving every level's path geometry
            # strictly increases at least one positive level failure charge.
            # Thus every budget-feasible product is globally nondominated.
            frontier.append((*metrics, scenario))
            frontier.sort(key=lambda entry: entry[:-1])
            if len(frontier) > max_results:
                frontier.pop()
        else:
            if any(
                existing[:-1] == metrics or _dominates(existing[:-1], metrics)
                for existing in frontier
            ):
                continue
            frontier = [entry for entry in frontier if not _dominates(metrics, entry[:-1])]
            frontier.append((*metrics, scenario))
    frontier.sort(key=lambda entry: entry[:-1])
    rendered = [entry[-1] for entry in frontier[:max_results]]
    for index, scenario in enumerate(rendered):
        scenario["name"] = f"pareto-{index}"
    return {
        "schema": "akita-jl-budget-search-v1",
        "planningStatus": plan["planningStatus"],
        "productionWarning": (
            "UNTRUSTED EXPLORATION: search results are not certified protocol parameters"
            if plan["planningStatus"] != "certified"
            else None
        ),
        "scope": scope,
        "failureBudget": _fraction_json(budget),
        "objective": ["failureCost", "pathDistortion[*]"],
        "optimalityClaimAboutUniformBits": False,
        "inputSearchPairCount": input_pair_count,
        "locallyNondominatedPairCount": len(pairs),
        "combinationCount": combination_count,
        "budgetSatisfiedCombinationCount": admitted_count,
        "paretoFrontierSize": admitted_count if scope == "level" else len(frontier),
        "resultsTruncated": (
            admitted_count > max_results if scope == "level" else len(frontier) > max_results
        ),
        "results": rendered,
    }


def _dominates(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="audit census and named scenarios")
    audit.add_argument("plan", type=pathlib.Path)
    search = subparsers.add_parser("search", help="exact finite Pareto search")
    search.add_argument("plan", type=pathlib.Path)
    search.add_argument("--scope", choices=("level", "use"), default="level")
    search.add_argument("--budget-bits", type=int)
    search.add_argument("--max-combinations", type=int, default=2_000_000)
    search.add_argument("--max-results", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.command == "audit":
            report = audit_report(plan)
        else:
            if args.max_combinations <= 0 or args.max_results <= 0:
                raise PlanError("search limits must be positive")
            if args.budget_bits is None:
                budget = plan["budget"]
            elif args.budget_bits < 0:
                raise PlanError("--budget-bits must be non-negative")
            else:
                budget = Fraction(1, 1 << args.budget_bits)
            report = search_report(
                plan, args.scope, budget, args.max_combinations, args.max_results
            )
    except PlanError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
