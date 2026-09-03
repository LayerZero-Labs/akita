#!/usr/bin/env python3
"""Resolve profile CI schedule features across workspace crate boundaries."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


MANIFESTS = {
    "akita-pcs": Path("crates/akita-pcs/Cargo.toml"),
    "akita-config": Path("crates/akita-config/Cargo.toml"),
    "akita-schedules": Path("crates/akita-schedules/Cargo.toml"),
}


def load_feature_graph(repo: Path) -> dict[str, dict[str, list[str]]]:
    return {
        crate: {
            name: list(members)
            for name, members in tomllib.loads(
                (repo / manifest).read_text(encoding="utf-8")
            )["features"].items()
        }
        for crate, manifest in MANIFESTS.items()
    }


def resolve_feature(
    graph: dict[str, dict[str, list[str]]],
    crate: str,
    feature: str,
    active: tuple[tuple[str, str], ...] = (),
) -> set[tuple[str, str]]:
    node = (crate, feature)
    if crate not in graph or feature not in graph[crate]:
        raise ValueError(f"feature {crate}/{feature} was not found")
    if node in active:
        cycle = " -> ".join(f"{c}/{f}" for c, f in (*active, node))
        raise ValueError(f"Cargo feature cycle: {cycle}")

    resolved = {node}
    for member in graph[crate][feature]:
        if member.startswith("dep:"):
            continue
        if "/" in member:
            member_crate, member_feature = member.split("/", 1)
            if member_crate not in graph:
                continue
        elif member in graph[crate]:
            member_crate, member_feature = crate, member
        else:
            continue
        resolved.update(
            resolve_feature(graph, member_crate, member_feature, (*active, node))
        )
    return resolved


def schedule_features(
    graph: dict[str, dict[str, list[str]]], crate: str, feature: str
) -> set[str]:
    return {
        resolved_feature
        for resolved_crate, resolved_feature in resolve_feature(graph, crate, feature)
        if resolved_crate == "akita-schedules"
        and resolved_feature != "default"
    }


def schedule_symbol(feature: str) -> str:
    return f"{feature.upper().replace('-', '_')}_SCHEDULES"


def all_schedule_features(graph: dict[str, dict[str, list[str]]]) -> set[str]:
    return {
        feature
        for feature in graph["akita-schedules"]
        if feature.startswith(("fp32-", "fp64-", "fp128-"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("allowed-symbols", "all-symbols"))
    parser.add_argument("feature", nargs="?")
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()

    graph = load_feature_graph(args.repo)
    if args.command == "allowed-symbols":
        if args.feature is None:
            parser.error("allowed-symbols requires an akita-pcs feature")
        features = schedule_features(graph, "akita-pcs", args.feature)
    else:
        features = all_schedule_features(graph)
    for feature in sorted(features):
        print(schedule_symbol(feature))


if __name__ == "__main__":
    main()
