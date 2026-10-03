#!/usr/bin/env bash

# Regenerate the checked-in schedule artifacts.
#
# With `--dev` as the first argument, regenerate the dev-only set in
# `artifacts/schedules-dev`, priced for the `recompute-last-block` wire
# format. Otherwise regenerate `artifacts/schedules`.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$repo_root" ]; then
    echo "error: must be run inside a git repository" >&2
    exit 2
fi

cd "$repo_root"

planner_features="catalog-gen"
artifact_dir="artifacts/schedules"
if [ "${1:-}" = "--dev" ]; then
    shift
    planner_features="catalog-gen,recompute-last-block"
    artifact_dir="artifacts/schedules-dev"
fi
for arg in "$@"; do
    if [ "$arg" = "--check-catalog" ]; then
        planner_features="$planner_features,catalog-check"
        break
    fi
done

cargo run --release -p akita-planner --features "$planner_features" --bin gen_schedule_artifacts -- \
    "$artifact_dir" "$@"
