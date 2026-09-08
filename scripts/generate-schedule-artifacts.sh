#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$repo_root" ]; then
    echo "error: must be run inside a git repository" >&2
    exit 2
fi

cd "$repo_root"

planner_features="catalog-gen"
for arg in "$@"; do
    if [ "$arg" = "--check-catalog" ]; then
        planner_features="catalog-gen,catalog-check"
        break
    fi
done

cargo run --release -p akita-planner --features "$planner_features" --bin gen_schedule_artifacts -- \
    artifacts/schedules "$@"
