#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$repo_root" ]; then
    echo "error: must be run inside a git repository" >&2
    exit 2
fi

cd "$repo_root"

cargo run --release -p akita-planner --features catalog-gen --bin gen_schedule_tables -- \
    crates/akita-schedules/src/generated "$@"
