#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
cd "$repo_root"

failures=0

report_matches() {
    local message="$1"
    shift
    local matches
    matches="$("$@" || true)"
    if [ -n "$matches" ]; then
        echo "error: $message" >&2
        printf '%s\n' "$matches" >&2
        failures=1
    fi
}

if [ -e crates/akita-schedules/src/generated ]; then
    echo "error: generated schedule-row machinery is forbidden; commit .aks artifacts instead" >&2
    failures=1
fi

report_matches \
    "legacy schedule-table Cargo features are forbidden" \
    git grep -n -E \
        '(^|[^[:alnum:]_-])(all-schedules|schedules-default|schedules-fp[[:alnum:]_-]*)([^[:alnum:]_-]|$)' \
        -- ':(glob)**/Cargo.toml' ':(glob)**/*.yml' ':(glob)**/*.yaml'

report_matches \
    "artifact generator family names must come from CommitmentConfig" \
    git grep -n -E 'module_name|family_row!\([^)]*"fp(32|64|128)_' -- \
        crates/akita-planner/src/generated_families.rs

report_matches \
    "schedule artifacts must not be embedded in Rust binaries" \
    git grep -n -E 'include_bytes!\(.*(artifacts/schedules|\.aks)' \
        -- ':(glob)crates/**/*.rs' ':(glob)profile/**/*.rs'

report_matches \
    "production library sources must not discover the workspace artifact directory" \
    git grep -n -E 'artifacts/schedules|from_workspace_schedule_artifact' -- \
        crates/akita-config/src crates/akita-pcs/src crates/akita-prover/src \
        crates/akita-schedules/src crates/akita-setup/src crates/akita-verifier/src \
        ':(exclude,glob)**/tests/**' ':(exclude,glob)**/test_support.rs'

report_matches \
    "live setup-offloading docs must not assign schedule ownership to compiled Rust tables" \
    git grep -n -i -E \
        'GeneratedFrozenGroup|GeneratedRecursiveFold|generated[- ]table|generated module|table constructor|generated replay|Rust schedule table|committed schedule table' \
        -- specs/setup-offloading-planner.md book/src/how/setup-offloading.md \
        book/src/usage/profiling.md book/src/introduction/reviewing-akita.md \
        book/src/introduction/built-for-production.md \
        crates/akita-config/src/proof_optimized.rs crates/akita-config/src/lib.rs

if [ "$failures" -ne 0 ]; then
    exit 1
fi

echo "external schedule artifact source guards passed"
