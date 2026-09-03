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

report_matches \
    "generated schedule-row Rust modules are forbidden; commit .aks artifacts instead" \
    find crates/akita-schedules/src/generated -maxdepth 1 -type f -name 'fp*.rs' -print

report_matches \
    "legacy schedule-table Cargo features are forbidden" \
    rg -n '(^|[^[:alnum:]_-])(all-schedules|schedules-default|schedules-fp[[:alnum:]_-]*)([^[:alnum:]_-]|$)' \
        --glob 'Cargo.toml' --glob '*.yml' --glob '*.yaml' . profile crates

report_matches \
    "schedule artifacts must not be embedded in Rust binaries" \
    rg -n 'include_bytes!\([^\n]*(artifacts/schedules|\.aks)' crates profile --glob '*.rs'

report_matches \
    "production library sources must not discover the workspace artifact directory" \
    rg -n 'artifacts/schedules|from_workspace_schedule_artifact' \
        crates/akita-config/src crates/akita-pcs/src crates/akita-prover/src \
        crates/akita-schedules/src crates/akita-setup/src crates/akita-verifier/src \
        --glob '*.rs' --glob '!**/tests/**' --glob '!**/test_support.rs'

if ! rg -Uq '#\[cfg\(feature = "planner-support"\)\][[:space:]]+pub mod generated;' \
    crates/akita-schedules/src/lib.rs; then
    echo "error: compact generated-row machinery must remain gated behind planner-support" >&2
    failures=1
fi

if [ "$failures" -ne 0 ]; then
    exit 1
fi

echo "external schedule artifact source guards passed"
