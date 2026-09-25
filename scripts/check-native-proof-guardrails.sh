#!/usr/bin/env bash
# Guard the small reviewed surface that may bypass native Spongefish state APIs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail_if_match() {
    local pattern="$1"
    shift
    if search_lines "$pattern" "$@"; then
        echo "error: forbidden native proof-input construct matched: $pattern" >&2
        exit 1
    fi
}

search_lines() {
    local pattern="$1"
    shift
    local status
    if [ "${AKITA_GUARDRAIL_SEARCH:-}" = "grep" ]; then
        if grep -EnR "$pattern" "$@"; then
            return 0
        else
            status=$?
        fi
    elif command -v rg >/dev/null 2>&1; then
        if rg -n "$pattern" "$@"; then
            return 0
        else
            status=$?
        fi
    else
        if grep -EnR "$pattern" "$@"; then
            return 0
        else
            status=$?
        fi
    fi
    [ "$status" -eq 1 ] && return 1
    echo "error: native guardrail search failed with status $status" >&2
    exit "$status"
}

search_files() {
    local pattern="$1"
    shift
    local status
    if [ "${AKITA_GUARDRAIL_SEARCH:-}" = "grep" ]; then
        if grep -ElR "$pattern" "$@"; then
            return 0
        else
            status=$?
        fi
    elif command -v rg >/dev/null 2>&1; then
        if rg -l "$pattern" "$@"; then
            return 0
        else
            status=$?
        fi
    else
        if grep -ElR "$pattern" "$@"; then
            return 0
        else
            status=$?
        fi
    fi
    [ "$status" -eq 1 ] && return 1
    echo "error: native guardrail search failed with status $status" >&2
    exit "$status"
}

proof_input_roots=(
    crates/akita-transcript/src
    crates/akita-types/src
    crates/akita-challenges/src
    crates/akita-sumcheck/src
    crates/akita-prover/src/protocol
    crates/akita-verifier/src
)

fail_if_match 'Validate::No|deserialize_[a-z_]*unchecked' "${proof_input_roots[@]}"
fail_if_match 'ProverState::default|VerifierState::default' "${proof_input_roots[@]}"

raw_state_files="$(search_files 'duplex_sponge_state' "${proof_input_roots[@]}" | sort)"
expected_raw_state_files='crates/akita-transcript/src/native.rs
crates/akita-transcript/src/native/sampling.rs
crates/akita-transcript/src/native/verifier.rs'
if [ "$raw_state_files" != "$expected_raw_state_files" ]; then
    echo "error: native raw-state access escaped its reviewed allowlist" >&2
    printf '%s\n' "$raw_state_files" >&2
    exit 1
fi

constructor_files="$(search_files '\.to_(prover|verifier)\(' "${proof_input_roots[@]}" | sort)"
expected_constructor_files='crates/akita-transcript/src/native/domain.rs'
if [ "$constructor_files" != "$expected_constructor_files" ]; then
    echo "error: Spongefish state construction escaped its reviewed allowlist" >&2
    printf '%s\n' "$constructor_files" >&2
    exit 1
fi

echo "Native proof guardrails passed."
