#!/usr/bin/env bash
# Guard the small reviewed surface that may construct proof transcripts or
# bypass checked proof decoding.
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
    crates/akita-types/src
    crates/akita-params/src
    crates/akita-challenges/src
    crates/akita-sumcheck/src
    crates/akita-prover/src/protocol
    crates/akita-verifier/src
)

fail_if_match 'Validate::No|deserialize_[a-z_]*unchecked' "${proof_input_roots[@]}"

# Protocol code runs on the caller's transcript. Only the standalone verifier
# boundary and unit-test fixtures may start one.
constructor_files="$(search_files '(Prover|Verifier)Transcript(::<[^>]*>)?::new' "${proof_input_roots[@]}" | sort)"
expected_constructor_files='crates/akita-prover/src/protocol/prove/suffix.rs
crates/akita-sumcheck/src/proof_stream.rs
crates/akita-types/src/transcript.rs
crates/akita-verifier/src/fold/verify.rs
crates/akita-verifier/src/stages/opening_claims/extension_claim.rs'
if [ "$constructor_files" != "$expected_constructor_files" ]; then
    echo "error: transcript construction escaped its reviewed allowlist" >&2
    printf '%s\n' "$constructor_files" >&2
    exit 1
fi

echo "Native proof guardrails passed."
