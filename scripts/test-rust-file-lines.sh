#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checker="$script_dir/check-rust-file-lines.sh"
tmp_root="$(mktemp -d)"

trap 'rm -rf "$tmp_root"' EXIT

fail() {
    echo "error: $*" >&2
    exit 1
}

new_repo() {
    local repo
    repo="$(mktemp -d "$tmp_root/repo-XXXXXX")"
    git -C "$repo" init -q
    printf '%s\n' "$repo"
}

write_lines() {
    local file="$1"
    local lines="$2"

    mkdir -p "$(dirname -- "$file")"
    awk -v lines="$lines" 'BEGIN { for (i = 1; i <= lines; i++) print "// line " i }' > "$file"
}

write_baseline() {
    local repo="$1"
    shift

    : > "$repo/baseline.tsv"
    local entry
    for entry in "$@"; do
        printf '%s\n' "$entry" >> "$repo/baseline.tsv"
    done
}

track_all() {
    local repo="$1"
    git -C "$repo" add .
}

run_checker() {
    local repo="$1"
    (
        cd "$repo"
        "$checker" --max-lines 5 --baseline baseline.tsv
    )
}

expect_success() {
    local name="$1"
    local repo="$2"

    if ! output="$(run_checker "$repo" 2>&1)"; then
        echo "$output" >&2
        fail "$name: expected success"
    fi
}

expect_failure() {
    local name="$1"
    local repo="$2"
    local expected="$3"
    local status output

    set +e
    output="$(run_checker "$repo" 2>&1)"
    status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        echo "$output" >&2
        fail "$name: expected failure"
    fi

    case "$output" in
        *"$expected"*)
            ;;
        *)
            echo "$output" >&2
            fail "$name: expected output to contain: $expected"
            ;;
    esac
}

repo="$(new_repo)"
write_lines "$repo/src/legacy big.rs" 6
write_lines "$repo/src/small.rs" 5
write_baseline "$repo" $'6\tsrc/legacy big.rs'
track_all "$repo"
expect_success "baseline allows current offender and path with spaces" "$repo"

repo="$(new_repo)"
write_lines "$repo/src/new offender.rs" 6
write_baseline "$repo"
track_all "$repo"
expect_failure "new unbaselined offender" "$repo" "above the cap of 5"

repo="$(new_repo)"
write_lines "$repo/src/legacy.rs" 7
write_baseline "$repo" $'6\tsrc/legacy.rs'
track_all "$repo"
expect_failure "baseline growth" "$repo" "above its baseline of 6 lines"

repo="$(new_repo)"
write_lines "$repo/src/legacy.rs" 5
write_baseline "$repo" $'6\tsrc/legacy.rs'
track_all "$repo"
expect_failure "stale baseline" "$repo" "remove its baseline entry"

repo="$(new_repo)"
write_lines "$repo/src/small.rs" 5
write_baseline "$repo" $'6\t./src/small.rs'
track_all "$repo"
expect_failure "noncanonical baseline path" "$repo" "baseline path is not an exact tracked Rust file"

repo="$(new_repo)"
write_lines "$repo/src/small.rs" 5
write_baseline "$repo" $'6\t*.rs'
track_all "$repo"
expect_failure "pathspec-shaped baseline path" "$repo" "baseline path is not an exact tracked Rust file"

repo="$(new_repo)"
write_lines "$repo/src/legacy.rs" 6
write_baseline "$repo" $'6\tsrc/legacy.rs' $'6\tsrc/legacy.rs'
track_all "$repo"
expect_failure "duplicate baseline path" "$repo" "duplicate baseline path"

repo="$(new_repo)"
write_baseline "$repo" $'6\tsrc/missing.rs'
track_all "$repo"
expect_failure "untracked baseline path" "$repo" "baseline path is not an exact tracked Rust file"

repo="$(new_repo)"
write_lines "$repo/src/legacy.rs" 6
write_baseline "$repo" $'abc\tsrc/legacy.rs'
track_all "$repo"
expect_failure "invalid baseline count" "$repo" "invalid line count"

repo="$(new_repo)"
printf 'not rust\n' > "$repo/README.md"
write_baseline "$repo" $'6\tREADME.md'
track_all "$repo"
expect_failure "non-Rust baseline path" "$repo" "baseline path is not a Rust file"

repo="$(new_repo)"
write_baseline "$repo" $'6\t/tmp/outside.rs'
track_all "$repo"
expect_failure "absolute baseline path" "$repo" "path must be repo-relative"

repo="$(new_repo)"
write_baseline "$repo" $'6\t../outside.rs'
track_all "$repo"
expect_failure "parent baseline path" "$repo" "path must be repo-relative"

echo "Rust file line-cap self-tests passed."
