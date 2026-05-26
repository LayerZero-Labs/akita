#!/usr/bin/env bash

set -u
set -o pipefail

usage() {
    cat <<'EOF'
usage: scripts/check-rust-file-lines.sh [--max-lines N] [--baseline PATH] [--no-baseline]

Checks tracked Rust files against a physical line-count cap.

By default this enforces a 1500-line cap with the explicit ratchet baseline in
scripts/rust-file-line-cap-baseline.tsv. Baseline entries are not permanent
exemptions: they may not grow, and they must be removed once the file is under
the cap.
EOF
}

max_lines="${AKITA_RUST_FILE_LINE_CAP:-1500}"
baseline_path="scripts/rust-file-line-cap-baseline.tsv"
use_baseline=true

while [ "$#" -gt 0 ]; do
    case "$1" in
        --max-lines)
            if [ "$#" -lt 2 ]; then
                echo "error: --max-lines requires a value" >&2
                exit 2
            fi
            max_lines="$2"
            shift 2
            ;;
        --baseline)
            if [ "$#" -lt 2 ]; then
                echo "error: --baseline requires a path" >&2
                exit 2
            fi
            baseline_path="$2"
            use_baseline=true
            shift 2
            ;;
        --no-baseline)
            use_baseline=false
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$max_lines" in
    '' | *[!0-9]*)
        echo "error: max line count must be a positive integer, got: $max_lines" >&2
        exit 2
        ;;
esac

if [ "$max_lines" -le 0 ]; then
    echo "error: max line count must be positive, got: $max_lines" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$repo_root" ]; then
    echo "error: must be run inside a git repository" >&2
    exit 2
fi

cd "$repo_root" || exit 2

baseline_paths=()
baseline_counts=()
errors=0

record_error() {
    echo "error: $*" >&2
    errors=$((errors + 1))
}

physical_line_count() {
    awk 'END { print NR }' "$1"
}

baseline_index() {
    local target="$1"
    local i=0
    while [ "$i" -lt "${#baseline_paths[@]}" ]; do
        if [ "${baseline_paths[$i]}" = "$target" ]; then
            echo "$i"
            return 0
        fi
        i=$((i + 1))
    done
    return 1
}

load_baseline() {
    local file="$1"
    local line_number=0
    local line count path existing

    if [ ! -f "$file" ]; then
        record_error "baseline file not found: $file"
        return
    fi

    while IFS= read -r line || [ -n "$line" ]; do
        line_number=$((line_number + 1))
        line="${line%$'\r'}"

        case "$line" in
            '' | '#'*)
                continue
                ;;
        esac

        if [ "$line" = "${line#*$'\t'}" ]; then
            record_error "$file:$line_number: expected '<line-count><TAB><path>'"
            continue
        fi

        count="${line%%$'\t'*}"
        path="${line#*$'\t'}"

        case "$count" in
            '' | *[!0-9]*)
                record_error "$file:$line_number: invalid line count: $count"
                continue
                ;;
        esac

        case "$path" in
            '' | /* | ../* | */../*)
                record_error "$file:$line_number: path must be repo-relative: $path"
                continue
                ;;
        esac

        case "$path" in
            *.rs)
                ;;
            *)
                record_error "$file:$line_number: baseline path is not a Rust file: $path"
                continue
                ;;
        esac

        if existing="$(baseline_index "$path")"; then
            record_error "$file:$line_number: duplicate baseline path also recorded with ${baseline_counts[$existing]} lines: $path"
            continue
        fi

        if ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
            record_error "$file:$line_number: baseline path is not a tracked file: $path"
            continue
        fi

        baseline_paths+=("$path")
        baseline_counts+=("$count")
    done < "$file"
}

if [ "$use_baseline" = true ]; then
    load_baseline "$baseline_path"
fi

scanned=0

while IFS= read -r -d '' file; do
    scanned=$((scanned + 1))
    lines="$(physical_line_count "$file")"
    idx=""

    if idx="$(baseline_index "$file")"; then
        recorded="${baseline_counts[$idx]}"
    else
        recorded=""
    fi

    if [ "$lines" -gt "$max_lines" ]; then
        if [ -n "$recorded" ]; then
            if [ "$lines" -gt "$recorded" ]; then
                record_error "$file has $lines lines, above its baseline of $recorded lines and cap of $max_lines"
            fi
        else
            record_error "$file has $lines lines, above the cap of $max_lines"
        fi
    elif [ -n "$recorded" ]; then
        record_error "$file has $lines lines, at or below the cap of $max_lines; remove its baseline entry"
    fi
done < <(git ls-files -z -- '*.rs')

if [ "$errors" -ne 0 ]; then
    cat >&2 <<EOF

Rust file line-cap check failed.

Split over-cap files into smaller modules. For current offenders listed in
$baseline_path, do not raise the recorded count. Remove the entry once the file
reaches the cap.
EOF
    exit 1
fi

echo "Rust file line-cap check passed: scanned $scanned tracked Rust files; cap=$max_lines; baseline_entries=${#baseline_paths[@]}."
