#!/usr/bin/env bash
# Compare the existing binary root with the immutable pinned dev revision.
# Linux only: retain every capture and log under target/binary-root-bytes/.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo 'This gate is Linux-only.' >&2
    exit 1
fi
repo="$(git rev-parse --show-toplevel)"
cd "$repo"
git diff --exit-code HEAD
git merge-base --is-ancestor 357d12b748b356b8da69eb20fe556b79b3234bec HEAD
unset AKITA_BINARY_BASELINE_DIR AKITA_BINARY_CAPTURE_DIR AKITA_BINARY_BYTE_GOLDENS
unset AKITA_CAPTURE_BINARY_BYTE_GOLDENS AKITA_BINARY_BYTE_GOLDEN_RECEIPTS
mkdir -p "$repo/target/binary-root-bytes"
run_dir="$(mktemp -d "$repo/target/binary-root-bytes/run.XXXXXX")"
base_revision=357d12b748b356b8da69eb20fe556b79b3234bec
head_revision="$(git rev-parse HEAD)"
echo "$head_revision" > "$run_dir/head.txt"
echo "$base_revision" > "$run_dir/base.txt"
echo "Binary byte gate logs: $run_dir"
mkdir "$run_dir/base" "$run_dir/head"
git archive "$base_revision" | tar -x -C "$run_dir/base"
git archive "$head_revision" | tar -x -C "$run_dir/head"
# Both revisions use frozen, separate source checkouts and build directories.
# Only the observation/assertion hook is added to the archived baseline.
# No candidate production code, tests or expectations replace baseline code.
python3 - "$run_dir/head" "$run_dir/base" <<'PY'
from pathlib import Path
import sys

repo, base = map(Path, sys.argv[1:])
relative = Path('crates/akita-pcs/tests/binary_source_end_to_end.rs')
head = (repo / relative).read_text()
start = head.index('    // Compared against the immutable pinned dev commit')
end_marker = '    eprintln!("{golden}");\n'
assert head.count(end_marker) == 1
end = head.index(end_marker, start) + len(end_marker)
hook = head[start:end]
assert 'byte_contract::' not in hook and 'packed_prover' not in hook
path = base / relative
source = path.read_text()
needle = '    let continuation: [u8; 32] = prover.verifier_message();\n'
if source.count(needle) != 1:
    raise SystemExit('baseline observation site is not unique')
# The old inline known vectors cover Keccak only. Leave them enabled there;
# Blake2b is compared with this run's independently compiled baseline instead.
known = '    compatibility::assert_base_bytes(&proof, mixed, word_width);'
assert source.count(known) == 1
source = source.replace(known, '    #[cfg(feature = "transcript-keccak")]\n' + known)
declaration = '#[path = "binary_source_end_to_end/compatibility.rs"]'
assert source.count(declaration) == 1
source = source.replace(declaration, '#[cfg(feature = "transcript-keccak")]\n' + declaration)
path.write_text(source.replace(needle, needle + hook))
PY

tests=(
    projection_binary_group_opens_once_inside_the_callers_transcript
    binary_and_ordinary_groups_share_one_opening
    field_switch_narrow_source_mutations_authenticate
)
for backend in transcript-blake2b transcript-keccak; do
    for protocol in current dev; do
        features="labinius-trinomial,$backend"
        if [[ "$protocol" == dev ]]; then features+=,dev-protocol; fi
        reference="$run_dir/base-$backend-$protocol.golden"
        # Separate target directories are mandatory: an archive can preserve
        # older mtimes and otherwise reuse another revision's Cargo artifacts.
        for variant in base head-sequential head-parallel; do
            checkout="$run_dir/head"
            selected="$features"
            target="$run_dir/target-head"
            expected="$reference"
            if [[ "$variant" == base ]]; then
                checkout="$run_dir/base"
                target="$run_dir/target-base"
                expected=''
                selected+=,parallel
            elif [[ "$variant" == head-parallel ]]; then
                selected+=,parallel
            fi
            capture="$run_dir/$variant-$backend-$protocol"
            for test_name in "${tests[@]}"; do
                (
                    cd "$checkout"
                    if [[ -n "$expected" ]]; then
                        export AKITA_BINARY_GOLDEN="$expected"
                    else
                        unset AKITA_BINARY_GOLDEN
                    fi
                    # Capture raw output for the parser without an RTK dependency.
                    CARGO_TARGET_DIR="$target" cargo test --locked --release -p akita-pcs \
                        --test binary_source_end_to_end --no-default-features \
                        --features "$selected" -- --exact "$test_name" --nocapture \
                        > "$capture-$test_name.log" 2>&1
                )
            done
            python3 - "$capture" "$expected" "${tests[@]}" <<'PY'
from pathlib import Path
import sys

capture, expected, *tests = sys.argv[1:]
rows = []
for test in tests:
    log = Path(f'{capture}-{test}.log').read_text()
    if 'test result: ok. 1 passed;' not in log:
        raise SystemExit(f'{test}: expected one passing test')
    records = [line for line in log.splitlines() if line.startswith('binary-golden:')]
    if len(records) != 1:
        raise SystemExit(f'{test}: expected one binary golden, got {len(records)}')
    rows.extend(records)
rows.sort()
keys = {tuple(row.split(':')[1:3]) for row in rows}
if keys != {('false', '128'), ('true', '128'), ('false', '8')}:
    raise SystemExit(f'incomplete binary fixtures: {keys}')
Path(f'{capture}.golden').write_text('\n'.join(rows) + '\n')
if expected and rows != Path(expected).read_text().splitlines():
    raise SystemExit('binary proof/commitment/continuation drift')
PY
        done
    done
done
echo "Binary byte comparison passed; retain $run_dir"
