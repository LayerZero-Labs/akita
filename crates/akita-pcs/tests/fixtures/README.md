# Research binary-source fixtures

The P64 complete replay tests read trusted native catalog JSON from
`binary-source-p64/` and `binary-source-p64-dev/`. These are research test
inputs, separate from the supported schedule artifacts under `artifacts/`.
The example revalidates each catalog's identity, root geometry and suffix through
the native catalog constructor before use.

Each directory contains `{22,23}-<shape>.json` for these shapes:

- `binary-claim`
- `prime-claim`
- `binary-and-prime-claims`
- `binary-and-prime-claims-plus-ordinary`

Proof pins share `scripts/binary-source-full-size-digests.tsv` with P128.
P64 rows are `p64/<log> <shape> <proof length> <digest> sha3-256`; one eight-key
roster serves both protocol modes. The field-qualified key keeps the P128 rows
and their legacy four-column parser unchanged. P64 uses the default descriptor
digest; selecting `keccak256-digests` fails with the selected algorithm named.
Both modes have identical proof pins but distinct catalogs. Required CI replays
log 22, the smallest size measured (logs 4–21 have not been tried), and compares
proof length, digest and algorithm immediately after honest proving, before
verification and tampering.
The cheap `p64_committed_catalogs_admit_both_sizes` test admits all eight catalogs
for the compiled mode; the two invocations cover all 16 without proving at log 23.
The required Python suite runs `scripts/tests/test_p64_fixtures.py`, checking the
complete shared pin roster and all 16 catalog hashes in `P64-SHA256SUMS`.
Missing files, missing or duplicate entries, malformed pins and changed catalog
bytes fail these checks. Hash integrity does not establish that a log-23 proof
pin matches current protocol code: honest log-23 replay in both modes is a
separate Linux gate whenever protocol bytes change. Replay never refreshes pins.

The selected planner/preset refused log 3 because the full-field typed prefix's
contribution to the complete successor witness does not satisfy the strict
`next_witness_len < request.current_witness_len` rule in
`crates/akita-planner/src/schedule_params/candidate/recursive/frontier.rs`.
For word-only log 3, distinguish 77,824 natural prefix field cells, a 131,072-cell
padded prefix domain and 82,176 outgoing physical digit cells. The comparison
uses the complete successor and current physical witness, not raw prefix length.
This is a property of the selected planner/preset, not a general impossibility
result for small P64 protocols.

The catalogs and pins were generated on Linux at `afe36fd1` and reproduced in two
independent roots: all 80 catalog, commitment, verifier setup, claims and proof
comparisons matched. `P64-SHA256SUMS` preserves the supplied SHA-256 entries for
the 16 catalogs; proof lengths and digests are preserved in the shared TSV. Keep
induced setups, proofs and measurement logs in the Linux evidence bundle. The research preset retains the P128 degree-64 search
dimensions and basis ranges, with Q64Offset59 and 64-bit opening bounds for P64.

## Replay on Linux

The dedicated `test-binary-p64` CI job runs the entire example sequentially in
both protocol modes on `GitHubRunner-Large-Public`, using the optimized `ci-test`
profile with debug assertions and overflow checks. Its 60-minute timeout is an
initial budget; expected hosted runtime is unmeasured. It uses a dedicated Rust
build cache. The existing Clippy job lints all measurement targets in both modes.
Run the complete replay with:

```bash
cargo test --profile ci-test -p akita-pcs --example binary_source_full_size --no-default-features --features parallel,binary-source-measurements,transcript-keccak -- --test-threads=1
cargo test --profile ci-test -p akita-pcs --example binary_source_full_size --no-default-features --features parallel,binary-source-measurements,transcript-keccak,dev-protocol -- --test-threads=1
```

Provisional measurements used a 64-thread Threadripper PRO 7975WX on Ubuntu
24.04 with Rust 1.95.0. Honest log-22 proving took about 2–4.3 s and warm
verification about 11–23 ms (mixed repeats reached 27 ms). At `72ef7dbc`, the
complete release suite passed five tests per mode, with none ignored or filtered.
One suite run per mode took 104.28 s (current) and 112.55 s (dev); command wall
times including compilation were 221 s and 216 s. `421b93a0` applies the two
formatting fixes reported by that run. The newly added cheap admission test
brings the expected example total to six tests per mode; its Linux result and
the new `ci-test` profile timings are pending. These host-specific observations replace the earlier runtime budgets.

The final-relation bypass control for each K/prime fault is still open, as are
green final-head hosted CI and hosted runtime/peak-memory measurements. The
adversarial backend materializes dense weights, so the large-host suite totals
do not establish hosted-runner resource use. Production security, composition
and extraction obligations remain open.

## Regenerate on Linux

Use `scripts/binary-source-full-size.sh` for both fields. For P64, `--all` runs
all four shapes, both logs and both protocol modes. It unsets the catalog cache,
refuses existing case directories, runs plan/prove/verify in separate processes
and checks every verified proof against the shared TSV. It retains artifacts and
logs under `--output-dir`; it never updates installed references. `CARGO_TARGET_DIR`
selects the executable location, so a non-default build directory is supported.

Run twice in fresh roots; the second call also compares all five artifacts per
case against the first (80 comparisons):

```bash
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"
export P64_REFERENCE="$(mktemp -d /tmp/akita-p64-reference.XXXXXX)"
export P64_EVIDENCE="$(mktemp -d /tmp/akita-p64-evidence.XXXXXX)"
BINARY_SOURCE_VERIFY_REPEAT=5 scripts/binary-source-full-size.sh --field p64 --all --output-dir "$P64_REFERENCE"
BINARY_SOURCE_VERIFY_REPEAT=5 scripts/binary-source-full-size.sh --field p64 --all --output-dir "$P64_EVIDENCE" --reference-dir "$P64_REFERENCE"
```

Require equal `native-catalog.json`, `commitments.bin`, `verifier.bin`, `claims.json`
and `proof.bin`, and successful verification in both roots. Compare each generated
catalog with its installed mode-specific `<log>-<shape>.json` and manifest entry
before copying it. Investigate any difference; never silently replace a reference.
For honest replay using installed catalogs, select `--replay-catalogs`; `--shape all`
selects all four shapes at one log, and `--dev-protocol` selects the other mode.
P128 keeps its existing script defaults, output parsing and literal pins unchanged.
