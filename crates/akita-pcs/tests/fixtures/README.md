# Research binary-source fixtures

The P64 complete replay tests read trusted native catalog JSON and proof pins
from `binary-source-p64/` and `binary-source-p64-dev/`. These are research test
inputs, separate from the supported schedule artifacts under `artifacts/`.
The example revalidates each catalog's identity, root geometry and suffix through
the native catalog constructor before use.

Each directory contains `{22,23}-<shape>.json` for these shapes:

- `binary-claim`
- `prime-claim`
- `binary-and-prime-claims`
- `binary-and-prime-claims-plus-ordinary`

`large-pins.txt` contains eight lines, each `<log>-<shape> <proof length> <SHA3-256>`.
The digest is the default descriptor digest; do not enable `keccak256-digests`.
Both protocol modes have identical proof pins but distinct catalogs. Required CI
replays log 22, the smallest feasible measured size, and compares proof length
and digest against the exact pin entry. The example's cheap
`p64_committed_catalogs_admit_both_sizes` test loads and admits all eight catalogs
for the compiled mode; the two CI invocations cover all 16 catalogs without
proving at log 23. The required Python suite runs
`scripts/tests/test_p64_fixtures.py`, checking the full eight-key roster in each
pin file and every installed file against the complete 18-entry SHA-256 manifest.
Missing files, missing or duplicate entries, malformed pins and changed installed
bytes fail these checks. Hash integrity does not establish that a log-23 proof
pin matches current protocol code: honest log-23 replay for all four shapes in
both modes is a separate Linux gate whenever protocol bytes change. Replay must
compare existing pins and must never refresh them silently.

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
comparisons matched. `P64-SHA256SUMS` preserves the supplied SHA-256 manifest for
all 18 installed files. Keep induced setups, proofs and measurement logs in the
Linux evidence bundle. The research preset retains the P128 degree-64 search
dimensions and basis ranges, with Q64Offset59 and 64-bit opening bounds for P64.

## Replay on Linux

The CI step runs the entire example in release, sequentially to bound peak memory,
in both protocol modes:

```bash
cargo test --release -p akita-pcs --example binary_source_full_size --no-default-features --features parallel,binary-source-measurements,transcript-keccak -- --test-threads=1
cargo test --release -p akita-pcs --example binary_source_full_size --no-default-features --features parallel,binary-source-measurements,transcript-keccak,dev-protocol -- --test-threads=1
```

Provisional measurements used a 64-thread Threadripper PRO 7975WX on Ubuntu
24.04 with Rust 1.95.0. Honest log-22 proving took about 2–4.3 s and warm
verification about 11–23 ms (mixed repeats reached 27 ms). At `72ef7dbc`, the
complete release suite passed five tests per mode, with none ignored or filtered.
One suite run per mode took 104.28 s (current) and 112.55 s (dev); command wall
times including compilation were 221 s and 216 s. `421b93a0` applies the two
formatting fixes reported by that run. The newly added cheap admission test
brings the expected example total to six tests per mode; its Linux result is
pending. These host-specific observations replace the earlier runtime budgets.

The final-relation bypass control for each K/prime fault is still open, as are
green final-head hosted CI and hosted runtime/peak-memory measurements. The
adversarial backend materializes dense weights, so the large-host suite totals
do not establish hosted-runner resource use. Production security, composition
and extraction obligations remain open.

## Regenerate on Linux

Use the existing example with a fresh directory and unset the catalog cache.
Run this block twice with distinct `P64_EVIDENCE` roots, retaining the first as
`P64_REFERENCE`. Plan, prove and verify every shape at both logs and in both modes.

```bash
export P64_EVIDENCE="$(mktemp -d /tmp/akita-p64-evidence.XXXXXX)"
unset BINARY_SOURCE_CATALOG_CACHE
for mode in current dev; do
  p64_features=parallel,binary-source-measurements,transcript-keccak
  if [ "$mode" = dev ]; then p64_features="$p64_features,dev-protocol"; fi
  cargo build --release -p akita-pcs --example binary_source_full_size --no-default-features --features "$p64_features" || exit
  for log in 22 23; do
    for shape in binary-claim prime-claim binary-and-prime-claims binary-and-prime-claims-plus-ordinary; do
      fixture="$P64_EVIDENCE/$mode-$log-$shape"
      mkdir -p "$fixture"
      for phase in plan prove verify; do
        target/release/examples/binary_source_full_size --field p64 --log-carriers "$log" --shape "$shape" --phase "$phase" --fixture "$fixture" > "$fixture/$phase.log" 2>&1 || exit
      done
    done
  done
done
```

Before installing, require successful verification in both roots and byte equality
of `native-catalog.json`, `commitments.bin`, `verifier.bin`, `claims.json` and
`proof.bin` for every case. Copy only `native-catalog.json` to `<log>-<shape>.json`
in the corresponding mode directory. Write `large-pins.txt` in lexicographic key
order using each verified proof's byte length and `hashlib.sha3_256` digest.
Require regenerated catalogs and pins to match the installed references; investigate
any difference before replacing a reference. Verify every installed file with
`hashlib.sha256` against `P64-SHA256SUMS`. Never derive new pins from failing or
unverified proofs, and never silently skip missing fixtures.
