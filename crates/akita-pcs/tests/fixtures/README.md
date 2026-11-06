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
Both protocol modes have identical proof pins but distinct catalogs. The complete
replay tests use log 22, the smallest feasible measured size, and compare both
proof length and digest against the exact pin entry. Missing catalogs, missing
pins, duplicate entries or changed bytes fail the test.

Log 3 is infeasible because the full-field typed setup prefix exceeds the native
witness while `crates/akita-planner/src/schedule_params/candidate/recursive/frontier.rs:83`
requires `next_witness_len < request.current_witness_len`.

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

On a 64-thread Threadripper PRO, measured honest log-22 proving takes about
2–4.3 s and warm verification about 11–23 ms (mixed repeats reached 27 ms).
The following complete-test estimates include the additional forged proofs and
repeated cold verification; they are budgets, not measured replay-test results:

| Test | Proofs (honest + forged) | Expected release runtime on that host |
| --- | --- | --- |
| `p64_word_complete_schedule_and_tampering` | 1 + 1 | 10–30 s |
| `p64_prime_complete_schedule_and_tampering` | 1 + 2 | 20–60 s |
| `p64_word_and_prime_complete_schedule_and_tampering` | 1 + 2 | 20–60 s |
| `p64_mixed_complete_schedule_and_tampering` | 1 + 2 | 30–90 s |

Catalog identity checks should take well under a second. Cached catalogs avoid
planning; each fixture-dependent test should remain below a few minutes in CI.
Measure the restored suite on Linux, including the adversarial backend's dense
weight materialization, before treating those runtime estimates as evidence.

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
