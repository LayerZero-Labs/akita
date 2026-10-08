# Feature flags and build recipes

Akita uses Cargo features for compute support and diagnostics. Schedule
rows are external runtime artifacts, not Cargo features. The default
`akita-pcs` build is a parallel CPU configuration for ordinary use.

## Default features

| Feature | What it provides |
| --- | --- |
| `parallel` | Rayon execution across field arithmetic, setup, proving, sumcheck, and verification |

The normal build uses it:

```bash
cargo build -p akita-pcs --release
```

## Common build recipes

### Sequential CPU build

Remove Rayon:

```bash
cargo build -p akita-pcs --release --no-default-features
```

This build produces the same protocol results. It changes local execution and
performance.

### Schedule families

Choosing `fp128::DenseBounded`, `RecursiveCommitmentConfig<_>`, or a multi-chunk
preset does not change the feature graph. Load the matching `.aks` artifact from
application-owned storage and pass its bytes to
`AkitaCommitmentScheme::from_schedule_artifact`. The config validates the family
name and planner-policy digest before any row can be used.

### Disk backed public setup

```bash
cargo build -p akita-pcs --release --features disk-persistence
```

This stores public matrix coefficients and setup prefix artifacts. Prepared NTT
caches remain local memory state and rebuild from the public setup.

## Transcript sponge

The sponge is not a Cargo feature. The core prove and verify entry points are
generic over a `jolt_transcript::Sponge` and run on the caller's transcript;
the standalone API uses `AkitaSponge` (Blake2b-512). The sponge is bound into
the transcript's protocol identifier, so prover and verifier must use the same
sponge and protocol revision.

## Schedule catalog storage

Tracked development artifacts live under `artifacts/schedules/`. Production
applications may use a filesystem, database, object store, or another trusted
parameter channel; Akita itself accepts bytes or a validated catalog and does
not choose that storage policy. The [configuration guide](./configuration.md)
explains which family to choose.

## Diagnostic features

| Feature | Purpose |
| --- | --- |
| `logging` (`akita-pcs`) | Records every transcript operation with its site and argument-string range, for the transcript-hardening suites and the profile example's wire report |
| `response-model-diagnostics` | Measures complete source and response energies for planner model calibration |

`response-model-diagnostics` scans witness data that normal proving does not
scan. Use it for model calibration runs, not for ordinary performance numbers.

The transcript-hardening suites compare the prover's and verifier's event
streams under `logging`:

```bash
cargo test --release -p akita-pcs --features logging --test transcript_hardening
```

## Profile CI features

The benchmark workflow uses narrow features such as `profile-ci-fp32` and
`profile-ci-distributed`. Each feature compiles only the modes in one CI shard.
`profile-ci` is their compatibility union, and `profile-bench-selected` is an
internal marker used by those groups.

Application builds should treat profile CI features as repository-only mode
selectors. Production schedule coverage comes from the external catalog, not
the feature graph. The [benchmark report guide](./benchmark-reports.md) explains
how the workflow uses these selectors.

## Pin one feature contract

Pin every Akita crate to the same commit or release and record the accepted
feature set and approved catalog digest with the deployment. This gives the
prover and verifier the same catalog identity, transcript backend, public types,
and proof format.
