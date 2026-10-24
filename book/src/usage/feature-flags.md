# Feature flags and build recipes

Akita uses Cargo features for transcript backends and compute support. Schedule
rows are external runtime artifacts, not Cargo features. The default
`akita-pcs` build is a parallel CPU configuration for ordinary use.

## Default features

| Feature | What it provides |
| --- | --- |
| `parallel` | Rayon execution across field arithmetic, setup, proving, sumcheck, and verification |
| `transcript-keccak` | The default Spongefish transcript backend |

The normal build uses both:

```bash
cargo build -p akita-pcs --release
```

## Common build recipes

### Sequential CPU build

Keep the default transcript while removing Rayon:

```bash
cargo build -p akita-pcs --release \
  --no-default-features \
  --features transcript-keccak
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

Backend routing requires no Cargo feature; see [backend routing](../how/architecture.md#backend-routing).

## Transcript backends

Production builds enable exactly one transcript backend. Keccak is the
default; Blake2b stays available as an opt-in feature. To select Blake2b,
build with `--no-default-features` and enable `transcript-blake2b`.

| Feature | Backend |
| --- | --- |
| `transcript-keccak` | Keccak based Spongefish transcript (default) |
| `transcript-blake2b` | Blake2b based Spongefish transcript (opt-in) |

The transcript backend is part of proof compatibility. Prover and verifier
must use the same backend and protocol revision.

### Prover-side transcript feature

`akita-transcript`, `akita-challenges`, and `akita-types` each have a `prover`
feature. It enables the proof-of-work nonce search and the fold-response
preview. Both evaluate candidate nonces on a copy of the sponge state, and
Spongefish exposes that state only under its `yolocrypto` feature. `akita-prover`
turns `prover` on; `akita-verifier` leaves it off. The dependency hygiene check
`scripts/check-crate-deps.sh akita-verifier` fails if the verifier's own
dependency graph enables `yolocrypto`.

Cargo unifies features within one build. A binary that links the prover
therefore compiles its verifier with `yolocrypto` as well. The feature changes
which Spongefish items are visible; it does not change proof bytes.

## Schedule catalog storage

Tracked development artifacts live under `artifacts/schedules/`. Production
applications may use a filesystem, database, object store, or another trusted
parameter channel; Akita itself accepts bytes or a validated catalog and does
not choose that storage policy. The [configuration guide](./configuration.md)
explains which family to choose.

## Diagnostic features

| Feature | Purpose |
| --- | --- |
| `logging-transcript` | Records transcript schedule events and checks that wire values are absorbed before challenges |
| `response-model-diagnostics` | Measures complete source and response energies for planner model calibration |

`response-model-diagnostics` scans witness data that normal proving does not
scan. Use it for model calibration runs, not for ordinary performance numbers.

The `transcript_schedule` example uses `logging-transcript`:

```bash
cargo run -p akita-pcs \
  --features logging-transcript \
  --example transcript_schedule
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
