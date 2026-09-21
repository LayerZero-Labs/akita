# Feature flags and build recipes

Akita uses Cargo features for executable PCS field tiers, transcript backends,
and compute support. Schedule rows are external runtime artifacts, not Cargo
features. The default `akita-pcs` build is a parallel CPU configuration with
all protocol field tiers available.

## Default features

| Feature | What it provides |
| --- | --- |
| `field-fp32` | Executable dispatch for PCS fields up to 32 bits |
| `field-fp64` | Executable dispatch for PCS fields from 33 through 64 bits |
| `field-fp128` | Executable dispatch for PCS fields wider than 64 bits |
| `parallel` | Rayon execution across field arithmetic, setup, proving, sumcheck, and verification |
| `transcript-blake2b` | The default Spongefish transcript backend |

The normal build uses all five:

```bash
cargo build -p akita-pcs --release
```

## Common build recipes

### Sequential CPU build

Keep the default transcript while removing Rayon:

```bash
cargo build -p akita-pcs --release \
  --no-default-features \
  --features field-fp32,field-fp64,field-fp128,transcript-blake2b
```

This build produces the same protocol results. It changes local execution and
performance.

### Field-tier pruning

Applications that use only one PCS field tier can omit the other dispatch
branches before typechecking and monomorphization. For example, an Fp64-only
build is:

```bash
cargo build -p akita-pcs --release \
  --no-default-features \
  --features field-fp64,parallel,transcript-blake2b
```

Field features are positive and additive. Selecting `field-fp64` retains every
Fp64 ring degree in the canonical protocol tables; it does not select a ring
degree, schedule family, or application profile. Selecting no field feature is
valid for shared-type and offline-planning crates, but setup, proving,
verification, and dispatch reject every PCS field with `InvalidSetup`.

Cargo unifies features across the dependency graph. If any dependency enables
the default features or another field tier on `akita-types`, `akita-prover`,
`akita-verifier`, `akita-setup`, or `akita-pcs`, the final binary includes that
tier. Use `default-features = false` consistently on direct Akita execution
dependencies when pruning matters. Offline schedule planning still uses the
complete protocol tables and is not restricted by the host executable's field
features.

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

## Transcript backends

Production builds enable exactly one transcript backend.

| Feature | Backend |
| --- | --- |
| `transcript-blake2b` | Blake2b based Spongefish transcript with SHA3 support |
| `transcript-keccak` | Keccak based Spongefish transcript |

The transcript backend is part of proof compatibility. Prover and verifier
must use the same backend and protocol revision.

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
