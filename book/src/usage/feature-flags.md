# Feature flags and build recipes

Akita uses Cargo features to compile the exact protocol catalogs, transcript
backend, and compute support needed by a host. The default `akita-pcs` build is
a complete parallel CPU configuration for development and ordinary use.

## Default features

| Feature | What it provides |
| --- | --- |
| `parallel` | Rayon execution across field arithmetic, setup, proving, sumcheck, and verification |
| `schedules-default` | The standard generated schedule catalogs |
| `transcript-blake2b` | The default Spongefish transcript backend |

The normal build uses all three:

```bash
cargo build -p akita-pcs --release
```

## Common build recipes

### Sequential CPU build

Keep the default catalogs and transcript while removing Rayon:

```bash
cargo build -p akita-pcs --release \
  --no-default-features \
  --features schedules-default,transcript-blake2b
```

This build produces the same protocol results. It changes local execution and
performance.

### Bounded fp128 dense commitments

Enable the generated catalog for `fp128::DenseBounded`:

```bash
cargo build -p akita-pcs --release \
  --features schedules-fp128-dense-bounded
```

This configuration commits to centered values within a signed 65 bit bound,
which contains the complete `u64` range. The bound is enforced during
commitment and becomes part of commitment identity.

### Recursive setup offloading

Enable the catalog that matches the chosen recursive configuration. The common
fp128 one hot path uses:

```bash
cargo build -p akita-pcs --release \
  --features schedules-fp128-onehot-recursive
```

The multi chunk recursive companion has its own feature. Generated catalog
features are intentionally separate so a deployment can compile only the proof
families it accepts.

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

## Schedule catalog features

The configuration names the field and data representation. A schedule feature
ships the generated rows accepted for that family. Examples include:

- `schedules-fp32-dense` and `schedules-fp32-onehot`.
- `schedules-fp64-dense` and `schedules-fp64-onehot`.
- `schedules-fp128-dense` and `schedules-fp128-onehot`.
- Recursive and multi chunk fp128 companions.

`schedules-default` bundles the standard direct catalogs. Specialized bounded,
recursive, and partitioned profiles are opt in. An enabled catalog adds
approved rows to the binary. It does not run planner search at verification
time.

The [configuration guide](./configuration.md) explains which family to choose.
The generated feature definitions in `crates/akita-config/Cargo.toml` are the
source of truth for the exact catalog bundle.

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

Application builds should choose normal schedule catalog features instead of
profile CI features. The [benchmark report guide](./benchmark-reports.md)
explains how the workflow uses them.

## Pin one feature contract

Pin every Akita crate to the same commit or release and record the accepted
feature set with the deployment. This gives the prover and verifier the same
catalogs, transcript backend, public types, and proof format.
