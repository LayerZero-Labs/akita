# Feature flags and build recipes

Akita uses Cargo features for transcript backends and compute support. Schedule
rows are external runtime artifacts, not Cargo features. The default
`akita-pcs` build is a parallel CPU configuration for ordinary use.

## Default features

| Feature | What it provides |
| --- | --- |
| `parallel` | Rayon execution across field arithmetic, setup, proving, sumcheck, and verification |
| `transcript-blake2b` | The default Spongefish transcript backend |

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
  --features transcript-blake2b
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

## Backend routing

Backend routing is always available through `batched_prove`; it requires no
Cargo feature. A fold reduces a private witness to a smaller successor witness.
Its producer commits that successor once, then a caller-defined route chooses
its next owner. On a switch, the source exports an opaque capability and the
destination adopts the complete witness and commitment material synchronously.
The source finishes its sumchecks before the destination executes its fold.
The terminal fold can have its own owner as well.

Construct `BackendRegistry<Cfg>`, register each backend with its local
`akita_prover::SetupPrefixProverRegistry`. Registration returns a `BackendId`.
Pass the registry, typed opening inputs, and route to `batched_prove`.
`FixedFoldRoute::new(vec![a, b, a])` assigns folds 0, 1, and 2 to those registry
entries. Custom `FoldExecutionPolicy` implementations select every level through
`choose_backend`; its current owner is absent at level 0. The registry dispatches
the root to the selected executor, checks its commitment handle family, and
validates that the handles belong to that instance. The route stores only IDs.
Prefix handles must belong to each
local instance and have identical public commitments. Registration rejects
duplicate instances; IDs from another registry reject. Route implementations
own any cost model; Akita does not request or compare cost estimates.

Backends implement `SuccessorExportKernel` and `SuccessorImportKernel` alongside
ordinary typed kernels. Each directed type pair implements `SuccessorBridge`
on `Edge<A, B>`. Register the bridge explicitly with
`registry.register_bridge::<A, B>()?` before selecting a switch. Same-type
instance switches (two `CpuBackend`s) still need
`register_bridge::<CpuBackend<F, E>, CpuBackend<F, E>>()?`. Heterogeneous
routes register each directed edge (for example `PrivateCpu` ↔ `CpuBackend`).
A missing bridge rejects the selected route; Akita never silently reroutes it.
A single instance needs no bridge: same-ID choices retain native handles without
export, conversion, or import.

The handoff is blocking: export a typed packet, convert it to the destination's
packet, then validate and import both handles. The coordinator checks shape
metadata and dispatches typed bridges without reading private payloads. CPU
self-edges share immutable packed witness and material storage. External bridges
can construct `CpuImportPacket::new(descriptor, sections)` with packed or signed
digits, canonical field coefficients, and compression sections. CPU exports
provide `into_sections()` for conversion to another representation. Import checks
section lengths, the admitted handoff, digit bounds, and canonical encodings.
Switching retains the producer's handles until its sumchecks complete.
The CPU scheme method uses this same prover with every level assigned to its
supplied backend. Routing does not change the setup, commitment, transcript, or
proof format.

Before any proof computation, the coordinator calls `prepare_executor` on every
registered backend to open its local proof session. Before each assigned level,
`begin_fold` validates that level and its successor commitment work. Preparation
does not require a backend to execute unassigned levels. All prepared sessions
finish on success or abort on failure, including those on unselected backends.
A registry supports repeated proofs, with one active proof at a time.

Implementation: `crates/akita-prover/src/protocol/prove/execution.rs`,
`crates/akita-prover/src/backend/transfer.rs`,
and
`crates/akita-cpu-backend/src/opaque/transfer.rs`.
Regression coverage: `crates/akita-pcs/tests/dynamic_backends.rs`.

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
