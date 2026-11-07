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

### Digest hash

`keccak256-digests` (on `akita-pcs` and `akita-transcript`) replaces SHA3-256
with Keccak-256 for every digest Akita binds: the instance descriptor's
digests, commitment digests, and the schedule and planner-policy digests. It
is for an application whose recursion guest checks those digests with Jolt's
Keccak-256 inline. The protocol tag names the selection, so a proof made under
one digest hash never verifies under the other. The feature is independent of
the transcript backend.

The checked-in `.aks` catalogs carry SHA3-256 digests. A build with
`keccak256-digests` rejects them with `AkitaError::InvalidSetup` at the
planner-policy digest check, so such an application must regenerate its
catalogs with `gen_schedule_artifacts` built under the same feature (for
example `--features akita-transcript/keccak256-digests`).

Cargo unifies features across a build, so any crate in the dependency graph
that enables `keccak256-digests` turns it on for every Akita crate in that
build. Check the resolved feature set with `cargo tree -e features` when a
catalog that loads elsewhere fails this check.

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

## Guest inline backends

These features matter only to a Jolt RISC-V guest verifier. On `riscv64` they
route hot primitives through Jolt inlines with the same outputs as the portable
paths; every other target keeps the portable path.

| Feature | Inline |
| --- | --- |
| `ntt-inline` | 64-point i32 negacyclic NTT and six-product pointwise dot (`jolt-inlines-ntt`) |

## Diagnostic features

| Feature | Purpose |
| --- | --- |
| `logging-transcript` | Records transcript schedule events and checks that wire values are absorbed before challenges |
| `response-model-diagnostics` | Measures complete source and response energies for planner model calibration |
| `compression-weights-crosscheck` | Checks every factored compression relation weight evaluation against the per-event evaluation and counts its evaluation tiers |

`response-model-diagnostics` scans witness data that normal proving does not
scan. Use it for model calibration runs, not for ordinary performance numbers.

`compression-weights-crosscheck` (on `akita-types`, forwarded by `akita-pcs`)
evaluates every compression relation weight table both by factor and event by
event, at the verifier's point and at seeded random points, and compares each
grouped builder family with the per-event builder. A difference fails
verification. Each evaluation emits one `tracing` event at `INFO` with its
evaluation tiers and builder outcomes. It is for native diagnostic runs; it
changes no proof, transcript or default build.

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

## External setup accumulation

Setup accumulation needs no Cargo feature. The verifier takes a `SetupPolicy`:
`Check(None)` for ordinary verification, `Check(Some(&mut accumulator))` for
inherited claims or an empty accumulator, or `Defer(&mut accumulator)` to export
direct setup evaluations. For ordinary verification, pass
`akita_verifier::SetupPolicy::Check(None)`.
Every policy checks Stage 3 completely; only direct setup evaluations are deferred.
Attach auxiliary archived-proof advice with `SetupClaimAccumulator::with_advice`.
See [external setup accumulation](../how/recursion.md#external-setup-accumulation)
for inheritance and final-discharge requirements.

## P64 native research catalogs

The public PCS accepts P64 roots only through an explicitly selected native catalog
row. Field-pair validation remains separate from catalog admission. Ordinary catalog
rows cannot admit a P64 root. Native ordinary producers require outer dimension 64.

The `binary_source_full_size` measurement example selects P64 with `--field p64`.
Its trusted research catalogs live under `crates/akita-pcs/tests/fixtures/`, outside
production schedule artifacts. The `test-binary-p64` job replays the example in both
protocol modes, including public-statement, setup, proof and committed-witness faults.
Catalog freshness is checked by
`scripts/binary-source-full-size.sh --field p64 --check-catalogs`:
every installed case must equal fresh planner output. The gate disables caches and
never installs files; admission and replay alone are insufficient. It is a release
check run when catalogs change, not a CI job.

P64 prime groups accept one or two distinct ordered E64 points using
`PrimeGroupClaim` records with `PrimeClaimValue::Extension` coordinates. Each
polynomial value contributes two physical planes and two independent E64 identities.
Two points on one polynomial therefore use four planes and four relation batches.
The shared source is stored once; point collisions reject without resampling.

`PolynomialGroupClaims::new_two_points` also supplies two ordinary readings in a
mixed P64 root. Both readings constrain the same ordinary source. Native ordinary
roots admit exactly one polynomial; signed multi-slot arithmetic does not establish
native admission. The existing single-point transcript encoding remains unchanged.
Common-source extraction and composition for two points remain open obligations.
