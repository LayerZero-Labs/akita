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

The research `binary-source-measurements` example accepts `--field p64` to
select the p64/E64 native root; omitting it preserves P128. Shapes are
`binary-claim`, `prime-claim`, `binary-and-prime-claims`, and
`binary-and-prime-claims-plus-ordinary`. P64 fixtures use `--log-carriers 22`
and 23, both with C=256; complete replay tests use log 22, the smallest size
measured (logs 4–21 have not been tried). The selected planner/preset refused log 3 under the strict
`next_witness_len < request.current_witness_len` rule in
`crates/akita-planner/src/schedule_params/candidate/recursive/frontier.rs`.
That comparison includes the full-field typed prefix's contribution to the
complete successor witness. Word-only log 3 has 77,824 natural prefix field
cells, a 131,072-cell padded prefix domain and 82,176 outgoing physical digit
cells; raw prefix length is not the comparison. This is a property of the
selected planner/preset, not a theorem excluding all small P64 protocols.
Use separate `--phase plan`,
`--phase prove`, and `--phase verify` invocations with a `--fixture` directory.
Planning writes a research native catalog; proving writes commitments, induced
verifier setup and proof. Verification reconstructs and validates that catalog.
Catalogs must come from trusted local preprocessing. None is a production
admission certificate.

The measurement driver reuses `<fixture>/native-catalog.json` before consulting
`BINARY_SOURCE_CATALOG_CACHE`, for both fields. A reused directory therefore skips
planning when its catalog passes identity and row validation. Use a fresh directory
and unset the cache for fresh planning. Cache names now include the field family;
legacy P128 cache filenames are not reused. A valid alternate cached row can change
setup and proof bytes, so byte comparisons must use identical selected rows.
Research replay catalogs at logs 22/23 live under
`crates/akita-pcs/tests/fixtures/binary-source-p64{,-dev}/`, outside the supported
schedule artifacts. Their shared SHA3-256 proof pins live alongside P128 in
`scripts/binary-source-full-size-digests.tsv`. Required CI replays log 22 in both
protocol modes, admits all 16 catalogs without proving and checks the shared pin
roster and installed catalog hashes. Honest log-23 replay against existing pins
is a separate Linux gate whenever protocol bytes change. The existing
`scripts/binary-source-full-size.sh --field p64 --all` runs fresh generation and
pin checks; its `--reference-dir` option compares two independent executions.
See the [fixture README](../../../crates/akita-pcs/tests/fixtures/README.md) for the
single regeneration procedure and evidence requirements.
