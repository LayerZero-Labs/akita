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
| `experiment-capture` | Host-only capture of challenge messages and ranks for recursive checked advice; also requires `AKITA_CAPTURE_CHALLENGES` |
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

`experiment-capture` is a feature of `akita-challenges`. On non-`riscv64`
hosts, setting `AKITA_CAPTURE_CHALLENGES` (to any value) enables capture. The
environment setting is cached on first use; set it before decoding and use a
separate process to change modes. Capture forces serial decoding and emits
these space-separated stderr lines:

```text
EXPERIMENT_MESSAGE d a b count filtered message_hex
EXPERIMENT_RANK d a b rank_one rank_two
```

Here `d` is the ring dimension, `a` and `b` count coefficients of magnitude 1
and 2, `count` is the requested number of challenges, and `filtered` is `0` for
an unfiltered draw or `1` for an operator-norm draw. `message_hex` is the entire
squeezed message in byte order, with two lowercase hex digits per byte and no
prefix. Other fields are decimal integers. `rank_one` is the colexicographic
rank of the ±1 positions; `rank_two` ranks the ±2 positions within their
complement.

To produce advice, capture from a process that only verifies. Prover draws,
including previews, also emit capture lines; their output does not have the
verifier's tape-consumption order.

For a verification-only capture, a message line precedes its candidate rank
lines. Ranks are emitted serially in the order position advice will be consumed:
one per in-range candidate examined, including candidates later rejected by the
operator-norm predicate.
Out-of-range candidates emit no rank line, and decoding stops after the
requested number of accepted challenges. Unfiltered D64/31/10 draws requesting
at least 1024 challenges emit no rank lines because fold verification routes
them to quartet advice. A capture consumer must preserve draw and candidate
order when producing the two advice tapes. The
[advice-frame contract](../../../specs/sparse-challenge-stream.md#checked-rank-advice-frame)
defines their binary layout and authentication requirements.
Installing captured advice requires a trusted byte budget; see
[checked challenge advice](../how/verification.md#checked-challenge-advice).

Capture is disabled on `riscv64` and in builds without the feature. It changes
host execution and diagnostic output; challenge values and transcript bytes
remain the same.

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
