# Architecture overview

How the workspace is organized and how a single `commit → prove → verify` call
flows through it.

## Crate map

Workspace members live under `crates/`.
There is **no** `akita-scheme` crate: end-to-end `AkitaCommitmentScheme`
orchestration lives in `akita-pcs`.

| Crate | Role |
|-------|------|
| `akita-field` | Field traits, prime/extension fields, unreduced/packed helpers, FFT, parallel macros |
| `akita-witness` | Shared borrowed witness/polynomial view vocabulary (`PolynomialView`, `WitnessProvider`) for sumcheck and polyops paths |
| `akita-serialization` | Serialization, validation, and compression traits |
| `akita-algebra` | Modules, vectors, NTTs, cyclotomic rings, sparse challenges, polynomials |
| `akita-transcript` | Spongefish-backed Fiat-Shamir transcript, descriptor preamble, logging checks |
| `akita-challenges` | Fiat-Shamir challenge sampling helpers |
| `akita-sumcheck` | Sumcheck proofs, drivers, compact folding, batching, accumulation |
| `akita-types` | Proof, setup, schedule, layout, commitment, and transcript-append shapes; SIS floors; layout and proof-size helpers |
| `akita-planner` | `Cfg`-free schedule engine: generated table types, catalog validation, compact→`LevelParams` expansion, offline DP |
| `akita-schedules` | Feature-gated shipped schedule table data (types from `akita-planner`) |
| `akita-config` | Runtime presets, the `CommitmentConfig` trait, `policy_of::<Cfg>()`, schedule catalog wiring, transcript bind helper |
| `akita-setup` | Config-backed setup construction and optional setup cache |
| `akita-verifier` | Verifier replay without prover-only polynomial backends; directly `<Cfg>`-generic |
| `akita-prover` | Commitment, proving, setup expansion, witnesses, polynomial backends, compute operation traits |
| `akita-pcs` | Umbrella crate: `AkitaCommitmentScheme`, re-exports, examples, benches, integration tests |

**Dependency graph and ownership rules:** [`docs/crate-graph.md`](../../../docs/crate-graph.md).
CI enforces one-way boundaries via `scripts/check-crate-deps.sh`.

Key structural facts:

- `akita-planner` owns offline schedule search and table emission. It names no
  `CommitmentConfig` type and is not on the verifier runtime dependency path.
- `akita-verifier` depends on `akita-config`, which resolves rows through
  `akita-schedules`. Verification reaches generated row expansion, not planner
  search.
- Verifier-only integrations should use `akita-verifier` + `akita-types` + `akita-config`, not the umbrella `akita-pcs` package.

## End-to-end lifecycle

1. **Preset selection.** The caller picks a `CommitmentConfig` preset (`fp32` / `fp64` / `fp128` families). `CommitmentConfig::resolve_catalog_row_for_key` resolves one complete row from the shipped catalog. Planner search remains offline. Each row selects `SubringCoefficientPacking` or `EvaluationTrace` for every nonterminal fold. EOR is present only for an evaluation trace opening over a proper extension field. See [Fold path and field geometry](./proving/fold-path.md).
2. **Setup.** `akita-setup` expands the config-backed setup (Ajtai matrices, stride envelopes). Setup capacity must cover the requested `num_vars`.
3. **Commit.** The context-aware `commit` entry point (in `akita-prover`, orchestrated by `akita-pcs`) produces one committed polynomial group using `GroupContext`. Scheduler mode selects the scalar row when the group has no precommitted groups, or the exact grouped row when it does; explicit mode validates caller-supplied root parameters. A group committed under a scalar row may later be supplied as a precommitted group.
4. **Claims.** The caller supplies ordered `PolynomialGroupClaims`; each group owns its complete point, evaluations, and commitment.
5. **Prove.** `batched_prove` walks the schedule level by level. It prepares each group with the scheduled opening method, runs the sumchecks, performs EOR when required, and hands the last folded witness to the direct terminal proof.
6. **Verify.** `batched_verify` re-derives the schedule, replays nonterminal sumchecks and relation-matrix evaluations, then closes the terminal with direct consistency/A and weighted trace checks. Prover and verifier share `bind_transcript_instance_descriptor` so Fiat-Shamir challenges match.

Entry points: `crates/akita-pcs/src/scheme/mod.rs`, `crates/akita-prover/src/protocol/core/prove.rs`, `crates/akita-verifier/src/protocol/core/verify.rs`.

Further reading: [Configuration and planning](./configuration.md), [Proving](./proving/proving.md), [Verification](./verification.md).

Recursive setup offloading adds one setup-only `SetupSumcheckProof` at each
nonterminal producer whose successor consumes a setup prefix.
Its wire payload is the setup claim, the setup-prefix evaluation, and one
degree-two sumcheck over the native setup domain.
Its round count and planned size do not depend on the successor witness length.

## Ring-dimension ownership

The cyclotomic ring dimension is **schedule-derived shape metadata, not a
type parameter of the protocol**. Protocol data — commitments, hints, proofs,
claims, and root polynomial storage (`DensePoly<F>`, `OneHotPoly<F, I>`, and
their enum wrapper) — is flat field-element vectors (`RingVec<F>`). Per-level
`CommitmentRingDims` (`d_a` / `d_b` / `d_d` on `LevelParams::role_dims`) is
the operation authority for how those vectors are interpreted; levels may
differ. Here, *role* is the historical protocol name for a commitment matrix's
fixed job: A carries the relation witness, B commits the next witness, and D
commits the opening digits. The matrices do not switch roles when their ring
dimensions change. User-facing prose therefore calls a non-uniform tuple such
as `128/64/64` **per-matrix ring dimensions** and a change between levels a
**ring-dimension transition**. [`validate_schedule_ring_dims`] checks every
scheduled dimension directly against the field's dispatch and NTT support.
The public setup is one flat field stream with no ring dimension.

A, B, and D matrix dimensions form a separate admission domain and are all at
least 64. Compressed commitments derive their two smaller dimensions directly
from the modulus profile (`q128: 16/8`, `q64: 32/16`, `q32: 64/32`). Those
compression-only dimensions never become `CommitmentRingDims` and never reduce
the ordinary relation's common coefficient block.

Every function on the prove/verify path has one of two roles:

- **Orchestration** reads schedule types, drives the transcript, and moves
  D-free storage. It never carries `const D`.
- **Kernels** (NTT, digit decomposition, commit/opening folds,
  ring-switch arithmetic) are const-generic over `D` and receive extracted
  numbers, never schedule types.

The bridge is the *operation adapter*: a D-free function that extracts the
ring dimension of the specific data one operation touches and enters the
kernel through `akita_types::dispatch_for_field!` exactly once,
returning D-free storage. Dispatch is per operation — never per level or per
proof — so that per-matrix ring dimensions inside one fold (`d_a`/`d_b`/`d_d`,
see `specs/runtime-ring-cutover.md`) reduce to feeding different
dimensions to different adapters. `CommitmentRingDims` on `LevelParams::role_dims`
names the per-matrix ring dimensions; prove/verify hot paths dispatch on
`d_a()`, `d_b()`, or `d_d()` per operation, not on a single fused dimension.

The normative contract (discriminator rule, forbidden facade/level-
monomorphization patterns) lives in `specs/runtime-ring-cutover.md`.
Mixed-dimension malformed proof rejection is covered by
`crates/akita-verifier/tests/mixed_d_rejections.rs` through the verifier API.

## Core types

| Type | Role |
|------|------|
| `AkitaCommitmentScheme<Cfg>` | Top-level PCS `commit` / `prove` / `verify` orchestration (`akita-pcs`) |
| `AkitaProverSetup<F>` | Prover setup wrapper around a materialized prefix of the dimension-free public field stream |
| `Commitment<F>`, `RingVec<F>` | protocol commitment and field-vector storage |
| `CommitmentRingDims`, `validate_schedule_ring_dims` | A/B/D commitment-matrix ring dimensions and schedule validation |
| `CommitmentConfig` | Single user-facing trait for every per-config policy hook (algebra, exact SIS profile, decomposition, layout, schedule, transcript bind, prove/commitment params). Verifier-reachable hooks return `Result<_, AkitaError>` |
| `LevelParams` | Per-level recursion layout and config (ring/ext degrees, decomposition depth, `role_dims`) |
| `PlannerPolicy` | `Cfg`-free projection of a preset for `akita_planner::find_schedule`; derive via `akita_config::policy_of::<Cfg>()` |
| `DensePoly`, `OneHotPoly`, `Root*Source`, compute-backend traits | Polynomial sources and operation capabilities consumed by the scheme |
| `WitnessLayout`, `WitnessUnitLayout` | Canonical digit-innermost group-and-chunk ranges ([opening layout](./proving/opening-points-layout.md)) |
| `AkitaBatchedProof`, `FoldLevelProof`, `TerminalLevelProof` | Structural serialized proof: root fold, recursive folds, and one terminal witness (singleton openings are the 1×1 batched case) |
| `PolynomialGroupClaims` | One commitment group's complete opening point, evaluations, and commitment |
| `OpeningClaims` | Ordered group-owned public claims in transcript order |
| `OpeningClaimsLayout` | Value-free group arities and polynomial counts for setup and schedule lookup |
| `CommittedGroupProfile`, `CommittedGroup` | Source-free public commitment geometry and its commitment rows |
| `PreparedProverGroup` | Coarse borrowed prover group; applications may use one concrete enum polynomial type for heterogeneous representations |
| `ProverOpeningData`, `SelectedProverOpeningData` | Private ordered group-local hint/polynomial records bound to public claims, then paired once with one exact schedule selection |
| `OpeningScheduleSelection`, `GroupBatchStatement` | Exact generated-row identity and verifier-side self-describing opening statement |
| `AkitaTranscript`, `Transcript` | Spongefish-backed Fiat-Shamir layer |
| `AkitaInstanceDescriptor` | Canonical transcript preamble binding algebra, setup, plan, and call shape |
