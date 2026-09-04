# Architecture overview

How the workspace is organized and how a single `commit → prove → verify` call
flows through it.

## Crate map

Workspace members live under `crates/`.
There is **no** `akita-scheme` crate: end-to-end `AkitaCommitmentScheme`
orchestration lives in `akita-pcs`.

| Crate | Role |
|-------|------|
| `akita-error` | Shared protocol error and reusable checked integer formulas for exact sizes, offsets, and ranges |
| `jolt-field` (external) | Shared field traits, prime and extension fields, packed and unreduced kernels, parallel helpers |
| `akita-witness` | Shared borrowed witness/polynomial view vocabulary (`PolynomialView`, `WitnessProvider`) for sumcheck and polyops paths |
| `akita-serialization` | Serialization, validation, and compression traits |
| `akita-algebra` | Modules, vectors, NTTs, cyclotomic rings, sparse challenges, polynomials |
| `akita-transcript` | Spongefish-backed Fiat-Shamir transcript, descriptor preamble, logging checks |
| `akita-challenges` | Fiat-Shamir challenge sampling helpers |
| `akita-sumcheck` | Sumcheck proofs, drivers, compact folding, batching, accumulation |
| `akita-types` | Proof, setup, schedule, layout, commitment, and transcript-append shapes; SIS floors; layout and proof-size helpers |
| `akita-planner` | `Cfg`-free offline schedule search and artifact emission |
| `akita-schedules` | Versioned schedule artifacts, semantic row audit, and validated owned catalogs |
| `akita-config` | Runtime presets, the `CommitmentConfig` trait, trusted artifact loading, `policy_of::<Cfg>()`, and transcript binding |
| `akita-setup` | Config-backed setup construction and optional setup cache |
| `akita-verifier` | Verifier replay without prover-only polynomial backends; directly `<Cfg>`-generic |
| `akita-prover` | Commitment, proving, setup expansion, witnesses, polynomial backends, compute operation traits |
| `akita-pcs` | Umbrella crate: `AkitaCommitmentScheme`, re-exports, examples, benches, integration tests |

**Dependency graph and ownership rules:** [`docs/crate-graph.md`](../../../docs/crate-graph.md).
CI enforces one-way boundaries via `scripts/check-crate-deps.sh`.

Key structural facts:

- `akita-error` is the lowest shared failure layer. It owns `AkitaError` and
  reusable exact `usize` formulas in `akita_error::checked`. These formulas
  return `Option`; each caller maps failure to the `AkitaError` variant that
  describes its protocol boundary. Field arithmetic lives in the shared
  external `jolt-field` crate.
- `akita-planner` owns offline schedule search and artifact emission. Normal
  search is `Cfg`-free and is not on the verifier runtime dependency path. The
  optional `catalog-gen` feature enables `akita-config`, so artifact-emission
  binaries may name concrete `CommitmentConfig` presets.
- `akita-verifier` depends on `akita-config`, `akita-schedules`, and
  `akita-types`. It receives a validated trusted catalog and never reaches
  planner search.
- Verifier-only integrations should use `akita-verifier` + `akita-types` + `akita-config`, not the umbrella `akita-pcs` package.

## End-to-end lifecycle

1. **Preset and trusted catalog selection.** The caller picks a `CommitmentConfig` preset and loads the matching schedule artifact from the same trusted parameter source used for setup or preprocessing. The caller constructs one `AkitaCommitmentScheme<Cfg>` with that catalog. The catalog contains complete expanded rows. Planner search remains offline. Each row selects `SubringCoefficientPacking` or `EvaluationTrace` for every nonterminal fold. EOR is present only for an evaluation trace opening over a proper extension field. See [Fold path and field geometry](./proving/fold-path.md).
2. **Setup.** `akita-setup` scans the rows in the scheme's trusted catalog and expands the setup (Ajtai matrices and stride envelopes) to cover the requested capacity.
3. **Commit.** The context-aware `commit` entry point (in `akita-prover`, orchestrated by the same scheme instance) produces one committed polynomial group using `GroupContext`. Scheduler mode selects the scalar row when the group has no precommitted groups, or the exact grouped row when it does. Explicit mode validates caller-supplied root parameters. A group committed under a scalar row may later be supplied as a precommitted group.
4. **Claims.** The caller supplies ordered `PolynomialGroupClaims`; each group owns its complete point, evaluations, and commitment.
5. **Prove.** `batched_prove` walks the schedule level by level. It prepares each group with the scheduled opening method, runs the sumchecks, performs EOR when required, and hands the last folded witness to the direct terminal proof.
6. **Verify.** `batched_verify` resolves the proof row digest in the trusted catalog, replays nonterminal sumchecks and relation-matrix evaluations, then closes the terminal with direct consistency/A and weighted trace checks. The proof never supplies schedule bytes. Prover and verifier share `bind_transcript_instance_descriptor` so Fiat-Shamir challenges match.

Entry points: `crates/akita-pcs/src/scheme/mod.rs`, `crates/akita-prover/src/protocol/core/prove.rs`, `crates/akita-verifier/src/protocol/core/verify.rs`.

Further reading: [Configuration and planning](./configuration.md), [Setup
offloading](./setup-offloading.md), [Proving](./proving/proving.md), and
[Verification](./verification.md).

Recursive setup offloading adds one setup-only `SetupSumcheckProof` at each
nonterminal producer whose successor consumes a setup prefix.
Its wire payload is the setup claim, the setup-prefix evaluation, and one
degree-two sumcheck over the native setup domain.
Its round count and planned size do not depend on the successor witness length.
The [setup offloading chapter](./setup-offloading.md) follows this path from
offline planning through the recursive verifier handoff.

## Ring-dimension ownership

The cyclotomic ring dimension is **schedule-derived shape metadata, not a
type parameter of the protocol**. Protocol data — commitments, hints, proofs,
claims, and root polynomial storage (`DensePoly<F>`, `OneHotPoly<F, I>`, and
their enum wrapper) — is flat field-element vectors (`RingVec<F>`). Per-level
`CommitmentRingDims` (`d_a` / `d_b` / `d_d` from
`CommittedGroupParams::role_dims()`) is
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
dimensions to different adapters. `CommittedGroupParams::role_dims()` names
the per-matrix ring dimensions; prove and verify hot paths dispatch on
`d_a()`, `d_b()`, or `d_d()` per operation, not on a single fused dimension.

The normative contract (discriminator rule, forbidden facade/level-
monomorphization patterns) lives in `specs/runtime-ring-cutover.md`.
Mixed-dimension malformed proof rejection is covered by
`crates/akita-verifier/tests/mixed_d_rejections.rs` through the verifier API.

## Core types

| Type | Role |
|------|------|
| `AkitaError`, `akita_error::checked` | Shared protocol failures and reusable checked formulas for sizes, offsets, ranges, alignment, and exact division |
| `AkitaCommitmentScheme<Cfg>` | Stateful top-level PCS orchestration that owns one trusted catalog for setup, commitment, proving, and verification (`akita-pcs`) |
| `AkitaProverSetup<F>` | Prover setup wrapper around a materialized prefix of the dimension-free public field stream |
| `Commitment<F>`, `RingVec<F>` | protocol commitment and field-vector storage |
| `CommitmentRingDims`, `validate_schedule_ring_dims` | A/B/D commitment-matrix ring dimensions and schedule validation |
| `CommitmentConfig` | Single user-facing trait for every per-config policy hook (algebra, exact SIS profile, decomposition, layout, schedule, transcript bind, prove/commitment params). Verifier-reachable hooks return `Result<_, AkitaError>` |
| `CommittedGroupParams` | One fold's ordered groups, shared D matrix, payload mode, source encoding, and witness chunk layout |
| `FoldParams`, `TerminalFoldParams`, `FoldSchedule` | Verifier-visible nonterminal, terminal, and complete schedule structure |
| `PlannerPolicy` | `Cfg`-free projection of a preset for `akita_planner::find_schedule`; derive via `akita_config::policy_of::<Cfg>()` |
| `DensePoly`, `OneHotPoly`, `Root*Source`, compute-backend traits | Polynomial sources and operation capabilities consumed by the scheme |
| `WitnessLayout`, `WitnessUnitLayout` | Canonical digit-innermost group-and-chunk ranges ([opening layout](./proving/opening-points-layout.md)) |
| `AkitaBatchedProof`, `FoldLevelProof`, `TerminalLevelProof` | Structural serialized proof: root fold, recursive folds, and one terminal witness (singleton openings are the 1×1 batched case) |
| `PolynomialGroupClaims` | One commitment group's complete opening point, evaluations, and commitment |
| `OpeningClaims` | Ordered group-owned public claims in transcript order |
| `OpeningClaimsLayout` | Value-free group arities and polynomial counts for setup and schedule lookup |
| `GroupCommitPhaseParams`, `CommittedGroup` | Source-free public commitment geometry and its commitment rows |
| `PreparedProverGroup` | Coarse borrowed prover group; applications may use one concrete enum polynomial type for heterogeneous representations |
| `ProverOpeningData`, `SelectedProverOpeningData` | Private ordered group-local hint/polynomial records bound to public claims, then paired once with one exact schedule selection |
| `OpeningScheduleSelection`, `GroupBatchStatement` | Exact generated-row identity and verifier-side self-describing opening statement |
| `TrustedScheduleCatalog` | Validated trusted parameter that owns expanded rows and serves both prover key lookup and verifier digest lookup |
| `AkitaTranscript`, `Transcript` | Spongefish-backed Fiat-Shamir layer |
| `AkitaInstanceDescriptor` | Canonical transcript preamble binding algebra, setup, plan, and call shape |
