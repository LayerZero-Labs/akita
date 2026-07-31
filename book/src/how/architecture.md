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

- `akita-planner` sits **below** `akita-config` and names no `CommitmentConfig` type.
- `akita-verifier` depends on `akita-config` and therefore reaches `akita-planner` transitively; the schedule DP is verifier-reachable.
- Verifier-only integrations should use `akita-verifier` + `akita-types` + `akita-config`, not the umbrella `akita-pcs` package.

## End-to-end lifecycle

1. **Preset selection.** The caller picks a `CommitmentConfig` preset (`fp32` / `fp64` / `fp128` families). `CommitmentConfig::runtime_schedule` resolves the recursion schedule from a shipped table or the offline DP (`akita_planner::resolve_schedule`). The preset also fixes the protocol geometry: when the claim field coincides with the coefficient field (`EXT_DEGREE == 1`, today's `fp128` families) the fold path never runs extension-opening reduction; when claims live in a proper extension (`fp32` / `fp64`), root EOR follows `akita_types::root_tensor_projection_enabled` and suffix EOR follows `EXT_DEGREE > 1`. See [Fold path and field geometry](./proving/fold-path.md).
2. **Setup.** `akita-setup` expands the config-backed setup (Ajtai matrices, stride envelopes). Setup capacity must cover the requested `num_vars`.
3. **Commit.** `commit` / `batched_commit` (in `akita-prover`, orchestrated by `akita-pcs`) produce commitments over root polynomials at the opening layout implied by the schedule.
4. **Claims.** The caller supplies ordered `PolynomialGroupClaims`; each group owns its complete point, evaluations, and commitment.
5. **Prove.** `batched_prove` walks the folded-only schedule level by level: per-group opening preparation, sumcheck stages, extension-opening reduction, recursive suffix work, and the final direct terminal handoff.
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
claims, and root polynomial storage (`DensePoly<F>`, `OneHotPoly<F, I>`,
`SparseRingPoly<F>`) — is flat field-element vectors (`RingVec<F>`). Per-level
`CommitmentRingDims` (`d_a` / `d_b` / `d_d` on `LevelParams::role_dims`) is
the operation authority for how those vectors are interpreted; levels may
differ. Here, *role* is the historical protocol name for a commitment matrix's
fixed job: A carries the relation witness, B commits the next witness, and D
commits the opening digits. The matrices do not switch roles when their ring
dimensions change. User-facing prose therefore calls a non-uniform tuple such
as `128/64/32` **per-matrix ring dimensions** and a change between levels a
**ring-dimension transition**. [`validate_schedule_ring_dims`] checks
every level dimension against the setup's generation dimension.

Every function on the prove/verify path has one of two roles:

- **Orchestration** reads schedule types, drives the transcript, and moves
  D-free storage. It never carries `const D`.
- **Kernels** (NTT, digit decomposition, commit/opening/tensor folds,
  ring-switch arithmetic) are const-generic over `D` and receive extracted
  numbers, never schedule types.

The bridge is the *operation adapter*: a D-free function that extracts the
ring dimension of the specific data one operation touches and enters the
kernel through `akita_types::dispatch_for_field!` exactly once,
returning D-free storage. Dispatch is per operation — never per level or per
proof — so that per-matrix ring dimensions inside one fold (`d_a`/`d_b`/`d_d`,
see `specs/mixed-row-ring-dimensions.md`) reduce to feeding different
dimensions to different adapters. `CommitmentRingDims` on `LevelParams::role_dims`
names the per-matrix ring dimensions; prove/verify hot paths dispatch on
`d_a()`, `d_b()`, or `d_d()` per operation, not on a single fused dimension.

The normative contract (discriminator rule, forbidden facade/level-
monomorphization patterns) lives in `specs/runtime-ring-cutover.md`.
Mixed-dimension execution is exercised end-to-end by
`crates/akita-pcs/tests/mixed_d_per_level_e2e.rs` and
`crates/akita-verifier/tests/mixed_d_rejections.rs` through the normal public API.

## Core types

| Type | Role |
|------|------|
| `AkitaCommitmentScheme<Cfg>` | Top-level PCS `commit` / `prove` / `verify` orchestration (`akita-pcs`) |
| `AkitaProverSetup<F>` | Prover setup wrapper; `gen_ring_dim` is runtime shape metadata |
| `Commitment<F>`, `RingVec<F>` | protocol commitment and field-vector storage |
| `CommitmentRingDims`, `validate_schedule_ring_dims` | A/B/D commitment-matrix ring dimensions and schedule validation |
| `CommitmentConfig` | Single user-facing trait for every per-config policy hook (algebra, exact SIS profile, decomposition, layout, schedule, transcript bind, prove/commitment params). Verifier-reachable hooks return `Result<_, AkitaError>` |
| `LevelParams` | Per-level recursion layout and config (fold shape, ring/ext degrees, decomposition depth, `role_dims`) |
| `PlannerPolicy` | `Cfg`-free projection of a preset for `akita_planner::find_group_batch_schedule`; derive via `akita_config::policy_of::<Cfg>()` |
| `DensePoly`, `OneHotPoly`, `Root*Source`, compute-backend traits | Polynomial sources and operation capabilities consumed by the scheme |
| `WitnessLayout`, `WitnessUnitLayout` | Canonical digit-innermost group-and-chunk ranges ([opening layout](./proving/opening-points-layout.md)) |
| `AkitaBatchedProof`, `FoldLevelProof`, `TerminalLevelProof` | Structural serialized proof: root fold, recursive folds, and one terminal witness (singleton openings are the 1×1 batched case) |
| `PolynomialGroupClaims` | One commitment group's complete opening point, evaluations, and commitment |
| `OpeningClaims` | Ordered group-owned public claims in transcript order |
| `OpeningClaimsLayout` | Value-free group arities and polynomial counts for setup and schedule lookup |
| `CommittedGroupProfile`, `CommittedGroup` | Source-free public commitment geometry and its commitment rows |
| `WholeGroupSourceProvider`, `PreparedProverGroup`, `EitherPreparedGroup` | Prover-only group validation and heterogeneous whole-group execution with monomorphized kernels |
| `ProverGroupInput`, `ProverOpeningData` | Ordered group-local hint/source records bound to public claims and one exact schedule selection |
| `OpeningScheduleSelection`, `GroupBatchStatement` | Exact generated-row identity and verifier-side self-describing opening statement |
| `AkitaTranscript`, `Transcript` | Spongefish-backed Fiat-Shamir layer |
| `AkitaInstanceDescriptor` | Canonical transcript preamble binding algebra, setup, plan, and call shape |
