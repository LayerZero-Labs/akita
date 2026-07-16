# Architecture overview

How the workspace is organized and how a single `commit → prove → verify` call
flows through it.

## Crate map

Workspace members live under `crates/`.
There is **no** `akita-scheme` crate: end-to-end `AkitaCommitmentScheme`
orchestration lives in `akita-pcs`.

| Crate | Role |
|-------|------|
| `jolt-field` (Jolt repository) | Shared field traits, prime/extension fields, unreduced/packed helpers, FFT, parallel macros |
| `akita-error` | Akita protocol error definition |
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

- `jolt-field` is owned by Jolt and imported directly; Akita has no field wrapper crate.
- `akita-planner` sits **below** `akita-config` and names no `CommitmentConfig` type.
- `akita-verifier` depends on `akita-config` and therefore reaches `akita-planner` transitively; the schedule DP is verifier-reachable.
- Verifier-only integrations should use `akita-verifier` + `akita-types` + `akita-config`, not the umbrella `akita-pcs` package.

## End-to-end lifecycle

1. **Preset selection.** The caller picks a `CommitmentConfig` preset (`fp32` / `fp64` / `fp128` families). `CommitmentConfig::runtime_schedule` resolves the recursion schedule from a shipped table or the offline DP (`akita_planner::resolve_schedule`).
2. **Setup.** `akita-setup` expands the config-backed setup (Ajtai matrices, stride envelopes). Setup capacity must cover the requested `num_vars`.
3. **Commit.** `commit` / `batched_commit` (in `akita-prover`, orchestrated by `akita-pcs`) produce commitments over root polynomials at the opening layout implied by the schedule.
4. **Prove.** `batched_prove` walks the resolved schedule level by level: sumcheck stages, fold or direct openings, extension-opening reduction, and recursive suffix work as dictated by each `LevelParams` step. The same batched API dispatches to ZeroFold, terminal-root, and fold+recursive families purely from schedule shape.
5. **Verify.** `batched_verify` re-derives the schedule, replays each level's sumcheck and opening checks, and evaluates the relation matrix at the derived point. Prover and verifier share `bind_transcript_instance_descriptor` so Fiat-Shamir challenges match.

Entry points: `crates/akita-pcs/src/scheme/mod.rs`, `crates/akita-prover/src/protocol/core/prove.rs`, `crates/akita-verifier/src/protocol/core/verify.rs`.

Further reading: [Configuration and planning](./configuration.md), [Proving](./proving/proving.md), [Verification](./verification.md).

## Ring-dimension ownership

The cyclotomic ring dimension is **schedule-derived shape metadata, not a
type parameter of the protocol**. Protocol data — commitments, hints, proofs,
claims, and root polynomial storage (`DensePoly<F>`, `OneHotPoly<F, I>`,
`SparseRingPoly<F>`) — is flat field-element vectors (`RingVec<F>`). Per-level
`CommitmentRingDims` (`d_a` / `d_b` / `d_d` on `LevelParams::role_dims`) is
the operation authority for how those vectors are interpreted; levels may
differ. [`validate_schedule_ring_dims`] checks every level
dimension against the setup's generation dimension.

Every function on the prove/verify path has one of two roles:

- **Orchestration** reads schedule types, drives the transcript, and moves
  D-free storage. It never carries `const D`.
- **Kernels** (NTT, digit decomposition, commit/opening/tensor folds,
  ring-switch arithmetic) are const-generic over `D` and receive extracted
  numbers, never schedule types.

The bridge is the *operation adapter*: a D-free function that extracts the
ring dimension of the specific data one operation touches and enters the
kernel through `akita_types::dispatch_ring_dim_result!` exactly once,
returning D-free storage. Dispatch is per operation — never per level or per
proof — so that per-role ring dimensions inside one fold (`d_a`/`d_b`/`d_d`,
see `specs/mixed-row-ring-dimensions.md`) reduce to feeding different
dimensions to different adapters. `CommitmentRingDims` on `LevelParams::role_dims`
names the per-role dimensions; prove/verify hot paths dispatch on `d_a()`, `d_b()`,
or `d_d()` per operation, not on a single fused dimension.

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
| `CommitmentRingDims`, `validate_schedule_ring_dims` | Per-role ring dimensions and schedule validation |
| `CommitmentConfig` | Single user-facing trait for every per-config policy hook (algebra, SIS family, decomposition, layout, schedule, transcript bind, prove/commitment params). Verifier-reachable hooks return `Result<_, AkitaError>` |
| `LevelParams` | Per-level recursion layout and config (fold shape, ring/ext degrees, decomposition depth, `role_dims`) |
| `PlanPolicy` | Value-typed inputs to `akita_types::schedule_plan_from_table` |
| `PlannerPolicy` | `Cfg`-free projection of a preset for `akita_planner::find_group_batch_schedule`; derive via `akita_config::policy_of::<Cfg>()` |
| `DensePoly`, `OneHotPoly`, `AkitaPolyOps` | Polynomial backends consumed by the scheme |
| `BlockOrder` | Root-vs-recursive opening split convention ([`docs/block-order.md`](../../../docs/block-order.md)) |
| `AkitaBatchedProof`, `AkitaLevelProof`, `AkitaProofStep` | Serialized proof structure (singleton openings are the 1×1 batched case) |
| `OpeningClaims` / `OpeningClaimsLayout` | Public single-point opening claims and layout-only batch geometry for prove/verify, setup, and schedule lookup ([`specs/shared-opening-claims-api.md`](../../../specs/shared-opening-claims-api.md)) |
| `AkitaTranscript`, `Transcript` | Spongefish-backed Fiat-Shamir layer |
| `AkitaInstanceDescriptor` | Canonical transcript preamble binding algebra, setup, plan, and call shape |
