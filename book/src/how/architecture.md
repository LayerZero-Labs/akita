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
| `akita-r1cs` | Deferred R1CS relations for the `zk` path only (not on the transparent path) |
| `akita-pcs` | Umbrella crate: `AkitaCommitmentScheme`, re-exports, examples, benches, integration tests |

**Dependency graph and ownership rules:** [`docs/crate-graph.md`](../../../docs/crate-graph.md).
CI enforces one-way boundaries via `scripts/check-crate-deps.sh`.

Key structural facts:

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

## Core types

| Type | Role |
|------|------|
| `AkitaCommitmentScheme` | Top-level PCS `commit` / `prove` / `verify` orchestration (`akita-pcs`) |
| `CommitmentConfig` | Single user-facing trait for every per-config policy hook (algebra, SIS family, decomposition, layout, schedule, transcript bind, prove/commitment params). Verifier-reachable hooks return `Result<_, AkitaError>` |
| `LevelParams` | Per-level recursion layout and config (fold shape, ring/ext degrees, decomposition depth) |
| `PlanPolicy` | Value-typed inputs to `akita_types::schedule_plan_from_table` |
| `PlannerPolicy` | `Cfg`-free projection of a preset for `akita_planner::find_schedule`; derive via `akita_config::policy_of::<Cfg>()` |
| `DensePoly`, `OneHotPoly`, `AkitaPolyOps` | Polynomial backends consumed by the scheme |
| `BlockOrder` | Root-vs-recursive opening split convention ([`docs/block-order.md`](../../../docs/block-order.md)) |
| `AkitaBatchedProof`, `AkitaLevelProof`, `AkitaProofStep` | Serialized proof structure (singleton openings are the 1×1 batched case) |
| `OpeningBatch` | Single-point batch descriptor for batched prove/verify ([`specs/single-point-opening-batch.md`](../../../specs/single-point-opening-batch.md)) |
| `AkitaTranscript`, `Transcript` | Spongefish-backed Fiat-Shamir layer |
| `AkitaInstanceDescriptor` | Canonical transcript preamble binding algebra, setup, plan, and call shape |
