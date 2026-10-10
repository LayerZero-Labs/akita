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
| `akita-serialization` | Serialization, validation, and compression traits |
| `akita-algebra` | Modules, vectors, NTTs, cyclotomic rings, sparse challenges, polynomials |
| `akita-transcript` | Spongefish-backed Fiat-Shamir transcript, descriptor preamble, logging checks |
| `akita-challenges` | Fiat-Shamir challenge sampling helpers |
| `akita-sumcheck` | Sumcheck proofs, drivers, compact folding, batching, accumulation |
| `akita-params` | Parameter geometry, sizing, SIS tables, schedules, compression plans, witness layout, dispatch, and grinding plans |
| `akita-types` | Proof, setup, and claim wire values, shared protocol math, and native transcript replay |
| `akita-planner` | `Cfg`-free offline schedule search and artifact emission |
| `akita-schedules` | Versioned schedule artifacts, semantic row audit, and validated owned catalogs |
| `akita-config` | Runtime presets, the `CommitmentConfig` trait, trusted artifact loading, `policy_of::<Cfg>()`, and transcript binding |
| `akita-setup` | Config-backed setup construction and optional setup cache |
| `akita-verifier` | Verifier replay without prover-only polynomial backends; directly `<Cfg>`-generic |
| `akita-prover` | Protocol sequencing, public geometry checks, proof assembly, opaque operation contracts |
| `akita-cpu-backend` | Backend-owned sources, commitment and witness arithmetic, prepared setup resources, caches |
| `akita-pcs` | Umbrella crate: `AkitaCommitmentScheme`, re-exports, examples, benches, integration tests |
| `akita-labinius-verifier` (`dev`) | Opt-in LaBinius extension: admitted root setup, lowered root relation, and verification of the root reduction |
| `akita-labinius-prover` (`dev`) | Opt-in LaBinius extension: binary source commitment, lowered root witness, and the root reduction prover |

**Dependency graph and ownership rules:** [`docs/crate-graph.md`](../../../docs/crate-graph.md).
CI enforces one-way boundaries via `scripts/check-crate-deps.sh`.
The LaBinius prover crate depends on its verifier crate, never the reverse.
No Akita proof-path crate depends on this opt-in extension.

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
  `akita-params`. It receives a validated trusted catalog and never reaches
  planner search.
- Verifier-only integrations should use `akita-verifier` + `akita-types` + `akita-config`, not the umbrella `akita-pcs` package.

## End-to-end lifecycle

1. **Preset and trusted catalog selection.** The caller picks a `CommitmentConfig` preset and loads the matching schedule artifact from the same trusted parameter source used for setup or preprocessing. The caller constructs one `AkitaCommitmentScheme<Cfg>` with that catalog. The catalog contains complete expanded rows. Planner search remains offline. Each row selects `SubringCoefficientPacking` or `EvaluationTrace` for every nonterminal fold. EOR is present only for an evaluation trace opening over a proper extension field. See [Fold path and field geometry](./proving/fold-path.md).
2. **Setup.** `akita-setup` scans the rows in the scheme's trusted catalog and expands the setup (Ajtai matrices and stride envelopes) to cover the requested capacity.
3. **Commit.** The application consumes its polynomials into a reusable `CpuBackend` and commits the imported source using `GroupContext`. The result contains a public commitment and an opaque handle retaining the exact source and commitment parameters. Scheduler mode selects the scalar row when the group has no precommitted groups, or the exact grouped row when it does. Explicit mode validates caller-supplied root parameters. A group committed under a scalar row may later be supplied as a precommitted group.
4. **Claims.** The caller supplies ordered `PolynomialGroupClaims`; each group owns its complete point, evaluations, and commitment.
5. **Prove.** `batched_prove` receives retained commitment handles and public claims. The backend checks that each claim matches its commitment before preparing a fresh proof session. The generic prover walks the schedule, absorbs backend messages, samples challenges, and assembles the proof. Source coefficients and witness arithmetic remain inside the backend.
6. **Verify.** `batched_verify` resolves the proof row digest in the trusted catalog, replays nonterminal sumchecks and relation-matrix evaluations, then closes the terminal with direct consistency/A and weighted trace checks. The proof never supplies schedule bytes. Prover and verifier share `bind_transcript_instance_descriptor` so Fiat-Shamir challenges match.

Entry points: `crates/akita-pcs/src/scheme/mod.rs`, `crates/akita-prover/src/protocol/prove/root.rs`, `crates/akita-verifier/src/fold/verify.rs`.

Further reading: [Configuration and planning](./configuration.md), [Setup
offloading](./setup-offloading.md), [Proving](./proving/proving.md), and
[Verification](./verification.md).

Recursive setup offloading adds one setup-only Stage 3 message group at each
nonterminal producer whose successor consumes a setup prefix.
Its wire payload is the setup claim, the setup-prefix evaluation, and one
degree-two sumcheck over the native setup domain.
Its round count and planned size do not depend on the successor witness length.
The [setup offloading chapter](./setup-offloading.md) follows this path from
offline planning through the recursive verifier handoff.

## Prover consumer boundary

Protocol orchestration passes private state as associated handles through
`OpaqueProverConsumer`. It owns transcript order, public plans, commitments,
claims, and round messages. The selected consumer owns opening buffers, fold
responses, compression witnesses, recursive witnesses, and mutable sumcheck
and extension-opening sessions.

One `CpuBackend` implements the opaque contracts directly. Physical routing,
prepared setup storage, and cache policy live in `akita-cpu-backend`. The
generic prover has no production dependency on that crate. Source import and
persistence are application/backend operations. Setup prefixes enter generic
proving as ordinary reusable commitment handles with checked public metadata.

A shared `ConsumerIdentity` binds all fold levels to the same setup. A
`ProofContext` identifies an active proof, fold level, and source group before
private preparation starts. Witness assembly checks those identities before
consuming its inputs, and checks that each accepted fold carries its original
public challenges. EOR openings also bind to the exact witness operation that
produced them. Finishing or abandoning a proof invalidates its retained scope
leases; round operations check those leases without locking the scope registry.

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
kernel through `akita_params::dispatch_for_field!` exactly once,
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
| `DensePoly`, `OneHotPoly`, `CommitmentSource`, `CommitmentExecutor` | D-free polynomial storage, commitment representations, and the checked split-or-fused commitment boundary |
| `ProverBackend`, focused opaque kernel traits | Public protocol operations with backend-owned opening, EOR, fold, and sumcheck state |
| `WitnessLayout`, `WitnessUnitLayout` | Canonical digit-innermost group-and-chunk ranges ([opening layout](./proving/opening-points-layout.md)) |
| `Vec<u8>` returned by `batched_prove` | Canonical Spongefish argument stream, consumed in protocol order by `batched_verify` with EOF and grinding-plan completion |
| `PolynomialGroupClaims` | One commitment group's complete opening point, evaluations, and commitment |
| `OpeningClaims` | Ordered group-owned public claims in transcript order |
| `OpeningClaimsLayout` | Value-free group arities and polynomial counts for setup and schedule lookup |
| `GroupCommitPhaseParams`, `CommittedGroup` | Source-free public commitment geometry and its commitment rows |
| `SourceHandle`, `CommitmentHandle` | Backend-owned immutable source and reusable source-retaining commitment |
| `ProverOpeningData`, `SelectedProverOpeningData` | Ordered opaque commitment handles bound to public claims and one exact schedule selection |
| `OpeningScheduleSelection`, `GroupBatchStatement` | Exact generated-row identity and verifier-side self-describing opening statement |
| `ValidatedScheduleCatalog` | Config-free, semantically audited expanded rows with canonical lookup indexes and artifact I/O |
| `TrustedScheduleCatalog<Cfg>` | Config-bound trusted parameter passed to setup, prover, and verifier APIs |
| Spongefish prover/verifier states | Fiat--Shamir state, proof emission/receipt, domain separation, challenges, and EOF checking |
| `AkitaInstanceDescriptor` | Canonical transcript preamble binding algebra, setup, plan, and call shape |

Opening batch kernels validate one authoritative challenge partition against every
source and return one aggregate witness per requested chunk. The protocol combines
their `z` values into the global fold witness; non-fused backends reuse the public
checked aggregator within each chunk.

## Per-fold execution ownership

The `batched_prove` API keeps one transcript while assigning each
successor fold to a registered backend instance. A typed executor owns its local
session and setup-prefix handles. The coordinator carries a public continuation
(level, commitment binding, challenges, and openings) separately from an opaque,
owner-tagged witness/material pair. Handle pairs and transfer packets are erased
only at dispatch; all arithmetic kernels keep their associated handle types.

## Backend routing

Backend routing is always available through `batched_prove`; it requires no
Cargo feature. A fold reduces a private witness to a smaller successor witness.
Before preparing any executor, a caller-defined policy resolves the owner of
every fold. Each producer commits its successor once. On a switch, the source
exports a typed packet, a registered bridge converts it, and the destination
imports local handles synchronously. The source finishes its sumchecks before
the destination executes its fold. The terminal fold can have its own owner as
well.

Construct `BackendRegistry<Cfg>`, register each backend with its local
`akita_prover::SetupPrefixProverRegistry`. Registration returns a `BackendId`.
Pass the registry, typed opening inputs, and route to `batched_prove`.
`FixedFoldRoute::new(vec![a, b, a])` assigns folds 0, 1, and 2 to those registry
entries. Custom `FoldExecutionPolicy` implementations select every level through
`choose_backend`, called once per level in order before preparation; its current
owner is absent at level 0. Choices remain fixed throughout the proof. The
registry dispatches the root to the selected executor, checks its commitment
handle family, and validates that the handles belong to that instance. The route
stores only IDs. Each instance supplies the prefix handles needed by its folds
and successor commitments. Registration builds a union of public prefix slots
and rejects conflicting public values for a shared slot ID. Public agreement is
checked once for the immutable registrations; each selected executor must still
have its required local handles. Registration also rejects duplicate instances;
IDs from another registry reject. Route implementations own any cost model;
Akita does not request or compare cost estimates.

Backends implement `SuccessorExportKernel` and `SuccessorImportKernel` alongside
ordinary typed kernels. Backend-provided conversions implement `SuccessorBridge`
on `Edge<A, B>`. Register the bridge explicitly with
`registry.register_bridge::<A, B>()?` before selecting a switch. Same-type
instance switches (two `CpuBackend`s) still need
`register_bridge::<CpuBackend<F, E>, CpuBackend<F, E>>()?`. Heterogeneous routes
register each directed edge (for example `PrivateCpu` ↔ `CpuBackend`). For an
application-owned converter, use `registry.register_bridge_with::<A,
B>(convert)?`. It accepts a function or closure from `A::ExportPacket` and the
handoff plan to `B::ImportPacket`, allowing applications to connect
dependency-owned backends without implementing a foreign trait. Both
registration APIs share the same type-pair key and reject duplicates. A missing
bridge or foreign ID rejects before any executor is prepared. Only selected
instances are prepared, and their prefix handles and fold capacities are checked
before opening the transcript. Akita never silently reroutes a proof. A single
instance needs no bridge: same-ID choices retain native handles without export,
conversion, or import.

The handoff is blocking: export a typed packet, convert it to the destination's
packet, then validate and import both handles. Every import packet exposes
`HandoffMetadata`: the identity of this transfer and the logical witness
geometry. The destination checks this metadata against the handoff plan before
adopting private payloads. The coordinator dispatches typed bridges without
prescribing payload encoding. CPU self-edges share immutable packed witness and
material storage. External bridges can construct
`CpuImportPacket::new(descriptor, sections)` with packed or signed digits,
canonical field coefficients, and compression sections described by
`CpuPacketDescriptor`. These section types and encodings belong to
`akita_cpu_backend`; other backend pairs can use their own packet formats. CPU
exports provide `into_sections()` for conversion to another representation.
Import checks section lengths, the admitted handoff, digit bounds, and canonical
encodings. Portable packets carry logical witness digits only; CPU import
derives the tensor representation when needed. Native CPU self-edges retain the
private transformed cache through shared storage. For outer successors, import
trusts that the bridge's inner rows and compression material reproduce the
public commitment; it checks geometry and canonical encoding, and the verifier
checks consistency. Terminal successors also cross-check the imported inner rows
against the public fields. Switching retains the producer's handles until its
sumchecks complete. The CPU scheme method uses this same prover with every level
assigned to its supplied backend. Routing does not change the setup, commitment,
transcript, or proof format.

Before any proof computation, the coordinator calls `prepare_executor` on each
selected backend to open its local proof session. It then calls `begin_fold` for
every assigned level to validate that level and its successor commitment work
before opening the transcript. Preparation does not require a backend to execute
unassigned levels. All prepared sessions use `ProofScope` to finish on success
or abort on failure, including scopes whose completion fails. A registry
supports repeated proofs, with one active proof at a time.

Implementation: `crates/akita-prover/src/protocol/prove/registry.rs`,
`crates/akita-prover/src/protocol/prove/execution.rs`,
`crates/akita-prover/src/backend/transfer.rs`, and
`crates/akita-cpu-backend/src/opaque/transfer.rs`. Design record:
`specs/dynamic-backend-bridges.md`. Regression coverage:
`crates/akita-pcs/tests/dynamic_backends.rs`.
