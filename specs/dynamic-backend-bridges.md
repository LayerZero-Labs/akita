# Spec: Dynamic Backend Bridges

| Field         | Value                                            |
|---------------|--------------------------------------------------|
| Author(s)     | Amir Kh                                          |
| Created       | 2026-10-07                                       |
| Status        | implemented                                      |
| PR            | [#193](https://github.com/LayerZero-Labs/akita/pull/193) |
| Book-chapter  | book/src/how/architecture.md                  |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14 when,
and only when, they appear in all capitals.

## Summary

Previously, one backend executed every fold of a proof. Akita now assigns each
fold level to a registered prover backend while keeping homogeneous proof bytes
unchanged. Typed export and import packets carry the private continuation through
application-registered bridges. Routing uses instance `BackendId`s; packet
conversion is keyed by backend types.

## Intent

### Goal

Allow one `batched_prove` call to execute successive folds on different
`ProverBackend` instances (including distinct handle families) while keeping:

- setup, commitment, transcript, and proof bytes identical to a homogeneous run
  of the same schedule;
- protocol code free of private witness layouts;
- external backends able to bridge to and from `CpuBackend` using only public
  packet constructors.

### Key abstractions

| Abstraction | Role |
| --- | --- |
| `BackendRegistry` | Registers backend instances and type-keyed bridges |
| `BackendId` | Runtime owner of a fold level |
| `FoldExecutionPolicy` | Resolves all fold owners before preparation (e.g. `FixedFoldRoute`) |
| `SuccessorExportKernel` | Source backend → owned `ExportPacket` |
| `SuccessorImportKernel` | Destination backend ← `ImportPacket` → local handles |
| `HandoffMetadata` | Packet identity and witness geometry, independent of encoding |
| `SuccessorBridge` on `Edge<A, B>` or application converter | `A::ExportPacket` → `B::ImportPacket` |
| `FoldHandoff` | After successor commit, retain local handles or export→convert→import to the resolved next owner |
| `CpuExportPacket` / `CpuImportPacket` | Public CPU sectioned IR |

### Invariants

1. **Same-instance short-circuit.** When the policy selects the current
   `BackendId`, the coordinator MUST NOT export, convert, or import. It MUST
   prepare that fold on its selected instance before proving. Protected by
   homogeneous routes in `crates/akita-pcs/tests/dynamic_backends.rs`.
2. **Bridge required on type switch.** When the selected id differs from the
   current id, the registry MUST look up
   `(TypeId::of::<Src>(), TypeId::of::<Dst>())`. A missing bridge MUST reject
   the route before any executor is prepared.
3. **Type-keyed, not id-keyed, bridges.** `register_bridge::<A, B>()` registers
   one directed conversion for all instances of those types. Distinct instances
   of the same type (two `CpuBackend`s) still NEED
   `register_bridge::<CpuBackend<F, E>, CpuBackend<F, E>>()`.
   Applications MAY instead register a typed function or closure through
   `register_bridge_with::<A, B>(convert)`, including when both backend types
   come from dependencies. Both APIs share the same key and reject duplicates.
4. **Handoff precedes producer sumchecks.** After the producer commits the
   successor witness, `FoldHandoff::handoff_successor` MUST run before that
   producer's stage-1/2/3 work. Every selected fold MUST be prepared before the
   transcript is opened; the destination MUST NOT run fold sumchecks until
   `batched_prove` dispatches it via `resident.owner`.
5. **Wire invariance.** Routing MUST NOT change setup, commitment, proof, or
   transcript bytes relative to a same-schedule homogeneous proof. Protected by
   byte-equality checks against a homogeneous reference in `dynamic_backends.rs`.
6. **Public CPU adopt IR.** External bridges to CPU MUST be able to build
   `CpuImportPacket` through `CpuImportPacket::new(descriptor, sections)`
   without naming private CPU handles. CPU import MUST validate plan shape,
   encodings, and digit bounds. Portable digit payloads MUST contain only logical
   witness digits; CPU import MUST derive any tensor representation locally.
   Native CPU self-edges MAY share the private CPU-produced transformed cache.
   Portable outer successors trust the bridge to supply inner rows and compression
   material consistent with the public commitment; import validates geometry and
   canonicity, while the verifier checks commitment consistency. Terminal imports
   MUST also cross-check the imported inner rows against the public fields.
7. **Session isolation.** Export borrows the live producer session; import runs
   on the destination executor's prepared session. The coordinator MUST NOT
   re-enter the producer's executor slot for export.
8. **Public prefix union.** Registration MUST merge public prefix slots from all
   registered instances and reject conflicting values for the same slot ID
   without changing the registry. This establishes public agreement for the
   lifetime of the immutable registrations; selected executors MUST retain
   local handles for their fold and successor-commit work.
9. **Packet format ownership.** The destination MUST validate `HandoffMetadata`
   against the admitted plan before adoption. CPU imports MUST additionally validate
   `CpuPacketDescriptor` section encodings and lengths before allocation or
   decoding. Other backends MAY use packets without portable byte sections.
10. **Eager route validation.** The coordinator MUST call the policy once per
    level, in order, before preparing executors. It MUST validate every selected
    ID and every required bridge, prepare only selected instances, and validate
    their local prefix handles and fold capacity before opening the transcript.
    Policy choices are fixed for the proof; they cannot depend on fold progress.

### Non-Goals

- Automatic discovery of bridges; every directed type edge is explicit.
- Cost models or scheduler heuristics inside Akita (routes own policy).
- Changing verifier behavior or proof format.
- Sharing one backend instance across different `(F, E)` pairs.
- Streaming or async handoff; transfer remains synchronous and blocking.

## Evaluation

### Acceptance Criteria

- [x] Backends implement `SuccessorExportKernel` and `SuccessorImportKernel`
  to exchange committed successor state.
- [x] `Edge<A, B>: SuccessorBridge` or an application callback supplies `convert`;
  both registration APIs use the same coordinator path
  (`export_successor` → `convert` → `import_successor`).
- [x] `BackendRegistry::register_bridge::<A, B>()` keys by `(TypeId<A>, TypeId<B>)`.
- [x] Same `BackendId` skips transfer; different ids of the same type still
  transfer through the registered self-edge.
- [x] Heterogeneous `PrivateCpu` ↔ `CpuBackend` routes match homogeneous proof
  bytes (`dynamic_backends.rs`).
- [x] Integration test `dynamic_backends.rs` alternates `PrivateCpu` and
  `CpuBackend` with both directed bridges registered.
- [x] Book usage documents `register_bridge` and packet handoff
  (`book/src/how/architecture.md`).

### Testing Strategy

- `crates/akita-pcs/tests/dynamic_backends.rs` — four one-hot polynomials,
  PrivateCpu → CPU routing through portable packets and CPU → PrivateCpu routing
  through the native CPU edge, homogeneous proof-byte equality, and verification.
- `crates/akita-prover/tests/external_backend.rs` — external backend still
  compiles against the new kernels (stub export/import).
- `crates/akita-pcs/tests/dynamic_backends/portable.rs` — portable fp32 tensor
  successors and offloaded setup with executor-local prefixes, checked against
  verified homogeneous proof bytes. Missing local prefixes reject before transfer.
- `dynamic_backend_failures_cleanup_and_retry` — eager route rejection,
  preparation and handoff failures, cleanup, and reuse of the same registry.

### Performance

Handoff is not on the verifier path. Prover overhead is one typed packet copy
(or CPU native self-edge conversion) per backend switch. No proof-size or
security-parameter change is expected; equality tests against homogeneous
proofs are the gate.

## Design

### Architecture

```text
batched_prove
  └─ resolve all owners; validate IDs and bridges
  └─ prepare selected executors and their assigned folds
  └─ registry.slot(owner).root / .recursive / .terminal
        └─ prove_fold(..., handoff callback)
              ├─ commit successor witness
              ├─ FoldHandoff::handoff_successor
              │     ├─ same BackendId → retain local handles
              │     └─ else registered typed converter
              │           export → convert → import → ResidentSuccessor
              └─ producer sumchecks
  └─ next fold uses resident.owner
```

**Crate layout**

| Location | Owns |
| --- | --- |
| `akita-prover/src/backend/transfer.rs` | Kernels, `Edge`, packet traits, handoff metadata |
| `akita-prover/src/protocol/prove/registry.rs` | Registry, bridges map, `FoldHandoff` |
| `akita-prover/src/protocol/prove/execution.rs` | `TypedExecutor`, `batched_prove` |
| `akita-cpu-backend/src/opaque/transfer.rs` | CPU packets, portable section format, CPU↔CPU bridge, import validation |

Typed folds export directly from their live producer session after checking the
bridge exists. Each prepared executor stores a `ProofScope`, which owns scope
completion and abort cleanup, including when completion itself fails.

**Why type-keyed bridges.** Convert depends only on packet types. Instance
identity is already carried by `BackendId` when choosing who runs the fold and
which session imports. Registering `(A, B)` once covers every A→B instance pair.

**Why not pairwise typed handles in the protocol.** The registry is type-erased
after `register`. Dynamic `BackendId` routes cannot name `A::WitnessHandle` at
the coordinator. Packets are the shared IR; bridges are the typed adapters.

### Alternatives Considered

1. **Global byte-section pull registry.** Rejected: destinations would pull bytes
   through a shared API, forcing every source to implement that representation.
2. **Exporter builds destination handles (`ExportToward<B>`).** Rejected:
   sources would depend on every destination type (orphan-rule and N×M
   coupling). Applications instead register a backend-provided `Edge<A, B>`
   bridge or supply their own typed converter without implementing a foreign trait.
3. **Configuration-aware fold hook.** Rejected: the fold needs only field and
   handle types. A caller-owned callback runs after commitment and captures
   routing state outside the protocol fold.

## Documentation

- Book usage: `book/src/how/architecture.md` (Backend routing) — keep as the
  integrator narrative; update Implementation paths to include `registry.rs`.
- This spec — design record and acceptance list for the per-fold backend routing.
- No verifier-contract or schedule-artifact changes.

## Execution

Implemented on branch `codex/dynamic-backend-bridges`. Follow-ups MAY include
auto-registering `Edge<B, B>` on `register` for ergonomics.

## References

- Book: [`book/src/how/architecture.md`](../book/src/how/architecture.md)
- Related: [`specs/opaque-prover-consumer.md`](opaque-prover-consumer.md),
  [`specs/family-agnostic-cpu-backend.md`](family-agnostic-cpu-backend.md)
- Tests: `crates/akita-pcs/tests/dynamic_backends.rs`
