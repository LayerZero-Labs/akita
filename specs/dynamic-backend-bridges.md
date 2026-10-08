# Spec: Dynamic Backend Bridges

| Field         | Value                                            |
|---------------|--------------------------------------------------|
| Author(s)     | Amir Kh                                          |
| Created       | 2026-10-07                                       |
| Status        | implemented                                      |
| PR            | [#193](https://github.com/LayerZero-Labs/akita/pull/193) |
| Supersedes    | opaque endpoint-id successor handoff             |
| Superseded-by |                                                  |
| Book-chapter  | book/src/usage/feature-flags.md                  |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14 when,
and only when, they appear in all capitals.

## Summary

Akita can assign each fold level of one proof to a different registered prover
backend. Earlier handoff used an opaque export capability and a global endpoint
registry with byte-section `read` pulls. That coupled destinations to a shared
pull API and made external backends reconstruct CPU-private layouts through
`akita-backend-transfer`. This change replaces that path with typed export and
import packets plus app-registered `Edge<A, B>` bridges. The protocol still
routes by `BackendId`; packet conversion is keyed only by backend **types**.

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
| `FoldExecutionPolicy` | Chooses the next `BackendId` (e.g. `FixedFoldRoute`) |
| `SuccessorExportKernel` | Source backend → owned `ExportPacket` |
| `SuccessorImportKernel` | Destination backend ← `ImportPacket` → local handles |
| `SuccessorBridge` on `Edge<A, B>` | `A::ExportPacket` → `B::ImportPacket` |
| `FoldHandoff` | After successor commit, same-id `begin_fold` or export→convert→import |
| `CpuExportPacket` / `CpuImportPacket` | Public CPU sectioned IR |

### Invariants

1. **Same-instance short-circuit.** When the policy selects the current
   `BackendId`, the coordinator MUST NOT export, convert, or import. It MUST
   call `begin_fold` on that instance only. Protected by homogeneous routes in
   `crates/akita-pcs/tests/dynamic_backends.rs`.
2. **Bridge required on type switch.** When the selected id differs from the
   current id, the registry MUST look up
   `(TypeId::of::<Src>(), TypeId::of::<Dst>())`. A missing bridge MUST reject
   the route before export. Protected by the no-bridge failure case in
   `dynamic_backends.rs`.
3. **Type-keyed, not id-keyed, bridges.** `register_bridge::<A, B>()` registers
   one directed conversion for all instances of those types. Distinct instances
   of the same type (two `CpuBackend`s) still NEED
   `register_bridge::<CpuBackend<F, E>, CpuBackend<F, E>>()`.
4. **Handoff precedes producer sumchecks.** After the producer commits the
   successor witness, `FoldHandoff::handoff_successor` MUST run before that
   producer's stage-1/2/3 work. The destination MAY begin its fold during
   handoff; it MUST NOT run fold sumchecks until `batched_prove` dispatches it
   via `resident.owner`.
5. **Wire invariance.** Routing MUST NOT change setup, commitment, proof, or
   transcript bytes relative to a same-schedule homogeneous proof. Protected by
   byte-equality checks against a CPU-only reference in `dynamic_backends.rs`.
6. **Public CPU adopt IR.** External bridges to CPU MUST be able to build
   `CpuImportPacket` through `CpuImportPacket::new(descriptor, sections)`
   without naming private CPU handles. CPU import MUST validate plan shape,
   encodings, and digit bounds.
7. **Session isolation.** Export borrows the live producer session; import runs
   on the destination executor's prepared session. The coordinator MUST NOT
   re-enter the producer's executor slot for export.

### Non-Goals

- Automatic discovery of bridges; every directed type edge is explicit.
- Cost models or scheduler heuristics inside Akita (routes own policy).
- Changing verifier behavior or proof format.
- Sharing one backend instance across different `(F, E)` pairs.
- Streaming or async handoff; transfer remains synchronous and blocking.

## Evaluation

### Acceptance Criteria

- [x] `OpaqueSuccessorExport`, endpoint ids, and `akita-backend-transfer` are
  removed from the successor path.
- [x] Backends implement `SuccessorExportKernel` and `SuccessorImportKernel`
  instead of a combined opaque transfer kernel.
- [x] `Edge<A, B>: SuccessorBridge` supplies `convert`; the framework
  `ErasedBridge::transfer` is fixed for all edges
  (`export_as` → `convert` → `import_as`).
- [x] `BackendRegistry::register_bridge::<A, B>()` keys by `(TypeId<A>, TypeId<B>)`.
- [x] Same `BackendId` skips transfer; different ids of the same type still
  transfer through the registered self-edge.
- [x] Heterogeneous `PrivateCpu` ↔ `CpuBackend` routes match homogeneous proof
  bytes (`dynamic_backends.rs`).
- [x] Example `dynamic_onehot` alternates `PrivateCpu` and `CpuBackend` with
  both directed bridges registered.
- [x] Book usage documents `register_bridge` and packet handoff
  (`book/src/usage/feature-flags.md`).

### Testing Strategy

- `crates/akita-pcs/tests/dynamic_backends.rs` — routes, missing bridges,
  fault injection, portable packets, cleanup on failure.
- `crates/akita-pcs/examples/dynamic_onehot.rs` — PrivateCpu ↔ CPU ping-pong.
- `crates/akita-prover/tests/external_backend.rs` — external backend still
  compiles against the new kernels (stub export/import).

### Performance

Handoff is not on the verifier path. Prover overhead is one typed packet copy
(or CPU native self-edge conversion) per backend switch. No proof-size or
security-parameter change is expected; equality tests against homogeneous
proofs are the gate.

## Design

### Architecture

```text
batched_prove
  └─ policy.choose_backend → BackendId
  └─ registry.slot(owner).root / .recursive / .terminal
        └─ prove_fold(..., &mut FoldHandoff)
              ├─ commit successor witness
              ├─ FoldHandoff::handoff_successor
              │     ├─ same BackendId → begin_fold only
              │     └─ else Edge::<Src,Dst>::transfer
              │           export → convert → import → ResidentSuccessor
              └─ producer sumchecks
  └─ next fold uses resident.owner
```

**Crate layout**

| Location | Owns |
| --- | --- |
| `akita-prover/src/backend/transfer.rs` | Kernels, `Edge`, packets traits, section metadata |
| `akita-prover/src/protocol/prove/registry.rs` | Registry, bridges map, `FoldHandoff` |
| `akita-prover/src/protocol/prove/execution.rs` | `TypedExecutor`, `batched_prove` |
| `akita-cpu-backend/src/opaque/transfer.rs` | CPU packets, CPU↔CPU bridge, import validation |

**Why type-keyed bridges.** Convert depends only on packet types. Instance
identity is already carried by `BackendId` when choosing who runs the fold and
which session imports. Registering `(A, B)` once covers every A→B instance pair.

**Why not pairwise typed handles in the protocol.** The registry is type-erased
after `register`. Dynamic `BackendId` routes cannot name `A::WitnessHandle` at
the coordinator. Packets are the shared IR; bridges are the typed adapters.

### Alternatives Considered

1. **Opaque endpoint + `SuccessorEndpoint::read`.** Rejected for external CPU
   adopt: destinations pulled bytes through a global registry and decoded
   inside CPU import, forcing every source to implement the pull API.
2. **Exporter builds destination handles (`ExportToward<B>`).** Rejected:
   sources would depend on every destination type (orphan-rule and N×M
   coupling). Bridges invert that: apps that compose A and B own `Edge<A, B>`.
3. **`SuccessorHandoff` trait seam for fold.** Removed; only `FoldHandoff`
   existed. Fold now takes `&mut FoldHandoff` from `registry.rs`.

## Documentation

- Book usage: `book/src/usage/feature-flags.md` (Backend routing) — keep as the
  integrator narrative; update Implementation paths to include `registry.rs`.
- This spec — design record and acceptance list for the typed-bridge cutover.
- No verifier-contract or schedule-artifact changes.

## Execution

Shipped on branch `codex/dynamic-backend-bridges`. Follow-ups MAY include
auto-registering `Edge<B, B>` on `register` for ergonomics, and folding more of
this narrative into a dedicated Book architecture subsection if routing grows
beyond the feature-flags page.

## References

- Book: [`book/src/usage/feature-flags.md`](../book/src/usage/feature-flags.md)
- Related: [`specs/opaque-prover-consumer.md`](opaque-prover-consumer.md),
  [`specs/family-agnostic-cpu-backend.md`](family-agnostic-cpu-backend.md)
- Tests: `crates/akita-pcs/tests/dynamic_backends.rs`
- Example: `crates/akita-pcs/examples/dynamic_onehot.rs`
