# Simplified commitment backend composition

Status: implemented on `feat/backend-commit-simplified`

Date: 2026-09-09

Scope: commitment inner, outer, fused inner/outer, compression, retained state,
and routing known before execution.

This design replaces the state-registry and composite-dispatcher parts of the
original [commitment execution design](../crates/akita-prover/docs/design.md).
It keeps the existing arithmetic plans, source compiler, backend capability
checks, NTT resource controls, public commitment bytes, and transcript behavior.

## Decisions

The design has four operation traits:

- `InnerCommitOperation` performs A and returns opaque retained inner state.
- `OuterCommitOperation` consumes that state, directly or through an explicit
  host export, and returns `u`.
- `FusedInnerOuterOperation` performs A, decomposition, slicing, and B in one
  backend call. It returns opaque retained inner state and host `u`.
- `CompressionOperation` consumes `u` and returns the public terminal payload
  plus opaque compression state.

Every operation is independently registered. A full route is either:

```text
split: sources -> inner -> retained inner state -> outer -> host u
fused: sources -> fused inner/outer -------------------------> host u
                        |
                        +-> retained inner state

either route: host u -> compression -> terminal payload
                                  +-> retained compression state
```

Compression never receives the inner witness. The common boundary between an
inner/outer route and any compressor is `RingVec<F>` for `u`.

Dynamic trait dispatch remains inside Akita. It is useful because the selected
backends can be runtime configuration even though the complete selection is
known before execution. External users implement operation traits and describe
the routing; they do not implement type erasure, sequencing, or composite state
assembly.

## Complete routing before execution

The caller creates each usable stage combination with
`CommitmentExecutorBuilder`, then declares independent round ranges with
`CommitmentExecutionScheduleBuilder`.

```rust,ignore
let mut schedule = CommitmentExecutionScheduleBuilder::new(round_count)?;

// Each executor is built from independently registered operations. These
// three entries represent the combinations produced by the two cutovers.
schedule
    .register_executor("gpu-fused", "metal", &gpu_fused_metal)?
    .register_executor("gpu-fused", "cpu", &gpu_fused_cpu)?
    .register_executor("cpu-split", "cpu", &cpu_split_cpu)?;

schedule
    .inner_outer(0..i, "gpu-fused", InnerOuterRouteKind::Fused)?
    .inner_outer(i..round_count, "cpu-split", InnerOuterRouteKind::Split)?
    .compression(0..j, "metal")?
    .compression(j..round_count, "cpu")?;

let schedule = schedule.compile()?;

// Inspect every resolved round before commitment or proof work starts.
for step in schedule.steps() {
    inspect(step.round(), step.inner_outer_kind(),
            step.inner_outer(), step.compression());
}

schedule.preflight(round, &round_plan, &sources)?;
let output = schedule.execute_full(round, &round_plan, &sources)?;
```

For four full rounds with `i = 2` and `j = 1`, compilation resolves:

| Round | Inner/outer | Compression |
| --- | --- | --- |
| 0 | GPU fused | Metal |
| 1 | GPU fused | CPU |
| 2 | CPU split | CPU |
| 3 | CPU split | CPU |

The two boundaries are independent. Therefore every pair they produce must
have a registered executor. Akita rejects the schedule before execution when:

- `round_count` is zero;
- a range is outside the schedule;
- a round is assigned twice for the same component;
- any round has no inner/outer or compression assignment;
- a selected inner/outer and compression pair has no registered executor;
- the declared fused/split kind disagrees with the registered executor.

Empty ranges are valid, so `i` and `j` may be zero or `round_count`. Compiling
the schedule does not run a backend. Execution uses the resolved executor for
the requested round and does not retry or select a fallback.

`schedule.preflight` validates the selected round's source admission, protocol
mode, ring dimensions, and compression relation support without materializing
sources or calling arithmetic. Applications can preflight every round for
which source metadata is already available before starting the run.

`CommitmentExecutionPlan` remains the checked arithmetic plan for a particular
round. The schedule selects the executor; the plan supplies protocol geometry.
Existing `TieredProveStacks` continues to select the complete prover stack by
fold level. Thus the application can decide all commitment and proof-stack
ranges before calling commitment/proving without introducing another runtime
state enum.

## Split and fused executor construction

A split executor registers inner, outer, and compression separately:

```rust,ignore
let mut builder = CommitmentExecutorBuilder::new::<InnerContext>(
    expanded,
    inner_backend_kind,
    accepted_source_types,
    ResidentStatePolicy,
);

builder.register_inner(inner, inner_owner, inner_context,
                       inner_dimensions, optional_host_export)?;
builder.register_outer(outer, outer_owner, outer_context,
                       outer_dimensions)?;
builder.register_compression(compression, compression_context,
                             compression_capabilities,
                             optional_portable_export)?;
let executor = builder.build()?;
```

If inner and outer use the same `StateOwnerCapability<InnerImage>`, outer gets
the resident state directly. If their owners differ, construction requires an
explicit `InnerImageExportOperation`; execution exports canonical host rows and
passes them to outer. There is no implicit transfer or fallback.

A fused-only executor does not register dummy split operations:

```rust,ignore
let mut builder = CommitmentExecutorBuilder::new_fused(
    expanded,
    ResidentStatePolicy,
);

builder.register_fused(fused, fused_context, fused_capabilities,
                       fused_dimensions, optional_portable_export)?;
builder.register_compression(compression, compression_context,
                             compression_capabilities,
                             optional_portable_export)?;
let executor = builder.build()?;
```

An executor may also contain split operations plus a fused operation. Full and
uncompressed A/B plans use fused. Inner-only plans use the registered split
inner operation. A fused-only executor therefore supports full/uncompressed
commitment and correctly rejects an inner-only request.

## Direct state ownership

`BackendStateRef<K>` is an opaque, checked owner of one backend value. The
value may be:

- CPU witness vectors;
- a Metal or CUDA allocation lease;
- a remote object lease;
- any other owned backend handle satisfying the operation boundary.

The backend creates it through its `StateOwnerCapability<K>`:

```rust,ignore
let state = owner.bind(binding, retained_bytes, MyDeviceLease { ... });
```

The concrete value is erased inside Akita. Only the matching owner can borrow
it with `owner.value::<MyDeviceLease>(&state)`. External callers do not
downcast the state. The binding records setup identity, inner plan, source
count, and relation mode, and output constructors validate it when stages are
joined.

The state directly owns its concrete value. Dropping the last state reference
drops that value, so a GPU lease can release or recycle its allocation through
ordinary `Drop`. Akita no longer keeps a global slot table, generation counter,
pending deposit, or cleanup callback for ordinary commitment state. A device
backend may still maintain its own allocation pool when it needs one.

CPU inner state directly owns its `Vec<CommitInnerWitness<F>>`. CPU compression
state directly owns its `CpuCompressionRetention<F>`. The resident composite
directly owns its inner and optional compression state references.

The state wrapper may be cloned as a shared lease, but the backend value itself
does not implement `Clone`, serialization, `Debug`, or conversion to ring rows.
This allows a move-only device lease without copying the witness.

## Fused GPU communication contract

A fused GPU implementation can keep the entire inner witness on the device:

```text
GPU: A + decomposition + slicing + B
  ├── device inner witness -> opaque BackendStateRef<InnerImage>
  └── u                    -> host RingVec<F>
```

This version explicitly permits communication of `u`. It forbids downloading
the large inner witness merely to finish commitment:

- no inner-row download between A and B;
- no inner-row download before or during compression;
- no inner-row download during result assembly or binding validation;
- no `Vec<RingVec<F>>` or portable hint required from a fused backend;
- no inner exporter required for a fused-only resident route.

Compression receives only the binding, compression plan, relation mode, and
owned `u`. It has no parameter through which it could read the inner witness.
The final resident state retains the fused backend's opaque inner lease and the
compressor's opaque state side by side.

Portable export remains optional and happens only at an explicit consumer such
as setup-prefix persistence or the existing CPU proving path. Whether future
GPU opening and relation operations keep the witness device-resident will be
designed later. It is not a requirement on fused commitment in this version.

## What became simpler

| Previous component | Replacement | Result |
| --- | --- | --- |
| `BackendStateStore` slot table | `BackendStateRef<K>` directly owns an erased value | No reserve, seal, lookup, generation, or universal cleanup callback. |
| `StateRegistrar` and pending deposits | `StateOwnerCapability::bind` | A backend returns one checked owned value. |
| CPU token-to-buffer maps | Values stored directly in the state | No map lookup or lock to reach CPU witnesses. |
| `CommitmentStateDispatcher` and composite registry entry | `ResidentCommitmentState` directly owns `CommitmentStateComponents` | No second registry object pointing at two other handles. |
| Builder always requiring split stages | `CommitmentExecutorBuilder::new_fused` | A fused backend needs no unused inner or outer implementation. |
| One executor choice for a run | Immutable round schedule | Fused/split and compression cutovers are independently declared and inspected before execution. |
| Mandatory portable hint shape | Optional explicit exporters | Resident GPU state need not become `Vec<RingVec<F>>`. |

The arithmetic kernels, source validation, setup checks, output shape checks,
relation-mode checks, and resource lifecycle checks remain because they enforce
protocol correctness rather than state bookkeeping.

## External backend responsibilities

An external backend implements only the operations it supplies and captures
its prepared setup/resources in those operation objects.

- A fused backend implements `FusedInnerOuterOperation<F>` and returns checked
  `UncompressedCommitmentOutput<F>` containing opaque inner state and `u`.
- A compressor implements `CompressionOperation<F>` and returns checked
  `CompressionStageOutput<F>`.
- A split inner backend implements `InnerCommitOperation<F>`.
- A split outer backend implements `OuterCommitOperation<F>` and either shares
  the inner owner or accepts the explicit host-row boundary.

Akita handles source resolution, calls the selected operation, checks state and
output geometry, connects `u` to compression, applies the selected state policy,
and assembles the public result. A fused implementation and compressor do not
know each other's concrete retained-state types, so they can be mixed freely.

## Verification

The implementation has focused tests for:

- direct state ownership and final-reference cleanup;
- rejection of a foreign owner or wrong concrete representation;
- direct CPU inner and compression storage;
- split routes with resident and explicit host-transfer connections;
- fused execution parity with CPU split execution;
- a fused-only executor with no split registrations and no inner exporter;
- no inner export during fused commitment, compression, or result assembly;
- independent `i` and `j` boundaries resolving distinct executor objects;
- schedule gaps, overlaps, missing pairs, and fused/split mismatches;
- existing setup-prefix, portable, resident, and no-retained state policies.

Subsequent versions will decide the interface for device-resident opening and
relation proving. This design deliberately stops at the retained opaque state
boundary required by commitment.
