# Akita compute backends

The generic prover in `akita-prover` owns protocol sequencing, public geometry,
transcript absorption, challenges, and proof assembly. `akita-cpu-backend` owns
CPU sources, commitment material, witnesses, arithmetic, setup resources,
caches, and independent proof lifetimes. The generic crate has no production
dependency on the CPU crate.

## Application ownership

Create one backend for a setup and trusted configuration. Share it explicitly
through `Arc` when several callers need it:

```rust
let scheme = AkitaCommitmentScheme::<Cfg>::from_schedule_artifact(&artifact_bytes)?;
let setup = scheme.setup_prover(nv, num_polys)?;
let backend = std::sync::Arc::new(CpuBackend::<Cfg>::new(
    setup.expanded.clone(),
    scheme.schedules(),
)?);
let source = backend.import_source(polys)?;
let committed = backend.commit(
    &source,
    GroupContext::scheduler_without_precommitted_groups(),
)?;
```

The backend implements neither `Clone` nor `Copy`. Constructing another backend
creates another owner identity, even for the same public setup. Source import
consumes the application's polynomial storage. Commitment handles share
immutable retained source and commitment contents; dropping the source handle
does not invalidate its commitments.

Opening pairs an ordered vector of commitment handles with public claims.
There is no separately supplied polynomial table. Admission checks setup,
owner, public commitment, frozen profile, and ordered group. Each proof then
gets an opaque session handle with independent state. The handle owns cleanup
on completion, failure, or early return; it cannot be cloned or constructed
from a numeric scope ID. Scope IDs in public operation contexts serve only as
correlation metadata. Finishing one proof leaves other proofs and reusable
commitments valid.

See the [commitment API](../book/src/usage/commitment-api.md) for complete caller
examples and the [architecture](../book/src/how/architecture.md) for the crate
map.

## Opaque protocol operations

`CpuBackend` directly implements the focused contracts in
`akita-prover::compute`. Contracts carry public plans, validated geometry,
challenges, and proof context. Opaque handles retain source lineage, opening
points, accepted challenges, intermediate rows, witness construction state,
and mutable sumcheck sessions inside the backend.

Readable results are public commitments, scheduled proof messages, public
metadata, acceptance decisions, and errors. Fold probes expose only acceptance
or rejection. Stage 1, Stage 2, and Stage 3 sessions emit their designated round
and completion messages. Terminal commitment fields are released at their
original transcript position and remain tied to the final terminal response.

Physical representation dispatch and commitment execution components remain
inside the CPU crate. They can exchange ordinary slices and ring buffers
internally. Generic proving does not select physical routes or manage NTT
resources.

## CPU resource controls

`CpuBackend::with_ring_switch_cache_limit` accepts the expanded setup, trusted
catalog, and maximum cached ring-switch elements. A zero limit streams
supported operations; `usize::MAX` retains all supported ring-switch
operations. The CPU kernel sizes one-hot commitment scratch automatically from
the commitment geometry; see the
[CPU resource policy](../book/src/how/optimizations.md#cpu-resource-limits).

These controls change CPU work and memory retention without changing public
parameters or protocol messages. Ring dimensions come from validated schedule
parameters. Cached and streamed ring-switch paths use the same quotient
arithmetic and exact field fallback.

The backend lazily prepares NTT prefixes and can prewarm a schedule. Shared
caches survive completed proofs. Applications can use `trim_caches()` to remove
cached entries; active operations retain their owned references. Concurrent
construction can repopulate a cache during trimming. Cache metrics and witness
energy diagnostics are CPU/application capabilities, outside generic proving
contracts.

## Setup-prefix persistence

The CPU backend exports portable setup-prefix artifacts for application-managed
storage. Import checks the setup seed, exact prefix parameters, public
commitment, and retained material against recomputed setup data before issuing
fresh current-owner commitment handles. Serialized process-local identities
never grant authority.

Generic proving receives a registry of public prefix commitments and opaque
ordinary commitment handles. Missing prefixes are prepared through its
backend-neutral setup-prefix kernel. It neither builds dense sources nor
extracts or deserializes retained CPU material.

Cross-backend commitment transfer is similarly explicit:
`backend.import_commitment(&foreign_handle)` validates the retained source and
commitment under the receiving setup before creating a new owner-bound handle.
