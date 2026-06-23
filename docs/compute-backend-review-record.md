# Compute Backend Review Record

> **Historical snapshot.** Pre-implementation design-review record; the CPU
> cutover has landed. The current boundary is in `docs/compute-backends.md`; the
> remaining Metal work is tracked in `specs/akita-compute-backend-metal.md`.
> Scheduled to move to `docs/archive/` (see `specs/PRUNING.md`).

This records the design review that narrowed
`specs/akita-compute-backend-metal.md` into the first implementation slice.

## Status

- Spec: `specs/akita-compute-backend-metal.md`
- Spec commit: `324d14b7` (`docs(compute): add backend cutover spec`)
- CPU / `AkitaPolyOps` cutover: **landed** in PR [#206](https://github.com/LayerZero-Labs/akita/pull/206)
- Deferred target: accelerator and integration stack (Metal)

## Review Outcome

The first PR should not be a thin inventory PR, and it should not introduce
Metal beside the existing CPU-shaped path. It should land one coherent code
change:

- make prover compute an explicit backend operation boundary;
- move setup-owned CPU NTT caches into CPU prepared setup;
- change the public prover API to accept an explicit backend plus its typed
  prepared setup;
- cut `AkitaPolyOps` through the new boundary instead of bypassing it;
- remove direct `NttSlotCache` plumbing from `akita-scheme`;
- preserve CPU proof behavior and transcript order.

Metal skeleton, production Metal kernels, field/MLE kernels, sumcheck backend
hooks, true hybrid scheduling, and the Jolt adapter stay in one coarse
remaining-work bucket until this cutover merges.

## Evidence Reviewed

This evidence was collected from the pre-cutover base commit. It records what
the current implementation slice must remove, replace, or confine to CPU
backend internals.

- `crates/akita-prover/src/api/setup.rs`: `AkitaProverSetup` owns
  `NttSlotCache<D>` and builds it in both setup constructors.
- `crates/akita-prover/src/api/scheme.rs`: `CommitmentProver` is generic over
  a cache type defaulting to `NttSlotCache<D>`.
- `crates/akita-prover/src/lib.rs`: `AkitaPolyOps` exposes `type CommitCache`
  and passes that cache through `commit_inner` and `commit_inner_witness`.
- `crates/akita-prover/src/api/commitment.rs`: root commit functions consume
  `setup.ntt_shared` directly and call CPU NTT mat-vec kernels.
- `crates/akita-scheme/src/lib.rs`: scheme orchestration imports
  `NttSlotCache`, `MultiDNttCaches`, and `dispatch_with_ntt!`.
- `crates/akita-prover/src/protocol/flow.rs`: root and recursive folded paths
  take `&NttSlotCache<D>` and construct `MultiDNttCaches` locally.
- `crates/akita-prover/src/protocol/ring_switch.rs`: ring-switch witness
  construction and recursive witness commitment take direct NTT cache handles.
- `crates/akita-prover/src/protocol/quadratic_equation.rs`: prover `v` rows and
  split-eq residual rows call CPU NTT kernels behind direct cache parameters.
- `crates/akita-prover/src/backend/*.rs`: dense, one-hot, sparse-ring,
  root-projection, multilinear, and recursive witness implementations encode
  the current CPU cache in their `AkitaPolyOps` commit paths.

## Blocking Risks Resolved In The Spec

- **Backend beside backend:** adding `akita-metal` first would create a second
  execution layer while protocol code still owned CPU NTT caches. Resolved by
  making the CPU backend operation cutover the first implementation.
- **Half-cutover API:** splitting setup, public API, and `AkitaPolyOps` into
  separate mergeable PRs would leave old and new paths alive at once. Resolved
  by making the current PR one atomic CPU compute cutover.
- **Setup cache ownership:** keeping a `cpu_cache` field on
  `AkitaProverSetup` would preserve the wrong ownership model. Resolved by
  making prepared setup backend-owned.
- **Representation loss:** routing every backend plan through dense `Vec<F>`
  materialization would erase one-hot, sparse-ring, compact digit, and
  recursive witness structure. Resolved by requiring representation-aware
  compute plans.
- **Transcript coupling:** backend code must not own transcript absorbs or
  challenge squeezes. Resolved by keeping transcript order in protocol flow and
  making backend outputs keyed by protocol plan slots.
- **Jolt dependency shape:** verifier-side Jolt integration must not pull in
  prover compute or accelerator crates. Resolved by keeping `akita-verifier`
  and `akita-types` free of prover/backend/device dependencies.

## Remaining Implementation Risks

- Operation plan structs must remain consumable by future out-of-crate
  backends. Public trait methods cannot require a Metal crate to see
  `akita-prover` crate-private one-hot or sparse-ring entries.
- Dynamic-D dispatch must prepare typed backend contexts inside the matched
  const-generic arm and return `AkitaError` rather than cache-specific dispatch
  panics on migrated paths.
- Q128 benchmark and parity coverage must be kept representative, not
  exhaustive for every modulus family.
- CPU benchmark variance may exceed the target threshold on shared hardware;
  the baseline document must record machine and command details.

## Rubric Snapshot

Rubric: `specs/SPEC_REVIEW.md`

| Dimension | Score | Notes |
| --- | ---: | --- |
| Goal Clarity | 0.95 | Current PR is CPU backend/setup/API/`AkitaPolyOps` cutover. |
| Constraint Clarity | 0.95 | Protocol, transcript, verifier, setup, and accelerator boundaries are explicit. |
| Evaluation Clarity | 0.90 | Acceptance criteria name tests, guards, parity paths, and benchmarks. |
| Context Clarity | 0.95 | Affected crates, modules, traits, and deferred work are listed. |
| **Ambiguity** | **7%** | Ready for implementation, with risks above tracked. |

Hard gates pass for the current CPU cutover. The spec does not change proof
format, verifier equations, transcript order, setup descriptor bytes, or
serialization semantics.
