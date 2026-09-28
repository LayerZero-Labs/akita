# Spec: Backend-Generic Commitment Execution

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Quang Dao                      |
| Created       | 2026-09-28                     |
| Status        | proposed                       |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  | book/src/roadmap/compute-backends.md |

## Summary

Commitment computation (the inner `A` commitment of dense, one-hot and recursive
sources, the decomposition into outer subrings, the outer `B` commitment over
slices, and compression) runs only on the CPU backend today. The stage design in
`crates/akita-cpu-backend/src/commitment/` is already backend-neutral in shape
(object-safe inner, outer, compression and export operations over opaque
backend state), but the stage traits were crate-private and every production
call site built the CPU executor. This spec stages the work that lets a compute
backend (Metal first; CUDA or WebGPU later) run commitment stages behind one
boundary, with the CPU backend as the byte-for-byte reference.

## Intent

### Goal

Every commitment stage can be supplied by a compute backend through
`CommitmentStageProvider`, selected per request by capability, with the CPU
backend owning handles, retained state and proof orchestration.

### Invariants

1. A backend never changes plans, schedules, transcript order, serialization,
   or setup, commitment and proof bytes (`docs/branches.md` rules 1, 2 and 5).
   Protected by the provider parity tests in
   `crates/akita-cpu-backend/src/opaque/commitment_provider_tests.rs`, which
   compare committed groups, retained state and `batched_prove` bytes.
2. With no provider installed, the CPU route is unchanged.
3. A provider declines a request before any source is materialized
   (`supports_plan` and the preflight), and the CPU executor takes over. A
   device failure after selection is returned, not silently retried.
4. The verifier depends on no backend crate (`scripts/check-crate-deps.sh`).
5. One boundary per stage: no parallel old and new executor APIs.

### Non-Goals

- Changing the commitment protocol or its parameters.
- Opening-side kernels (decompose-fold, coefficient packing, sumcheck); each
  gets its own spec in stage 5.

## Evaluation

### Acceptance Criteria

- [x] Stage 1: the seam, with parity and fallback tests.
- [ ] Stage 2: Metal inner and outer stages with byte-identical proofs for the
      fp128 and fp64 dense and one-hot catalogs.
- [ ] Stage 3: a residency abstraction covering coefficient-form matrices.
- [ ] Stage 4: device-resident sources, including recursive short-norm
      witnesses.
- [ ] Stage 5: compression and opening-side seams, each with its own spec.
- [ ] Stage 6: the Jolt adapter.

### Testing Strategy

Each stage adds CPU differential tests per representation, ring degree and
field, plus proof-byte equality through `CpuBackend::with_commitment_stage_provider`.
GPU tests run on macOS and are reported in each pull request.

### Performance

Each backend stage reports device against CPU time on production schedule
shapes, with the command and hardware, as `specs/akita-compute-backend-metal.md`
requires.

## Design

### Architecture

Stage 1 (landed with this spec) adds `CommitmentStageProvider<F>` in
`crates/akita-cpu-backend/src/commitment/provider.rs`:

- `stages(expanded)` returns optional prepared inner and outer operations; the
  CPU fills any stage left empty, and compression stays on the CPU.
- `CpuBackend::with_commitment_stage_provider` installs a provider at runtime.
- `CpuBackend::commitment_executor` selects the route per request with the
  existing side-effect-free portable-export preflight, which now also checks
  `supports_plan` on the inner and outer operations.
- A hidden `commitment_backend` module re-exports what an out-of-crate provider
  needs.

The remaining stages:

2. **Metal inner and outer stages** (`crates/akita-metal`, on the kernel stack):
   device matrices prepared once per setup from the coefficient-form matrix
   (limb-split NTT matrices for dense and outer, coefficient form for one-hot),
   an inner operation for dense and one-hot sources that registers a host-row
   exporter, and an outer operation that decomposes into subrings, gathers the
   slices and runs the `B` matvec, downloading only `u` and, for portable
   retained state, `t`.
3. **Residency.** Generalize the NTT cache requirement into a stage-matrix
   requirement (`stage, ring degree, rows, cols, form`) so coefficient-form
   matrices are prewarmed and released like NTT caches, and route cache trimming
   to providers.
4. **Device sources.** Recursive short-norm witnesses for `commit_witness`,
   and device-resident external sources through the context-typed external
   inner commitment capability.
5. **Compression and opening-side seams.** Route root commitments through the
   registered compression stage, then give decompose-fold, coefficient packing
   and the sumcheck stages their own seams, consuming device-resident inner
   images instead of exported rows.
6. **Jolt adapter.** A device one-hot source for Jolt's trace-packed
   polynomials.

### Alternatives Considered

- **A Metal feature inside `akita-cpu-backend`.** Rejected: the dependency
  checks keep device crates out of the CPU backend, and a feature must not
  select an extension.
- **A new prover-level root-commit trait.** Deferred: the composable execution
  spec places accelerator composition inside the owning backend, which the
  provider does without widening the prover contract.

## Documentation

`book/src/roadmap/compute-backends.md` gains the provider boundary once stage 2
lands.

## Execution

Open questions for the maintainer:

1. Crate placement of device adapters: `akita-metal` (which now depends on
   `akita-cpu-backend` for source types) or a separate adapter crate.
2. Whether `AkitaError` should gain a typed unsupported-backend variant instead
   of treating every preflight failure as a decline.
3. Whether root commitments should move to the registered compression stage.
4. Whether setup-prefix commitments may take accelerated routes (their bytes
   are identical either way; the seam allows it when the provider accepts).
5. Whether device backends prepare their own matrices (as the Metal limb-split
   matrices do) or the CPU prepared setup lends its NTT caches.
6. The drift in `specs/composable-commitment-execution.md` (its code map and
   the fused external API, which is test-only in the code).

## References

- `specs/akita-compute-backend-metal.md`
- `specs/composable-commitment-execution.md`
- `crates/akita-cpu-backend/src/commitment/`
- `docs/branches.md`
