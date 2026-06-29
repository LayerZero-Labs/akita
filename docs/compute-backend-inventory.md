# Compute Backend Inventory

> **Historical snapshot.** Pre-cutover symbol inventory (commit `324d14b7`); the
> cutover has landed and this references removed symbols (e.g. `akita-scheme`,
> `NttSlotCache`). The current compute-backend boundary is documented in
> `docs/compute-backends.md` and the Akita Book
> (`book/src/how/optimizations/compute-backends.md`). Scheduled to move to
> `docs/archive/` (see `specs/PRUNING.md`).

This inventory captures the pre-cutover CPU-cache and direct-kernel surface
that the first compute-backend PR must cut over. It was gathered at commit
`324d14b7` with:

```bash
rg -n "NttSlotCache|MultiDNttCaches|dispatch_with_ntt!|mat_vec_mul_ntt_|commit_w|commit_next_w_with_policy|compute_v_rows|compute_r_split_eq" crates -g '*.rs'
rg -n "trait AkitaPolyOps|impl<.*AkitaPolyOps|impl .*AkitaPolyOps|CommitCache|commit_inner|commit_inner_witness|decompose_fold|evaluate_and_fold" crates -g '*.rs'
```

## Symbol Counts

These counts include definitions, comments, tests, benches, and internal CPU
kernel modules. They are a sizing aid, not a removal target.

| Symbol | Count |
| --- | ---: |
| `NttSlotCache` | 126 |
| `MultiDNttCaches` | 24 |
| `dispatch_with_ntt!` | 3 |
| `mat_vec_mul_ntt_` | 63 |
| `commit_w` | 48 |
| `commit_next_w_with_policy` | 7 |
| `compute_v_rows` | 3 |
| `compute_r_split_eq` | 7 |

## File Counts

| File | Hits | Classification |
| --- | ---: | --- |
| `crates/akita-prover/src/api/setup.rs` | 2 | Current cutover: setup must stop owning prepared CPU NTT state. |
| `crates/akita-prover/src/api/scheme.rs` | 2 | Current cutover: public prover trait must stop defaulting to `NttSlotCache<D>`. |
| `crates/akita-prover/src/api/commitment.rs` | 16 | Current cutover: root commit path becomes compute-plan driven. |
| `crates/akita-prover/src/lib.rs` | 3 | Current cutover: `AkitaPolyOps::CommitCache` is the trait-level CPU cache leak. |
| `crates/akita-prover/src/backend/dense.rs` | 12 | Current cutover: dense representation must build commit compute plans. |
| `crates/akita-prover/src/backend/onehot.rs` | 4 | Current cutover: one-hot representation must preserve sparse/one-hot planning. |
| `crates/akita-prover/src/backend/sparse_ring.rs` | 4 | Current cutover: sparse-ring representation must avoid dense materialization. |
| `crates/akita-prover/src/backend/field_reduction.rs` | 2 | Current cutover: root tensor projection delegates through migrated commit path. |
| `crates/akita-prover/src/backend/multilinear_polynomial.rs` | 4 | Current cutover: enum wrapper forwards to dense/one-hot migrated plans. |
| `crates/akita-prover/src/backend/recursive_witness.rs` | 8 | Current cutover: recursive witness commit path becomes plan-backed. |
| `crates/akita-prover/src/protocol/flow.rs` | 40 | Current cutover: root and recursive flow accept backend-prepared setup. |
| `crates/akita-prover/src/protocol/ring_switch.rs` | 25 | Current cutover: `commit_w`, next-w commitment, and ring-switch build use backend plans. |
| `crates/akita-prover/src/protocol/quadratic_equation.rs` | 19 | Current cutover: `compute_v_rows` and `compute_r_split_eq` use backend operations. |
| `crates/akita-prover/src/protocol/dispatch.rs` | 2 | Current cutover or deletion: dynamic-D cache dispatch should become typed backend preparation. |
| `crates/akita-scheme/src/lib.rs` | 22 | Current cutover: scheme must stop importing CPU cache/kernel helpers. |
| `crates/akita-scheme/src/tests.rs` | 3 | Current cutover: tests update to backend-backed helpers. |
| `crates/akita-setup/src/lib.rs` | 2 | Current cutover: setup constructors remain expanded-setup-only. |
| `crates/akita-pcs/tests/*.rs` | 4 | Current cutover: direct calls update with backend/prepared setup. |
| `crates/akita-pcs/benches/*.rs` | 10 | Current cutover: benchmark call sites update with explicit backend. |
| `crates/akita-pcs/examples/profile/workload.rs` | 2 | Current cutover: profiling constructs explicit CPU backend. |
| `crates/akita-prover/src/kernels/crt_ntt.rs` | 10 | CPU backend internals: stay behind `CpuBackend`. |
| `crates/akita-prover/src/kernels/ntt_cache.rs` | 15 | Historical only: the cutover should delete this dynamic-D cache wrapper rather than preserve it as a public escape hatch. |
| `crates/akita-prover/src/kernels/linear.rs` | 61 | CPU backend internals: direct kernels remain private implementation details. |
| `crates/akita-prover/src/kernels/mod.rs` | 2 | Current cutover: public re-exports should narrow if callers no longer need them. |
| `crates/akita-prover/src/protocol/mod.rs` | 1 | Current cutover: public `commit_next_w_with_policy` exposure may change. |

## Current Cutover Surface

The following is the pre-cutover surface. After the implementation, direct
`NttSlotCache` usage should remain only inside CPU backend internals, low-level
kernel modules, and low-level kernel benchmarks. `MultiDNttCaches`,
`dispatch_with_ntt!`, `AkitaPolyOps::CommitCache`, and setup-owned
`ntt_shared` should have no live code references.

### Setup and public API

- `crates/akita-prover/src/api/setup.rs:3`: imports `build_ntt_slot` and
  `NttSlotCache`.
- `crates/akita-prover/src/api/setup.rs:20`: stores `pub ntt_shared:
  NttSlotCache<D>`.
- `crates/akita-prover/src/api/setup.rs:50`: `generate_with_capacity` builds
  shared CPU NTT state.
- `crates/akita-prover/src/api/setup.rs:91`: validated expanded-setup
  constructors compute total ring elements without owning shared CPU NTT state.
- `crates/akita-prover/src/api/scheme.rs:15`: `CommitmentProver` defaults its
  cache generic to `NttSlotCache<D>`.
- `crates/akita-prover/src/api/scheme.rs:60`, `:86`, `:110`: `commit`,
  `batched_commit`, and `batched_prove` constrain `P::CommitCache` to the
  trait cache parameter instead of accepting a backend.

### Root commit path

- `crates/akita-prover/src/api/commitment.rs:65`: `commit_with_params` takes
  setup and requires `P: AkitaPolyOps<CommitCache = NttSlotCache<D>>`.
- `crates/akita-prover/src/api/commitment.rs:87`: calls
  `poly.commit_inner_witness` with `&setup.ntt_shared`.
- `crates/akita-prover/src/api/commitment.rs:110`: computes outer commitment
  rows with `mat_vec_mul_ntt_single_i8(&setup.ntt_shared, ...)`.
- `crates/akita-prover/src/api/commitment.rs:145`: `commit_with_policy`
  preserves the same CPU cache trait bound.
- `crates/akita-prover/src/api/commitment.rs:249`: `batched_commit_with_policy`
  preserves the same CPU cache trait bound.
- `crates/akita-prover/src/api/commitment.rs:275`: `batched_commit_with_params`
  loops through the same CPU-backed `commit_with_params`.

### `AkitaPolyOps`

- `crates/akita-prover/src/lib.rs:125`: trait definition.
- `crates/akita-prover/src/lib.rs:127`: `type CommitCache` is the main trait
  leak of CPU NTT state.
- `crates/akita-prover/src/lib.rs:466`: `commit_inner` accepts
  `&Self::CommitCache`.
- `crates/akita-prover/src/lib.rs:484`: `commit_inner_witness` accepts
  `&Self::CommitCache` and delegates to `commit_inner`.
- `crates/akita-prover/src/lib.rs:529`: reference impl forwards the same cache
  type and methods.

Representation impls participating in root commit, recursive witness commit,
or ring-switch commit:

| Impl | Current cache hook | Notes |
| --- | --- | --- |
| `DensePoly<F, D>` in `backend/dense.rs` | `type CommitCache = NttSlotCache<D>`; `commit_inner`; `commit_inner_witness` | Dense digit and dense coefficient mat-vec source. |
| `OneHotPoly<F, D, I>` in `backend/onehot.rs` | `type CommitCache = NttSlotCache<D>`; `commit_inner`; `commit_inner_witness` | Must preserve one-hot sparse planning. |
| `SparseRingPoly<F, D>` in `backend/sparse_ring.rs` | `type CommitCache = NttSlotCache<D>`; `commit_inner`; `commit_inner_witness` | Must preserve sparse-ring row planning. |
| `RootTensorProjectionPoly<F, D>` in `backend/field_reduction.rs` | `type CommitCache = NttSlotCache<D>`; forwards to inner poly | Used by transformed root commit path. |
| `MultilinearPolynomial<'_, F, D, I>` in `backend/multilinear_polynomial.rs` | `type CommitCache = NttSlotCache<D>`; forwards dense/one-hot | Wrapper must not reintroduce CPU cache bounds. |
| `SuffixWitness<'_, F, D>` in `backend/recursive_witness.rs` | direct `NttSlotCache<D>` parameters on recursive commit helpers | Used by `commit_w` and recursive folded levels. |

### Scheme orchestration

- `crates/akita-scheme/src/lib.rs:12`: imports `NttSlotCache`.
- `crates/akita-scheme/src/lib.rs:16`: imports `MultiDNttCaches`.
- `crates/akita-scheme/src/lib.rs:19`: imports `dispatch_with_ntt!`.
- `crates/akita-scheme/src/lib.rs:153`: `dispatch_prove_level` accepts mutable
  cache maps and `setup_ntt_shared`.
- `crates/akita-scheme/src/lib.rs:217`: dynamic-D path calls
  `dispatch_with_ntt!`.
- `crates/akita-scheme/src/lib.rs:267`: recursive suffix passes cache maps and
  `&setup.ntt_shared`.
- `crates/akita-scheme/src/lib.rs:351`: terminal recursive level dispatch uses
  the same CPU cache shape.
- `crates/akita-scheme/src/lib.rs:470`, `:493`, `:525`: public
  `CommitmentProver` impl constrains `P::CommitCache = NttSlotCache<D>`.
- `crates/akita-scheme/src/lib.rs:579`: folded prove path passes
  `&setup.ntt_shared` into root proving.
- `crates/akita-scheme/src/lib.rs:587`: root next-w commitment passes
  `&setup.ntt_shared` into `commit_next_w_with_policy`.

### Ring-switch and quadratic equation

- `crates/akita-prover/src/protocol/quadratic_equation.rs:191`:
  `compute_v_rows` takes `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/quadratic_equation.rs:202` and `:206`:
  `compute_v_rows` calls `mat_vec_mul_ntt_single_i8`.
- `crates/akita-prover/src/protocol/quadratic_equation.rs:242`: root
  `QuadraticEquation::new_prover` accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/quadratic_equation.rs:444`: root prover
  computes and absorbs `v` before stage-1 challenge sampling.
- `crates/akita-prover/src/protocol/quadratic_equation.rs:590`: recursive
  multipoint prover accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/quadratic_equation.rs:669`: recursive
  multipoint prover computes and absorbs `v`.
- `crates/akita-prover/src/protocol/quadratic_equation.rs:1231`:
  `compute_r_split_eq` accepts `&NttSlotCache<D>` and computes split-eq
  residual rows.
- `crates/akita-prover/src/protocol/ring_switch.rs:87`:
  `ring_switch_build_w` accepts `&NttSlotCache<D>` and calls
  `compute_r_split_eq`.
- `crates/akita-prover/src/protocol/ring_switch.rs:491`: `commit_w` accepts
  `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/ring_switch.rs:528`: `commit_w` calls
  recursive witness `commit_inner_witness`.
- `crates/akita-prover/src/protocol/ring_switch.rs:548`: `commit_w` computes
  the outer rows with `mat_vec_mul_ntt_single_i8`.
- `crates/akita-prover/src/protocol/ring_switch.rs:582`:
  `dispatch_commit_w_with_layout_policy` dynamically obtains NTT state through
  `dispatch_with_ntt!`.
- `crates/akita-prover/src/protocol/ring_switch.rs:636`:
  `commit_next_w_with_policy` exposes both same-D direct cache and cross-D
  dynamic cache plumbing.

### Protocol flow

- `crates/akita-prover/src/protocol/flow.rs:620`:
  `prove_folded_batched_with_policy` accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:647`: root folded path still
  requires `P::CommitCache = NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:712`: folded path constructs local
  `MultiDNttCaches`.
- `crates/akita-prover/src/protocol/flow.rs:908`: folded recursive level
  accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:934`: folded recursive level calls
  `ring_switch_build_w`.
- `crates/akita-prover/src/protocol/flow.rs:1079`: terminal recursive fold
  accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:1294`: recursive fold with params
  accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:1477`: terminal recursive fold
  with params accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:1657`: recursive level policy
  accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:1723`: terminal recursive level
  policy accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:2202`: root fold helper accepts
  `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:2252`: root helper passes cache
  into `QuadraticEquation::new_prover`.
- `crates/akita-prover/src/protocol/flow.rs:2312`: root fold with params
  requires `P::CommitCache = NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:2626`: terminal root fold with
  params requires the same cache bound.
- `crates/akita-prover/src/protocol/flow.rs:2921`: terminal root helper accepts
  `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:3029`: root fold from quadratic
  accepts `&NttSlotCache<D>`.
- `crates/akita-prover/src/protocol/flow.rs:3187`: terminal root fold from
  quadratic accepts `&NttSlotCache<D>`.

## CPU Backend Internals To Preserve

These modules should remain CPU implementation details after the cutover rather
than disappearing:

- `crates/akita-prover/src/kernels/crt_ntt.rs`: CRT/NTT parameter selection and
  `NttSlotCache` construction.
- `crates/akita-prover/src/kernels/linear.rs`: direct CPU mat-vec kernels,
  single-row cyclic/negacyclic variants, quotient kernels, and cached mat-vec
  implementations.

## Follow-Up Or Non-Code References

- Existing specs and docs mention old CPU cache names for historical context.
  Do not chase those unless the current PR edits the referenced behavior.
- Production Metal, field/MLE kernels, sumcheck backend hooks, and Jolt adapter
  work are out of the current cutover.

## Guard Candidates

After the cutover, add checks that fail when:

- `akita-scheme/src/lib.rs` imports `NttSlotCache`, `MultiDNttCaches`,
  `dispatch_with_ntt!`, or `mat_vec_mul_ntt_*`;
- migrated paths in `api/commitment.rs`, `protocol/flow.rs`,
  `protocol/ring_switch.rs`, or `protocol/quadratic_equation.rs` accept
  `&NttSlotCache<D>`;
- public `CommitmentProver` methods can be called without an explicit backend
  plus typed prepared setup;
- migrated `AkitaPolyOps` impls expose `CommitCache = NttSlotCache<D>`.
