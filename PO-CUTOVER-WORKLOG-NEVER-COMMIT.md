**NEVER COMMIT THIS FILE.**

# PO-CUTOVER Worklog

## Goal & Scope
Track the single long-context PolyOps cutover PR described by `/Users/quang.dao/Documents/SNARKs/akita-polyops-cutover-spec/specs/akita-polyops-cutover.md`.
The work removes `AkitaPolyOps` and the old monolithic prover compute backend boundary after adding source-typed views/kernels for the remaining built-in representations and cutting over all prover/PCS call sites.

## Starting State
- Branch: `quang/po-cutover`
- Worktree: `/Users/quang.dao/Documents/SNARKs/akita-po-cutover`
- Base commit: `d620999e` (`feat(prover): dense root views + CpuBackend kernels (PO-dense)`)
- Related spec: `/Users/quang.dao/Documents/SNARKs/akita-polyops-cutover-spec/specs/akita-polyops-cutover.md`
- Coordination: `PO-CUTOVER` claimed in `/Users/quang.dao/Documents/SNARKs/akita-stack/MANIFEST.md`; locks `FLOW`, `QEQ`, `RSPROTO`, `PROVER-LIB`, and `PO-COMPUTE` held by `PO-CUTOVER` as of `2026-06-08T04:50Z`.

## Plan
1. Inventory PO-dense's source-typed boundary and remaining `AkitaPolyOps` call sites.
2. Add one-hot views and `CpuBackend` kernels, with equivalence tests against `AkitaPolyOps` while it still exists.
3. Add sparse-ring, root tensor projection, and multilinear dispatch views/kernels.
4. Add recursive witness opening/commit support without making recursive witnesses root polynomials.
5. Add ring-switch relation/quotient source views and `CpuBackend` kernels.
6. Cut over prover API/protocol call sites to operation contexts and source-typed kernels.
7. Cut over setup/PCS/examples/benches/tests, including custom-source and mixed-stack contract tests.
8. Delete `AkitaPolyOps`, its blanket impl, per-backend impls, and the old monolithic backend trait ladder.
9. Run forbidden greps, targeted tests, full green gate, byte-equality fixture, no-default-features checks, and profiles.

## Decisions
- **[2026-06-08] Ring-switch kernel module location.**
  Chose `crates/akita-prover/src/backend/ring_switch.rs` for the new source views and `CpuBackend` ring-switch kernel impls.
  Reason: the manifest gives PO-CUTOVER ownership of both the backend kernel module and `protocol/ring_switch.rs`, and placing the kernel beside other backend representation views keeps protocol files focused on flow state.

## Additional Perspectives
- **[2026-06-08] Streaming-commit boundary + Jolt trace-only prior art.**
  Added an appendix to the polyops-cutover spec ("Streaming Commitment And The Commit-Source Boundary"): the source-typed kernel boundary doubles as the streaming-commit seam for a future Jolt-style trace-driven commit. Rules captured: no separate streaming trait, the generic source `S` is the seam, the backend owns the block-sweep strategy, and the row-encoding palette stays provisional (never load-bearing because `S` is open).
  Concrete prior art for this exact shape is the `lz/integrate-hachi` branch in the `jolt` repo (worktree `/Users/quang.dao/Documents/SNARKs/jolt-hachi`), files `jolt-core/src/poly/commitment/hachi/{commitment_scheme.rs,packed_poly.rs,packed_layout.rs}` + `jolt-core/src/zkvm/prover.rs:167` (`LazyOneHotSource`). There a `JoltPackedPoly` implements the old monolithic `HachiPolyOps` over closures (`index_fn`/`batch_fn`) backed by `LazyOneHotSource { trace: &[Cycle], polys: &[CommittedPolynomial], ... }`, committing the whole one-hot mega-poly directly from the trace with zero poly materialization.
  Cutover relevance: the packed-poly's overridden `commit_inner` strategies (fast-singleton / column-sweep / tiled, `packed_poly.rs:751-1157`) are what must live in the `CpuBackend` commit kernel keyed on the source's traversal kind; the layout-driven trace walk (`PackedBitLayout::locate` + `for_each_entry_in_block`, `packed_poly.rs:249-327`) is exactly the per-block `CommitTraversal` the source must own. This source does NOT fit a small fixed `CommitRow` enum (its block "rows" are layout-mapped `(pos_in_block, coeff_idx)` sparse entries with a monomial-rotation accumulate, not contiguous slices), which is concrete support for keeping the encoding palette non-load-bearing and the generic `S` the real extension point.

## Deviations
- **[2026-06-08] Temporary `RootCommitPoly` algorithm methods (reverted this session).**
  The coherent commit-API slice introduced `RootCommitPoly::{commit_inner_witness, root_tensor_projection}` with per-type impls in `backend/root_commit_poly.rs` that delegated to `AkitaPolyOps`, because generic `commit<P>` could not prove `CpuBackend: RootCommitBackend<F, P, …>` when calling `RootCommitKernel` directly.
  This was a **public-bound cutover with an internal shim**, not the spec's kernel cutover. It reintroduced algorithm dispatch on the polynomial side, which the spec explicitly rejects (algorithms belong on backend kernels over `commit_view()` / `tensor_view()`).
  **Fix in flight:** remove algorithm methods from `RootCommitPoly` (marker bundle only); wire `api/commitment.rs` through `commit_view()` → `RootCommitKernel` and `tensor_view()` → `TensorProjectionKernel`; keep `RootCommitPolys` for rustc inference; use `RootCommitBackend` on scheme entry points and `RootCommitSource` + `for<'a> RootCommitKernel<…>` on `commit_with_params`.

## Tradeoffs
- **[2026-06-08] Worklog filename.**
  The default `WORKLOG-NEVER-COMMIT.md` is ignored by a shared git exclude rule, so this worklog uses `PO-CUTOVER-WORKLOG-NEVER-COMMIT.md`.
  This preserves a visible, untracked, non-ignored scratch file without changing any ignore configuration.

## Open Questions
- **[2026-06-08] Council review surfaced two items previously marked "none blocking".**
  Batch kernels in `sparse_ring.rs` and `field_reduction.rs` UFCS-delegate into `AkitaPolyOps`; fix before cutover continues so PO4 deletion is not blocked.
  `multilinear_polynomial.rs` and `field_reduction.rs` kernel dispatch have near-zero unit coverage; add oracle tests before call-site cutover.

## Council Review (2026-06-08)

Prior agent session: [PO-CUTOVER handoff](5f6ad251-7e3e-431d-8117-97c010cdf936). Additive backend boundary pass complete (~2146 uncommitted lines); cutover/delete not started (~30% of MANIFEST node).

### MANIFEST alignment
| Done | Missing |
|---|---|
| one-hot / sparse-ring / multilinear / field-reduction / recursive-witness views+kernels | `api/commitment.rs`, `api/scheme.rs`, `protocol/flow/**`, `ring_relation.rs` cutover |
| `backend/ring_switch.rs` (85-line shim) | Delete `AkitaPolyOps` + old backend ladder (`lib.rs`, `compute/backend.rs`) |
| 51 backend unit tests green | `commitment_contract.rs`, mixed-backend test, byte-equality fixture, full green gate, profiles |

No manifest violations: all edits stay in owned `backend/**`.

### Design verdict
- **Right direction:** open source-typed kernel boundary (`compute/kernels.rs`), `Copy` borrowed views, recursive witness not modeled as root poly, ring-switch at `backend/ring_switch.rs`.
- **Intentional additive debt:** dual `AkitaPolyOps` + kernel impl surfaces until call-site cutover.
- **Fix now (not lazy):** batch kernels UFCS-delegating back into `AkitaPolyOps` in `sparse_ring.rs` / `field_reduction.rs`.
- **Watch:** `sparse_ring.rs` at 1378 lines; `multilinear_polynomial.rs` doc still says it preserves `AkitaPolyOps` interface.

### Performance verdict
Neutral in additive phase: zero-cost views, monomorphized traits, batched fast paths preserved. Verify with profile/bench before PR open.

### Immediate todos (this session)
1. [done] Fix batch-kernel UFCS delegation in `sparse_ring.rs` and `field_reduction.rs`.
2. [done] Add `multilinear_polynomial.rs` dispatch unit tests (homogeneous dense, homogeneous one-hot, mixed fallback).
3. [done] Add `field_reduction.rs` positive-path kernel equivalence tests.
4. [done] Re-run `cargo test -p akita-prover --lib backend` → 57 passed (was 51).
5. [done] Extract `sparse_ring/ops.rs`; update `multilinear_polynomial.rs` module doc.
6. [skipped] Shared dispatch helper for field_reduction/multilinear (defer until after cutover).
7. [next] Call-site cutover (`commitment.rs` → `flow/**` → delete `AkitaPolyOps`).
8. [deferred] Byte-equality fixture + canonical profile before PR.

## Slice Retrospectives

### 2026-06-08 retrospective: one-hot source boundary

**Bottom line:** no blockers. One-hot now exposes root shape, commit, opening, tensor, and direct-witness capabilities, with `CpuBackend` kernels delegating to the existing one-hot paths while `AkitaPolyOps` remains as the oracle.

- `Risk:` The kernel impls currently delegate through the old trait methods for byte identity. This is intentional for the additive phase, but the final delete phase must move or inline those bodies rather than leave a hidden compatibility dependency.
- `Deferred:` Commit-kernel equivalence is covered structurally by delegation but does not yet have a dedicated one-hot commit kernel test with a prepared setup. Add or subsume this when commitment call sites move to operation contexts.
- `Non-issue checked:` Re-exporting one-hot view types through `backend/mod.rs` is required so public associated source types are nameable; after doing so, the `unreachable_pub` warnings disappeared.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-prover backend::onehot`
    → `test result: ok. 23 passed; 0 failed; 0 ignored; 0 measured; 142 filtered out; finished in 0.18s`

### 2026-06-08 retrospective: sparse-ring source boundary

**Bottom line:** no blockers. Sparse-ring now exposes root shape, commit, opening, tensor, and direct-witness capabilities, with `CpuBackend` kernels preserving the current dense tensor fallback and tensor-shaped sparse decompose-fold path.

- `Risk:` Sparse-ring tensor projection still materializes the direct root witness through the old default behavior. This matches current semantics, but the final `AkitaPolyOps` deletion must preserve the same `AkitaError` paths when that fallback moves under the tensor kernel.
- `Deferred:` Commit-kernel equivalence remains structurally covered by delegation but lacks a prepared-setup test, same as the one-hot slice. This should be covered when commit call sites and contract tests move onto operation contexts.
- `Non-issue checked:` Sparse-ring shape getters needed to be inherent to avoid the temporary `AkitaPolyOps`/`RootPolyShape` method ambiguity during the additive phase.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-prover backend::sparse_ring`
    → `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 161 filtered out; finished in 0.00s`

### 2026-06-08 retrospective: Akita-owned dispatch wrappers

**Bottom line:** no blockers. `RootTensorProjectionPoly` and `MultilinearPolynomial` now expose the new capability/kernel boundary through Akita-owned dispatch views.

- `Risk:` `RootTensorProjectionPoly` still delegates most operations through the old trait during the additive phase. The final delete phase must route those dispatch arms through dense/sparse kernels directly.
- `Non-issue checked:` `MultilinearPolynomial` tensor kernels dispatch directly to dense/one-hot tensor kernels instead of old defaults, so homogeneous one-hot batches preserve sparse tensor witness behavior.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-prover backend::field_reduction`
    → `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 168 filtered out; finished in 0.00s`
  - `cargo test -p akita-prover backend::multilinear_polynomial`
    → `test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 169 filtered out; finished in 0.00s`

### 2026-06-08 retrospective: recursive witness operation view

**Bottom line:** no blockers. Recursive witnesses now have a layout-carrying `RecursiveWitnessOpeningView` consumed by opening and commit kernels, without implementing any root-polynomial capability.

- `Risk:` The view validates `num_blocks` as a nonzero power of two, matching the recursive layout assumptions. Call-site cutover must pass the same `num_blocks` already used by recursive fold/decompose/commit helpers, not recompute a different layout.
- `Deferred:` Recursive commit kernel has no prepared-setup equivalence test yet. It should be covered when `commit_w` / `commit_next_w` are cut over to operation contexts and mismatch tests are added.
- `Non-issue checked:` Opening kernel equivalence is direct against `RecursiveWitnessView::evaluate_and_fold_ring`, so no root-poly trait was introduced for recursive witness data.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-prover backend::recursive_witness`
    → `test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 167 filtered out; finished in 0.00s`

### 2026-06-08 retrospective: ring-switch source views

**Bottom line:** no blockers. Added `backend/ring_switch.rs` with borrowed relation and quotient views plus `CpuBackend` source-typed relation/quotient kernels that reduce to the current fused row plans.

- `Risk:` The kernels still call the old fixed `RingSwitchComputeBackend` methods as lower-level helpers. This is acceptable for the additive phase, but the public fixed trait must still be removed when protocol call sites are cut over.
- `Non-issue checked:` The new views carry only borrowed witness data and norm metadata; scalar row counts and log basis remain in `RingSwitchRelationPlan` / `RingSwitchQuotientPlan`.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-prover ring_switch`
    → `test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 168 filtered out; finished in 0.01s`

### 2026-06-08 checkpoint: additive backend boundary pass

**Bottom line:** no blockers. The additive representation/kernel pass compiles across backend filters before any public call-site cutover.

- `Non-issue checked:` Touched backend files remain under the 1500-line cap; largest is `crates/akita-prover/src/backend/sparse_ring.rs` at 1378 lines.
- `Verification:`
  - `cargo test -p akita-prover backend`
    → `test result: ok. 51 passed; 0 failed; 0 ignored; 0 measured; 119 filtered out; finished in 0.17s`
  - `wc -l crates/akita-prover/src/backend/field_reduction.rs crates/akita-prover/src/backend/multilinear_polynomial.rs crates/akita-prover/src/backend/sparse_ring.rs crates/akita-prover/src/backend/onehot/ops.rs crates/akita-prover/src/backend/onehot/tests.rs crates/akita-prover/src/backend/recursive_witness.rs crates/akita-prover/src/backend/ring_switch.rs`
    → `619`, `827`, `1378`, `947`, `1149`, `536`, `85`

### 2026-06-08 retrospective: batch-kernel UFCS removal

**Bottom line:** no blockers. Batch kernels in `sparse_ring.rs` and `field_reduction.rs` no longer UFCS-delegate into `AkitaPolyOps`; they route through inner kernel traits or inherent poly methods.

- `Risk:` `field_reduction.rs` `OpeningBatchKernel` / `TensorProjectionBatchKernel` now mirror `multilinear_polynomial.rs` homogeneity dispatch (~90 lines duplicated). Acceptable for now; a shared dispatch helper could shrink this at cutover time, but only if it does not obscure per-representation fast paths.
- `Deferred:` `RootTensorProjectionPoly` `AkitaPolyOps` impl still UFCS-delegates tensor batched folds to inner dense/sparse (lines ~572–590). That block deletes with `AkitaPolyOps`; not in the new kernel path.
- `Non-issue checked:` Sparse-ring sparse `decompose_fold_batch` correctly returns `Ok(None)` (no batched sparse kernel exists); tensor path calls `SparseRingPoly::decompose_fold_tensor_batched` directly.
- `Verification:` `cargo fmt -q`; `cargo test -p akita-prover --lib backend` → 51 passed before new tests landed.

### 2026-06-08 retrospective: dispatch unit tests

**Bottom line:** no blockers. Added 4 multilinear dispatch tests + 2 field-reduction kernel tests; backend filter now 57 passed.

- `Risk:` Mixed multilinear batch test required matching `num_vars` between dense and one-hot (one-hot `num_vars=6` from 8×8 layout, not 5). Easy to write a bogus mixed-batch fixture that fails for shape reasons unrelated to dispatch.
- `Risk:` Field-reduction batch test must oracle against `RootTensorProjectionPoly` batch methods, not inner `DensePoly` batch (transformed roots have different partials). Initial test draft compared the wrong layers; caught at compile/run time.
- `Deferred:` No sparse-arm positive-path test for `RootTensorProjectionPoly` kernels yet (only dense tensor paths). Sparse projection roots need a one-hot→sparse-ring fixture.
- `Deferred:` Commit-kernel prepared-setup equivalence still absent across all backends.
- `Verification:` `cargo test -p akita-prover --lib backend` → 57 passed.

### 2026-06-08 retrospective: sparse_ring ops extraction + multilinear doc

**Bottom line:** no blockers. `sparse_ring.rs` dropped from 1378 → 1051 lines; views/kernels live in `sparse_ring/ops.rs` (351 lines). `MultilinearPolynomial` doc now describes the source-typed boundary instead of "preserves AkitaPolyOps".

- `Risk:` `DirectRootWitnessSource` body is duplicated between `ops.rs` and `AkitaPolyOps::direct_root_witness` in the parent file. Acceptable until `AkitaPolyOps` delete; then keep only the capability trait impl.
- `Deferred:` Shared homogeneous-batch dispatch helper for `field_reduction` / `multilinear_polynomial` (explicitly after cutover).
- `Non-issue checked:` `sparse_ring` tests needed trait imports in the `#[cfg(test)]` module after moving impls to `ops`.
- `Verification:` `cargo test -p akita-prover --lib backend` → 57 passed.

## Spec Comparison (2026-06-08, post coherent commit slice)

**Verdict:** directionally aligned on public commit surface; **not** spec-complete; **partially contradicts design intent** via the temporary `RootCommitPoly` shim (reverted this session).

### Aligned with spec
| Item | Status |
|---|---|
| Public `commit` / `batched_commit` off `P: AkitaPolyOps` | Done (trait bound) |
| Capability traits (`RootPolyShape`, `RootCommitSource`, …) | Done (PO1 skeleton) |
| `OperationCtx` exists | Partial (ctx threaded; kernel consumption was missing on hot path) |
| No `StreamingCommitment` trait | OK |
| `CommitView<'a>` unconstrained GAT | OK |
| Verifier does not name poly ops | OK |
| Built-in reps compile on new boundary | OK at commit API |
| Appendix: no closed source enum in kernel sig | OK |
| Appendix: `CommitTraversal` blanket not added speculatively | OK |

### Gaps vs spec acceptance criteria
| Criterion | Status |
|---|---|
| `rg "AkitaPolyOps" crates` → no matches | FAIL (~30+ files) |
| Delete `AkitaPolyOps` + monolithic backend ladder | Not started |
| Commit APIs: `RootCommitSource` + `RootCommitKernel` bounds | **Done** (Slice 1) |
| `commit_with_params` capability-minimal (`RootCommitSource` only) | **Done** |
| Heterogeneous operation stack / mixed-backend test | Not done |
| `commitment_contract.rs` custom source canary | Not updated |
| Prove/flow cutover | Not started |
| Tensor via `TensorProjectionKernel` not poly methods | **Done** on commit path (Slice 1) |

### Appendix surprises / limitations
1. **Hot commit path bypassed kernels** — `commitment.rs` called `poly.commit_inner_witness` → `AkitaPolyOps`, not `RootCommitKernel`. Built-in kernels existed but were dead on scheme commit path.
2. **`RootCommitPoly` algorithm methods** — wrong layer per spec; rustc inference workaround that must not ship. Spec-correct fix: `RootCommitPolys` + `RootCommitBackend` at monomorphized scheme sites; `commit_view()` + kernel in free functions.
3. **Over-bound `RootCommitPoly`** — bundled `RootTensorSource` for all commits; spec says tensor only when transform runs. Scheme `commit` may keep `RootCommitPoly` marker; `commit_with_params` should be `RootCommitSource` only.
4. **Streaming appendix item 3 (MUST)** — commit must consume source via kernel through operation context; was not satisfied until kernel rewire.
5. **Orphan rule / Jolt trace-only** — unchanged; downstream still needs `OneHotBlocks` single-materialization or future `CommitTraversal` blanket. Not blocking cutover.
6. **`CpuBackend` pinned on `CommitmentProver::commit`** — fixed in Slice 1 (generic `B`); see locked design above.

## Locked design: Alternative 3 commit path (2026-06-08)

**Decision:** Open extension via generic `P` + generic `B` + source-typed kernels. **No `RootCommitDispatch`.** **Never pin `CpuBackend` on generic `commit<P>` / `batched_commit<P>`.**

### HRTB (Higher-Ranked Trait Bound)

`for<'a> RootCommitKernel<<P as RootCommitSource>::CommitView<'a>, F, D>` on **`B`** means: for every borrow lifetime `'a`, backend `B` implements the commit kernel for `P`'s borrowed view at that lifetime.

### Rust rule (load-bearing)

| Pattern | Compiles? |
|---------|-----------|
| `fn commit<P, B>(backend: &B, …) where B: for<'a> RootCommitKernel<P::CommitView<'a>, …>` | ✅ obligation at call site when `P` and `B` are concrete |
| `fn commit<P>(backend: &CpuBackend, …) where CpuBackend: for<'a> RootCommitKernel<P::CommitView<'a>, …>` | ❌ `P` abstract → cannot prove kernel for all `CommitView`s |
| `RootCommitDispatch` per-type shim | ❌ rejected interim; delete |

Call sites still pass `&CpuBackend`; `B` is inferred. Pinning `CpuBackend` only in leaf code (kernel impls, tests) is fine.

### API shape

| Surface | `P` bound | `B` bound |
|---------|-----------|-----------|
| `commit_with_params` | `RootCommitSource` | `CommitmentComputeBackend` + HRTB `RootCommitKernel<P::CommitView<'a>, …>` |
| `commit` / `batched_commit` / `CommitmentProver::commit` | `RootCommitPoly` | `RootCommitBackend<F, P, E, D>` |
| Tensor transform | `RootCommitPoly` (source) | same `B`; commit transformed via `RootCommitKernel` on `RootTensorProjectionPoly::CommitView` (included in `RootCommitBackend`) |

### Implementation path

- **Slice 1 (this session):** delete `commit_dispatch.rs`; unify `commitment.rs` on one `commit_with_validated_params<P, B>`; generic `B` on `commit` / `batched_commit` / `CommitmentProver`; `commit_view()` → `RootCommitKernel`, `tensor_view()` → `TensorProjectionKernel`.
- **Slice 2:** `protocol/flow/**` prove cutover, same `P` + `B` pattern with opening/tensor kernel bundles.
- **Slice 3:** delete `AkitaPolyOps`, move kernel bodies off delegation.
- **Slice 4:** `commitment_contract.rs`, mixed-backend test, Jolt orphan-rule doc.

### Explicit non-goals for Slice 1

- No enum bundle at scheme boundary (closes extension).
- No poly-side algorithm traits (`RootCommitPoly` stays marker-only).
- Kernel impls may still delegate to `AkitaPolyOps` internally until Slice 3.

## Slice 1 complete (2026-06-08)

### Shipped
- Deleted `backend/commit_dispatch.rs` and all `RootCommitDispatch` exports/call sites.
- `commit` / `batched_commit` / `CommitmentProver::{commit,batched_commit}` are generic over `P: RootCommitPoly` + `B: RootCommitBackend<F, P, E, D>`.
- `commitment.rs` unified on `commit_with_validated_params<P, B>`; tensor path uses `tensor_view()` → `TensorProjectionKernel`, inner path uses `commit_view()` → `RootCommitKernel`.
- `RootCommitPolys::commit_with` pins `P` from `self` for inference.
- **`CommitmentProver` parameter order:** `bundle` / `polys_per_point` before `backend` + `prepared` so rustc fixes `P` before checking `B: RootCommitBackend` (trait-solver limitation).
- Added `commit_multilinear_polynomials` for borrowed [`MultilinearPolynomial`] batches: generic `for<'a> RootCommitKernel<P::CommitView<'a>>` still forces `'static` on wrapper lifetimes in the current solver; the helper ties `'p` explicitly via UFCS on `MultilinearPolynomialView<'_, 'p, …>`.

### Verification
```bash
cargo build -p akita-pcs --tests
cargo test -p akita-pcs --lib scheme::tests
cargo test -p akita-pcs --test single_poly_e2e --test batched_aggregated_e2e
rg "RootCommitDispatch|commit_inner_witness_via_kernel|commit_with_validated_params_cpu" crates  # empty
```

All above green on `quang/po-cutover` worktree.

### Must-fix polish (2026-06-08, pre-commit)
- Rustdoc on `CommitmentProver::{commit,batched_commit}` (param order + multilinear pointer).
- Rustdoc on `commit_multilinear_polynomials` (HRTB / `'static` solver note, tensor rejection).
- `commit_matches_commit_with_params_on_dense_poly` in `commitment.rs` tests.
- `commit_multilinear_polynomials_rejects_tensor_projection_schedule` in `fp32_ring_subfield.rs`.

### Multilinear wrapper fix (2026-06-08, post–Slice 1)

**Problem:** single-lifetime view collapse (`MultilinearPolynomialView<'a>`) was insufficient. HRTB `for<'view> RootCommitKernel<P::CommitView<'view>>` still forced stack borrows to `'static` when `P` carried a lifetime (`MultilinearPolynomial<'p, …>`).

**Fix:** owned-by-move enum (no lifetime on `P`):
- `MultilinearPolynomial<F, D, I>` owns `Dense(DensePoly)` or `OneHot(OneHotPoly)`; constructors `dense(poly)` / `onehot(poly)` take by move (no clone).
- Views are `MultilinearPolynomialView<'a>` borrowing the owned enum; GATs use `<'view>` only.
- Deleted `commit_multilinear_polynomials` and duplicated commit orchestration; mixed batches use generic `commit<P, B>` / `CommitmentProver::commit` like `DensePoly`.
- Call sites updated: `MultilinearPolynomial::dense(dense_a)` not `dense(&dense_a)`; e2e/fp32 tests use layout-aligned one-hot construction.

**Semantic note:** wrapping moves the inner poly; callers needing both wrapped and unwrapped handles must keep separate references before wrap.

### Slice 1 debt / follow-ups
- ~~`batched_prove` still on `AkitaPolyOps` (Slice 2).~~ Done in Slice 2 (`cb0459d7`).
- Kernel bodies still delegate to `AkitaPolyOps` (Slice 3, in flight).
- Update spec comparison table rows marked "fix in flight" → done for commit path.

## Slice 2 complete (2026-06-08)

### Shipped (`cb0459d7`)
- `RootProvePoly` / `RootProveBackend<F, P, ClaimE, ChallengeE, D>` prove cutover mirroring Slice 1 commit.
- `protocol/flow/poly_kernels.rs` prove dispatch; `compute/dispatch.rs` shared `tensor_root_projection`.
- `CommittedPolynomials { polynomials: &[&P] }` for HRTB inference (decision A deferred).
- `root_fold/` split: `eval.rs`, `finish.rs`, `relation.rs`, `mod.rs` orchestration.
- Extension post-transform dedup (`eval_extension_reduction_post_transform`).
- Review polish: dual-field trait absorbs claim-field batch HRTB (no extra bounds at scheme/inputs).

### Verification
```bash
cargo clippy --all -D warnings
cargo test -p akita-prover --lib
cargo test -p akita-pcs --lib scheme::tests
```

## `F: 'static` on backend bundle traits (2026-06-08)

### Cause
`RootCommitBackend` / `RootProveBackend` close over `for<'a> OpeningFoldKernel<<P as RootOpeningSource>::OpeningView<'a>, …>` (and the same for `RootTensorProjectionPoly`). GATs on [`RootOpeningSource`] use `type OpeningView<'a> where Self: 'a`. For the HRTB to hold for every `'a`, the polynomial type must be `'static`, which propagates to `F: 'static` inside `RootTensorProjectionPoly<F, D>`.

Compiler evidence: removing `F: 'static` from the traits yields **E0311** on `impl RootProveBackend for CpuBackend` with note pointing at `poly.rs` line `Self: 'a`.

### What was *not* needed (experimentally verified)
Removing these still compiles (clippy + prover lib tests):
- `ClaimE: 'static`, `ChallengeE: 'static` on `RootProveBackend`
- `E: 'static` on `prove_root_fold_with_params` / `eval_extension_reduction_post_transform`
- `C: 'static` on challenge-field generic params (when removed consistently across `finish_*` / `root_extension` / `inputs`)
- `Self::ClaimField: 'static` on `CommitmentProver::batched_prove`

Those were bound **contagion** from briefly having extension-field `'static` on the trait, not independent requirements.

### Resolution applied
- Trait docs on `RootProveBackend` / `RootCommitBackend` explain `F: 'static` only.
- Dropped redundant `ClaimE`/`ChallengeE`/`E`/`C` `'static` from trait and call sites.
- **Deferred:** relax GAT `Self: 'a` design (larger refactor; no preset impact).

## Slice 3 (complete)

### Done
- `rg "AkitaPolyOps" crates` → empty (specs/docs only).
- Deleted `AkitaPolyOps` trait + blanket `&P` impl from `lib.rs`; removed `recompose_commit_inner_blocks` (was only used by trait default).
- Backend `impl AkitaPolyOps for T` → inherent `impl T` (dense, onehot, sparse_ring, field_reduction commit dispatch, multilinear wrapper kernels only).
- Dropped dead `MultilinearPolynomial` inherent dispatch block (kernels call inner dense/onehot views).
- `zk_hiding_commit` → `RootCommitKernel` + `CommitInnerPlan`.
- PCS tests/examples/benches/recursion updated to `RootOpeningSource` / `OpeningFoldKernel` / `RootProvePoly`.
- Clippy clean; `cargo test -p akita-prover --lib` → 178 passed.

### Deferred (A / C)
- `RootProveClaims` wrapper vs `&[&P]`
- `batched_prove` / `batched_commit` API symmetry

## Follow-ups
- Rebase this branch as base PRs land.
- Slice 4: `commitment_contract.rs`, mixed-backend test, delete monolithic backend ladder in `compute/backend.rs`, forbidden grep gates, full workspace `cargo test`/clippy.
