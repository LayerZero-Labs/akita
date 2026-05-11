# Spec: Extension-Field Trace Cutover and Verifier Surface Tightening

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-08 |
| Status | implementation |
| PR | #71 (`quang/general-field-final`) |
| Predecessor | `specs/extension-claim-incidence-cutover.md` (#69) |
| Companion completion plan | `specs/extension-field-opening-batching.md` (#71 expanded scope) |

## Summary

First slice of Phase 4 of the extension-field opening cutover: the production Hachi `psi` packing, the production fixed-subfield `embed_subfield` element embedding, and the production `Tr_H` inner-product check, all const-generic over the ring dimension `D` and extension degree `K`. The verifier root trace check is rewritten to consume typed `Cfg::ClaimField` openings end to end and dispatched at runtime to the matching `K` monomorphization. The explicit degree-one bridge for the trace check (`claim_values_to_base`) is dropped per the no-backward-compat policy.

PR #71 has since been expanded to include the remaining extension-field opening completion work. The companion spec `specs/extension-field-opening-batching.md` is the implementation plan for that expansion and establishes the field tower naming convention:

```text
F ⊆ E ⊆ L
```

where `F = Cfg::Field`, `E = Cfg::ClaimField`, and `L = Cfg::ChallengeField`.

A separate behavior-preserving theme rides with this PR: an audit of the `akita-verifier` crate's public surface, demoting 18 of 26 re-exports to `pub(crate)` and shrinking the lib-root re-export list to the 5 items downstream crates actually use plus 3 documented test-only items. The audit was triggered by concern that the original crate-decomposition PR mechanically lifted everything that used to be `pub` to crate-public without re-tightening, and was shipped together because the Phase 4 trace-check work touched the same files and would otherwise have grown the surface further.

PR #71 Part 1 after this trace slice has now landed the proof-scalar payload
reshape for stage-1/stage-2 and recursive proof state: proof containers carry
`F` ring material and `L` proof-scalar material, and
`AkitaCommitmentScheme` exposes `AkitaBatchedProof<F, Cfg::ChallengeField>`.
The companion completion spec is now PR #71 Part 2. Since this trace-cutover
slice, Part 2 has also lifted root folded `gamma` into `L`, added a root-direct
fallback for valid extension openings outside the packed-inner folded shape,
and landed field-family SIS sizing plus larger small-field profile candidates.
The remaining design-sensitive work is true recursive extension-valued
materialization, bridge removal, norm documentation, early parameter
validation, Frobenius compression, and generated planner defaults.

## Intent

### Goal

Replace the K=1-only Hachi trace-check shortcut with the production K-generic implementation, route the verifier through it without losing the K=1 hot-path performance, and drop the explicit degree-one bridge that was making the trace check refuse `K > 1` claim fields. Use the same code path for the recursive levels' trace check so the verifier surface has one entry point rather than two parallel implementations.

Independently, restore the verifier crate's surface to "only what downstream actually uses" without changing behavior.

### Scope Boundary

- The K-generic primitives in `akita-types::field_reduction` (`embed_subfield<F, D, K>`, `psi_embed<F, D, K>`, `check_trace_inner_product<F, D, K>`, `dispatch_trace_inner_product_check<F, D>`, `SubfieldParams<D, K>`) become the production embedding/trace path; reference helpers from PR #60 are graduated, not parallel-tracked.
- The root verifier (`verify_root_level`) computes the per-point gamma-batched opening in `E = Cfg::ClaimField` directly via `opening.mul_base(g)`, then projects to base coordinates with `to_base_vec()` for the trace identity. In this first slice `gamma` remains in `F`; the companion completion spec lifts it to `L = Cfg::ChallengeField`.
- The recursive-level verifier (`verify_one_level`) routes its K=1 check through the same runtime-K dispatcher so the verifier has one trace-check entry point. Recursive proof scalars stay base-field in this first slice; the companion completion spec makes payload structs generic over `L`.
- The verifier crate's submodules (`proof`, `protocol`, `stages`) become `mod` rather than `pub mod`; only the externally-used items remain `pub` re-exports. Three items (`prepare_m_eval`, `PreparedMEval`, `AkitaStage1Verifier`) stay `pub` solely for integration tests in `akita-pcs` and are documented as such.

### Invariants

- fp128 production behavior is unchanged. The K=1 path inside `check_trace_inner_product` keeps the O(1) scalar shortcut.
- The verifier has exactly one trace-check entry point on the live path: `dispatch_trace_inner_product_check`.
- `claim_values_to_base` is removed entirely (no callers remain after this PR; the function existed only as the explicit K=1-or-die bridge for the trace check).
- `psi_embed` and `embed_subfield` are const-generic over `D` and `K`, so the inner loops monomorphize and unroll. There is no runtime-K version of the embedding or `trace_h` inside the helper module; runtime-K dispatch lives only at the verifier-extension boundary.
- The verifier crate's `Cargo.toml` continues to depend only on `akita-algebra`, `akita-challenges`, `akita-field`, `akita-sumcheck`, `akita-transcript`, `akita-types`, and `tracing`. The surface tightening does not add new incoming surface from `akita-types`.
- Behavior is preserved across the surface tightening: `cargo nextest run --workspace` produces 572 passed / 0 failed / 3 skipped before and after.

### Historical First-Slice Non-Goals

- This slice did not lift `gamma` (root same-point batching) or
  `batching_coeff` (stage-2) into `L = Cfg::ChallengeField`; PR #71 Part 1 has
  since lifted `batching_coeff` and stage payload scalars to `L`, while
  `gamma` remains part of the Part 2 design.
- This slice did not make `AkitaStage1Proof`, `AkitaStage2Proof`,
  `AkitaLevelProof`, or `AkitaBatchedProof` generic over `L`; PR #71 Part 1
  has since landed that reshape.
- This slice did not lift recursive suffix opening points to `E` or `L`; PR
  #71 Part 1 now stores recursive suffix challenges/openings as `L`, while the
  materialization of true extension-valued recursive ring openings remains a
  Part 2 task.
- This slice does not document norm behavior for `k = 1` vs `k > 1`; that is tracked by the companion completion spec.
- This does not add direct algebra tests against extension-field inner products at the verifier-orchestration level (Phase 4 spec line 793). Direct algebra coverage at the helper level is in place from PR #60.
- This does not implement early rejection of invalid ring/extension parameter combinations at the scheme/setup boundary (Phase 4 spec line 794).
- This slice does not implement the Frobenius-conjugate base/ext optimization (Phase 5).

## Evaluation

### Acceptance Criteria

Production trace primitives:

- [x] `SubfieldParams<D, K>` is a zero-sized witness type whose `new()` validates `D` is a nonzero power of two, `K` is nonzero and divides `D / 2`, and `4K + 1` is invertible mod `2D`.
- [x] `embed_subfield<F, D, K>(params, &[F; K]) -> CyclotomicRing<F, D>` writes `2K - 1` coefficients (slot 0 plus the `K - 1` shifted positive/negative pairs) and is the production single-element embedding.
- [x] `psi_embed<F, D, K>(params, &[[F; K]; D/(2K)]) -> CyclotomicRing<F, D>` is branchless and unrolls over `K` and `D/(2K)`.
- [x] `check_trace_inner_product<F, D, K>` evaluates `Tr_H(trace_input) == (D/K) * embed_subfield(opening_coords)` in `R_q`, with a `const if K == 1` shortcut that reduces to a single scalar equality on `coefficient[0]` and avoids the `trace_h` traversal.
- [x] `dispatch_trace_inner_product_check<F, D>(trace_input, opening_coords, error)` selects the matching `check_trace_inner_product<F, D, K>` for `K ∈ {1, 2, 4, 8}` based on the runtime length of `opening_coords` and returns the supplied error for unsupported `K`.

Verifier root trace check:

- [x] `verify_root_level` no longer calls `claim_values_to_base`; openings stay typed as `&[E]` where `E = Cfg::ClaimField`.
- [x] The per-point gamma-batched opening is summed in `E` via `opening.mul_base(g)` (`gamma` itself stays in `F`).
- [x] The trace check feeds `batched_opening.to_base_vec()` into `dispatch_trace_inner_product_check::<F, D>`. `K` equals `<E as ExtField<F>>::EXT_DEGREE` at runtime.
- [x] `claim_values_to_base` is removed from `akita-types`; no remaining callers.

Verifier recursive-level trace check:

- [x] `verify_one_level` routes its K=1 trace check through the same `dispatch_trace_inner_product_check` rather than calling `check_trace_inner_product::<F, D, 1>` directly.
- [x] The K=1 shortcut inside `check_trace_inner_product` continues to be exercised in this path, so recursive levels keep the O(1) scalar comparison.

Verifier surface tightening:

- [x] `crates/akita-verifier/src/lib.rs` declares `proof`, `protocol`, `stages` as `mod` (not `pub mod`).
- [x] The lib-root `pub use` list exposes exactly: `CommitmentVerifier`, `CommittedOpenings`, `VerifierClaims` (re-exports from `akita-types`); `verify_batched_with_policy`; `direct_witness_opening_matches`; plus `prepare_m_eval`, `PreparedMEval`, `AkitaStage1Verifier` documented as test-only carve-outs.
- [x] The 18 internal-only items previously exposed as `pub use` (per the audit table in this PR's commit message for `e858d79`) are demoted to `pub(crate)` declarations on their definitions.
- [x] Intra-crate imports use explicit submodule paths (`crate::protocol::levels::Foo` etc.) rather than re-exports through the crate root.
- [x] Workspace-wide `cargo clippy --all -- -D warnings` is green; the `unreachable_pub` lint catches accidental backsliding.

Compatibility and CI:

- [x] All 572 workspace tests pass on each of the three thematic commits independently.
- [x] `cargo fmt -q` clean.
- [x] `cargo clippy --all --message-format=short -q -- -D warnings` clean.
- [x] GitHub CI green on PR head.

### Testing Strategy

Existing tests that must continue passing:

- `cargo nextest run --workspace --no-fail-fast`
- `cargo test -p akita-types field_reduction` (anchors `embed_subfield`, `psi_embed`, `check_trace_inner_product` for K ∈ {1, 2, 4})
- `cargo test -p akita-verifier`
- `cargo test -p akita-scheme fp128_degree_one_batched_proof_roundtrip_is_stable`

Targeted tests added in this PR:

- [x] Trace identity tests for K ∈ {1, 2, 4} over `RingSubfieldFp4` exercising `Tr_H(Y * sigma_{-1}(V)) == (D/K) * embed_subfield(<s, v>)` directly in `crates/akita-types/src/field_reduction.rs`.
- [x] `dispatch_trace_inner_product_check` is exercised indirectly through every existing batched/recursive verifier test (572 tests).

### Performance

The K=1 hot path keeps its O(1) scalar comparison via the `const if K == 1` shortcut inside `check_trace_inner_product`. The runtime-K dispatch at the verifier boundary adds a single `match` on a small enum of supported `K` values (compiled to a jump table) and is not on a hot inner loop; one trace check is performed per opening point per level.

The K-generic embedding and trace-h paths are exercised only for `K > 1`, which is not yet on a production e2e profile in this repo. They are designed to monomorphize: with `K` const-generic, the inner loops in `psi_embed`, `embed_subfield`, and `trace_h` unroll fully.

The verifier surface tightening is behavior-preserving and has no measurable performance impact.

## Design

### K-Generic Trace Primitives

`SubfieldParams<const D, const K>` is a zero-sized validation witness. Both `D` and `K` are compile-time constants, so every loop in the helpers (`psi_embed`, `embed_subfield`, `trace_h`) monomorphizes per `(D, K)`. The validator rejects malformed parameters before the algorithms run; downstream callers can `unwrap` or propagate the validation error once and not re-validate at every call site.

`embed_subfield<F, D, K>(params, &[F; K]) -> CyclotomicRing<F, D>` writes the `2K - 1` non-zero coefficients of `c_0 + sum_{j=1}^{K-1} c_j (X^{j*step} + X^{-j*step})` directly. The slot-0 specialization avoids the full `psi_embed` traversal when only one subfield element is being embedded (the verifier's case).

`check_trace_inner_product<F, D, K>` is the live verifier predicate. The `const if K == 1` shortcut reduces the check to `D * trace_input.coefficients()[0] == D * opening_coords[0]`, matching the previous `trace::<F, D>` shortcut bit-for-bit. For `K > 1` it falls back to `trace_h(...) == (D/K) * embed_subfield(...)` as a ring equality.

`dispatch_trace_inner_product_check<F, D>` provides the boundary between the const-generic algebra and the verifier's run-time-known `K`. It dispatches on `opening_coords.len()` to one of the four currently-supported monomorphizations (`K ∈ {1, 2, 4, 8}`) and rejects unsupported values with the supplied error. Adding a new `K` is a one-line `match` arm.

### Root Trace Check Without The Degree-One Bridge

The pre-PR sequence in `verify_root_level` was:

1. Project openings to base scalars: `let base_openings = claim_values_to_base::<F, E>(openings, ...)` (rejects `K > 1` at runtime).
2. Sum gamma-batched openings in `F`: `batched_openings_per_point[point_idx] += gamma * base_opening`.
3. Trace check: `check_trace_inner_product::<F, D, 1>(params, &trace_input, &[batched_opening])`.

The post-PR sequence is:

1. (No projection step.)
2. Sum gamma-batched openings in `E`: `batched_openings_per_point[point_idx] += opening.mul_base(g)` where `gamma_i: F` and `opening_i: E`. Result lives in `E` because `MulBase<F>` returns `E`.
3. Trace check: `coords = batched_opening.to_base_vec(); dispatch_trace_inner_product_check::<F, D>(&trace_input, &coords, AkitaError::InvalidProof)?`.

For `E = F` (degree-one bridge in fp128), `to_base_vec()` returns one coordinate, the dispatcher routes to `check_trace_inner_product::<F, D, 1>`, and the K=1 shortcut runs. The fp128 path is bit-identical.

For `E = Fp_{q^k}` with `k ∈ {2, 4, 8}`, `to_base_vec()` returns `k` coordinates and the dispatcher routes to the corresponding monomorphization. No runtime check fails; the trace identity is verified in `R_q` exactly as the spec describes.

### Historical Note: Why Gamma Initially Stayed In `F`

`gamma` in `F` makes `gamma * opening_E` live in `E`, which is what the trace
identity needs in the current base-ring `y_rings` relation. PR #71 Part 1 has
already lifted `s_claim`, `next_w_eval`, stage proofs, and recursive proof
state into `L`, but it deliberately does not decide how an `L`-valued root
batching coefficient should be materialized against the base-ring relation.
That decision belonged to the Part 2 extension-opening completion spec and has
now landed for the folded-root path.

The soundness implication is that root batching still gives only `|F|`-bit
soundness until Part 2 lifts `gamma` or replaces the materialization path with
an explicitly justified reduction. For fp128 this is still 128-bit batching
soundness. For fp32-base configs with extension-valued `E`, this would only be
32-bit root-batching soundness. The current PR head no longer relies on that
historical bridge for folded roots; recursive extension materialization remains
the outstanding bridge.

### Verifier Surface Audit

The audit found the verifier crate re-exporting 26 items, of which only 5 had external callers (`CommitmentVerifier`, `CommittedOpenings`, `VerifierClaims` from `akita-types`; `verify_batched_with_policy` for `akita-scheme`; `direct_witness_opening_matches` for `akita-scheme` tests). 18 items were `pub` but used only inside the crate. 3 more were `pub` solely so integration tests in `akita-pcs` could reach replay primitives in isolation (`prepare_m_eval`, `PreparedMEval`, `AkitaStage1Verifier`).

The cleanup:

1. `pub mod` → `mod` for the three submodules in `lib.rs`.
2. The lib-root `pub use` list keeps the 5 + 3 items above, with the test-only carve-outs documented.
3. The 18 internal-only items are demoted to `pub(crate)` on their declarations, satisfying the workspace's `unreachable_pub = "warn"` lint and reflecting actual visibility in rustdoc.
4. Intra-crate imports use explicit module paths (`crate::protocol::levels::Foo` instead of `crate::Foo`), since the lib-root re-exports the items go through are gone.

Behavior-preserving. All 572 workspace tests pass. The `unreachable_pub` lint catches future accidental over-exposure.

### Alternatives Considered

**Make `gamma` `Cfg::ChallengeField`-valued in this PR.**
Originally rejected for the first trace-cutover slice because it forces the
proof-payload reshape (`AkitaStage1Proof<F>` becomes
`AkitaStage1Proof<F, L>` etc.). That reshape touches every proof struct's
serialization, every prover/verifier callsite, and the scheme orchestration.
The companion completion spec now deliberately brings that work into #71.

**Keep `claim_values_to_base` as a soft-fail for `K > 1` instead of removing it.**
Rejected because the no-backward-compat policy in this repo says to remove dead bridges, and there are zero remaining callers after the trace-check cutover.

**Use `feature(generic_const_exprs)` so the verifier can pass `<E as ExtField<F>>::EXT_DEGREE` directly as a const generic argument to `check_trace_inner_product`.**
Rejected because the workspace pins stable Rust 1.88. The runtime-K dispatcher with a small `match` is the stable workaround and has no measurable cost.

**Defer the verifier surface tightening to a separate PR.**
Rejected because the trace-check work touches the same files (`verify_root_level`, `verify_one_level`, the surrounding imports). Tightening the surface in the same PR avoids growing the surface during the trace cutover and then having to re-shrink it.

## Documentation

- This spec is the per-PR documentation artifact for #71.
- The PR description on GitHub mirrors the four-themed commit structure (ring-subfield Fp4 arithmetic, production psi/trace primitives, field-extension hierarchy on `CommitmentConfig`, K-generic verifier trace check + surface tightening) and explicitly flags what is *not* in this PR.
- The umbrella spec `specs/extension-field-opening-batching.md` is shrunk in the same PR to cover only the remaining work.

## Execution

### Phase 4 (this PR slice): K-Generic Trace Primitives And Verifier Cutover

- [x] Implement production extension-to-subfield basis embedding (`embed_subfield<F, D, K>`).
- [x] Implement production `psi` packing for `(R_q^H)^{D/K} -> R_q` (`psi_embed<F, D, K>`).
- [x] Implement trace-scaling handling for `(D/K)` (inside `check_trace_inner_product<F, D, K>`).
- [x] Add the runtime-K dispatcher `dispatch_trace_inner_product_check<F, D>` for `K ∈ {1, 2, 4, 8}`.
- [x] Cut over `verify_root_level` to typed `Cfg::ClaimField` openings and the runtime-K dispatcher; drop `claim_values_to_base`.
- [x] Cut over `verify_one_level` to the runtime-K dispatcher.
- [x] Remove `claim_values_to_base` from `akita-types` per no-backward-compat.
- [x] Add `LiftBase<Fp2<F>>`, `MulBase<Fp2<F>>`, `ExtField<Fp2<F>>` for `TowerBasisFp4` so the tower `F ⊆ Fp2 ⊆ TowerBasisFp4` is statable in trait bounds.
- [x] Add `CLAIM_EXT_DEGREE` and `CHAL_EXT_DEGREE` associated constants on `CommitmentConfig`.
- [x] Enforce `Cfg::ChallengeField: ExtField<Cfg::ClaimField>` at the trait level.

### Verifier Surface Tightening (this PR slice)

- [x] Audit `crates/akita-verifier` public surface; classify each re-export by external usage.
- [x] Change `pub mod` to `mod` for `proof`, `protocol`, `stages` in `lib.rs`.
- [x] Trim lib-root `pub use` to the 5 truly external items plus 3 documented test-only items.
- [x] Demote 18 internal-only items to `pub(crate)` on their declarations; satisfy `unreachable_pub` workspace lint.
- [x] Update intra-crate imports to use explicit submodule paths.
- [x] Verify all 572 workspace tests pass and `cargo clippy --all -- -D warnings` is clean.

### Files Modified In This PR

- `crates/akita-field/src/fields/{ring_subfield,packed_ring_subfield,packed_ext,lift,ext}.rs`
- `crates/akita-types/src/field_reduction.rs`
- `crates/akita-types/src/proof/{batch,mod}.rs`
- `crates/akita-types/src/lib.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-verifier/src/lib.rs`
- `crates/akita-verifier/src/{proof,protocol,stages}/{mod,*}.rs`
- `crates/akita-pcs/benches/field_arith*` (extended ext4 inverse and multiplication rows)
- `specs/extension-field-trace-cutover.md` (this spec)
- `specs/extension-claim-incidence-cutover.md` (retroactive #69 spec authored in the same scoping pass)
- `specs/extension-field-opening-batching.md` (shrunk to remaining-work-only in the same scoping pass)

## References

- Predecessor spec (Phases 1-3): `specs/extension-claim-incidence-cutover.md`
- Companion completion spec (Phase 4 payload reshape, Phase 5 Frobenius, Phase 6/7): `specs/extension-field-opening-batching.md`
- Field-role baseline: `specs/general-field-support.md`
- Production trace primitives: `crates/akita-types/src/field_reduction.rs`
- Verifier root trace check: `crates/akita-verifier/src/protocol/levels.rs`
- Verifier crate surface: `crates/akita-verifier/src/lib.rs`
- Field-extension chain: `crates/akita-field/src/fields/lift.rs`
- Config field convention: `crates/akita-config/src/lib.rs`
