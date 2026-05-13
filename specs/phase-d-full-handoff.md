# Phase D-full Hand-off Notes (v2)

**Status**: In progress on `feat/tensor-challenges`. **Slices A + B + C.1 + C.2.a + C.2.b infrastructure + C.2.c partial cascade-aware planner are in the working tree, not yet committed.** The v1 routing seam AND the v1 cascade-with-padding scaffolding (which only made sense under the v1 max-based cascade) have both been reverted this session.

**Last touched**: 2026-05-13 (design doc rewritten as v2; v1 routing seam reverted; v1 cascade-with-padding scaffolding discarded; planner cascade formula updated to the book's additive `w_fold_L + |S|/f`; implementation plan reframed to extend existing batched-Hachi primitives instead of introducing parallel "split commitment" infrastructure).

This document is the resume point for the next session. Read it together with `specs/phase-d-full-design.md` v2 (rewritten this session). The book-aligned design is what we follow now: the recursive `(w, S)` open IS multi-group batched Hachi extended for per-group `(m, r, B, digit_count)` and mixed witness types — NOT a novel construct parallel to existing infra.

**Compatibility notice**: Per repo `AGENTS.md`, no backward-compatibility guarantees apply. Breaking changes are expected throughout these slices.

---

## 1 — What is committed

| Commit | Purpose |
|---|---|
| `9089d667` | Security baseline (planner + challenge family fixes; 128-bit MSIS + 128-bit CWSS clear). |
| `95e79c54` | Phase D-full design doc v1 (now retracted in favor of v2 — see `specs/phase-d-full-design.md` §10). |

The branch is 11 ahead of `origin/feat/tensor-challenges`. Do **not** push until Phase D-full v2 slices A through F (k=1) are landed as the planned single PR.

---

## 2 — What is in the working tree (uncommitted)

### Useful foundations to KEEP

These map cleanly onto the v2 design doc's Slices A through C.2.c.

**Slice A (DONE)**: Tiered setup foundation + `OnceLock`-lazy commitments per design doc §4.1. (`crates/akita-types/src/proof/tiered_setup.rs`, `crates/akita-prover/src/api/tiered_setup.rs`, the cache hooks on prover/verifier setup.)

**Slice B (DONE)**: `Vec<RecursiveOpeningClaim>` carrier and `Vec<RecursivePolyHandle>` on prover/verifier states per design doc §4.2.

**Slice C.1 (DONE)**: `prove_recursive_multi_fold_with_params`, `verify_one_level` `claims.len() > 1` branch, per-claim shape checks. **Today this requires (a) all claims share ONE `LevelParams` and (b) all claims have the same i8-digit witness shape**; slices D and E lift these restrictions.

**Slice C.2.a (DONE)**: `s_opening_value` on the wire per design doc §4.3. Closing-oracle equality validated; the transitional `setup_view.mle == s_opening_value` check is retained pending slice F's conditional drop.

**Slice C.2.b (DONE — infrastructure only)**: tiered types, multi-claim transcript, multi-ring shape check infrastructure.

**Slice C.2.c partial (DONE)**:

- `PlannerConfig::planner_setup_polynomial_size` trait method (default returns 0) and `planner_setup_shrink_factor` (default 1).
- `WCommitmentConfig::PlannerConfig::planner_setup_polynomial_size` delegates to the production `S` size.
- `derive_candidate_level_params_with_shape` and `derive_root_candidate_with_shape` apply the cascade penalty using the **book's additive formula** `w_fold_L + |S|/f` (NOT the v1 max-based formula). The planner currently always picks `num_claims = 1` (cascade off in production); slice F activates cascade by routing the S-group through the next level.
- `current_num_claims` plumbing as memoization key on `derive_optimal_suffix_schedule` and `derive_root_candidate_with_shape`.
- `planned_w_ring_element_count_with_claims` and `planned_next_w_len_with_claims` helpers in `proof_size.rs`.
- `prove_recursive_level_with_policy` refactored to dispatch through `prove_recursive_multi_fold_with_params` (bit-equivalent for N=1).

### v1 routing seam — REVERTED this session

The following were attempts at the wrong primitive (treating `t̂_S` as the next-level recursive witness via the existing multi-claim machinery, with `S` sharing `w`'s LP). They have been **removed** from the working tree this session:

- `RecursiveSMaterial<F, D>` struct.
- `build_s_recursive_material` helper.
- The `prover_setup_for_s_routing` and `next_lp_for_s_routing` parameters on `prove_fold_level_from_quadratic`, `prove_root_fold_from_quadratic`, `prove_recursive_multi_fold_with_params`, `prove_recursive_fold_with_params`, `prove_recursive_level_with_policy`, `prove_root_fold_with_params`.
- `verify_setup_claim_reduction`'s `routes_recursively: bool` flag — also reverted; the production conditional mle drop will be re-added in slice F as part of routing the S-group through the next level.
- `verify_one_level` / `verify_root_level` returning `(Vec<F>, Option<(Vec<F>, F)>)` — reverted to the original return shape.
- `verify_batched_recursive_suffix` / `verify_fold_batched_proof` `cascade_c_s` / `root_c_s` parameters — reverted.
- Scheme-side `derive_c_s` closure, `route_s_recursively` parameter, `prover_setup_for_s_routing` parameter — reverted.

### v1 cascade-with-padding scaffolding — REVERTED this session

The following items only made sense under the v1 max-based cascade `max(natural, |S|*D)` where the W-witness had to be inflated to absorb S's contribution under one shared LP. The book's additive `w_fold_L + |S|/f` formula plus v2's per-handle LP design (slice E) means each handle keeps its natural size — no cross-group padding is needed. These have been **removed** from the working tree this session:

- `RecursiveWitnessFlat::pad_to_len` method (the file is now back to baseline).
- `expected_w_len` arg threaded through `prove_fold_level_from_quadratic`, `prove_root_fold_from_quadratic`, `prove_recursive_fold_with_params`, `prove_recursive_multi_fold_with_params`, `prove_recursive_level_with_policy`, `prove_root_fold_with_params`.
- `expected_next_w_len` arg through `dispatch_prove_level` and the `prove_level` closure in `prove_recursive_suffix_with_policy` (the closure type drops its trailing `usize` arg).
- The `next_step_is_fold` conditional padding logic in `prove_recursive_suffix_with_policy` and `prove_folded_batched_with_policy`.
- The `pad_to_len` calls + `w_padded` shadowing in `prove_fold_level_from_quadratic` and `prove_root_fold_from_quadratic` (`commit_w_for_next` now receives the natural `w` directly, and the next-state handle holds the natural `w`).
- The `direct_step.current_w_len < handle.w.len()` relaxation in `resolve_final_log_basis` (restored to strict equality `==`).

The codebase is back to a clean baseline. Slice D builds forward from there using the extension approach in design doc §3.3 and §4.4-§4.5.

---

## 3 — Why the v1 routing attempt failed

Documented for the record so the next session does not repeat the misstep.

1. **Symptom**: with prover-side routing enabled (push `S` as `handles[1]` with `w = t̂_S`), the verifier rejects with the per-point trace check failure:

   ```
   verify_one_level: trace check failed at point 0:
     lhs=Fp128([2529290720997145258, 6445612749238708398]),
     rhs=Fp128([8236330285525643634, 10045945666772276238])
   ```

   `lhs = trace(y_S · σ_{-1}(v_S))` where `y_S` is computed from the `t̂_S` digit witness folded as a polynomial. `rhs = d · γ · s_opening_value`. They disagree because `mle(t̂_S)(r_setup_padded) ≠ mle(S)(r_setup) = s_opening_value`.

2. **Root cause**: the multi-claim recursive infrastructure folds each handle's `w` as a polynomial via `F::from_i8`-lifted coefficients. So the "polynomial" the next level proves an opening for is the digit-poly encoded by `w`, NOT the source field-element polynomial. For the recursive `w` from the previous fold, this works because the previous level's stage-2 sumcheck closes on the SAME digit-poly. For `S` as a fresh field-element polynomial, this does NOT work.

3. **Book's resolution** (§5.3 lines 643-660): `S` enters the L+1 commit as a fresh field-element polynomial in its own commitment group with its own `(m_S, r_S, B_S, δ_commit,S)`. `D` and `A` matrices are shared across groups; `B` is per-group. The book calls this "split commitment" but structurally it is multi-group batched Hachi.

4. **What v2 does instead**: extends the existing multi-claim path to lift its two restrictive invariants (shared LP, homogeneous witness type). Slice D adds per-group LP shape; slice E adds mixed witness types via `RecursiveWitnessAsPoly`. No parallel construct is introduced.

---

## 4 — Resume checklist for the next session

1. `git log -2 --oneline` — confirm the latest commits are `9089d667` (security) and `95e79c54` (v1 design, retracted).
2. `git status` — confirm the foundations from §2 are in the working tree (and that `crates/akita-prover/src/backend/recursive_witness.rs` is NOT in the modified list — it was reverted to baseline this session).
3. `cargo test -p akita-types tiered_setup --release` — should show 5/5 pass.
4. `cargo test -p akita-prover tiered_setup --release` — should show 2/2 pass.
5. `cargo test -p akita-pcs --test recursive_multi_claim --release` — should show 2/2 pass.
6. `cargo test -p akita-pcs --test setup_claim_reduction_e2e --release` — should show 5/5 pass with cascade off (the test config does NOT override `planner_setup_polynomial_size` after the v2 revert; the `s_opening_value` tamper test from slice C.2.a is included).
7. `cargo clippy --all -- -D warnings` — should be clean.
8. `cargo test --release` — full workspace green.

If any of these fail, the working tree drifted; start by reverting any unintended changes.

After the basic checks, read `specs/phase-d-full-design.md` v2 in full (especially §3 "Protocol shape" and §6 "Implementation plan"). The book references in §2 are load-bearing; do NOT proceed without reading the cited passages of `5_fourth_root_verifier.tex`.

---

## 5 — Recommended next session focus

**Begin Slice D from `specs/phase-d-full-design.md` v2**: extend the batched commit kernel to support per-commitment-group `(m, r, B, digit_count)` with shared `D, A`. Single new primitive that subsumes both root batched opens (current case: all groups share `(m, r)`) and the L+1 recursive open (slice F: groups differ).

Sub-steps for slice D's first commit:

1. **Extend `LevelParams` in place** (design doc §4.4, chosen representation): add an optional `groups: Option<Vec<GroupSpec>>` field where `GroupSpec` carries `(m_vars, r_vars, b_key, num_digits_open)`. `None` preserves the existing single-group shape (every existing call site is unaffected); `Some(vec)` activates the multi-group code path with shared `D, A` from the outer `LevelParams` and per-group `(m, r, B, digit_count)` from each `GroupSpec`.

2. **Extend the commit kernel**: extend `commit_with_params` (or add a sibling `commit_multi_group` that calls into the same per-row machinery) to handle multi-group inputs and produce `(v_joint, c_joint, per_group_u, per_group_hint)`. For `groups == None` (or `groups.len() == 1`), the output collapses to the existing single-group shape (regression-safe).

3. **Extend `prepare_m_eval` row layout** to support per-commitment-group sub-rows in the B and eval/fold groups. The D and Ajtai groups stay shared (one row family each, sized for the union). Mirror in `AkitaStage2Prover` and `materialize_setup_claim_tables`.

4. **Tests for slice D** (root level only — no recursive plumbing yet):
   - Multi-group commit at root with two polys at mismatched `(m, r)` produces a single `v_joint` and `c_joint` plus per-group `u_g`.
   - The relation closes correctly through `prepare_m_eval` and `AkitaStage2Verifier`.
   - For `groups == None`, the path is bit-equivalent to the existing single-group path (regression-safe).

After slice D lands, **slice E** (mixed witness types in recursive multi-claim path, via `RecursiveWitnessAsPoly` newtype + per-handle/per-claim `LevelParams` plumbing) is next. Then **slice F** wires the S-group into the next state at recursive levels (using slices D + E primitives) and drops the per-level `mle` check.

See v2 design doc §6 for the full slice plan (D through I) and the ~5-week effort estimate.

### What slice D should NOT do

To keep the slice scoped, avoid these tempting expansions:

- Do NOT introduce a `SplitRecursiveWitness` or `commit_split` parallel to existing infra. The whole point of v2 is to extend `commit_with_params` and the existing batched relation, not to add a parallel "split" type system.
- Do NOT touch the recursive multi-claim path yet (that's slice E).
- Do NOT activate cascade in production callers (that's slice F).
- Do NOT add tier-3 meta-commitment row family (that's slice G — implies `k = 64`, which is the tiered S-group case).

---

## 6 — Why the v1 design was wrong (TL;DR)

See `specs/phase-d-full-design.md` v2 §10 for the formal retraction. In short: the v1 attempt pushed `S` as a second `RecursivePolyHandle` in the existing multi-claim path with `w = t̂_S` (S's digit decomposition). That path requires (a) all claims share ONE `LevelParams` and (b) all claims have the same i8-digit witness shape. The push violated both invariants:

- `mle(t̂_S)(r_setup_padded) ≠ mle(S)(r_setup) = s_opening_value` → trace check rejects.
- `S` requires its own `(m_S, r_S, B_S, δ_commit,S)` distinct from `w`'s, and S is a fresh `DensePoly` (not an i8-digit witness) → shared-LP and homogeneous-witness invariants violated.

The book's design (§5.3 lines 643-660) is multi-group batched Hachi: `D` and `A` shared across groups (joint `v`, joint `c`), `B` per group, per-group `(m, r, digit_count)`, mixed witness types per group (`DensePoly` for the S-side, recursive-witness-as-poly for the W-side). The "split commitment" name reflects the per-group `B`-matrix; structurally it is the existing batched commit primitive with the two restrictions lifted.

The v2 fix is to **extend** the existing primitives (slices D and E) so the multi-claim path admits per-group LP and mixed witness types, NOT to introduce parallel "split" infrastructure. This is what design doc v2 §3-§6 describe.

---

## 7 — Files touched this session (resume checkpoint)

`specs/phase-d-full-design.md` rewritten as v2 (book-aligned design with the multi-group batched Hachi framing). Old v1 content deleted; see git log to recover if needed.

`specs/phase-d-full-handoff.md` rewritten as v2 (this file).

Code in the working tree is reverted to a clean baseline:

- v1 routing seam additions removed (`RecursiveSMaterial`, `build_s_recursive_material`, prover/verifier routing parameters, scheme-side `derive_c_s` closure).
- v1 cascade-with-padding scaffolding discarded (`pad_to_len` method, `expected_w_len`/`expected_next_w_len` plumbing, `next_step_is_fold` conditional padding, `w_padded` shadowing, relaxed `<` check in `resolve_final_log_basis`).
- Stale doc comments referencing `SplitRecursiveWitness` / `commit_split` / "10-check-group" / "split-commitment joint open" / "cascade-fit" / "Phase D-full slice C" updated to point at the v2 multi-claim framing and slice F (`specs/phase-d-full-handoff.md`).
- Foundations preserved: tiered setup types, `Vec<RecursiveOpeningClaim>`, multi-claim plumbing (with `current_num_claims` memoization), `s_opening_value` wire, cascade-aware planner with the book's additive formula `w_fold_L + |S|/f`.
- Planner currently picks `num_claims = 1` always (cascade off in production); slice F activates it.

Verification: `cargo fmt -q && cargo clippy --all -- -D warnings && cargo test --release` all green this session.

---

## 8 — Decisions still open

None. All v2 design choices have been resolved:

- **Per-group LP representation** (design doc §4.4 / §5 D-5): chosen — extend `LevelParams` in place with an optional `groups: Option<Vec<GroupSpec>>` field. `None` preserves the existing single-group shape; `Some(vec)` activates the multi-group path with shared `D, A` and per-group `(m, r, B, digit_count)`. No `MultiGroupLevelParams` wrapper is introduced.
- **Slice E shape**: `RecursiveWitnessAsPoly` is a thin newtype implementing the same trait surface as `DensePoly`; per-handle/per-claim `LevelParams` is plumbing, not design. The shape follows directly from "extend the multi-claim path to admit per-group LP + mixed witness types".
- **Slice F shape**: routing the S-group at recursive levels is a direct application of slices D + E primitives. The conditional mle drop is a single boolean flag on `verify_setup_claim_reduction`.
- **Slice G shape**: tiered (k = 64) is a sub-chunking + meta-tier extension of the S-group only; the W-group is unaffected. The proof shape gains `(c_meta, v_meta, u_meta)` (independent of k per book §5.4 line 698).

The `SplitRecursiveWitness`, `commit_split`, `SplitLevelParams`, "where new types live", "split kernel signature" questions from the prior v1 §8 are MOOT under v2 because no parallel construct is introduced — the existing primitives are extended in place.

Slice D may proceed immediately on the next session with no additional design sign-off needed.
