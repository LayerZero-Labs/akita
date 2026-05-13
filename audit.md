# Fourth-Root Verifier Optimization Audit (`feat/tensor-challenges` vs `main`)

**Date**: 2026-05-13
**Branch reviewed**: `feat/tensor-challenges` (HEAD = `b8bf437e`)
**Comparison base**: `origin/main` (HEAD = `1a3e0bf2`)
**Reference spec**: `Lattice_Jolt/sections/akita/5_fourth_root_verifier.tex`
**Workspace rules referenced**: `.cursor/rules/blockers.mdc`, `.cursor/rules/coda_changes.mdc`

## Scope

The branch carries 90 commits on top of `main`, touching 199 files (+24 470 / −25 575 lines). The stated goal of the work is the Section 5 fourth-root verifier optimization:

1. **Technique 1**: tensor-structured stage-1 folding challenges (`c_{p\|q} = α_p · β_q`).
2. **Technique 2**: claim-reduction sumcheck for the setup-dependent `M`-table contribution.

Section 5 promises an asymptotic verifier reduction from `O(√N')` to `O(N'^{1/4})` per recursive level.

## Bottom Line

Functional correctness and short-term soundness look OK, but the **PR does not deliver the headline goal**, has **substantial unrelated drift**, and has **multiple coda_changes.mdc / blockers.mdc violations** that should be cleaned up before merge.

- **The verifier is ~2× slower than `main`** at the bench points the PR's own roadmap targets (NV=25 verify: branch 14.97 ms vs main 7.02 ms — `specs/recursive-s-opening-plan.md:547`).
- **The implementation log explicitly admits**: *"Until those two pieces land, the current branch is a correct but not-yet-asymptotically-faster claim-reduction prototype"* (`specs/fourth-root-verifier-optimization-roadmap.md:789-792`).
- The asymptotic fourth-root reduction (Phase D-full: recursive `S` opening) **is not implemented**, and the structural pieces that did land (tensor stage-1, materialized claim-reduction sumcheck) are not strictly better than the legacy fused stage-2 path in this codebase.

**Soundness summary**: the protocol modifications are individually sound; no cryptographic break was found. Below 128-bit security floors are preserved by the planner accounting (`stage1_extraction_relative_msis_degradation` of `4ω` is correctly propagated into SIS rank selection). The remaining concerns are about implicit (vs explicit) binding, env-var sensitivity, and the cumulative bit budget of stacking changes.

---

## Findings — Cryptographic Soundness

### Blocker / Risk classification

- `S` = **Soundness Concern**: would matter if exploited; usually a defense-in-depth ask.
- `B` = **Blocker**: must be addressed before this PR can deliver its claimed goal.
- `R` = **Risk**: subtle invariant or hidden assumption a future change can violate.

### S-1 — Stage-2 dispatch is proof-driven, not schedule-driven

**Severity**: Soundness Concern (defense-in-depth)

**Where**:
- `crates/akita-verifier/src/protocol/levels.rs:244-258` (`verify_root_level`)
- `crates/akita-verifier/src/protocol/levels.rs:419-433` (`verify_one_level`)

```242:259:crates/akita-verifier/src/protocol/levels.rs
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        if let Some(payload) = stage2.setup_claim_reduction.as_ref() {
            verify_stage2_with_setup_claim_reduction::<F, _, D>(
                &stage2.sumcheck,
                payload,
                &stage2_verifier,
                transcript,
            )?
        } else {
            verify_sumcheck::<F, _, F, _, _>(
                &stage2.sumcheck,
                &stage2_verifier,
                transcript,
                |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
            )?
        }
    };
```

The verifier branches on whether the proof carries a setup-claim-reduction payload, **not** on whether the schedule (`lp.use_setup_claim_reduction`) says claim reduction is supposed to run. There is no explicit equality check between `proof.setup_claim_reduction.is_some()` and `lp.use_setup_claim_reduction` anywhere in `crates/akita-verifier/`.

A malicious prover can attach or omit the payload independent of the schedule:
- If they attach when the schedule says "no": the verifier follows the CR path, which uses a smaller M-eval (algebraic only) and requires the prover to honestly produce `m_setup_eval = ⟨w_setup, S⟩`. The closing equality on the main sumcheck still pins `m_alg + m_setup_eval` so the prover gains nothing.
- If they omit when the schedule says "yes": the verifier computes the full setup-dependent `m_eval` directly. The prover is now binding the same identity, so no soundness is lost.

Both branches verify the *same* mathematical identity, just with different work distribution, so this is **not a security break**. It is, however, a fragile invariant.

**Refactor / fix**:
1. In `verify_root_level` (line 242) and `verify_one_level` (line 417), assert `stage2.setup_claim_reduction.is_some() == batched_lp.use_setup_claim_reduction` (and `lp.use_setup_claim_reduction` respectively) before dispatching. Reject with `AkitaError::InvalidProof` on mismatch.
2. Same check on the prover side in `crates/akita-prover/src/protocol/flow.rs:701` and `:1233`, even though the prover trivially obeys its own `lp` (defense-in-depth so the proof shape is unambiguous from the schedule alone).

### S-2 — Production schedules silently switched to Tensor mode (concrete-security migration)

**Severity**: Soundness Concern (real protocol change, not a fix)

**Where**:
- `crates/akita-config/src/proof_optimized.rs:54-79` (`fp128_stage1_challenge_config` + `stage1_challenge_shape_for_config`)
- `crates/akita-types/src/generated/mod.rs` (six `fp128_*` tables now declare `GeneratedStage1ChallengeShape::Tensor`)
- `crates/akita-types/src/generated/fp128_d{64,128}_{full,onehot}.rs` (regenerated tensor schedules)
- `crates/akita-config/src/proof_optimized.rs:425-427` (`allow_tensor_stage1_schedules() = true` for every fp128 preset)

```54:69:crates/akita-config/src/proof_optimized.rs
pub(crate) fn fp128_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
    match d {
        // Safe-margin tensor defaults selected from the May 2026 planner rerun:
        // keep a small buffer above the per-side minima without drifting back
        // toward the old flat-side masses.
        32 => SparseChallengeConfig::BoundedL1Norm,
        64 => SparseChallengeConfig::ExactShell {
            count_mag1: 18,
            count_mag2: 0,
        },
        128 => SparseChallengeConfig::Uniform {
            weight: 13,
            nonzero_coeffs: vec![-1, 1],
        },
        _ => panic!("unsupported fp128 ring dim {d}"),
    }
}
```

Main's default at D=64 is `ExactShell { count_mag1: 30, count_mag2: 12 }` (`ω = 54`, flat). The branch ships `ExactShell { count_mag1: 18, count_mag2: 0 }` *plus* tensor shape (effective L1 mass `18² = 324`). At D=128 the branch goes from `flat` to `Tensor` with `Uniform{weight: 13}`.

This is a **concrete-security migration**, not "tensor of the old protocol":
- The base sparse-challenge family is different (smaller `ω` per side).
- Tensor extraction adds `4ω` MSIS-norm degradation (`crates/akita-types/src/layout/params.rs:266-277`).
- Combined with the lighter per-side family, the SIS rank floors derived in `LevelParams::stage1_extraction_infinity_norm` and `crates/akita-types/src/layout/sis_derivation.rs` change.

**Audit status**:
- The 4ω propagation through SIS sizing **is** wired correctly (`Stage1ChallengeShape::Tensor` → `stage1_extraction_relative_msis_degradation` → `min_rank_for_secure_width`). Verified via `LevelParams::stage1_sis_extraction_report` tests.
- The MSIS floor for the production family is documented as `~280` bits (`specs/fourth-root-verifier-optimization-roadmap.md:236-239`) → `216 ≈ 7.8` bit penalty leaves `≈272` bits, comfortably above 128. Verified for `D=64` but **not re-derived** here for the new `D=128 Uniform{13}` family.
- The `BoundedL1Ball` sampler is hard-coded to `(D=32, M=8, B=121)` (`crates/akita-challenges/src/sampler/bounded_l1.rs`), which is the only family that remains flat. This means **D=32 takes none of the tensor benefit and keeps the legacy MSIS floor**.

**What is missing in this PR**:
- An explicit re-statement of the post-change concrete security margin for **every** production preset (D32, D64Full, D64OneHot, D128Full, D128OneHot, both flat and tensor where applicable).
- A note in `AGENTS.md` or a security policy doc that the production challenge families are *new*, not just a tensor-wrapped version of main's.
- Confirmation that the SIS rank floor tables in `crates/akita-types/src/generated/sis_floor.rs` cover the new extraction collision buckets that the lighter `ExactShell{18,0}` × `4ω` combination produces. The `Phase K` notes mention SIS table coverage was extended from rank 4 to rank 7, but the exact rank that the new `D=128 Uniform{13}` family hits should be sanity checked.

**Refactor / fix**:
1. Add a `security_analysis.md` (or extend `specs/fourth-root-verifier-optimizations.md`) that lists each production `(D, challenge_family, max_num_vars)` combination with: post-tensor MSIS bits, knowledge error per level, and concrete `|F|`/`|C|`-based knowledge-error budget. Cite the planner table that backs each bound.
2. Add a one-shot smoke test in `crates/akita-config` that asserts every production preset's `stage1_extraction_relative_msis_degradation` × `stage1_config.infinity_norm()` lands in a covered SIS bucket (i.e., `ceil_supported_collision` never returns `None` for any `max_num_vars` the preset advertises).

### S-3 — `HACHI_PLANNER_S1_WEIGHT` env var ships in production planner

**Severity**: Soundness Concern (deployment/configuration risk)

**Where**:
- `crates/akita-planner/src/schedule_params.rs:228-246`

```228:246:crates/akita-planner/src/schedule_params.rs
    let weight = std::env::var("HACHI_PLANNER_S1_WEIGHT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(Cfg::planner_stage1_prover_weight);
```

The env var changes the objective cost during schedule DP, which in turn changes which schedule the planner selects. The selected schedule affects:
- Tensor vs flat shape per level (when hybrid search is enabled).
- `log_basis` per level → digit depths → SIS rank floors.
- Total fold-level count → recursive opening cascade size.

If prover and verifier run with different env-var values and neither has a pinned/generated schedule, they will pick different schedules. Even when the schedule is bound by public inputs to a deterministic table lookup, the **planner-only paths** (used by tests/benches and by non-generated configs) are vulnerable to silent drift.

**Refactor / fix**:
1. Remove the env var entirely. The roadmap (`specs/recursive-s-opening-plan.md:441-448`) recommends `weight=3` default; that is `Cfg::planner_stage1_prover_weight()`. Anyone who wants to experiment with a different weight can set it via a custom `PlannerConfig` impl.
2. If the env-var override must be kept for development convenience, gate it behind `#[cfg(debug_assertions)]` and emit a loud warning when it is read.

### S-4 — Schedule cache key omits `use_setup_claim_reduction`

**Severity**: Risk (current usage is safe; invariant is fragile)

**Where**:
- `crates/akita-types/src/schedule.rs:210-221` (`AkitaScheduleLookupKey`)
- `crates/akita-types/src/proof/setup.rs:52-87` (`AkitaVerifierSetup::schedule_cache`)
- `crates/akita-verifier/src/protocol/batched.rs:218-230` (`verify_batched_with_policy`)

`AkitaScheduleLookupKey = (max_num_vars, num_vars, layout_num_claims, batch)`. It is missing the `use_setup_claim_reduction` flag (and, for hybrid configs, the per-level shape pattern is encoded in the cached `Schedule` but not in the key).

Today this is safe because each `AkitaVerifierSetup<F>` is per-`Cfg` and Cfg uniquely determines `use_setup_claim_reduction`. But it is the kind of invariant that breaks silently the moment someone adds a runtime "turn CR on/off" switch.

**Refactor / fix**:
- Either add `use_setup_claim_reduction: bool` to `AkitaScheduleLookupKey` (and the generated table lookup), or document/assert at construction time that the cache is per-`Cfg`.

### S-5 — `level_proof_bytes` ignores claim-reduction payload bytes

**Severity**: Risk (planner accuracy)

**Where**:
- `crates/akita-types/src/layout/proof_size.rs:74-98` (`level_proof_bytes`)

When `use_setup_claim_reduction = true`, the actual serialized proof contains an extra `m_setup_eval` field element plus a setup-claim-reduction `SumcheckProof` (`log m_row + log d` rounds at degree 2). The planner's byte budget does not include those bytes.

If a future planner enables CR by default (Phase E in the roadmap, currently DEFERRED), the planner would underestimate proof size by ~`(field_bytes + (log_m_row + log_d) * (degree+1) * field_bytes)` per level. That can change which schedule the DP selects.

**Refactor / fix**:
- Make `level_proof_bytes` and `recursive_level_proof_bytes` shape-aware: take `LevelParams` (already does) and inspect `lp.use_setup_claim_reduction`, then add the SCR sumcheck bytes when true.
- Update the `schedule.rs` tests `planned_level_bytes_match_two_stage_payload_at_all_bases` and `planned_batched_root_bytes_match_two_stage_payload_at_all_bases` to cover the CR-enabled case (currently they hard-code `setup_claim_reduction: None`).

### S-6 — Tensor exact aggregate evaluator: positive note

**Severity**: Verified correct

The book's simplified `c_α(p\|q) = c_α^L(p) · c_α^R(q)` is **not valid** for Hachi's generic ring-switch `α`. The implementation in `crates/akita-challenges/src/stage1.rs:224-320` (`eval_factored_aggregate_at_pows`) correctly applies the negacyclic quotient correction:

```text
S = product_eval - (α^D + 1) · quotient_eval
```

The unit tests in `crates/akita-pcs/tests/tensor_stage1_e2e.rs` and `crates/akita-verifier/src/protocol/ring_switch.rs` (around lines 1585-1628) compare the exact aggregate against the expanded reference for random `α`, multiple offsets, and odd `r`. This is a genuine correction of the book text and the code is right.

**No action needed**, but the documentation in `specs/fourth-root-verifier-optimizations.md` and the LaTeX book should be updated to reflect that the production code applies the negacyclic correction and the book's simplified identity is approximate only. (`specs/fourth-root-verifier-implementation-audit.md` already calls this out.)

### S-7 — Two-stage prover fold (book claim) is NOT implemented; impact on prover work

**Severity**: Risk (acknowledged in roadmap, but worth flagging)

**Where**:
- `crates/akita-challenges/src/stage1.rs:120-133` (`expand_integer`)
- `crates/akita-prover/src/protocol/quadratic_equation.rs:490-517` (batched fallback)
- `crates/akita-prover/src/backend/onehot.rs:1222-1235` (the only backend that *doesn't* fall back to full expansion)

The book describes (Round 4 of `fig:fourthroot-protocol`) a two-stage prover fold: `tmp_p = Σ_q β_q · s_{p,q}; z = Σ_p α_p · tmp_p`. The implementation for **dense** and **multilinear** backends materializes all `2^r` logical tensor products via `IntegerChallenge::tensor_product` before folding. Only the **one-hot** backend has a tensor-aware kernel (`single_chunk_onehot_accumulate_tensor` / `multi_chunk_onehot_accumulate_tensor`).

This is acknowledged in `specs/fourth-root-verifier-optimization-roadmap.md:120-127` and is not a soundness issue. It is, however, why the prover does not benefit from tensor stage-1 at the dense backend — and why the published bench numbers (NV=25 prover branch 510 ms vs main 1054 ms) come almost entirely from unrelated optimizations (multi-chunk Ajtai commit tiling, centered fold accumulators) rather than tensor stage-1 itself.

**Refactor / fix**:
- Either implement the two-stage tensor fold in dense / multilinear backends, or explicitly remove tensor stage-1 from dense schedules and only generate flat schedules for those configs.

---

## Findings — Goal Coverage

### B-1 — Verifier is slower than `main`, not faster

**Severity**: Blocker for the stated goal

**Evidence**: `specs/recursive-s-opening-plan.md:541-548` (Phase K.6 final apples-to-apples table):

| NV | Stage | main | branch tensor | branch hybrid | tensor vs main | hybrid vs main |
|---:|-------|-----:|--------------:|--------------:|---------------:|---------------:|
| 25 | verify | **7.021 ms** | 14.970 ms | 14.580 ms | +113.2% | +107.7% |
| 20 | verify | 2.402 ms | 4.121 ms | 3.692 ms | +71.6% | +53.7% |
| 15 | verify | 0.877 ms | 0.997 ms | 0.996 ms | +13.7% | +13.6% |

The verifier regression grows with `N`. The hybrid stage-1 shape search (Phase K.1-K.7) only shaves 5-18 percentage points off the gap.

**Root cause**:
1. The claim-reduction sumcheck's closing oracle still calls `setup_view.mle(r_row, r_col, r_coeff)` in `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:127-131`, which walks `num_rows × num_cols × D` of the live setup prefix. This is the same asymptotic order as the legacy fused stage-2 path's setup-matrix reads.
2. The setup-side claim-reduction adds **extra** sumcheck rounds (`log m_row + log d`) and an extra closing `S(r*)` evaluation that *does not exist in main*. So when the prover opts into CR (it is opt-in via `lp.use_setup_claim_reduction`), the verifier does **more** work, not less.
3. The four production presets that ship in `proof_optimized.rs` *also* tensorize stage-1, which adds the `4ω` SIS norm penalty and forces heavier digit depths per fold level. The verifier hot path consumes stage-2 random challenges that have no tensor structure, so the heavier per-level shape costs the verifier extra work without payback.

**What the spec promises is missing**:
- **Phase D-full (recursive `S` opening)**: not implemented (`specs/tensor-everywhere-implementation-plan.md:226-228`). The setup polynomial `S` is not folded into the next level's witness; instead it is materialized at the closing point. Without recursive opening, the fourth-root asymptotic does not materialize.
- **Tiered commitments (book §5.4)**: not implemented at all. No tier-3 meta-commitment, no split D/B commitment for `[w \| S]`, no per-chunk shared matrix.
- **Batched stage-1 (book §5.3, "Round 7 batched range + relation")**: not implemented. The prover still runs the legacy fused stage-2 sumcheck *and then* an extra claim-reduction sumcheck. The book's protocol replaces stage 2; ours appends to it.

**Recommended next slice** (from the roadmap, paraphrased):
- Land Phase D-full (recursive `S` opening) before claiming any fourth-root reduction.
- Alternatively, **revert** the tensor and claim-reduction code from production presets and ship this branch as the prover-only speedup it actually is (~2× at NV=25), with the tensor/CR code retained as opt-in for development.

### B-2 — Claim-reduction prover materializes full padded hypercube tables

**Severity**: Blocker for any verifier improvement

**Where**:
- `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:28-65` (`materialize_setup_claim_tables`)
- `crates/akita-prover/src/protocol/setup_claim_reduction.rs:46-49`

```28:32:crates/akita-verifier/src/protocol/setup_claim_reduction.rs
/// Materialize the setup weights and setup polynomial table used by the
/// claim-reduction sumcheck. Reference / debug helper: production callers
/// should prefer the structured evaluator
/// [`PreparedMEval::eval_setup_weight_at_point`] for the weight side and
/// [`SetupMatrixPolynomialView::mle`] for the setup side.
```

The function's own doc-comment labels it "Reference / debug helper" and tells production callers to use the structured evaluator. **The production prover calls it anyway**:

```46:48:crates/akita-prover/src/protocol/setup_claim_reduction.rs
    let (setup_weights, setup_table) =
        materialize_setup_claim_tables::<F, D>(prepared, x_challenges, setup, alpha)?;
    let mut prover = WeightedTableProver::new(setup_table, setup_weights)?;
```

Both vectors have length `2^(row_bits + col_bits + coeff_bits)` (padded hypercube). This is `O(rows · stride · D · padding)` per level for the prover, plus the same again for the verifier when it builds `setup_weights` via the same helper. This is the dominant cost at the bench points where claim-reduction is slower.

**Refactor / fix**:
- The verifier has a structured `eval_setup_weight_at_point` (`crates/akita-verifier/src/protocol/ring_switch.rs:1206-1344`) that already computes `w_setup(r)` in `O(log)` without materializing the table. **Add a corresponding structured `S(r)` evaluator (or proper `mle` over the live prefix) and remove the prover-side `materialize_table()` call entirely**.
- Either restructure `prove_setup_claim_reduction` to use a custom sumcheck instance that consumes `S` via streaming/structured access, or accept that the prover materializes both tables and stop calling it "Reference / debug helper" — the false labeling makes the code confusing.

---

## Findings — Code Bloat / coda_changes.mdc violations

### C-1 — `eval_at_point` vs `eval_algebraic_at_point` duplicate ~180 lines

**Severity**: Bloat (coda_changes.mdc: "fast path plus fallback" pattern)

**Where**:
- `crates/akita-verifier/src/protocol/ring_switch.rs:542-796` (`eval_split_at_point`)
- `crates/akita-verifier/src/protocol/ring_switch.rs:809-987` (`eval_algebraic_at_point`)

`eval_algebraic_at_point` is a near-line-for-line copy of `eval_split_at_point` that omits the `setup` parameter and the four lines that read shared setup-matrix rows (`eval_d_matrix_w_residual_direct`, `eval_b_matrix_t_residual_direct`, the `z_base_setup` branch, the `z_dense_setup` branch).

`eval_split_at_point` already returns `PreparedMEvalSplit { algebraic, setup }`, so `eval_algebraic_at_point` could be a 5-line wrapper:

```rust
pub fn eval_algebraic_at_point<const D: usize>(&self, ...) -> Result<F, AkitaError> {
    Ok(self.eval_split_at_point::<D>(...)?.algebraic)
}
```

The current implementation duplicates ~180 lines and has its own `#[allow(clippy::too_many_lines)]` (line 809).

**Refactor / fix**:
- Replace `eval_algebraic_at_point` with the 5-line wrapper, **unless** profiling shows that the wasted `setup` work is a verifier-time bottleneck. If profiling does show that, refactor `eval_split_at_point` to accept an `enum { Full, AlgebraicOnly }` selector and gate the setup-matrix iterations on that selector. Either way, do not keep two diverging copies.

### C-2 — Four eq-weighted sumcheck primitives, one used in production

**Severity**: Bloat (speculative API)

**Where**:
- `crates/akita-sumcheck/src/eq_weighted_table.rs` (311 lines, four pub structs)

| Primitive | Production caller | Test/debug caller |
|-----------|-------------------|-------------------|
| `WeightedTableProver` | `crates/akita-prover/src/protocol/setup_claim_reduction.rs:48` | — |
| `WeightedTableVerifier` | — | `crates/akita-pcs/tests/ring_switch.rs` |
| `EqWeightedTableProver` | — | `crates/akita-types/src/layout/flat_matrix.rs::setup_polynomial_claim_reduction_roundtrip` |
| `EqWeightedTableVerifier` | — | `crates/akita-types/src/layout/flat_matrix.rs::setup_polynomial_claim_reduction_roundtrip` |
| `eq_eval` (free fn) | — | (only internal use within `EqWeightedTableVerifier`) |

The verifier doesn't even use `WeightedTableVerifier`; it inlines `weight_at_point * setup_at_point` directly in `verify_setup_claim_reduction` (`crates/akita-verifier/src/protocol/setup_claim_reduction.rs:133`). So only one of the four types is in the production hot path, and only via the prover.

**Refactor / fix**:
- Move `EqWeightedTableProver`, `EqWeightedTableVerifier`, and `WeightedTableVerifier` behind `#[cfg(test)]`, *or* delete them and rewrite the affected tests around `WeightedTableProver` + an inline verifier check.
- `eq_eval` is only used internally; either delete the `pub` and inline it, or keep it `pub` only if there is a documented callsite.
- Net reduction: ~200 of 311 lines.

### C-3 — Sibling constructors `new_two_stage` vs `new_two_stage_with_setup_claim_reduction`

**Severity**: Bloat (coda_changes.mdc: "foo_safe / foo_tiled / foo_fallback" sibling pattern)

**Where**:
- `crates/akita-types/src/proof/mod.rs:1069-1085` (`AkitaLevelProof::new_two_stage`)
- `crates/akita-types/src/proof/mod.rs:1252-1268` (`AkitaBatchedRootProof::new_two_stage`)

```1069:1085:crates/akita-types/src/proof/mod.rs
    pub fn new_two_stage<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<F>,
        stage2_sumcheck: SumcheckProof<F>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: F,
    ) -> Self {
        Self::new_two_stage_with_setup_claim_reduction::<D>(
            y_ring,
            v,
            stage1,
            stage2_sumcheck,
            None,
            next_w_commitment,
            next_w_eval,
        )
    }
```

The bare `new_two_stage` exists only to call its sibling with `None`. coda_changes.mdc explicitly recommends *"prefer generalizing or strengthening the existing function/API under its current name over adding sibling variants"*.

**Refactor / fix**:
- Make the single constructor `new_two_stage` take `setup_claim_reduction: Option<SetupClaimReductionPayload<F>>`. Update the ~3 callers in `crates/akita-prover/src/protocol/flow.rs:397-405` and `:731-739` accordingly.
- Same for `AkitaBatchedRootProof`.

### C-4 — Reference/debug methods kept in production code

**Severity**: Bloat (debug paths shipped, no `#[cfg(test)]` gate)

**Where**:
- `crates/akita-verifier/src/protocol/ring_switch.rs:480-492` (`PreparedMEval::debug_expanded_challenge_evals`)
- `crates/akita-verifier/src/protocol/ring_switch.rs:998-1014` (`PreparedMEval::debug_split_eval_table`)
- `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:42-65` (`materialize_setup_claim_tables`, doc-string says "Reference / debug helper" but is the production prover's path — see B-2)

Methods prefixed `debug_*` and labeled "reference bridge" or "debug/reference" should be `#[cfg(test)]` or removed.

**Refactor / fix**:
- `debug_expanded_challenge_evals` and `debug_split_eval_table` are only called from tests in `crates/akita-pcs/tests/ring_switch.rs`. Gate them with `#[cfg(test)]` or move them into a `pub(crate)` test helper module.
- `materialize_setup_claim_tables`: see B-2. Either it is the production path (in which case the "Reference / debug helper" label is misleading and must be removed) or it should be `#[cfg(test)]` and a structured alternative used in production. Cannot be both.

### C-5 — `PlannerHybridCfg` is test-only but hybrid search infrastructure is in production crate

**Severity**: Bloat (test-only feature gated in production code)

**Where**:
- `crates/akita-planner/src/lib.rs` (trait method `planner_stage1_shapes_to_search`, default empty `Vec`)
- `crates/akita-planner/src/schedule_params.rs:278-294` (DP loop iterates over shape choices)
- `crates/akita-planner/src/schedule_params.rs:346-357` (`planner_shape_choices`)
- `crates/akita-pcs/tests/hybrid_stage1_e2e.rs` (test wrapper `PlannerHybridCfg`)
- `crates/akita-pcs/benches/akita_e2e.rs` (bench wrapper)

Production presets (`D64OneHot`, `D64Full`, `D128OneHot`, `D128Full`) **do not** override `planner_stage1_shapes_to_search`, so the hybrid DP search runs `vec![None]` only — i.e., it does not search shapes. The hybrid branch in `schedule_params.rs:278-294` is dead code in production today.

The Phase K.6 results (`specs/recursive-s-opening-plan.md:541-548`) show hybrid saves at most 18 percentage points of the gap to main, and it never beats tensor-only on both prover and verifier simultaneously.

**Refactor / fix**:
Pick one:
- Option A (delete): if hybrid never wins net, remove `planner_stage1_shapes_to_search`, `derive_candidate_level_params_with_shape`, `planner_shape_choices`, and the for-loop in `schedule_params.rs:278-294`. Keep the rest of the planner DP as it was. Remove `PlannerHybridCfg` test wrapper and `hybrid_stage1_e2e.rs`.
- Option B (enable in production): Make the production presets opt in to hybrid search and regenerate the schedule tables with the hybrid DP. This is a multi-week investment per the roadmap. Until then, the code is speculative.

The implementation log explicitly recommends keeping hybrid as a tunable knob "for deployments that prefer prover speed" (`specs/recursive-s-opening-plan.md:447-448`), but that contradicts coda_changes.mdc's preference for one consistent shape.

### C-6 — `eval_algebraic_at_point`'s `#[allow(clippy::too_many_lines)]` is a smell

**Severity**: Bloat

**Where**:
- `crates/akita-verifier/src/protocol/ring_switch.rs:809`

The annotation acknowledges that the function is too large but suppresses the lint instead of refactoring. coda_changes.mdc says *"If a fix seems to require a lot of new machinery, stop and reassess the abstraction boundary."* See C-1.

### C-7 — Documentation drift: spec references non-existent code

**Severity**: Bloat (docs are misleading)

**Where**:
- `specs/recursive-s-opening-plan.md:344-358` mentions `try_apply_planner_shape` and a "post-hoc swap" that has been removed from code.

**Refactor / fix**:
- Delete the K.1 post-hoc-swap section, or move the "Phase K" implementation log into a separate `specs/archive/` file so the live spec only contains current behavior.

### C-8 — `planner_shape_choices` doc comment is inverted

**Severity**: Bloat (doc bug)

**Where**:
- `crates/akita-planner/src/schedule_params.rs:346-357`

The doc comment says one thing (paraphrased: "empty vector means search all shapes returned by `planner_stage1_shapes_to_search`") and the code does the opposite (empty → single `None` entry, i.e., legacy shape-blind path).

**Refactor / fix**:
- Rewrite the comment to match the code (empty → no hybrid search; non-empty → search those shapes).

---

## Findings — Unrelated Code Drift

This PR mixes the fourth-root work with **at least four unrelated cleanups** that should be split into separate PRs. Each makes the diff harder to review and creates merge conflict surface for unrelated work.

### D-1 — ZK code wholesale deletion (~4500 lines)

**Severity**: Unrelated drift

**Where**: 22 deleted files
- `crates/akita-types/src/zk.rs` (-111)
- `crates/akita-pcs/tests/zk.rs` (-499)
- `crates/akita-prover/src/protocol/masking.rs` (-48)
- `crates/akita-types/src/generated/fp128_d32_full_zk.rs` (-726)
- `crates/akita-types/src/generated/fp128_d32_onehot_zk.rs` (-720)
- `crates/akita-types/src/generated/fp128_d64_full_zk.rs` (-774)
- `crates/akita-types/src/generated/fp128_d64_onehot_zk.rs` (-721)
- `crates/akita-types/src/generated/fp128_d128_full_zk.rs` (-788)
- `crates/akita-types/src/generated/fp128_d128_onehot_zk.rs` (-728)
- `specs/akita-zk-commitment-hiding.md` (-771)
- `specs/akita-zk-v-hiding.md` (-293)
- Plus removal of `#[cfg(feature = "zk")]` branches in `crates/akita-types/src/layout/proof_size.rs` and friends.

**Refactor / fix**:
- Move all ZK deletions into a separate PR with title like `chore(zk): remove ZK feature scaffolding`. The fourth-root PR should not be doing this work.

### D-2 — `profile/akita-recursion/` entirely deleted (~3200 lines)

**Severity**: Unrelated drift

13 files removed (RISC0 recursion harness). No connection to the fourth-root verifier.

**Refactor / fix**:
- Separate PR.

### D-3 — Field crate refactor (~1939 lines net delete in akita-field)

**Severity**: Unrelated drift

**Where**:
- `crates/akita-field/src/fields/ext.rs` (-952): `Fp2`/`Fp4` tower simplification, removal of `PowerBasisFp4`.
- `crates/akita-field/src/fields/packed_neon.rs` (-400): NEON refactor.
- `crates/akita-field/src/fields/pseudo_mersenne.rs` (~197 changes).
- `crates/akita-field/src/fields/lift.rs` (~147 changes).
- `crates/akita-field/src/jolt_traits.rs` (~147 changes; `PowerBasisFp4` jolt impls removed).
- `crates/akita-field/src/fields/packed_ext.rs` (~355 changes).

None of these are needed for the fourth-root verifier. The `crt_ntt_repr.rs` +101 lines IS needed (i64 centered-coeff helpers, see C-9 below), but the rest are unrelated cleanups.

**Refactor / fix**:
- Separate PR for the `Fp2`/`Fp4` tower simplification.

### D-4 — Bench file consolidation (`field_arith/*` → single file)

**Severity**: Unrelated drift

**Where**:
- `crates/akita-pcs/benches/field_arith/*` (12 files removed)
- `crates/akita-pcs/benches/field_arith.rs` (+1370)

Pure bench scaffolding refactor.

**Refactor / fix**:
- Separate PR.

### D-5 — `incidence.rs` deletion (-932 lines, planner refactor)

**Severity**: Unrelated drift

**Where**:
- `crates/akita-types/src/proof/incidence.rs` (deleted)
- `specs/planner-incidence-generalization.md` (-457)

`ClaimIncidenceSummary` was removed in favor of `AkitaScheduleLookupKey` consuming t/w/z vector counts directly (commit `53af2ae8`). Unrelated to fourth-root.

### D-6 — `CenteredCoeff = i64` global typing change

**Severity**: Unrelated drift (or at least: unrelated to the *verifier* goal)

**Where**:
- `crates/akita-prover/src/lib.rs:60-63`

The widening was added to support D=32 tensor where mass is `121² = 14641`, which overflows i32 accumulators. But the typing change affects **every** schedule, flat included.

```60:63:crates/akita-prover/src/lib.rs
pub type CenteredCoeff = i64;
/// Infinity norm type for centered folded coefficients.
pub type CenteredInfNorm = u64;
```

Cascade through `crates/akita-prover/src/backend/dense.rs`, `onehot.rs`, `poly_helpers.rs`, `recursive_witness.rs`, `kernels/linear.rs`, plus the new `crates/akita-algebra/src/ring/crt_ntt_repr.rs` i64 entrypoints.

**Refactor / fix**:
- If D=32 tensor is not in the production matrix (per `specs/fourth-root-verifier-optimization-roadmap.md:40-41`, *"D32 tensor is not ready in the benchmark matrix"*), consider keeping `CenteredCoeff = i32` and only widening to i64 inside the tensor-D32 path. Otherwise document explicitly that all centered-coeff math is now i64 and accept the regression in flat schedules.

### D-7 — Specs include implementation logs and benchmark tables

**Severity**: Documentation noise

The seven new spec files include extensive implementation logs and benchmark tables (~3000 total lines of new spec). These are not specifications; they are work logs.

**Files**:
- `specs/fourth-root-verifier-optimization-roadmap.md` (806 lines, mostly Phase 0-8 implementation log).
- `specs/tensor-everywhere-implementation-plan.md` (327 lines).
- `specs/recursive-s-opening-plan.md` (607 lines).
- `specs/tensor-exact-aggregate-evaluator.md` (758 lines).
- `specs/tensor-stage1-parameter-search.md` (108 lines).
- `specs/fourth-root-verifier-implementation-audit.md` (183 lines, which is a prior audit of this same code).
- `specs/fourth-root-verifier-optimizations.md` (508 lines).

The audit `fourth-root-verifier-implementation-audit.md` explicitly says the branch *"should not be described as completing the fourth-root verifier optimization because the claim-reduction sumcheck and setup-opening protocol are still missing"* — yet here it sits alongside CR code that was added later. The relationship between these documents and the actual code is unclear.

**Refactor / fix**:
- Consolidate the seven spec files into:
  - One clean `specs/fourth-root-verifier.md` with the agreed protocol (matching the LaTeX book) and the implementation status checklist.
  - Move implementation logs and benchmark tables into `specs/archive/` or delete them (the git history retains them).

### D-8 — Test fixtures `multipoint_batched_e2e.rs` changes (~650 lines)

**Severity**: Unclear — possibly necessary, possibly drift.

The test file has 651 changed lines. From the commit log it appears to be a mix of "forward tensor config field roles" (`7d267bce`) and unrelated refactors. Should be triaged: if necessary for tensor, keep but split into per-concern commits; if drift, separate PR.

---

## Findings — blockers.mdc Checklist

Going down the blockers.mdc risk checklist:

- ✅ **API shape forced an implementation into identity functions, dummy values, placeholder params, or `()` that hides a capability distinction**: see C-3 (sibling constructors).
- ✅ **A generic trait requires behavior that only some backends can implement naturally**: see S-7 (one-hot has tensor kernel; dense and multilinear fall back to full expansion).
- ✅ **A default implementation materializes data or changes asymptotic behavior in a hot path**: see B-2 (`materialize_setup_claim_tables`).
- ✅ **A fast path was preserved only for existing types, not for new source abstractions**: see C-1 (`eval_algebraic_at_point` is the fast path; only used when claim-reduction is enabled; otherwise the legacy `eval_at_point` runs).
- ✅ **Streaming, parallelism, batching, or memory layout moved from an explicit path into an implicit default**: see S-2 (production presets switched to tensor by default through the regenerated tables).
- ✅ **Tests cover the mock but not the real scheme, or single-proof paths but not batch paths**: tests for the claim-reduction tamper case (`crates/akita-pcs/tests/setup_claim_reduction_e2e.rs::rejects_modified_m_setup_eval`) cover the happy path and a single tamper case. Batch + recursive tampers are present. ZK-related path coverage was deleted in D-1 (ZK feature is gone).
- ✅ **The implementation compiled by adding trait imports, wrappers, or aliases that may obscure the intended boundary**: see C-3 (sibling constructors), C-4 (debug-labeled methods), C-5 (test wrappers in production crate).
- ✅ **A compatibility shim or fallback path was introduced despite a full-cutover requirement**: production schedules cut over to tensor (S-2), but flat is still supported through `Stage1ChallengeShape::Flat` and the hybrid-search infrastructure. There is no "single shape" enforced; instead we have at least three shape regimes (flat, tensor, hybrid).
- ✅ **A spec says stronger behavior than the current code implements**: the LaTeX book promises fourth-root verifier; the code delivers tensor stage-1 + materialized claim-reduction sumcheck. The internal spec `specs/fourth-root-verifier-optimization-roadmap.md` is honest about this gap.

---

## Recommended Refactor Order

If the goal is to land a clean fourth-root verifier PR, I would split the current branch as follows:

1. **PR-zk-removal** (separate, easy): all of D-1.
2. **PR-recursion-profile-removal**: all of D-2.
3. **PR-field-tower-cleanup**: all of D-3 plus relevant `jolt_traits.rs` and `packed_*` changes.
4. **PR-bench-consolidation**: all of D-4.
5. **PR-planner-incidence**: all of D-5 (already partially landed in `main` via merge).

Land 1–5 first. They are independent and the parent fourth-root PR rebases on top.

Then for the fourth-root work itself:

6. **PR-tensor-stage1-plumbing**: tensor challenge sampling (`stage1.rs`), exact aggregate evaluator, transcript labels, MSIS extraction accounting in `LevelParams`. Production presets remain on **flat** shape. Tests cover the algebraic equivalence. No verifier hot path changes.
7. **PR-tensor-stage1-onehot-kernel**: the one-hot tensor fold kernel only (S-7's exception). Dense and multilinear backends remain on `expand_integer` fallback. No production schedule cutover.
8. **PR-claim-reduction-prototype**: claim-reduction sumcheck, `SetupClaimReductionPayload`, proof-shape changes. Defaults remain off (`use_setup_claim_reduction = false`). Single test config opts in. Address C-1, C-2, C-3, C-4, C-6 as part of this PR. **Address S-1 explicitly** (proof-shape vs schedule binding) before merging.
9. **PR-structured-S-evaluator** (the real fourth-root win): implement the `O(log)` `S(r)` evaluator (Phase D-full / Phase C streaming) so that `materialize_setup_claim_tables` is no longer the production path. Re-run benches; only if verifier improves vs main, flip a production preset to tensor + CR.
10. **PR-production-cutover** (only after 9 shows a verifier win): regenerate tables, update `fp128_stage1_challenge_config`, update `security_analysis.md` (S-2). Address S-3 (env var) and S-4 (cache key) defensively.

If the project decides **not** to pursue the fourth-root verifier optimization in this codebase (because S-2's concrete-security ask is large and B-1's verifier regression is real), the alternative is:

- **PR-prover-perf**: keep the genuinely good prover changes (multi-chunk Ajtai tiling, i64 centered fold accumulators) and discard everything tensor/CR-related. Production presets stay flat with `ExactShell{30,12}` (unchanged from main). The branch then delivers a ~2× prover speedup at NV=25 with no verifier regression.

---

## Verification Quality

Checks run as part of *this* audit:
- ✅ Read `5_fourth_root_verifier.tex` (book spec).
- ✅ Read all seven new spec files in `specs/`.
- ✅ Examined `git diff main..HEAD` file-by-file for: planner, types, verifier (levels.rs, ring_switch.rs, stage2.rs, batched.rs, setup_claim_reduction.rs), prover (flow.rs, setup_claim_reduction.rs, quadratic_equation.rs spot-check), challenges (stage1.rs), config (proof_optimized.rs, lib.rs), transcript labels.
- ✅ Cross-referenced via four parallel sub-agents (planner, types, verifier orchestration, prover protocol, field/sumcheck/algebra) for completeness.
- ✅ `cargo fmt -q` (clean).

Checks **not** run as part of this audit:
- ❌ `cargo clippy --all -- -D warnings` — could not run from sandbox (sandbox lock on `target/`). The roadmap states clippy is clean at HEAD; verify locally.
- ❌ `cargo test --workspace` — same reason. The roadmap states the full test suite passes; verify locally.
- ❌ Independent re-derivation of the 128-bit security budget for the new `D=128 Uniform{13}` family (S-2 follow-up).
- ❌ Independent verification of the SIS rank tables for the new tensor extraction collision buckets (S-2 follow-up).
- ❌ Benchmark replay — relied on the numbers documented in `specs/recursive-s-opening-plan.md:541-548` and `specs/tensor-everywhere-implementation-plan.md:282-298`.

**Confidence**:
- Code review confidence: **High** for protocol shape, transcript ordering, and proof-payload threading.
- Cryptographic soundness confidence: **Medium-High** for tensor extraction propagation and stage-1 + CR composition; **Medium** for the concrete security margin of the new production families (S-2's follow-up).
- Performance confidence: **High** that the verifier is currently slower than main (the roadmap documents it); **Low** that any of the cleanup proposed here will move that number — the asymptotic fix is Phase D-full / recursive `S` opening, which is the real next slice.

---

## Appendix — Files With Substantial Tensor / CR Changes (production hot paths)

| File | Lines changed vs main | Role |
|------|----------------------:|------|
| `crates/akita-challenges/src/stage1.rs` | +555 (new) | Tensor challenge sampling, exact aggregate evaluator |
| `crates/akita-verifier/src/protocol/ring_switch.rs` | +1589 / -1589 (net big rewrite) | M-eval split, tensor carry summaries, structured weight evaluator |
| `crates/akita-verifier/src/protocol/setup_claim_reduction.rs` | +197 (new) | Claim-reduction verifier, rounds-only composition |
| `crates/akita-prover/src/protocol/setup_claim_reduction.rs` | +62 (new) | Claim-reduction prover (materialized) |
| `crates/akita-sumcheck/src/eq_weighted_table.rs` | +311 (new) | Four sumcheck primitives (one used in production) |
| `crates/akita-sumcheck/src/drivers.rs` | +58 | `verify_sumcheck_rounds_only` |
| `crates/akita-types/src/proof/mod.rs` | ~766 changes | `SetupClaimReductionPayload`, `LevelProofShape.stage2_setup_claim_reduction` |
| `crates/akita-types/src/layout/params.rs` | +270 | Tensor extraction accounting, `with_setup_claim_reduction` |
| `crates/akita-types/src/layout/flat_matrix.rs` | +346 | `SetupMatrixPolynomialView`, `materialize_table` |
| `crates/akita-config/src/proof_optimized.rs` | +348 changes | Production preset cutover to tensor |
| `crates/akita-config/src/schedule_policy.rs` | +458 changes | Shape-aware layout derivation |
| `crates/akita-config/src/lib.rs` | +181 changes | `use_setup_claim_reduction` CommitmentConfig hook |
| `crates/akita-planner/src/schedule_params.rs` | +641 changes | Hybrid stage-1 shape DP search |
| `crates/akita-prover/src/protocol/flow.rs` | +541 changes | Setup-claim-reduction call site, tensor handoff |
| `crates/akita-prover/src/protocol/quadratic_equation.rs` | +671 changes | Tensor expansion at prover side |
| `crates/akita-prover/src/backend/onehot.rs` | +736 changes | Tensor-aware one-hot fold kernel |
| `crates/akita-transcript/src/labels.rs` | +14 | `CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND`, tensor-left/right labels |

## Appendix — Test Files Added

| File | Lines |
|------|------:|
| `crates/akita-pcs/tests/setup_claim_reduction_e2e.rs` | 548 |
| `crates/akita-pcs/tests/tensor_stage1_e2e.rs` | 537 |
| `crates/akita-pcs/tests/hybrid_stage1_e2e.rs` | 922 |

The first two are necessary. `hybrid_stage1_e2e.rs` exercises the test-only `PlannerHybridCfg` that is itself unused in production (C-5).
