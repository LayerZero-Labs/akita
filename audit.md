# Fourth-Root Verifier Optimization — Comprehensive Audit

**Date**: 2026-05-18
**Branch reviewed**: `feat/tensor-challenges` (HEAD = `cb36143a`, "prover: tier-aware D-row quotient + cross-check tests")
**Comparison base (main)**: `d7dd31ed2c513d4090cdb0c306a19140ee61a393`
**Commits on top of main**: 113
**Diff stat**: 231 files changed, +151 378 / −17 259 lines
**Reference spec**: book §5 (fourth-root verifier)
**Workspace rules**: `.cursor/rules/blockers.mdc`, `.cursor/rules/coda_changes.mdc`
**Goal**: implement book §5 "Fourth-Root Verifier" (Technique 1 tensor-structured challenges + Technique 2 claim-reduction sumcheck + §5.4 tiered commitment design) while preserving ≥128-bit security.

This report supersedes the earlier audit produced at branch checkpoint `b8bf437e`. The earlier audit (now retained inline as referenced findings S-1..S-7 / B-1..B-2 / C-1..C-8 / D-1..D-8 wherever its conclusions still hold) was written before the Phase D-full v2 work (Slices D, E, F.1, G prep, H staging, the recent CRT-capacity fix). Many of its findings are still open; several have changed character because of the new work; a few have been superseded. Every item below is re-derived against the current HEAD.

---

## 0 — Bottom line

1. **The protocol is not yet end-to-end at production parameters.** Phase D-full v2 has landed Slices A–E + F.1 + G prep + the partial H staging commit `5ccde2a0`, but `specs/phase-d-full-handoff.md` §8 still flags F.2–F.5 ("heterogeneous `prepare_m_eval` + S-routing wiring + cascade activation + E2E milestone") as blockers, and slices H and I (SIS table regen + post-cascade security re-audit) are gated on F.5. Two tiered tests pass at small NV (`tiered_onehot_prove_verify_small`, `tiered_dense_prove_verify_small`), but `tiered_production_prove_verify` SIGKILLs (likely OOM) and `tiered_rejects_tampered_meta_material` does **not** reject (see B-3 below).
2. **Soundness: no cryptographic break was found.** The protocol modifications (tensor stage-1, claim-reduction sumcheck, multi-group batched commit, tiered routing) are individually sound and the 128-bit MSIS / |C| ≥ 2^128 baseline is re-established in `specs/security_analysis.md`. However, multiple **defense-in-depth gaps**, **silent invariants**, and **schedule-vs-proof-shape mismatches** can become breakage vectors as soon as the cascade is activated in production. The most urgent items are S-1 (proof-shape vs schedule), S-2 (CRT capacity is gated by `is_tiered`, not by a numeric overflow bound), S-3 (tiered cache vs deterministic re-derivation), and S-7 (`HACHI_PLANNER_S1_WEIGHT` env var still in release code).
3. **Bloat is substantial.** Per `coda_changes.mdc`'s "smallest coherent change" rule, at least ~1 800 LOC of debug helpers, sibling APIs, materialized reference helpers shipped in production, and a 270-line `level == 1` diagnostic block in `flow.rs` should be deleted or `#[cfg(test)]`-gated. Several speculative APIs (`RecursiveWitnessAsPoly`, `tiered_s_cache` on the verifier, `EqWeighted*` siblings, hybrid stage-1 search) have no production caller today and should be removed or fenced.
4. **Documentation drift.** Multiple docs in `specs/`, in source comments, and in the test name `multi_fold_rejects_heterogeneous_per_claim_lp` describe behavior the code no longer implements. The `phase-d-full-handoff.md` itself is partly stale (it says heterogeneous `prepare_m_eval` "rejects loudly", but the code has since been extended to handle it; meanwhile real heterogeneity in stage-2 weight eval still materializes the full hypercube).

The remaining sections are structured as:

- §1 **Phase D-full status**: what each slice actually delivers today.
- §2 **Soundness findings (S-1 … S-13)**: anything that could become a cryptographic break.
- §3 **Goal-coverage / blockers (B-1 … B-5)**: stops the PR from honestly claiming the 4th-root optimization is shipped.
- §4 **Code bloat / `coda_changes.mdc` violations (C-1 … C-14)**: refactors to apply before merge.
- §5 **Documentation drift (D-1 … D-6)**.
- §6 **Recommended refactor order**.
- §7 **Verification quality / what was not checked**.

Severity tags:

- **S** = Soundness Concern (defense-in-depth; if exploited matters).
- **B** = Blocker for the stated goal.
- **R** = Risk (subtle invariant a future change can violate).
- **C** = Code-bloat / `coda_changes.mdc` violation.
- **D** = Documentation drift.

---

## 1 — Phase D-full v2 slice status

Cross-checked against `specs/phase-d-full-handoff.md` (which is itself partly stale; see D-2).

| Slice | What it claimed | Today's reality |
|------:|-----------------|-----------------|
| A | Tiered setup foundation (`TieredSetupParams`, `TieredSetupCommitments<F,D>`, `TieredSetupProverExtras<F,D>`, `OnceLock`-lazy caches) | ✅ landed (`ccbbb8e2`). Verifier cache is **unused on the verify path** (S-3, C-3). |
| B | `Vec<RecursiveOpeningClaim>` + `Vec<RecursivePolyHandle>` | ✅ landed. |
| C.1 | `prove_recursive_multi_fold_with_params`, `verify_one_level` multi-claim branch | ✅ landed. Homogeneous-witness restriction still in force (Slice F.3 not done). |
| C.2.a | `s_opening_value` on the wire | ✅ landed. |
| C.2.b | Tiered types + multi-claim transcript | ✅ landed. |
| C.2.c | Cascade-aware planner with book formula `w_fold_L + |S|/f` | ✅ landed (additive, not v1 `max`). `planner_setup_polynomial_size = 0` in production, so cascade is off at production callers (B-1). |
| D | Multi-group commit kernel + LP shape (`LevelParams.groups`) | ✅ landed (`a669f8b4`). `verify_root_direct_commitments_with_params` still ignores `groups` (S-9). |
| E | Per-handle / per-claim LP plumbing | ✅ landed (`454409fc`). Test `multi_fold_rejects_heterogeneous_per_claim_lp` asserts the wrong thing (D-4); the prover no longer rejects heterogeneous overrides — it groups them, per Slice D. |
| F.1 | `routes_recursively` flag on `verify_setup_claim_reduction` | ✅ landed (`ce8ecf00`). Dispatch is **proof-driven, not schedule-driven** (S-1). |
| F.2 | Heterogeneous `prepare_m_eval` / stage-2 / materialize | Partially landed via `eval_split_at_point_grouped` (Slice G prep). For non-homogeneous *digit depths* across groups, the verifier **falls back to full hypercube materialization** (S-4, B-2). |
| F.3 | Mixed witness types (`RecursiveWitnessAsPoly` + `DensePoly` in same batch) | **NOT landed.** `RecursiveWitnessAsPoly` is a shape carrier without an `AkitaPolyOps` impl (`crates/akita-prover/src/backend/recursive_witness.rs:73-93`). |
| F.4 | Cascade routing activated in production | **NOT landed.** Production presets keep `use_setup_claim_reduction = false` by default (S-12). |
| F.5 | E2E milestone | **NOT landed at production parameters.** Only `tiered_*_prove_verify_small` work today. |
| G | Tiered `k = 64` | Code path exists (Slice H staging `5ccced20`), `tiered_*_prove_verify_mid_f4` work at NV=18; production `tiered_production_prove_verify` at NV=32 SIGKILLs (B-5). |
| H | SIS table regen | **NOT landed.** `sis_floor.rs` has no `meta`/`D_meta` entries; Slice H "staging" is the partial `5ccced20`. |
| I | Post-cascade security re-audit | **NOT landed.** `specs/security_analysis.md` predates Slice G's potential new SIS-rank cells. |

Net: the branch is honest as a **Phase D-full v2 *foundation+prototype***, not as a shipped fourth-root verifier. The earlier audit's B-1 (verifier slower than `main`) still holds at production parameters because the asymptotic win is gated on F.4+F.5 (recursive S opening actually firing in production), which is not done.

---

## 2 — Soundness findings

### S-1 — Stage-2 dispatch is proof-driven, not schedule-driven *(unfixed, identical to old S-1)*

**Severity**: Soundness Concern (defense-in-depth)

**Where**:
- `crates/akita-verifier/src/protocol/levels.rs:842-897` (`verify_root_level`)
- `crates/akita-verifier/src/protocol/levels.rs` analogous block inside `verify_one_level` (~1530-1620)

```842:898:crates/akita-verifier/src/protocol/levels.rs
        if let Some(payload) = stage2.setup_claim_reduction.as_ref() {
            let (stage2_challenges, r_setup, s_opening_value) =
                verify_stage2_with_setup_claim_reduction::<F, _, D>(
                    &stage2.sumcheck,
                    payload,
                    &stage2_verifier,
                    transcript,
                    routes_setup_recursively,
                )?;
            ...
        } else {
            let out = verify_sumcheck::<F, _, F, _, _>(
                &stage2.sumcheck,
                &stage2_verifier,
                transcript,
                |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
            )?;
            out
        }
```

A grep for `use_setup_claim_reduction` under `crates/akita-verifier/` returns **no matches**: the verifier branches purely on whether the proof carries a setup-claim-reduction payload. There is no assertion that `stage2.setup_claim_reduction.is_some() == lp.use_setup_claim_reduction` (and the schedule-driven `routes_setup_recursively` is computed from `s_field_len_emitted`, not from `next_lp.use_setup_claim_reduction`).

This is the same fragility flagged previously and remains unfixed. It is not a cryptographic break today because both branches verify the same identity, but it becomes one as soon as either branch acquires a side-effect the other doesn't (e.g., S-12 production cutover).

**Refactor / fix**:
1. In both `verify_root_level` and `verify_one_level` dispatch sites, assert `stage2.setup_claim_reduction.is_some() == effective_lp.use_setup_claim_reduction`, returning `AkitaError::InvalidProof` on mismatch. Mirror in the prover.
2. Additionally assert: if `s_field_len_emitted > 0` (schedule says route recursively) **then** `stage2.setup_claim_reduction.is_some()` must be true, and the next-level state must have an S-side `RecursiveOpeningClaim`. Today no such check exists.

### S-2 — CRT-capacity dispatch is keyed on `is_tiered`, not on a numeric overflow bound

**Severity**: Soundness Concern (latent silent breakage as schedules grow)

**Where**:
- `crates/akita-prover/src/protocol/quadratic_equation.rs:1636-1788` (`compute_r_split_eq` heterogeneous A-row Z-quotient setup)
- `crates/akita-prover/src/protocol/quadratic_equation.rs:1421-1475` (`compute_v_tier_aware` D-row setup)

The recent fix (`specs/tiered-d-row-m-relation-bug-handoff.md`) dispatched the chunks A-row Z-quotient to a field-domain `high_half(A · z_pre)` path when `has_tiered_group && is_tiered` because the NTT-cached `fused_split_eq_quotients` kernel silently wraps when the integer per-coefficient bound exceeds the CRT product `P ≈ 2^150`. The fix is correct **for the chunks group**.

The dispatch predicate is `spec.tier.is_some_and(|t| t.is_tiered())`, **not** a numerical comparison against `P`. The W-group and meta-group A rows still take `fused_split_eq_quotients`. Per the handoff's own §7:

> If a future schedule pushes the W or meta A rows over the CRT capacity, the same field-domain dispatch will need to be widened — in that case generalize the predicate from `is_tiered` to a per-group overflow check.

There is **no fail-loud assertion anywhere** that detects CRT capacity overflow regardless of `is_tiered`. A future schedule (larger `inner_width_g`, larger `|z|`, or a different `Q-variant dispatch`) can silently wrap and produce a prover that disagrees with the verifier — except that the verifier evaluates `−chunks_z = LIFTED(A·z_pre)(α)` via scalar field arithmetic at `α`, so prover and verifier disagree and the proof rejects. Mathematically this is "safe" in the sense that an honest prover hits the wraparound and fails to produce a valid proof; it is "unsafe" in the sense that **the relation no longer encodes what it should**, and a clever adversary could potentially construct a witness that satisfies the wrapped identity but not the true field identity (this would require more analysis than was done in this audit).

**Refactor / fix**:
1. Compute the integer bound `inner_width_g · D · |A_norm| · |z_pre_norm|` per group at dispatch and compare against the active CRT product `P`. If the bound exceeds `P`, dispatch the field-domain path. Use the same predicate for D-rows, W A-rows, meta A-rows, and chunks A-rows.
2. Independently, panic (or return `InvalidSetup`) if the bound exceeds `P` in any group where the NTT path is taken. This is the fail-loud invariant the handoff §7 asked for.
3. Add a regression test that constructs a synthetic NTT input at the boundary and confirms either dispatch or fail-loud, never silent wrap.

### S-3 — Verifier's tiered-cache tamper test does not reject *(book §5.4 lines 692-699 binding)*

**Severity**: Soundness Concern (real attack vector once cascade is active; **demonstrated by the test that doesn't reject**)

**Where**:
- `crates/akita-verifier/src/protocol/levels.rs:370-408` (`expand_tiered_setup_claims`)
- `crates/akita-verifier/src/protocol/levels.rs:527-602` (`derive_tiered_setup_material_for_verifier`)
- `crates/akita-pcs/tests/tiered_setup_e2e.rs:445-503` (`tiered_rejects_tampered_meta_material`)
- `crates/akita-types/src/proof/setup.rs:50-167` (`AkitaVerifierSetup::tiered_s_cache_get_or_init`)

The verifier **recomputes** the tiered B-side material at every verification call inside `derive_tiered_setup_material_for_verifier` — it does **not** read from `tiered_s_cache`. A workspace grep confirms `tiered_s_cache_get_or_init` has **zero call sites under `crates/akita-verifier/`**. The cache exists, is `pub`, and is documented as "lazy on first use", but verification ignores it.

This is good for *honest* operation (no malicious-cache attack today because the cache isn't consulted), but it has three problems:

1. **The tamper test `tiered_rejects_tampered_meta_material` mutates `tiered_s_cache`** (`tiered_setup_e2e.rs:489-492`) — i.e. it tampers with material that the verifier never reads. The test currently **does not reject** (per the handoff §6), not because there is a real security gap, but because the test poisons the wrong artifact. The test name and intent are wrong, and **there is currently no test that exercises actual meta-tier tamper resistance** of the verifier path.
2. The verifier's defense relies on re-deriving the meta material from `setup.expanded.shared_matrix` every time, which is **O(rows × cols × D)** per call — this is the asymptotic gap that recursive S opening was supposed to close. The cache was added to skip this, but the verifier path never uses it.
3. As soon as someone wires the cache into the verifier hot path (perfectly reasonable optimization once F.5 lands), the absence of a cross-check between cached and freshly-derived material becomes a real attack vector: a malicious prover that controls or influences the cache (e.g., via a network deserialization path or a test harness) can substitute arbitrary `chunk_b_commitments`.

**Refactor / fix**:
1. **Today**: either delete `tiered_s_cache` from `AkitaVerifierSetup` entirely (it's dead weight on the verifier and the prover has its own cache), or fence it behind a feature flag that is off by default.
2. **Fix the test**: change `tiered_rejects_tampered_meta_material` to corrupt the *proof payload* (e.g., the carried `u_meta` or `c_meta` on the wire if the proof shape supports it), not the cache. If the proof shape doesn't carry meta material yet (because slice F.5 is not done), the test should be deleted with a comment pointing at the missing infrastructure.
3. **When the cache is wired (post F.5)**: ensure first-write derives from the public matrix and any subsequent cache reads cross-check by hash against a deterministic re-derivation; fail with `InvalidSetup` on mismatch.

### S-4 — Tiered/multi-group claim-reduction verifier still materializes the full weight hypercube

**Severity**: Soundness Concern (it isn't an attack vector but it defeats the asymptotic — the verifier is *not* O(log) on tiered paths)

**Where**:
- `crates/akita-verifier/src/protocol/ring_switch.rs:2169-2173` inside `eval_setup_weight_at_point`

```2169:2173:crates/akita-verifier/src/protocol/ring_switch.rs
        if !self.uses_homogeneous_outer_layout() {
            let weights =
                self.setup_weight_table_at_point_grouped::<D>(x_challenges, setup, alpha)?;
            return akita_sumcheck::multilinear_eval(&weights, r_setup);
        }
```

The hot path that book §5.3 line 528-538 says should be `O(log m_row + log d)` is structured **only for the homogeneous-LP layout**. The instant `lp.groups.is_some()` with heterogeneous shape (which is exactly the cascade case the 4th-root verifier is supposed to optimize) the verifier falls back to materializing `2^(row_bits + col_bits + coeff_bits)` weights. This is the same asymptotic order as the legacy fused stage-2.

The prior audit's B-2 flagged the prover's `materialize_setup_claim_tables` analogue; the verifier-side equivalent was missed because earlier the `eval_setup_weight_at_point` structured path handled the homogeneous case (which is all the production presets reach today).

**Refactor / fix**:
- Implement a structured `eval_setup_weight_at_point_grouped` that factors the per-group sub-claims algebraically, the same way the homogeneous path does. The required ingredients are already on `PreparedMEval` (`group_layouts`, the per-group challenges); only the algebra has to be ported.
- Until that's done, **gate `lp.groups.is_some()` cascade-on production paths off** in `proof_optimized.rs`; otherwise enabling cascade for production presets would silently regress verifier performance to the legacy asymptotic at every level.

### S-5 — `routes_recursively == true` decouples cleartext check from any actual recursive discharge

**Severity**: Soundness Blocker if cascade ever turns on with mismatched proof shape

**Where**:
- `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:120-172` (`verify_setup_claim_reduction`)
- `crates/akita-verifier/src/protocol/levels.rs:1533-1537` (`routes_setup_recursively = level_step.s_field_len_emitted > 0`)

`routes_recursively = !is_last` is **not** the actual gate; the actual gate is `s_field_len_emitted > 0` from the planner's `FoldStep`. When `s_field_len_emitted > 0` the cleartext check `setup_view.mle == s_opening_value` is dropped (per book §5.3 line 627-642 "S enters L+1 unfolded as additional polynomial"); soundness is then anchored by the **next level's joint multi-group open of S** (Slice F.4/F.5).

Slices F.4 and F.5 are not done. If `s_field_len_emitted > 0` but the next-level state does not actually contain an S-claim that gets discharged at that level, **the verifier accepts whatever `s_opening_value` the prover provides**, modulo only the closing-oracle equality `weight_at_point · s_opening_value == final_running_claim`. A malicious prover can pick any `s_opening_value` consistent with a chosen `m_setup_eval`; the sumcheck closing equality is still satisfied by construction, so the proof is accepted.

**This is the F.2/F.4/F.5 soundness blocker the handoff §8 already calls out.** The point of including it here is that **today nothing in the verifier prevents a runtime / config from accidentally turning this on**. The planner picks `s_field_len_emitted = 0` for production presets (because `use_setup_claim_reduction = false`), so the path is unreachable in practice today. But as soon as someone flips one bit of a config preset, the verifier accepts unsound proofs.

**Refactor / fix**:
1. Add the equality assertion from S-1 (proof-shape ↔ schedule), plus a stronger assertion: if `routes_recursively == true` then the next-level state must contain an S-side `RecursiveOpeningClaim` (or a corresponding entry in `RecursiveProverState::handles` on the prover side). Reject if absent.
2. Alternatively, gate the `routes_recursively` branch behind a `#[cfg]` until F.5 lands, so it cannot be reached at runtime in shipped binaries.

### S-6 — Schedule cache key omits all of `use_setup_claim_reduction`, tier shape, cascade *(unfixed, identical to old S-4)*

**Severity**: Risk (current usage safe; invariant fragile)

**Where**:
- `crates/akita-types/src/schedule.rs:206-248` (`AkitaScheduleLookupKey`)

```206:221:crates/akita-types/src/schedule.rs
pub struct AkitaScheduleLookupKey {
    pub max_num_vars: usize,
    pub num_vars: usize,
    pub layout_num_claims: usize,
    pub batch: AkitaRootBatchSummary,
}
```

No `use_setup_claim_reduction`, no `tier_shrink_factor`, no cascade bit, no per-handle-LP digest. Safe today only because `AkitaVerifierSetup<F>` is per-`Cfg` and `Cfg` uniquely determines those bits. It breaks silently the moment someone caches schedules across modes.

**Refactor / fix**:
- Add the missing fields to `AkitaScheduleLookupKey` **or** assert at construction time that the cache is per-`Cfg`/per-mode. The first option is cheap (4 extra bytes per key) and removes the foot-gun.

### S-7 — `HACHI_PLANNER_S1_WEIGHT` env var still in release code *(unfixed, identical to old S-3)*

**Severity**: Risk (deployment/configuration drift)

**Where**:
- `crates/akita-planner/src/schedule_params.rs:359-368` (`stage1_prover_penalty`)

The env var override is unconditional (not `#[cfg(debug_assertions)]` gated). Prover and verifier picking different values pick different planner schedules; planner-only paths (tests, benches, non-generated configs) drift silently.

**Refactor / fix**:
1. Remove the env var entirely, or
2. Gate behind `#[cfg(debug_assertions)]` and emit a loud warning on first read.

### S-8 — `level_proof_bytes` still omits CR + tier stage-2 payload *(unfixed, expanded from old S-5)*

**Severity**: Risk (planner accuracy)

**Where**:
- `crates/akita-types/src/layout/proof_size.rs:578-602` (`level_proof_bytes`)

```578:602:crates/akita-types/src/layout/proof_size.rs
pub fn level_proof_bytes(...) -> usize {
    ...
    y_bytes
        + v_bytes
        + stage1_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + next_commit_bytes
        + next_eval_bytes
}
```

Still no branch on `lp.use_setup_claim_reduction`, no `m_setup_eval` byte, no setup-claim-reduction sumcheck rounds, no tiered `(c_meta, v_meta, u_meta)` proof bytes, no per-group sub-relation row scaling. Once the planner activates cascade in production (post F.5), it will pick the wrong schedule because the cost model underestimates the actual proof size.

**Refactor / fix**:
- Make `level_proof_bytes` shape-aware: take `LevelParams`, inspect `lp.use_setup_claim_reduction` and `lp.groups`/`lp.tier`, and add the correct payload bytes. Update the tests `planned_level_bytes_match_two_stage_payload_at_all_bases` to cover the CR-enabled and tiered-enabled cases.

### S-9 — `verify_root_direct_commitments_with_params` ignores `groups`

**Severity**: Risk (latent inconsistency in multi-group commit verification)

**Where**:
- `crates/akita-prover/src/api/commitment.rs:398-404`

The function loops over groups but calls `commit_with_params(..., params)` unmodified, ignoring the per-group lowered LP. For `params.groups == Some(...)`, the recomputed commitments can disagree with honest multi-group commits while the rest of the verification uses grouped logic. Currently no production caller hits this code with multi-group input, but it is a hidden bug that will manifest the moment multi-group root commitments are needed (e.g., for tampering tests or test fixtures).

**Refactor / fix**:
- Pass `params.group_specs(num_groups)?[i].lower_into_outer(params)` to the per-group `commit_with_params` call. Add a regression test for `verify_root_direct_commitments_with_params` with a heterogeneous multi-group LP.

### S-10 — `groups_are_homogeneous` returns true for `Some(vec![])`

**Severity**: Risk (edge case that breaks "homogeneous ⇒ fast path safe" callers)

**Where**:
- `crates/akita-types/src/layout/params.rs:622-633`

```622:633:crates/akita-types/src/layout/params.rs
pub fn groups_are_homogeneous(&self) -> bool {
    match &self.groups {
        Some(groups) => groups.iter().all(|g| g.matches_outer(self)),
        None => true,
    }
}
```

`Some(vec![])` returns `true` vacuously. Callers that use `groups_are_homogeneous() ⇒ legacy fast path` reach the fast path with zero groups, which is almost certainly wrong.

**Refactor / fix**:
- Treat empty `groups` as `InvalidSetup` in the constructor (or in `group_specs`), and document that `groups_are_homogeneous()` may only be called on a validated `LevelParams`.

### S-11 — `m_row_layout` panics on inconsistent tiered configs

**Severity**: Risk (panic on InvalidProof input, not an explicit error)

**Where**:
- `crates/akita-types/src/layout/params.rs:924-928`

`.expect("has_tiered implies grouped layout")` will panic at runtime if a malformed proof carries a tiered marker without a matching grouped layout. Per `coda_changes.mdc` "Contracts And Tests", contracts should be explicit and surfaceable as errors, not unreachable-but-actually-reachable panics.

**Refactor / fix**:
- Return `Result<MRowLayout, AkitaError>` and propagate. Mirror in `total_a_row_count` which uses a structurally-narrow heuristic `(2 + usize::from(has_w_group)) * n_a` (params.rs:877-889).

### S-12 — Production presets do not ship with claim-reduction enabled (corrects S-2 of prior audit)

**Severity**: Risk (the PR's narrative implies tensor+CR shipped; in reality only tensor stage-1 ships)

**Where**:
- `crates/akita-config/src/lib.rs:105-111` (`CommitmentConfig::use_setup_claim_reduction` defaults to `false`)
- `crates/akita-types/src/layout/params.rs:475-477` (`LevelParams::params_only` sets `use_setup_claim_reduction: false`)
- `crates/akita-config/src/proof_optimized.rs:54-94` (fp128 presets do not override the default)

The six fp128 production presets ship with:
- **Tensor stage-1 on D=64 and D=128** (per `crates/akita-types/src/generated/mod.rs:180-206`), challenge families `ExactShell{30,12}` at D=64 and `Uniform{32, ±1}` at D=128 (production families re-derived in `specs/security_analysis.md`).
- **Flat stage-1 on D=32** (only `BoundedL1Norm` family available — see prior audit S-2).
- **`use_setup_claim_reduction = false`** at every level. Claim-reduction sumcheck does not run in production today.

This is a substantive correction to the prior audit's S-2, which assumed CR was on in production. **CR is not on in production.** The prior audit's specific concrete-security concerns about `D=128 Uniform{13}` are also superseded — the production family was migrated to `Uniform{32, ±1}` and re-analyzed in `specs/security_analysis.md` §4 (all six presets clear 128-bit MSIS with margins ≥ 0.1 bit; smallest is `d64_*` at 128.1 bit, comfortable but thin).

The risk is the same as S-1/S-5: **flipping `use_setup_claim_reduction = true` on a production preset today would activate the unsound F.5-not-done path**. There is no compile-time or test-time check that prevents this misconfiguration.

**Refactor / fix**:
- Add a fail-loud constructor check on `Cfg::commitment_config()` for production fp128 presets: `use_setup_claim_reduction = true` requires the F.5 milestone (slice metadata bit) and the planner's `planner_setup_polynomial_size > 0`. Else `InvalidSetup`.

### S-13 — Per-level knowledge-error budget is not literal 2^-128 per level

**Severity**: Informational; sets expectations for what the security baseline actually proves

**Where**:
- `specs/security_analysis.md` §5 table (lines 199-205)
- Book `lem:fourthroot-knowledge-error` (§5.5 lines 1026-1043)

Per `specs/security_analysis.md` §5, the per-level CWSS / Fiat-Shamir term is:

- D=64 tensor: ε ≈ 2^-122.5 per level
- D=128 tensor: ε ≈ 2^-123.2 per level
- D=32 flat: ε ≈ 2^-104.0 per level

The book's standard reading (Lemma 3) is "negligible at λ = 128 once |C| ≥ 2^128", which is true and which `specs/security_analysis.md` §5 documents. But the wording "guarantee at least 128 bits of security" from the user query, if read literally as ε_per_level < 2^-128 in the statistical sense, is **not** what these numbers establish. The aggregated bound across `L ≈ 5–6` levels (union bound) is also a factor of ~`L` looser.

**This is not a bug.** It is the same security argument as the parent Hachi paper and is conventionally accepted as "128-bit secure" in the literature. The concern is that the audit asked for an explicit confirmation and the answer is: **yes**, under the conventional reading (CWSS Lemma 3 + |C| ≥ 2^128 + MSIS ≥ 128 bits at every Ajtai role), the protocol clears 128 bits. **No**, the bound is not a literal ε < 2^-128 per level; it is closer to ε ≈ 2^-104 (D=32) to 2^-123 (D=128).

**Refactor / fix (documentation only)**:
- Update any PR description / README to use the same language `specs/security_analysis.md` uses (`128-bit MSIS + |C| ≥ 2^128`), not "2^-128 per-level statistical".
- Add the post-Phase-D-full numbers when F.5 lands (Slice I), because the per-level error gains an additional `(log m_row + log d) · (deg + 1) / |F_q^k|` term.

---

## 3 — Goal coverage / blockers

### B-1 — The 4th-root verifier asymptotic is **not** delivered

**Severity**: Blocker for the stated goal

The book §5 asymptotic `O(N'^{1/4})` per level requires both Technique 1 (tensor stage-1 — done) **and** Technique 2 (claim-reduction sumcheck **with structured `S(r)` discharge via recursive opening at L+1** — not done).

What's missing today vs the book:

| Book section | Component | Status |
|--------------|-----------|--------|
| §5.2 Tensor stage-1 challenges | Tensor sampling + exact aggregate evaluator | ✅ landed |
| §5.3 Claim-reduction sumcheck | Sumcheck primitive + `s_opening_value` on the wire | ✅ landed |
| §5.3 Joint multi-group commit (`split commitment`) | `LevelParams.groups` + multi-group commit kernel | ✅ landed (root only; recursive cascade off) |
| §5.3 S enters L+1 unfolded as additional polynomial | Mixed-witness recursive batch + per-handle LP plumbing | ⚠ partial (per-handle LP done; mixed witness types Slice F.3 NOT done) |
| §5.3 Structured S(r) evaluator | `eval_setup_weight_at_point` for homogeneous LP only | ⚠ partial (heterogeneous falls back to materialized table, S-4) |
| §5.4 Tiered commitment (`f = 8`, `k = 64`) | Code path exists; small tests pass; production tests SIGKILL | ⚠ blocked on B-5 |
| §5.4 10 check groups | Block-diagonal D_chunk/B_chunk MLE collapse (`22bf8304`) | ✅ landed |
| §5.4 Tier-3 meta-commitment | Meta commitments derived; tamper test broken (S-3) | ⚠ partial |
| §5.5 Combined protocol rounds 1-8 | Transcript labels + ordering | ✅ landed |
| §5.5 Production cascade activation | `planner_setup_polynomial_size = 0` for production | ❌ NOT landed |
| §5.5 Theorem 5.4 soundness | Re-audit after Phase D-full v2 | ❌ NOT landed (Slice I) |

The branch is at "Phase D-full foundations + Slice G prep", not at "fourth-root verifier shipped". Per the user's acceptance criterion ("final protocol must be secure and guarantee at least 128 bits of security"), the protocol on this branch **does** clear 128 bits today (because the dangerous paths are unreachable in production presets — see S-12), but it **does not** deliver the headline 4th-root asymptotic.

**Recommended next slice**: land F.3 + F.4 + F.5 + G full + H + I sequentially per `phase-d-full-design.md` §6. Until that's done, do not flip `use_setup_claim_reduction = true` on any production preset.

### B-2 — Heterogeneous claim-reduction verifier is O(table size), not O(log)

See **S-4** above. The asymptotic win is gated on the structured evaluator landing for heterogeneous LPs. Without it, even with F.5 landed, the verifier remains at the same asymptotic order as `main`.

### B-3 — `tiered_rejects_tampered_meta_material` does not reject

See **S-3** above. The test asserts a security property that the test itself does not exercise (it tampers with verifier-side cache that is never read). There is currently no test that exercises real meta-tier tamper resistance.

### B-4 — Heterogeneous prover restrictions still active

**Severity**: Blocker for Slice F.5

**Where**:
- `crates/akita-prover/src/backend/recursive_witness.rs:73-117` (`RecursiveWitnessAsPoly` is a shape-only carrier without `AkitaPolyOps` impl)
- `crates/akita-prover/src/protocol/flow.rs:152-157` (`RecursiveHandlePoly` is Witness-vs-Dense enum only)

Slice F.3 ("mixed witness types in recursive multi-claim path") is not implemented. The prover can carry a `dense_poly: Option<DensePoly<F, D>>` next to the digit witness, but the multi-claim fold loop does not actually consume both polynomial types side-by-side through the same polynomial-ops interface. This blocks F.4 (cascade routing activated in production) and F.5 (E2E).

**Refactor / fix**:
- Implement the `AkitaPolyOps` trait for `RecursiveWitnessAsPoly` (the "column-major vs row-major fold orientation reconciliation" mentioned in `phase-d-full-handoff.md` §8 F.3) so the recursive batch can carry both `DensePoly` and `RecursiveWitnessView` handles uniformly.

### B-5 — `tiered_production_prove_verify` SIGKILLs at NV=32

**Severity**: Blocker for production-scale validation

**Where**:
- `crates/akita-pcs/tests/tiered_setup_e2e.rs:344-385`

Per `specs/tiered-d-row-m-relation-bug-handoff.md` §6, the test gets SIGKILL (likely OOM) at NV=32 dense, `f=8`, `k=64`. The field-domain chunks A quotient at production parameters is `n_A · inner_width_g · D² ≈ 1.6 × 10⁸` mults per level — that's a lot, but not OOM-large. The OOM is more likely from witness materialization elsewhere; needs profiling.

**Refactor / fix**:
1. Profile the OOM at NV=32 (memory ceiling, peak allocations).
2. If the field-domain quotient is the culprit, stream rows instead of accumulating all at once.
3. If witness materialization is the culprit, consider chunked / streaming `RecursiveWitnessAsPoly` (also feeds B-4).

---

## 4 — Code bloat / `coda_changes.mdc` violations

### C-1 — 270-line `level == 1` diagnostic block in `prove_fold_level_from_quadratic`

**Severity**: Bloat (production code shipping a dev-only diagnostic harness)

**Where**:
- `crates/akita-prover/src/protocol/flow.rs:1236-1508`

`prove_fold_level_from_quadratic` ships an inline ~270-line debug block (lines 1236-1508) that fires when `level == 1`, recomputing the relation per row, materializing `EqPolynomial::evals(&tau1)`, walking `claim_group_sizes`, building `layouts`, and printing `tracing::debug!` for `relation_claim_matches_direct`, `r_segment_matches_expected`, `neg_r_matches_expected`, `first bad M row`, `original A row group sums`, etc.

This is the diagnostic harness used to find the chunks A-row CRT bug (`specs/tiered-d-row-m-relation-bug-handoff.md`). The bug is fixed; the harness should be removed or moved to a `#[cfg(test)]` debug module. As shipped, it:
- Runs on every level-1 fold in **release** builds.
- Allocates and walks `live_x_cols × y_len` per row × per `m_row_count` rows — non-trivial.
- Generates 9+ `tracing::debug!` calls per fold, even though `tracing::debug!` is compiled in and the message-formatting cost is paid even when the subscriber filters it out.
- Violates `coda_changes.mdc`: *"Avoid defensive, retrospective, or PR-justifying comments. Do not narrate what went wrong historically."*

**Refactor / fix**:
- Delete the entire `if level == 1 { ... }` block (lines 1236-1508). The bug it diagnosed is shipped-fixed with a unit test (`tiered_grouped_m_rows_match_committed_witness_multi_a`); the diagnostic is no longer load-bearing.
- If diagnostics are still desired for the next investigation, move them to a `#[cfg(feature = "trace-folds")]` block or a stand-alone test helper.

### C-2 — Sibling tiered-derivation paths in prover code

**Severity**: Bloat (`coda_changes.mdc`: "prefer generalizing the existing function over adding sibling variants")

**Where**:
- `crates/akita-prover/src/api/tiered_setup.rs:198-279` (`derive_tiered_setup_handle_bundle`)
- `crates/akita-prover/src/protocol/flow.rs:378-519` (`build_tiered_handle_material`)

Both functions chunk `S` via `tiered_setup_chunk_index_map`, commit each chunk via `commit_dense_s_handle_direct`/`commit_with_params(chunk_lp)`, and bind the per-chunk B-side outputs via a meta commitment on the padded concatenation. They are not exact duplicates — `build_tiered_handle_material` carries a `TieredHandleMaterial` for the in-flight recursive prover state, while `derive_tiered_setup_handle_bundle` carries a `TieredSetupCommitments` for the setup cache — but the *derivation logic* is duplicated.

**Refactor / fix**:
- Factor a shared `derive_tiered_setup_full_commitments_inner` returning the chunk/meta materials, then have both call sites wrap their respective output structs around the shared core.
- This also tightens S-3's hardening story (only one place to add the cache-vs-fresh cross-check).

### C-3 — `tiered_s_cache` on `AkitaVerifierSetup` is dead code

**Severity**: Bloat (~120 LOC of cache machinery used only by the broken tamper test)

**Where**:
- `crates/akita-types/src/proof/setup.rs:50-167`

`AkitaVerifierSetup::tiered_s_cache`, `cached_tiered_s_commitments`, `tiered_s_cache_get_or_init`, the `OnceLock` + `Box<dyn Any>` downcast machinery — all of this has zero call sites under `crates/akita-verifier/`. The only consumer is the broken test in `tiered_setup_e2e.rs:489-492` (see S-3, B-3).

**Refactor / fix**:
- Delete the verifier-side cache (the prover side `AkitaProverSetup::tiered_s_cache_get_or_init` is genuinely used and stays). Update the broken test per B-3.

### C-4 — `eval_algebraic_at_point` vs `eval_split_at_point` duplication still present *(unfixed, identical to old C-1/C-6)*

**Severity**: Bloat (`coda_changes.mdc`: "do not keep two diverging copies")

**Where**:
- `crates/akita-verifier/src/protocol/ring_switch.rs:677+` (`eval_split_at_point`)
- `crates/akita-verifier/src/protocol/ring_switch.rs:1331+` (`eval_algebraic_at_point`, still annotated `#[allow(clippy::too_many_lines)]`)

Unchanged since the prior audit. `eval_algebraic_at_point` is the same shape as `eval_split_at_point` minus the setup-matrix reads.

**Refactor / fix**:
- Either collapse to `eval_split_at_point(... mode: SplitMode::AlgebraicOnly)`, or replace `eval_algebraic_at_point` with a 5-line wrapper that takes the `algebraic` field of `eval_split_at_point`'s output. Drop the `#[allow(clippy::too_many_lines)]`.

### C-5 — `eq_weighted_table.rs` siblings *(unfixed, identical to old C-2)*

**Severity**: Bloat

**Where**:
- `crates/akita-sumcheck/src/eq_weighted_table.rs` (311 lines, 4 pub structs)

| Primitive | Production caller | Test caller |
|-----------|-------------------|-------------|
| `WeightedTableProver` | `crates/akita-prover/src/protocol/setup_claim_reduction.rs:48` | — |
| `WeightedTableVerifier` | — | `crates/akita-pcs/tests/ring_switch.rs` |
| `EqWeightedTableProver` | — | `crates/akita-types/src/layout/flat_matrix.rs` |
| `EqWeightedTableVerifier` | — | same |
| `eq_eval` (free fn) | — | internal only |

Three of the four are test-only. The verifier inlines `weight_at_point * setup_at_point` directly in `verify_setup_claim_reduction:148`.

**Refactor / fix**:
- Move `EqWeightedTable*` and `WeightedTableVerifier` behind `#[cfg(test)]`, or rewrite the affected tests against `WeightedTableProver` + an inline verifier check. Net reduction ~200 of 311 lines.

### C-6 — `materialize_setup_claim_tables` labeled "Reference / debug helper" but is the production path *(unfixed, identical to old B-2/C-4)*

**Severity**: Bloat + misleading labeling

**Where**:
- `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:42-65` (definition)
- `crates/akita-prover/src/protocol/setup_claim_reduction.rs:52-53` (only call site, production)

The doc comment still says "Reference / debug helper: production callers should prefer the structured evaluator..." but the prover crate is its only consumer. Either:
- Replace the prover's call with the structured `S(r)` evaluator (this is the real fourth-root prover-side win), or
- Drop the misleading "Reference / debug helper" label and accept that the prover materializes the table.

Pick one; current state is the worst of both worlds.

### C-7 — `debug_expanded_challenge_evals` / `debug_split_eval_table` ship in production *(unfixed, identical to old C-4)*

**Severity**: Bloat (debug helpers without `#[cfg(test)]`)

**Where**:
- `crates/akita-verifier/src/protocol/ring_switch.rs:615-627` (`PreparedMEval::debug_expanded_challenge_evals`)
- `crates/akita-verifier/src/protocol/ring_switch.rs:1671-1687` (`PreparedMEval::debug_split_eval_table`)

Only callers are tests in `crates/akita-pcs/tests/ring_switch.rs`.

**Refactor / fix**:
- Gate behind `#[cfg(test)]` or move to a `pub(crate)` test helper module.

### C-8 — Hybrid stage-1 search infrastructure unused in production *(unfixed, identical to old C-5)*

**Severity**: Bloat (~500+ LOC of test-only feature in production crate)

**Where**:
- `crates/akita-planner/src/lib.rs:135-137` (`planner_stage1_shapes_to_search` defaults to empty)
- `crates/akita-planner/src/schedule_params.rs:510-521` (`planner_shape_choices` maps empty → `vec![None]` only)
- `crates/akita-pcs/tests/hybrid_stage1_e2e.rs` (test wrapper)

Per the prior audit's K.6 results, hybrid never wins net at production parameters. The infrastructure remains dead code for production presets.

**Refactor / fix**:
- Delete the hybrid DP loop and `PlannerHybridCfg`, or keep it test-only and remove it from `akita-planner`'s public surface.

### C-9 — `RecursiveWitnessAsPoly` is a shape carrier without trait impl

**Severity**: Bloat (speculative API; coda_changes.mdc: "no speculative fallback")

**Where**:
- `crates/akita-prover/src/backend/recursive_witness.rs:73-93`

`RecursiveWitnessAsPoly` exists for Slice F.3 but Slice F.3 is not implemented. The struct compiles, has no `AkitaPolyOps` impl, and is referenced only in tests and in the per-handle plumbing that doesn't use it productively. Until F.3 ships, this struct should not be in the codebase.

**Refactor / fix**:
- Delete `RecursiveWitnessAsPoly` and reintroduce it in the slice that actually implements the trait. The `RecursivePolyHandle.dense_poly: Option<DensePoly>` field already covers the present shape needs.

### C-10 — `multi_fold_rejects_heterogeneous_per_claim_lp` does not assert rejection

**Severity**: Bloat (test name and intent mismatch the implementation)

**Where**:
- `crates/akita-pcs/tests/per_handle_lp.rs:129-195`

The test name says "rejects"; the assertion checks the result is not an `InvalidSetup` containing the substring "per-claim LP override" — that error string does not appear anywhere in the enforcement code (grep shows it only in docs and the test). The prover *used to* reject heterogeneous overrides (Slice E original spec); after Slice D's `batch_groups` machinery, it *groups* them instead. The test was not updated to match.

The test also contains `commit_w_for_next = unreachable!`, which is a code smell.

**Refactor / fix**:
- Rename the test to match what it actually verifies (e.g., `multi_fold_groups_heterogeneous_per_claim_lps`), update the assertion to check the grouped behavior, and remove the dead `unreachable!` branch.

### C-11 — Stale rejection-narrative comments in flow.rs / recursive_opening_claim.rs

**Severity**: Bloat (documentation drift, see also D-4)

**Where**:
- `crates/akita-prover/src/protocol/flow.rs:1848-1858, 1879` (`prove_recursive_multi_fold_with_params` doc claims rejection of "override disagreement")
- `crates/akita-types/src/proof/recursive_opening_claim.rs:46-48` (`per_claim_lp` doc claims "verifier rejects when per-claim LPs are non-homogeneous (see `verify_one_level`'s multi-claim branch)")

Neither rejection happens. The verifier proceeds into ring-switch / stage-2 with grouped heterogeneous layouts (`verify_one_level` ~957-1007). The comments describe the Slice E original behavior, which was superseded mid-implementation by Slice D's group_specs path.

**Refactor / fix**:
- Either rewrite the comments to match the grouping behavior, or (if rejection was the intent for safety until F.3 lands) restore the explicit guards and update the test (C-10).

### C-12 — `probe_*` diagnostic tests with `eprintln!` in production test suite

**Severity**: Bloat (diagnostic probes shipped without `#[ignore]`)

**Where**:
- `crates/akita-pcs/tests/tiered_setup_e2e.rs:529-550` (`probe_min_viable_nv_for_tier_f4`) — uses `eprintln!`, not `#[ignore]`-gated
- `crates/akita-pcs/tests/tiered_setup_e2e.rs:217-218, 506-507` — `probe_dense_f4_scaling` and `probe_min_viable_nv_for_tier_f2` are `#[ignore = "diagnostic..."]`-gated correctly

**Refactor / fix**:
- Add `#[ignore = "diagnostic probe"]` to `probe_min_viable_nv_for_tier_f4` and replace `eprintln!` with `tracing::debug!` (then run with `RUST_LOG=debug`).

### C-13 — `tiered_grouped_m_rows_match_committed_witness_*` coverage gap

**Severity**: Bloat (false sense of coverage)

**Where**:
- `crates/akita-prover/src/protocol/quadratic_equation.rs:2512-3191` (the two `tiered_grouped_m_rows_match_committed_witness_*` tests)

The tests enumerate D-rows and B-rows but explicitly omit eval/fold rows and A-rows ("comment says A matrix rows are relied on via E2E"). The CRT-capacity fix is exercised by these tests (because `compute_r_split_eq` runs and the tiered A path executes inside `r`), but the assertions do not systematically pin each A-row M identity. A regression on the chunks A path could slip past these unit tests if E2E coverage is skipped.

**Refactor / fix**:
- Extend the unit test to enumerate A-row M identities for `original_a` / `meta_a` / chunk-A offsets, honoring the combined Ajtai story (book §5.4 lines 728-729). Then E2E becomes a sanity check, not the load-bearing one.

### C-14 — `level == 1` debug block in prover flow has matching debug-trace pattern in verifier `verify_one_level`

**Severity**: Bloat

**Where**:
- `crates/akita-verifier/src/protocol/levels.rs` (various `tracing::debug!` calls inside `verify_one_level` / `verify_root_level`)

Less severe than C-1 but in the same vein: many `tracing::debug!` calls were added during the chunks A-row bug investigation and remain. Per `coda_changes.mdc` "Comments should explain stable contracts ... avoid defensive, retrospective, or PR-justifying comments."

**Refactor / fix**:
- Audit the `tracing::debug!` calls added since `b8bf437e` and keep only those that explain stable contracts (timing spans, soundness checkpoints). Delete the chunks-A diagnostic ones.

---

## 5 — Documentation drift

### D-1 — `phase-d-full-handoff.md` says heterogeneous `prepare_m_eval` rejects loudly; the code now handles it

**Where**:
- `specs/phase-d-full-handoff.md:58` ("`prepare_m_eval` rejects loudly when `lp.groups.is_some() && !lp.groups_are_homogeneous()`")
- `crates/akita-verifier/src/protocol/ring_switch.rs` (`eval_split_at_point_grouped` and friends)

The handoff was written before Slice G prep. Code now routes through grouped helpers (`eval_split_at_point_grouped`, `setup_weight_table_at_point_grouped`) for the heterogeneous case — and in fact falls back to materialization for non-homogeneous *digit depths* (S-4). The handoff's claim is no longer accurate.

### D-2 — `phase-d-full-handoff.md` §1 status block says "Last touched 2026-05-13"; HEAD is 2026-05-18

**Where**:
- `specs/phase-d-full-handoff.md:5`

Five days of work (Slices F.1 → G prep → H staging → CRT fix) have happened since the handoff. The "Resume checklist for the next session" (§4) and "Recommended next session focus" (§5) are both out of date and contradict `specs/tiered-d-row-m-relation-bug-handoff.md`.

**Refactor / fix**:
- Either update `phase-d-full-handoff.md` to the current HEAD or supersede it with `specs/tiered-d-row-m-relation-bug-handoff.md` + a forward-looking note.

### D-3 — `specs/security_analysis.md` predates Slice G's potential new SIS cells

**Where**:
- `specs/security_analysis.md` (full file)

The doc analyzes the post-cutover production presets but does not cover Slice G's meta-tier `(D_meta, B_meta, A_meta)` SIS roles, because cascade is not active in production. When Slice F.4/G land, the security re-audit (Slice I) must regenerate the analysis. The current doc is correct for the current presets, but a reader could mistake it for "post-fourth-root security baseline".

### D-4 — Code comments describing rejection paths that no longer exist

See **C-11**. Cross-cutting documentation drift.

### D-5 — Spec proliferation in `specs/`

**Where**:
- `specs/phase-d-full-design.md` (1042 lines)
- `specs/phase-d-full-handoff.md` (314 lines)
- `specs/tiered-d-row-m-relation-bug-handoff.md` (214 lines)
- `specs/fourth-root-verifier-optimization-roadmap.md` (806 lines)
- `specs/fourth-root-verifier-optimizations.md` (508 lines)
- `specs/fourth-root-verifier-implementation-audit.md` (183 lines, itself an audit)
- `specs/recursive-s-opening-plan.md` (607 lines)
- `specs/tensor-everywhere-implementation-plan.md` (327 lines)
- `specs/security_analysis.md` (304 lines)
- `specs/tensor-exact-aggregate-evaluator.md` (758 lines)
- `specs/tensor-stage1-parameter-search.md` (108 lines)
- `specs/phase-d-full-design.md` (also includes a "v1 retraction" §10)
- `specs/extension-field-opening-batching.md` (573 lines)
- ~6 000 LOC of `specs/*.md` total

Most of these are implementation logs, not specifications. Reviewers cannot tell which is current. `coda_changes.mdc`: *"prefer source comments that still make sense after the PR discussion is forgotten"* — the same applies to specs.

**Refactor / fix**:
- Consolidate into a single `specs/fourth-root-verifier.md` describing the agreed protocol shape + a status checklist. Move implementation logs and bench tables to `specs/archive/` or delete (git history retains them).

### D-6 — `audit.md` references commit `b8bf437e` as the comparison baseline

This file (which you are reading) supersedes the earlier audit. The earlier audit was correct for the snapshot it analyzed but it claimed `b8bf437e` was the HEAD; HEAD is now `cb36143a`. The findings inheritance is documented in §0 of this file.

---

## 6 — Recommended refactor order

If the project wants to ship the 4th-root verifier with minimal review surface:

**Track A — Clean up surgically before continuing the implementation**

1. **PR-cleanup-1 (low risk, high signal-to-noise)**: address C-1 (delete the 270-line `level == 1` block), C-3 (delete verifier `tiered_s_cache`), C-7 (`#[cfg(test)]`-gate debug helpers), C-9 (delete speculative `RecursiveWitnessAsPoly`), C-10 (fix or delete the misleading `per_handle_lp` test), C-11 (rewrite stale rejection comments), C-12 (gate `probe_*`), C-14 (prune leftover diagnostic `debug!`s).
2. **PR-cleanup-2**: C-4, C-5, C-6 (the sibling/duplicate-helper cluster).
3. **PR-cleanup-3**: D-1, D-2, D-4, D-5 (documentation reconciliation).

After cleanup, the diff vs `main` is materially smaller and the remaining changes are the load-bearing protocol modifications.

**Track B — Land the defenses-in-depth that should ship regardless of F.4/F.5 timing**

4. **PR-soundness-fixes**: S-1 (proof-shape ↔ schedule assertion), S-5 (`routes_recursively` ↔ next-level S-claim assertion), S-6 (cache key extension), S-7 (env var removal/gating), S-8 (proof-size accounting), S-9 (`verify_root_direct_commitments_with_params` group fix), S-10 (empty-groups validation), S-11 (return Result instead of panic), S-12 (production-preset consistency check).
5. **PR-crt-bound** (S-2): generalize the chunks-A dispatch to a numeric CRT-overflow predicate and add the fail-loud invariant.

**Track C — Continue Phase D-full toward F.5**

6. **PR-structured-setup-eval-grouped** (S-4 / B-2): implement `eval_setup_weight_at_point_grouped` as a structured sum, not a materialized fallback.
7. **PR-slice-F.3** (B-4): `AkitaPolyOps` impl for `RecursiveWitnessAsPoly`.
8. **PR-slice-F.4+F.5** (B-1, B-3): wire cascade activation in production callers; fix the meta-tier tamper test (B-3); E2E milestone at NV ≥ 12.
9. **PR-slice-G**: tiered `k = 64` production tests passing without OOM (B-5).
10. **PR-slice-H+I**: SIS table regen + post-cascade security re-audit; update `specs/security_analysis.md` with the post-cascade numbers (D-3).

If the project decides **not** to pursue the 4th-root verifier optimization in this codebase (because B-1 + B-2 are real and large), the alternative is:

- **PR-prover-perf-only**: keep the genuinely good prover changes (multi-chunk Ajtai tiling, i64 centered fold accumulators, tensor stage-1 plumbing, exact aggregate evaluator) and discard everything cascade/CR-related. Production presets stay on `use_setup_claim_reduction = false` with tensor stage-1 active. The branch then delivers a ~2× prover speedup at NV=25 with no verifier regression and a clean security story.

---

## 7 — Verification quality / what was not checked

**Checks performed in this audit**:
- ✅ Read book `5_fourth_root_verifier.tex` §5.1-§5.5 in full.
- ✅ Read `specs/phase-d-full-design.md`, `specs/phase-d-full-handoff.md`, `specs/tiered-d-row-m-relation-bug-handoff.md`, `specs/security_analysis.md`, the prior `audit.md`.
- ✅ Examined `git diff d7dd31ed..HEAD` selectively for all critical files: `flow.rs`, `quadratic_equation.rs`, `ring_switch.rs` (prover + verifier), `levels.rs`, `setup_claim_reduction.rs` (prover + verifier), `params.rs`, `flat_matrix.rs`, `proof_size.rs`, `schedule.rs`, `tiered_setup.rs`, `recursive_opening_claim.rs`, `setup.rs`, `commitment.rs`, `tiered_setup.rs`, `eq_weighted_table.rs`, `proof_optimized.rs`, `schedule_params.rs`, `lib.rs` (planner/config), and all new tests.
- ✅ Cross-referenced via 5 parallel sub-agents (multi-group + per-handle LP, tiered slice G/H, verifier hot path + claim reduction, prover protocol + recursive S routing, security baseline + planner).
- ✅ Verified critical findings by direct read of file:line locations.
- ✅ Cited the book passage each cryptographic claim depends on.

**Checks not performed**:
- ❌ `cargo clippy --all -- -D warnings` (the user did not request a build; the audit is read-only per instructions).
- ❌ `cargo test --workspace` (the handoff doc claims tests pass at the noted parameters; not independently re-run).
- ❌ Re-running the lattice-estimator to independently confirm the 128-bit MSIS margins in `specs/security_analysis.md` §4.
- ❌ Profiling the `tiered_production_prove_verify` SIGKILL to confirm the root cause (B-5).
- ❌ Benchmark replay (the verifier-vs-`main` regression numbers cited in the prior audit were not re-run here; behavior likely the same since cascade is still off in production).

**Confidence**:
- Code review confidence: **High** for slice landings (D, E, F.1, G prep, H staging, the CRT fix) and the soundness implications of dispatch / cleartext-check / cache wiring.
- Cryptographic soundness confidence: **High** for "no break today, because the dangerous paths are unreachable in production". **Medium** for "the cascade-on path will be sound once F.5 lands" — depends on S-1, S-5, B-3, B-4 all being addressed before flipping `use_setup_claim_reduction = true` on any production preset.
- Asymptotic-win confidence: **High** that the verifier is not yet at `O(N'^{1/4})` per the book promise — B-1, B-2, S-4 are still open.
- Cleanup confidence: **High** that addressing C-1 through C-14 would shrink the diff by ~2 000 LOC without any behavior change.

---

## Appendix A — Files with substantial code-bloat targets

| File | LOC change vs main | Cleanup target | Action |
|------|-------------------:|----------------|--------|
| `crates/akita-prover/src/protocol/flow.rs` | +1 319 | C-1 (270 LOC), C-2 (~80 LOC), C-11 (comments) | Delete level==1 block, dedupe tiered material derivation, rewrite docs |
| `crates/akita-verifier/src/protocol/ring_switch.rs` | net big rewrite | C-4 (180 LOC), C-7 (40 LOC) | Collapse `eval_algebraic_at_point` → wrapper; gate debug helpers |
| `crates/akita-sumcheck/src/eq_weighted_table.rs` | +311 (new) | C-5 (~200 LOC) | Move 3-of-4 primitives under `#[cfg(test)]` |
| `crates/akita-types/src/proof/setup.rs` | new | C-3 (~120 LOC) | Delete verifier-side `tiered_s_cache` |
| `crates/akita-pcs/tests/per_handle_lp.rs` | new | C-10 | Fix test name and assertions; remove `unreachable!` |
| `crates/akita-pcs/tests/tiered_setup_e2e.rs` | new | C-12, B-3 | Gate probe diagnostics; fix tamper test |
| `crates/akita-planner/src/schedule_params.rs` | +641 | C-8 (~150 LOC), S-7 | Delete hybrid DP; remove env var |
| `crates/akita-prover/src/protocol/setup_claim_reduction.rs` | +21 | C-6 | Replace materialized helper or relabel honestly |
| `crates/akita-prover/src/backend/recursive_witness.rs` | +47 | C-9 (~25 LOC) | Delete `RecursiveWitnessAsPoly` until F.3 |
| `specs/*.md` | +6 000 across 10+ files | D-5 | Consolidate to one current spec, archive logs |

## Appendix B — Open production-preset misconfiguration vectors

Defense-in-depth: today the protocol clears 128-bit security because every production preset has `use_setup_claim_reduction = false`. The following one-line config flips would, today, produce an unsound or under-secure protocol:

1. Setting `Cfg::commitment_config().use_setup_claim_reduction = true` on any fp128 preset without first landing F.5 (S-1, S-5, B-1).
2. Setting `planner_setup_polynomial_size > 0` and `planner_setup_shrink_factor > 1` on a preset whose `LevelParams` are loaded from a generated table (the generated tables override `use_setup_claim_reduction = false` per `schedule.rs:273-342`) — cascade penalty would be priced into the schedule but the actual proof shape would not change, creating a planner/proof inconsistency.
3. Writing a new preset that enables tensor stage-1 at `D = 32` (currently only `D ≥ 64` has a production `ω` ≫ 3) — Lemma 5.2 (`lem:tensor-norm`) requires `ω ≫ 3`; `BoundedL1Ball{D=32}` has `ω` insufficient for the 4ω penalty to clear 128-bit MSIS (per `specs/security_analysis.md`).
4. Enabling `HACHI_PLANNER_S1_WEIGHT` divergently between prover and verifier (S-7) in a deployment where one side uses planner-derived schedules.

Each of these should be either fail-loud at config-construction time, or made unreachable by removing the corresponding code path.
