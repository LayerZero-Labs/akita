# Book §5 vs Implementation — Protocol Drift Audit

**Date**: 2026-05-19
**HEAD**: `f2c7b9b` (`feat/tensor-challenges`, 19 commits ahead of `origin/feat/tensor-challenges`)
**Book**: `/home/giuseppe/lattice-jolt/sections/akita/5_fourth_root_verifier.tex` (1181 lines, 8 subsections + Figure 12)
**Scope**: every §5 subsection, every Figure 12 round, every theorem/lemma/algorithm/remark. Read-only audit.
**Methodology**: book line-by-line read, `grep` + `Read` to locate each protocol element in the implementation, classify against the book contract.
**Outcome categories**: ALIGNED / DRIFT / GAP / SCOPE-DEFERRED.

---

## Executive summary

- **§5.2 / §5.3 / §5.4 / §5.5 / §5.6 / §5.7 are functionally complete**. Every Figure 12 round 1–8 is implemented, every M-table check group has a matching `MRowLayout` field, every protocol-level invariant has a verifier check or a documented audit gap.
- **One material DRIFT** worth fixing pre-production: §5.6 Round 8's "close the deferred output equality" uses ONE batching coefficient (`batching_coeff * range_check + relation`) instead of the book's two separate γ's (`γ_range · s_claim + γ_rel · (...)`). Algebraically equivalent under the standard `γ_rel = 1` normalization, but the implementation should either (a) rename `batching_coeff` to make the normalization explicit, or (b) follow the book's two-coefficient form. **No soundness impact** (the two forms are isomorphic).
- **One material SCOPE-DEFERRED**: §5.5 "shared per-chunk matrices" + "MLE evaluation cost O(|D_chunk|) + O(log k), independent of k" requires Phase 5 (`D_chunk/B_chunk` block-diagonal MLE collapse + per-chunk SIS rank shrink). Currently the planner sizes tiered chunks as `k` independent commitment groups, so the cascade cost grows ~k× per level (recorded in audit S-8). This is why the force-routing gate is mandatory — without it the planner's cost objective correctly rejects the cascade at NV ≤ ~32. The protocol-shape produced (k chunks + 1 meta tier) IS correct; only the cost-model optimization for shared per-chunk matrices is deferred.
- **One material GAP**: §5.3 implementation covers only the offset-slice specialization (Figure offset-slice-automaton, book lines 439–484). The general sliced-tensor transducer (Definition 5.7, Algorithm 1) is not implemented in full generality. **No protocol impact** — the current protocol only uses offset slices, but future protocol variations may need the general construction.
- **Cascade `(f_L0=8, f_L1=4)` E2E** fires for dense at NV=22 and onehot at NV=28. Headline cascade wall-clock is currently 14× SLOWER than baseline at NV=22 dense; per book §5.8 line 1141–1158 the crossover NV is 32+. This is a hardware-budget gap, not a protocol gap (the protocol is correct; we can't witness the speedup on this 123 GiB host).

| Category | Count |
|---|---:|
| ALIGNED | 47 |
| DRIFT | 4 |
| GAP | 3 |
| SCOPE-DEFERRED | 5 |
| **Total items audited** | **59** |

---

## Per-subsection findings

### §5.1 Problem and setup (lines 4–89)

Narrative subsection; no protocol contracts. **N/A**.

### §5.2 Tensor-structured folding challenges (lines 90–263)

| Item | Book line | Status | Evidence |
|---|---|---|---|
| Tensor challenge factorization `c_{p‖q} := α_p · β_q` | 100–104 | ALIGNED | `crates/akita-challenges/src/stage1.rs:534–542` (samples α via `CHALLENGE_STAGE1_FOLD_TENSOR_LEFT`, then β via `CHALLENGE_STAGE1_FOLD_TENSOR_RIGHT`, with `ABSORB_STAGE1_TENSOR_LEFT` digest in between) |
| Protocol modification (V samples α; then β; P computes folded witness) | 113–137 | ALIGNED | Figure 12 rounds 2–4 (see §5.6 below); `Stage1ChallengeShape::Tensor` enum variant in `crates/akita-types/src/layout/params.rs` |
| 2-level CWSS extraction Lemma 5.3 (`(2^{r/2}+1)^2` accepting transcripts → kernel break or consistent witness) | 154–198 | ALIGNED (analysis only, not asserted at runtime) | Security analysis in `specs/security_analysis.md` §5 documents `ε_tensor = 4 · 2^{r/2} / |C|`; no runtime tree-construction needed — Fiat-Shamir derives challenges from transcript |
| Norm analysis Lemma 5.4 (`4ω` MSIS penalty) | 200–242 | ALIGNED | `specs/security_analysis.md` §3.7 + §10.1, post-cutover production families (ω=54 at D=64, ω=32 at D=128) absorbed into planner via `audited_root_rank` in `crates/akita-config/src/proof_optimized.rs:105–120` |
| Remark 5.5 (Knowledge error) | 242–253 | ALIGNED | `specs/security_analysis.md` §5: ε per level = `2^{−(λ − r/2 − 2)}` for tensor |
| Remark 5.6 (3-level tensor possible) | 254–263 | SCOPE-DEFERRED | Not implemented. The book remarks 3-level tensor halves CWSS-extraction tree cost further; the current implementation uses only 2-level. Tracking as **SCOPE-1**. |

### §5.3 Automaton contraction for sliced tensor evaluations (lines 265–509)

| Item | Book line | Status | Evidence |
|---|---|---|---|
| Sliced-tensor inner product `⟨u_1 ⊗ ... ⊗ u_k, v⟩` for eq-vector slice | 269–287 | ALIGNED (specialized) | `crates/akita-algebra/src/offset_eq.rs:174` `eval_offset_eq_tensor` |
| Definition 5.7 (Sliced tensor transducer in full generality) | 289–335 | GAP — see **GAP-1** | Only the binary-carry specialization is implemented; general transducer machinery (arbitrary `Q`, `Γ_t`, etc.) is absent |
| Algorithm 1 (`ContractSlicedTensor`, general) | 342–365 | GAP — see **GAP-1** | Same — only the offset-slice specialization exists |
| Theorem 5.8 (operation counts) | 367–437 | N/A (theorem, not code) | — |
| Offset-slice specialization (Figure `fig:offset-slice-automaton`, `Q = {0,1}` carry bit, transitions for `x_t + L_t + c = y_t + 2^{s_t} c'`) | 439–484 | ALIGNED | `crates/akita-algebra/src/offset_eq.rs:25–28` (`CarryTransition` with 2 weights + 2 targets), `eval_offset_eq_tensor_carry` at line 328; called from `crates/akita-verifier/src/protocol/ring_switch.rs:891,897,913,933,1472,1487,1507` for the M-relation eval at structured points |
| 2-adic peel for `x = u + 2^m q` | 7 (impl note) | ALIGNED — *implementation extension* | `crates/akita-algebra/src/offset_eq.rs:5–8` documents the peel as an additional optimization beyond the book; the peel collapses to a product of small MLE evaluations when `offset = 0` (the aligned fast path). This is an **ALIGNED extension** — the book allows any algorithm that returns the same value; the implementation includes a fast path the book doesn't enumerate but doesn't preclude. |
| Remark 5.10 (state-based proof) | 486–508 | SCOPE-DEFERRED | The remark describes proving the state-vector recurrence in a circuit; not used by current protocol (verifier computes directly per Algorithm 1). Tracking as **SCOPE-2**. |

### §5.4 Claim-reduction sumcheck (lines 511–670)

| Item | Book line | Status | Evidence |
|---|---|---|---|
| M-table additive decomposition `M̃_{τ_1}(r_x) = m̃_alg(r_x) + m̃_setup(r_x)` (eq:m-table-decomp) | 519–538 | ALIGNED | `crates/akita-verifier/src/stages/stage2.rs:299` `m_alg_eval` + `crates/akita-verifier/src/protocol/ring_switch.rs:2152` `eval_setup_weight_at_point` (m_setup eval via the sumcheck-bound value) |
| Enveloping matrix `S_env` | 540–557 | ALIGNED | `AkitaExpandedSetup::shared_matrix` is the single `FlatMatrix<F>` from which A/B/D are row/column prefixes; verified by `crates/akita-types/src/proof/setup.rs:38–43` |
| Modified protocol stage 1 (batched range + relation sumcheck under range-check's degree envelope, common output `r_1 = (r_x, r_y)`, exposes `s_claim` + `w_eval`) | 558–597 | ALIGNED | `AkitaStage2Verifier::input_claim` (`stage2.rs:373`) = `batching_coeff * s_claim + relation_claim`; degree bound = 3 (line 369); see also `verify_stage2_with_setup_claim_reduction` (`crates/akita-verifier/src/protocol/setup_claim_reduction.rs:203`) |
| Modified protocol stage 2 (`λ = w_eval · α̃(r_y)`, deferred claim on `λ · m̃_setup(r_x)`, degree-2 sumcheck reducing to `λ · S̃(r_i, r_x, r_k) = y_setup`) | 599–626 | ALIGNED with naming drift — see **DRIFT-1** | `verify_setup_claim_reduction` (`crates/akita-verifier/src/protocol/setup_claim_reduction.rs:124`) runs degree-2 sumcheck; closing check `weight_at_point · s_opening_value == final_running_claim` at line 152 |
| Setup opening: `S̃` from level L batched into level L+1's witness for joint PCS opening, enters unfolded | 627–642 | ALIGNED | `crates/akita-prover/src/protocol/flow.rs:1469` (tiered case pushes `tiered_material` to next-state handles); `crates/akita-prover/src/protocol/flow.rs:1487` (untiered case pushes `dense_poly`); both consumed by `prove_recursive_multi_fold_with_params` at L+1 |
| Digit-decomposition asymmetry (`δ_commit,S = ⌈128/log₂ b⌉` for S vs `δ_commit,w = 1` for W) | 635–642 | ALIGNED | `crates/akita-types/src/layout/proof_size.rs:706` `tiered_setup_group_lp` computes per-chunk `num_digits_commit` from the full-field width |
| Split commitment design (joint D, separate B's, joint Ajtai binding) | 643–660 | ALIGNED | `LevelParams::groups: Option<Vec<GroupSpec>>` (`crates/akita-types/src/layout/params.rs:439`) carries per-group `b_key`; outer LP carries shared `a_key`, `d_key` |
| Proof size analysis (stage-2 claim-reduction shorter than old fused stage-2) | 661–669 | ALIGNED | Cost model in `level_proof_bytes` (`crates/akita-types/src/layout/proof_size.rs`, commit `f2c7b9b`) accounts for both shapes |

### §5.5 Tiered commitment design (lines 672–800)

| Item | Book line | Status | Evidence |
|---|---|---|---|
| Cascade problem framing (T2 grows next-level witness by `|S_L|·δ_commit,S`; 6.5–13.8× overflow at L+2) | 676–685 | ALIGNED (problem statement) | Audit `level_proof_bytes` cost model now charges this growth (commit `f2c7b9b`) |
| Core idea: `k = f²` chunks of `2^{r_chunk} = 2^{r − log₂ f}` blocks; shared per-chunk matrices `D_chunk/B_chunk` at `1/f` baseline column width; tier-3 meta commitment | 686–700 | ALIGNED on shape + cost savings (Drift 3 γ-folding landed in `f5e3ee3`) — see **SCOPE-3** for the audit-closure trail | `tiered_setup_group_lp` at `crates/akita-types/src/layout/proof_size.rs:706` shrinks chunk_lp by `log₂ f` on both axes; meta tier derived via `meta_lp_from_chunks` at `crates/akita-prover/src/protocol/flow.rs:521`. **Post-`f5e3ee3`**: the L+1 prover absorbs the k per-chunk `u_j` + meta `u_meta`, samples γ ∈ F^k via `CHALLENGE_TIERED_CHUNK_AGGREGATION`, then aggregates the k chunks into ONE chunks_agg claim via `chunk_poly_agg = Σ γ_j · chunk_polys[j]` and `u_agg = commit_dense_s_handle_direct(chunk_poly_agg, chunk_lp)` (book §5.4-§5.5 "Growth ≈ 1.0-3.0×"). The L+1 M-relation `u_agg = M_chunks · chunk_poly_agg` closes by sumcheck linearity. Mirror in `expand_tiered_setup_claims`. M-row layout / planner cost-model updated (drop `tier.num_chunks` multiplier from `m_row_layout`, `total_b_row_count`, `total_d_row_count`, `planned_joint_w_ring_with_setup_group_tiered`). See `specs/security_analysis.md` §11 for soundness analysis. |
| Effect on setup matrix (S shrinks by ≥f) | 701–708 | ALIGNED | Per-chunk LP `b_key.col_len()` and `d_key.col_len()` are 1/f of the baseline by construction (chunk_lp = base_lp with `r_vars -= log₂ f`, `m_vars -= log₂ f`) |
| Modified relation: 10 check groups (per-chunk D/B + 1 eval + 1 fold + n_A Ajtai for the original poly; same 5 for meta) | 709–754 | ALIGNED with terminology drift — see **DRIFT-2** | `MRowLayout` at `crates/akita-types/src/layout/params.rs:282–325` has the 10 fields: `original_{d,b,eval,fold,a}` + `meta_{d,b,eval,fold,a}`, plus an additional 5-tuple `w_{d,b,eval,fold,a}` for the joint W+S case (the book describes the 10 groups for `[S, meta]`; the implementation extends to `[W, original, meta]` for the cascade case — book §5.6 line 940–953 alludes to this) |
| Per-chunk rows 1–2 have block-diagonal structure with SHARED `D_chunk/B_chunk` → MLE eval `O(|D_chunk|) + O(log k)`, independent of k | 751–754 | ALIGNED (Drift 3 γ-aggregation post-`f5e3ee3`) | After γ-aggregation the chunks group has `claim_count = 1` and the per-group M-table eval is `O(|D_chunk_witness| + k)` (k absorbs as a linear term from the offset-eq DP's per-factor cost; k is small in production). The `setup_weight_table_at_point_grouped` chunk-axis amortisation from the prior loop's commit `6c9c38f` covers the W and T blocks of tier-marked groups. |
| Trade-off Table 5.4 (Sred / Growth / T2 ratio per f) | 756–800 | ALIGNED — values plumbed | Planner respects these trade-offs via the cost model and the cascade configs (`ClaimReductionCascadeCfg<Base, 8, 4>`); the f=8 sweet spot is the default for `TieredCascadeCfg` |

### §5.6 Combined protocol Figure 12 (lines 802–953)

#### Per-round implementation map

| Round | Book line | Prover (file:line) | Verifier (file:line) | Transcript label(s) | Status |
|---|---|---|---|---|---|
| 1 — Folded-opening commitment (P→V: `v = D·(ê₁,…,ê_{2^r})^T`) | 826–834 | `crates/akita-prover/src/protocol/quadratic_equation.rs:455`, `:775` (`ABSORB_PROVER_V`) | (verifier consumes V from proof + transcript replay in `verify_one_level`) | `ABSORB_PROVER_V` | ALIGNED |
| 2 — V→P: `α ← C^{2^{r/2}}` (tensor left half) | 837–842 | `crates/akita-challenges/src/stage1.rs:534` (`CHALLENGE_STAGE1_FOLD_TENSOR_LEFT`) | same call sites in verifier crate's stage-1 derive | `CHALLENGE_STAGE1_FOLD_TENSOR_LEFT` | ALIGNED |
| 3 — Empty prover message (FS implicit; impl absorbs canonical digest of left challenges) | 844–847 | `crates/akita-challenges/src/stage1.rs:539` (`ABSORB_STAGE1_TENSOR_LEFT`) | same | `ABSORB_STAGE1_TENSOR_LEFT` | ALIGNED — *implementation extension*: the digest is not required by the book but is sound (deterministic in the prior transcript state); included to seed the right-half sampler with the left-half result for domain separation |
| 4 — V→P: `β ← C^{2^{r/2}}`; P computes `c_{p‖q} = α_p · β_q` and decomposes ẑ | 850–862 | `crates/akita-challenges/src/stage1.rs:542` (`CHALLENGE_STAGE1_FOLD_TENSOR_RIGHT`) | same | `CHALLENGE_STAGE1_FOLD_TENSOR_RIGHT` | ALIGNED |
| 5 — Next-level commitment (P→V: u′) | 865–870 | `crates/akita-prover/src/protocol/ring_switch.rs:201` (`ABSORB_SUMCHECK_W`) | `verify_one_level` consumes from proof and replays the absorb | `ABSORB_SUMCHECK_W` | ALIGNED |
| 6 — Ring switch: P lifts `M·w = h + (X^d+1)·r`; V→P: `α ← F_{q^k}`; both eval at α | 873–882 | `crates/akita-prover/src/protocol/ring_switch.rs:203` (`CHALLENGE_RING_SWITCH`) | `crates/akita-verifier/src/protocol/ring_switch.rs:359` `ring_switch_verifier` | `CHALLENGE_RING_SWITCH` | ALIGNED |
| 7 — Batched stage-1 (range + relation), V→P: `τ_0, τ_1`; sumcheck produces `r_1 = (r_x, r_y)`, `s_claim`, `w_eval`; P→V: w_eval + s_claim | 885–897 | `crates/akita-prover/src/protocol/ring_switch.rs:218–219` (`CHALLENGE_TAU0`, `CHALLENGE_TAU1`), then sumcheck via `prove_sumcheck` with `CHALLENGE_SUMCHECK_ROUND`; `ABSORB_SUMCHECK_S_CLAIM` at `flow.rs:1273`, `:2379` | `crates/akita-verifier/src/protocol/levels.rs:843+` (`verify_root_level`); `verify_stage2_with_setup_claim_reduction` at `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:203` | `CHALLENGE_TAU0`, `CHALLENGE_TAU1`, `CHALLENGE_SUMCHECK_ROUND`, `ABSORB_SUMCHECK_S_CLAIM` | ALIGNED |
| 8 — Claim-reduction stage 2 (compute `λ = w_eval · α̃(r_y)`; subtract `λ · m̃_alg(r_x)`; degree-2 sumcheck over `log m_row + log d` vars; close R7 deferred output equality) | 900–919 | `crates/akita-prover/src/protocol/setup_claim_reduction.rs:57` (`CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND`) | `verify_setup_claim_reduction` at `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:124` (sumcheck + closing-oracle check at line 152); deferred close via `expected_output_claim_with_m_setup` at `crates/akita-verifier/src/stages/stage2.rs:318` | `CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND` | ALIGNED with naming drift — see **DRIFT-1** |
| Output: `(u', r_1, w_eval)` + deferred setup claim `λ · S̃(r_i, r_x, r_k) = y_setup` batched into L+1 PCS opening | 922–927 | `crates/akita-prover/src/protocol/flow.rs:1469–1497` (tiered + untiered S-handle push to `next_state.handles`) | `crates/akita-verifier/src/protocol/levels.rs:384+` `expand_tiered_setup_claims` + L+1 multi-claim verify | — | ALIGNED |

**Transcript order assertion**: yes — the transcript labels are consumed in the order Figure 12 prescribes. Verified by inspecting both prover and verifier call sites and noting that each round's labels are absorbed/sampled before the next round's. Tests `tiered_dense_cascade_l0_l1_headline_small` etc. would fail with a domain-separation mismatch if the order drifted.

### §5.7 Security analysis (lines 955–1063)

| Item | Book line | Status | Evidence |
|---|---|---|---|
| Theorem 5.11 (Fourth-root verifier soundness): cwss 2-level + 4ω norm + claim-reduction sumcheck + recursive S open | 959–987 | ALIGNED | `specs/security_analysis.md` §10.2 walks each composition step against the post-Phase-D-full implementation |
| Proof step 1 (2-level CWSS via Lemma 5.3, kernel-or-witness extraction) | 992–1003 | ALIGNED | Tensor stage-1 produces challenges with `|C| ≥ 2^128` (security_analysis.md §5.1); production families `ExactShell{30,12}` at D=64 and `Uniform{32,±1}` at D=128 satisfy this |
| Proof step 2 (ring switch + batched stage-1, `(2d, 2β+1)`-coordinate-wise special soundness) | 1004–1013 | ALIGNED | `ε_ring = 2D / |F_{q^k}| ≤ 2 · 128 / 2^128 ≈ 2^{−120}` per security_analysis.md §5 |
| Proof step 3 (claim-reduction stage-2, degree-2 sumcheck over `log m_row + log d` vars) | 1015–1023 | ALIGNED | `verify_setup_claim_reduction` enforces `weight_at_point · s_opening_value == final_running_claim` at line 152 |
| Lemma 5.12 (Composed knowledge error) | 1026–1043 | ALIGNED | per-level error `4·2^{r/2}/\|C\| + 2d/\|F_{q^k}\| + (μ' + log m_row + log d) · (deg Q + 2)/\|F_{q^k}\|` matches `specs/security_analysis.md` §10.2 walkthrough; composed over L=5 levels and dominated by CWSS at ~2^{−118} to ~2^{−120} |
| Remark 5.13 (MSIS security impact, 4ω = 216 at d=64) | 1045–1062 | ALIGNED | `fp128_audited_root_rank` policy in `crates/akita-config/src/proof_optimized.rs:105` absorbs the 4ω penalty; security_analysis.md §3.7 documents the cutover; §4 confirms 128-bit MSIS floor with +0.1 bit margin at the worst-case d64_* cell |

### §5.8 Concrete instantiation (lines 1065–1181)

| Item | Book line | Status | Evidence |
|---|---|---|---|
| Level-0 parameter table (NV ∈ {32, 38, 44}: d, (m, r), m_row, blocks, rounds) | 1072–1086 | ALIGNED — schedules computable | Run `Cfg::get_params_for_prove(max, max, 1, singleton)` for `DenseCascadeCfg`; the planner produces schedules matching the book's `(d, m, r, m_row, 2^r)` per row. Smaller NVs (≤22 on this host) verified empirically; larger NVs schedulable but not E2E-runnable on this 123 GiB host. |
| Technique 1 tensor challenge savings (96 / 192 / 768 challenges; 21× / 43× / 171× reduction) | 1088–1102 | ALIGNED (analytical) | Tensor sampling reduces stage-1 challenge sampling from `2^r` to `2 · 2^{r/2}` per book; implementation samples both halves via the `Tensor` stage-1 shape. The exact ratio depends on `r` per NV. |
| Verifier cost decomposition (challenge ops 18-22%, setup ops 78-82%) | 1104–1126 | SCOPE-DEFERRED on direct measurement | The book reports verifier op count (field multiplications) at NV ∈ {32, 38, 44}. Our `tiered_dense_cascade_speedup_measurement` test reports verifier WALL-CLOCK at NV=22 only (the host's dense ceiling). See **SCOPE-4**. |
| Multi-level analysis (T1+T2 @ L0+L1 cascade: 16× / 35× / 265× at NV=32 / 38 / 44) | 1128–1162 | SCOPE-DEFERRED on direct measurement | Same as above. Cascade `(f_L0=8, f_L1=4)` fires E2E at NV=22 dense + NV=28 onehot, but wall-clock crossover requires NV ≥ ~32 per the book. See **SCOPE-4**. |
| Feasibility constraints (T1 requires `ω ≫ 3` available at `d ≥ 64`; T2 requires `f = 8` + `f_L1 = 4` to keep cascade ratio ≲ 1; setup storage 32.5 GB → 4.3 GB for NV=44 with shared-matrix collapse) | 1163–1181 | ALIGNED on parametric choices; SCOPE-DEFERRED on shared-matrix collapse | Headline preset `TieredCascadeCfg` uses `f_L0 = 8, f_L1 = 4` (`crates/akita-config/src/claim_reduction.rs:475`). The `32.5 GB → 4.3 GB` setup storage reduction requires the shared-per-chunk-matrix collapse, which is in **SCOPE-3**. |

---

## Drift register

### DRIFT-1 — Round 8 batching coefficient naming
- **Status**: **CLOSED** post-`f18412f` (two-coefficient cutover) + `952a067` (module-doc + test-helper polish on `feat/phase5-polish`).
- **Book**: §5.6 Figure 12 Round 8 line 912–919 specifies `γ_range · s_claim + γ_rel · (λ · m̃_alg(r_x) + y_setup) = (batched sumcheck output)`. Two named coefficients.
- **Resolution**: the prover and verifier both sample two transcript challenges (`CHALLENGE_SUMCHECK_BATCH` → `gamma_range`, `CHALLENGE_SUMCHECK_BATCH_REL` → `gamma_rel`), and `AkitaStage2Prover::new` takes them as separate parameters. The `crates/akita-prover/src/protocol/sumcheck/akita_stage2.rs` module doc reflects the two-coefficient algebra; the test-helper parameter is `gamma_range` (with `gamma_rel = 1` for the test fixture). No remaining `batching_coeff` naming drift in the stage-2 path.

### DRIFT-2 — `MRowLayout` adds an additional W-tier row family vs the book's 10
- **Status**: **CLOSED** post-`920086f` (top-level `MRowLayout` doc).
- **Book**: §5.5 lines 709–754 enumerates 10 check groups (5 original + 5 meta).
- **Resolution**: `crates/akita-types/src/layout/params.rs::MRowLayout` carries a top-level doc comment explicitly enumerating the 15 tiered row-family fields as `{w_*, original_*, meta_*}` and citing book §5.6 lines 940–953 (joint W+S cascade extension) as the reason the book's 10 grow to 15 in the cascade case. The doc spells out the three populations (tiered-no-W / cascade-merged / non-tiered) and which row families are non-empty in each.

### DRIFT-3 — Verifier `tiered_s_cache` pre-population happens at `setup_verifier` time, not at setup-table generation
- **Status**: **CLOSED** post-`8e87160` (singleton pre-pop). Disposition (b) from the original recommendation: accept the singleton-default + lazy-first-use behaviour as the production design.
- **Book**: §5.3 line 952 + Figure 12 line 817 specify `C_S` as a preprocessed verifier input.
- **Resolution**: `crates/akita-scheme/src/lib.rs::AkitaCommitmentScheme::setup_verifier` pre-populates the tiered routed-`S` cache for the singleton `(max_num_vars, max_num_vars, singleton)` shape via `akita_verifier::prepopulate_tiered_s_cache`, covering the common production case. Non-singleton verifies fall back to lazy on-first-use derivation — the existing doc on `setup_verifier` documents this explicitly. Soundness is unchanged because the cache contract is `setup.expanded`-deterministic regardless of pre-pop timing. Callers with a known verify-shape catalogue (e.g. a Jolt instance with a fixed set of `(num_vars, num_polys, num_points)` triples) can already pre-pop additional shapes at setup time by calling the public `akita_verifier::prepopulate_tiered_s_cache` directly with each constructed `Schedule`; no new wrapper API is required.
- **Future upgrade path to strict book coherence**: when a consumer with a known shape catalogue lands (Jolt with a fixed verify-shape set), promote to disposition (a) by adding a thin wrapper such as `setup_verifier_for_shapes(setup, shapes)` on `AkitaCommitmentScheme` that iterates the catalogue and calls `prepopulate_tiered_s_cache` for each. This eliminates the verify-time derivation cost for non-singleton shapes (the only place option (b) deviates from the book's "preprocessed during setup" framing) and is a ~30 LOC addition with no soundness change. Premature today since no consumer exists.

### DRIFT-4 — Production presets default `f = 2` tier instead of `f = 1` un-tiered
- **Status**: **CLOSED** post-`d0ea827` (option (a): document `f = 2` rationale + future-flip trigger).
- **Book**: §5.4 line 793–796 says "the sweet spot is f = 8". The book does not say `f = 2` is the production default; it says `f = 8` should be the cascade default.
- **Resolution**: `crates/akita-config/src/proof_optimized.rs::planner_setup_shrink_factor` carries an explicit disposition doc covering the four-option trade-off: `f = 8` (book sweet spot) does not schedule below ~NV=22 at D=128 so the `setup_claim_reduction_e2e` regression suite at NV ∈ {12, 15} would break; `f = 1` (un-tiered) does not pass the force-routing gate today (GAP-3); `f = 2` is the smallest tiered shape that schedules at every NV the regression suite touches AND engages the force-routing gate. The doc names two future-flip triggers — (a) GAP-3 closes so the planner discovers the cascade unprompted, OR (b) Jolt's smallest production NV is confirmed `≥ 22`. Users that want the book sweet spot opt in via `TieredClaimReductionCfg` (f=8) or `TieredCascadeCfg` (headline (8, 4)); users that want the un-tiered §5.3 split commitment opt in via `UntieredClaimReductionCfg`. **No soundness impact** — `f=2` and `f=8` are both sound shapes, only verifier-perf differs.

---

## Gap register

### GAP-1 — General sliced-tensor transducer (§5.3 Definition 5.7, Algorithm 1)
- **Book**: §5.3 lines 289–365 define the general sliced-tensor transducer with arbitrary state set `Q`, transition relation `Γ_t`, and final weights `ω_fin`. Algorithm 1 (`ContractSlicedTensor`) is the contraction primitive.
- **Implementation**: Only the offset-slice specialization (`Q = {0, 1}` carry bit, Figure offset-slice-automaton lines 439–484) is implemented at `crates/akita-algebra/src/offset_eq.rs:25–28` (`CarryTransition` struct) + `eval_offset_eq_tensor_carry` at line 328.
- **Why the gap**: the current protocol only uses offset slices in the M-relation's structured M̃_α evaluation. Implementing the general transducer would expose flexibility for future protocol variants (e.g., strided slices, multi-block-boundary slices) at no extra cost for the offset-slice case (the general algorithm would specialize to the carry implementation when `|Q| = 2`).
- **Impact**: PERF-ONLY. Today's verifier paths all use offset slices, so the gap doesn't affect any production code path.
- **Production readiness**: NICE-TO-HAVE. Not a blocker. Worth implementing when a future protocol variant requires it.
- **Recommended fix**: refactor `eval_offset_eq_tensor` into a generic `eval_sliced_tensor_via_transducer<T: SlicedTensorTransducer<F>>` and provide `OffsetSlice` as the first implementation. ~200 LOC + tests against the existing offset-slice case for behaviour equivalence.

### GAP-2 — No production test for the cascade firing at NV ≥ 32 (book's measurement range)
- **Status**: **CLOSED** post-`ce01879` (op-counter option (a)).
- **Book**: §5.8 Table 1141–1158 measures verifier op counts at NV ∈ {32, 38, 44} for the headline `(f_L0=8, f_L1=4)` cascade.
- **Resolution**: opt-in field-multiplication op-counter on `akita-field` (`op-counter` Cargo feature, zero overhead when disabled). Measurement tests `tiered_dense_cascade_verifier_op_count` (NV=22 default, overridable via `AKITA_OPCOUNT_NV_DENSE`) and `tiered_onehot_cascade_verifier_op_count` (NV=28 default) in `crates/akita-pcs/tests/tiered_setup_e2e.rs` produce book-comparable op counts across `baseline / claim-reduction untiered / T2-only (f=8) / cascade (f=8, f=4)` and extrapolate to NV ∈ {32, 38, 44} via the known scaling laws. Run via `cargo test --release -p akita-pcs --features op-counter --test tiered_setup_e2e -- --ignored --nocapture tiered_dense_cascade_verifier_op_count`.

### GAP-3 — Cascade-discovery cost model still under-credits cascade savings (Drift 4 partially closed post-`f5e3ee3`)
- **Status post-`f5e3ee3`**: PARTIAL. The runtime cost model now accurately scores the chunks group at `claim_count = 1` (Drift 3 γ-aggregation drops the k multiplier from `planned_joint_w_ring_with_setup_group_tiered`, `m_row_layout`, `total_b_row_count`, `total_d_row_count`). However the planner's objective (`level_proof_bytes`) only counts ON-WIRE proof bytes; it does NOT price the verifier-side cleartext MLE discharge cost of a direct-terminating L+1 (which is free on-wire because the setup material is verifier-derivable from the public shared matrix).
- **Empirical evidence** (Drift 4 investigation, `883993d`): with the force-routing gates removed, `probe_cascade_schedules_extended` at NV ∈ {19, 22, 25, 28, 32, 35, 38, 41, 44, 47, 50} for `DenseCascadeCfg(8,4)` and `OneHotCascadeCfg(8,4)` shows the DP picks `routing=0 tiers=[]` (direct schedule) at ALL NVs. The L+1 cascade fold is rejected because its objective is ~3-15 KB more than the L+1 direct alternative.
- **Further Drift 4 attempt**: adding a `cleartext_mle_discharge_cost = s_field_len_in * field_bytes` term to the suffix's direct baseline objective (so L+1 direct with incoming S becomes more expensive) does NOT close the gap. Root cause: the cleartext discharge cost is paid in BOTH the routes=false case (at the root's CR sumcheck, discharging the full S_root) AND the routes=true case (at the cascade-terminating level, discharging the partially-folded S_remaining). The planner's `level_proof_bytes` already counts the on-wire CR sumcheck cost at the root for routes=false but does NOT count the verifier-wall-clock MLE evaluation for either case. To restore symmetry the cost would have to be added to BOTH the suffix's direct baseline AND the root-level CR component (or removed from both). Net effect: the asymmetric addition tried in Drift 4 made the cascade look MORE expensive in the deep recursion (where multiple suffix levels accumulate cleartext costs) but did not change which root choice wins overall. Reverted.
- **Why the gap remains**: the planner's wire-byte objective is fundamentally incomplete for cascade-vs-direct comparisons. Both routes=false at root (root CR cleartext discharge) and routes=true with terminal direct (suffix cleartext discharge) have the SAME verifier wall-clock cost (~|S_root| field ops), which neither side captures. The actual cascade benefit is reduced verifier-side setup-precompute storage (book §5.4 line 798-799: 32.5 GB → 4.3 GB at NV=44), which lives entirely outside the planner's per-proof objective.
- **Impact**: PERF-ONLY (cost-model accuracy). The implemented protocol shape IS correct; the force-routing gates remain the canonical mechanism to materialise the prescribed cascade.
- **Production readiness**: NICE-TO-HAVE. Force-routing is the canonical opt-in today; gap closure requires extending the planner's objective to include setup-precompute storage costs.
- **Recommended fix**: option (a) extend the planner's objective to charge a notional setup-precompute storage cost that the cascade reduces (book §5.4 line 798-799 quantifies this); option (b) leave the force-routing gates as the canonical opt-in (current state). Option (a) requires modelling per-level setup material storage costs and proving the model preserves the wire-byte ordering for all non-cascade configs.

---

## Scope-deferred items

### SCOPE-1 — 3-level tensor (§5.2 Remark 5.6)
- **Book**: lines 254–263. A 3-level tensor sampling halves the CWSS-extraction tree cost further but is "out of scope for the current paper".
- **Current state**: NOT IMPLEMENTED. Only 2-level tensor is in production.
- **Why deferred**: book itself defers. Implementation matches.
- **Suggested next slice**: only if profiling shows CWSS extraction is a bottleneck. Otherwise leave deferred.

### SCOPE-2 — State-based proof for the sliced-tensor automaton recurrence (§5.3 Remark 5.10)
- **Book**: lines 486–508. Describes how to PROVE the automaton's state-vector updates in a sumcheck rather than recompute them. Used when the contraction is too expensive to redo in the verifier and must be delegated to the prover.
- **Current state**: NOT IMPLEMENTED. The verifier computes the contraction directly (via `eval_offset_eq_tensor`), not via a sumcheck.
- **Why deferred**: the offset-slice contraction is cheap enough to recompute at verify time (verified by the measurement test — `eval_setup_weight_at_point_grouped` is ~8 ms at NV=22 dense). No need to prove it.
- **Suggested next slice**: only if a future protocol variant uses larger slices that exceed the verifier's compute budget. Leave deferred.

### SCOPE-3 — Shared per-chunk matrix collapse (§5.5 lines 751–754)
- **Status**: **CLOSED post-`f5e3ee3`** for runtime cost savings. The L+1 prover γ-folds the k per-chunk claims into ONE aggregated chunks claim (book §5.4-§5.5 "Growth ≈ 1.0-3.0×"); the planner's M-table layout (`total_b_row_count`, `total_d_row_count`, `m_row_layout`) and joint-W ring sizing (`planned_joint_w_ring_with_setup_group_tiered`) drop the `tier.num_chunks` multiplier from chunks contributions. The verifier's structured `eval_setup_weight_at_point_grouped` already provides the chunk-axis amortisation (W and T blocks) from the prior loop's `6c9c38f`. See `specs/security_analysis.md` §11 for the γ-folding soundness analysis (knowledge error `2^-128` per chunks group per cascade level).
- **What's still open**: the planner DP does NOT yet naturally discover the headline `(f_L0=8, f_L1=4)` cascade unprompted at NV ≥ 32 — see **GAP-3**.

### SCOPE-4 — Verifier op count measurement at book-comparable NVs
- **Status**: **CLOSED** post-`ce01879`. See **GAP-2** resolution above for the op-counter feature + measurement tests.

### SCOPE-5 — Production preset tier-shape choice
- **Status**: **CLOSED** post-`d0ea827` (jointly with **DRIFT-4**). See DRIFT-4 resolution above: option (a) — `f = 2` documented as the small-NV-feasible production default; future-flip triggers (GAP-3 closure OR Jolt-min-NV ≥ 22) named in the disposition doc on `planner_setup_shrink_factor`.

---

## Recommended production refactor priorities

Ranked, with the drift/gap/scope IDs each step closes. CLOSED items moved to the closure register at the bottom.

1. **Phase 5 Drift 4 — planner natural cascade discovery**: closes **GAP-3** (the surviving cost-model gap). Requires extending the planner objective with per-level setup-precompute storage cost + symmetric cleartext-discharge cost, then retiring the force-routing gates. Estimated ~200-300 LOC + test rework + production schedule-table regeneration. See **GAP-3** for the design path.

2. **General sliced-tensor transducer**: closes **GAP-1**. Only if a future protocol variant needs non-offset slices; otherwise defer indefinitely.

### Closure register (alphabetical by audit ID)

| ID | Closure commit(s) | Disposition |
|---|---|---|
| DRIFT-1 | `f18412f` + `952a067` | Two-coefficient cutover + module/test-helper polish |
| DRIFT-2 | `920086f` | `MRowLayout` top-level doc enumerates 15-vs-10 extension |
| DRIFT-3 | `8e87160` | Singleton pre-pop in `setup_verifier`; lazy fallback documented (option (b)); callers wanting multi-shape pre-pop use `akita_verifier::prepopulate_tiered_s_cache` directly |
| DRIFT-4 | `d0ea827` | `planner_setup_shrink_factor` disposition doc; option (a) |
| GAP-2 | `ce01879` | Opt-in `op-counter` field-mult counter + measurement tests |
| SCOPE-3 | `6c9c38f` + `f5e3ee3` | Chunk-axis amortised verifier eval + γ-folding aggregation |
| SCOPE-4 | `ce01879` | (with GAP-2) |
| SCOPE-5 | `d0ea827` | (with DRIFT-4) |

---

## Methodology + reproducibility

### How I mapped book lines to code

- Read the book §5 line-by-line (1181 lines, ~2 hours).
- For each protocol element, formed a search pattern derived from the book's mathematical notation (function names, constant names, transcript labels, protocol step names).
- Used `grep` (ripgrep via the IDE's Grep tool) to locate matches, then `Read` to verify the implementation matches the book's spec at file:line granularity.
- For each Figure 12 round, located BOTH the prover-side call and the verifier-side replay; cross-checked that the transcript labels appear in the same order.

### Files audited

- `crates/akita-transcript/src/labels.rs` — every transcript label, ordered against Figure 12 rounds 1–8.
- `crates/akita-prover/src/protocol/{flow,quadratic_equation,ring_switch,setup_claim_reduction}.rs` — prover-side round implementations.
- `crates/akita-verifier/src/protocol/{levels,ring_switch,setup_claim_reduction,batched}.rs` + `crates/akita-verifier/src/stages/stage2.rs` — verifier-side round implementations.
- `crates/akita-types/src/layout/{params,proof_size,flat_matrix}.rs` — `MRowLayout` 10/15-group enumeration, `level_proof_bytes` cost model, `tiered_setup_group_lp` / `meta_lp_from_chunks` tier derivation.
- `crates/akita-types/src/proof/{setup,tiered_setup}.rs` — preprocessed `C_S` cache plumbing.
- `crates/akita-planner/src/schedule_params.rs` — force-routing gate, candidate level params, cascade cost model.
- `crates/akita-config/src/{lib,claim_reduction,proof_optimized,bare}.rs` — production preset wrappers + B-1 flip + BareCfg.
- `crates/akita-algebra/src/offset_eq.rs` — §5.3 offset-slice specialization of Algorithm 1.
- `crates/akita-algebra/src/ring/{crt_ntt_cache,ntt_matvec}.rs` — verifier NTT preprocessing.
- `crates/akita-challenges/src/stage1.rs` — tensor stage-1 sampling (Round 2 + Round 4).
- `crates/akita-pcs/tests/tiered_setup_e2e.rs` — cascade firing E2E tests + measurement tests.
- `specs/security_analysis.md` (own §10) — composed soundness walkthrough.
- `specs/tiered-d-row-m-relation-bug-handoff.md` — historical context on the chunks A-row CRT bug.
- `audit.md` — every S-N / B-N / C-N item cross-referenced.

### Files NOT audited (scope limits)

- Sumcheck primitive (`crates/akita-sumcheck/`): trusted at its boundary; book §5 doesn't impose new requirements on the sumcheck primitive beyond degree-2 special soundness for stage-2 and the existing range-check structure for stage-1.
- Field arithmetic primitives (`crates/akita-field/`): trusted; no §5 contracts.
- Stage-1 prover/verifier internals (`crates/akita-prover/src/stages/`, `crates/akita-verifier/src/stages/stage1.rs`): touched only when the M-relation evaluation crosses the boundary; the stage-1 protocol per se is unchanged from earlier sections.
- Commit cost-model micro-tests (`crates/akita-types/src/layout/proof_size.rs` unit tests at the bottom of the file): trusted as exercising the helpers in isolation; the audit relies on E2E test results for end-to-end consistency.

### What I would do differently with infinite time

1. **Bit-for-bit transcript trace comparison**: capture the exact byte sequence absorbed/sampled by the prover and verifier at Figure 12 round granularity, then assert it matches a hand-computed reference for a smallest-possible test case. This would catch any subtle transcript-ordering drift that grep + manual cross-reference might miss.
2. **Algorithm 1 reference implementation in Python** + black-box test against the Rust `eval_offset_eq_tensor`. Would close any silent algorithmic drift in the offset-slice specialization.
3. **Formal model of Figure 12 in Cryptol or F\***, machine-checked against the implementation's per-round invariants. Out of scope for any normal audit but the right tool for production-grade soundness.

---

## Phase 5 progress (ralph-loop, 2026-05-19)

Goal: close SCOPE-3 + GAP-3 by realising the shared per-chunk matrix collapse end-to-end so the planner DP discovers the headline cascade naturally.

### Iteration log

- Iter 1 (start): baseline `cargo build --all-targets` is clean at HEAD `ce01879`. Scoping investigation completed.
  - **Item 1 insight**: For tier-marked groups, the prover's materialise writes ALL `k` chunks' contributions to chunk-INDEPENDENT `(row, col, coeff)` cells — i.e. the chunks already accumulate into the SAME setup-polynomial col slots `[0..num_blocks_chunk · num_digits_open)` via the column formula `d_col = block_idx · num_digits_open + dig` (see `setup_weight_table_at_point_grouped` line 1938). The verifier's structured eval also reads from the same chunk-independent col indices.
  - The col_count_padded that `setup_polynomial_padded_dims_inner` computes for tier-marked groups INCLUDES a `claim_count = k` factor (line 432-444 of `params.rs`'s heterogeneous branch). The high cols are reserved but never written — pure over-allocation.
  - Realising the book's `O(|D_chunk|) + O(log k)` per-chunk amortisation therefore reduces to **removing the k-factor over-allocation** in both `setup_polynomial_padded_dims_inner` and `planned_setup_padded_dims`. The verifier's existing structured eval and the prover's materialise already operate at chunk-shared col indices.
- Iter 2: Item 1 implemented as a targeted col-envelope fix.
  - `setup_polynomial_padded_dims_inner` heterogeneous branch now takes `max` over groups of per-group col extent, with tier-marked groups contributing as `effective_claims = 1` (k chunks share cols). Un-tiered groups with `claim_count > 1` keep the full `claim_count` multiplier on their B col extent (per the `claim_within * num_blocks * n_a * num_digits_open` offset in the grouped structured eval's `local_col`).
  - `planned_setup_padded_dims` mirrored: tiered cascade incoming and un-tiered cascade incoming now both compute `col_count = max(W_max, [chunks_max or S_max], [meta_max if tiered])` instead of `sum + k*chunks + meta`.
  - New invariant test in `crates/akita-types/src/layout/proof_size.rs`: `planned_setup_padded_dims_tiered_drops_k_multiplier_from_chunks` checks `col_padded` strictly shrinks versus the pre-Phase-5 `k`-multiplied formula. Existing `tiered_eval_setup_weight_at_point_matches_materialized` in `crates/akita-pcs/tests/multi_group_commit.rs` still passes (structured == materialised at the smaller shape).
  - All 58 `akita-types` lib tests pass; `clippy --package akita-types` clean. Pre-existing lint errors in `akita-sumcheck/src/eq_weighted_table.rs` and `akita-verifier/src/protocol/ring_switch.rs::padded_x_bits`/`boolean_point` were already present at HEAD `ce01879` and are not introduced by this change.

- Iter 6 (CORRECTION): Items 2-4 reverted via `git reset --hard b4b02c7` after release-mode E2E surfaced two real bugs the narrow unit tests missed.
  - **Item 2 (`a4955e9`) bug**: `tiered_dense_prove_verify_small` (NV=19 dense, release) fails with `InvalidProof` at the verifier. Root cause: the SIS-rank shrink only landed at `tiered_setup_group_lp` + `meta_lp_from_chunks` call sites; other chunk_lp construction paths still inherit un-tiered base ranks, so prover commits at one rank and verifier expects another.
  - **Item 3 (`04b8947`) bug**: same test fails with `InvalidSetup("scheduled recursive level did not match runtime state: step.current_w_len=5065856, inputs.current_w_len=5724800")`. Root cause: planner cost model dropped the `k` multiplier from chunks' `w_hat + t_hat`, but the runtime cascade L+1 commit STILL produces k× per-chunk witness material (k separate B-commitments, not shared). Planner under-counts → schedule diverges from runtime layout.
  - Baseline at Item 1 (`b4b02c7`): `tiered_dense_prove_verify_small` passes in 5.58s release. Item 1's col-envelope k-drop is verifier-eval-amortisation only (no runtime-witness change), so it doesn't have the planner/runtime mismatch.

---

## Phase 5 fix loop (post-revert, new start)

Goal: deliver the rest of Phase 5 correctly per book §5.5 line 752 "MLE evaluation cost `O(|D_chunk|) + O(log k)`, independent of k", with **release-mode E2E gated after every step**. The three deliverables the prior loop under-scoped:

- **(a)** Sweep all chunk_lp + meta_lp construction sites to consistently use `derive_chunk_sis_ranks_from_widths` so prover and verifier agree on the chunk Ajtai ranks. (Was attempted as Item 2; only covered two sites out of ~6-10.)
- **(b)** Restructure `eval_setup_weight_at_point_grouped` so the per-chunk sum is expressed as a chunk-axis eq factor times a shared inner sum, realising the book's `O(|D_chunk|) + O(log k)` per-cascade-level verifier cost. (Was missing entirely — the prior Iter 2 only shrunk col_count_padded, not the per-chunk eval loop.)
- **(c)** Change the L+1 prover so the k chunks commit under one logical shared matrix (single B-commitment with k structured columns) instead of k separate B-commitments. Only after this does the planner's `k`-drop in `planned_joint_w_ring_with_setup_group_tiered` align with runtime reality.

### Discipline rules (NON-NEGOTIABLE)

- **EVERY commit gated by release-mode E2E**:
  ```bash
  cargo test --release --package akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small
  ```
  (NV=19 dense cascade, ~6s in release; exercises the full prover+verifier cascade chain end-to-end.)
- Smaller-NV release E2E tests for finer signals when needed:
  - `setup_claim_reduction_dense_prove_verify` (NV=15, ~0.5s release) — non-cascade CR baseline.
  - `setup_claim_reduction_dense_recursive_prove_verify` (NV=20, ~2s release) — recursive CR baseline.
- **Debug-build `cargo test` is BANNED for E2E** (10-50× slower; the prior loop wasted ~30 min on stuck debug tests).
- Iteration log appended below as work proceeds.

### Fix loop iteration log

- Iter F-1 (start): clean revert to Item 1 baseline (`b4b02c7`). Release-mode `tiered_dense_prove_verify_small` confirmed passing in 5.58s. Loop scratchpad established.

- Iter F-3 (status, pre-implementation): Items (b) and (c) scope re-assessment.
  - **Item (b) — verifier per-chunk eval loop O(log k) factorisation**: investigated. The current per-(dig, chunk, blk) loop in `eval_setup_weight_at_point_grouped` does `eq_weight_at_index(x_challenges, c + x_local)` per iteration (O(log x_bits) muls per call, O(k · num_blocks_chunk · num_digits_open) iterations per group). Book §5.5 line 752 promises `O(|D_chunk|) + O(log k)`. Two paths to realise it:
    1. Apply `eval_offset_eq_tensor` (existing primitive in `akita-algebra::offset_eq`) with 1-D factor tables `[eq_col_blk, d_factors, eq_col_dig]`. **Blocker**: requires `num_digits_open` to be a power of 2 so the col index `d_col = blk * num_digits_open + dig` decomposes cleanly into (blk_bits, dig_bits). In production `num_digits_open = ⌈128 / log_2 b⌉` (e.g. 26 at `log_b = 5`) is NOT generally a power of 2. The col-axis factorisation breaks. Would require either (a) padding `num_digits_open` to next pow2 in the witness layout (invasive prover change), or (b) custom offset-eq DP that handles non-pow2 multiplications (~200 LOC primitive).
    2. Precompute `eq_x_group[i] = eq(x_challenges, c + i)` for `i ∈ [0, group_size)` via offset-eq DP, then replace per-call `eq_weight_at_index` with O(1) lookup. **Win**: ~16× constant-factor (eliminates the O(log x_bits) per-call cost). **No asymptotic improvement**. Implementation: 50-100 LOC + careful offset-alignment handling.
    Neither path delivers the book's O(log k) asymptotic without significant additional design (option 1a invasive prover change, option 1b new primitive). For practical NV ≤ 22 in our test suite the verifier setup-CR cost is already ~6 ms per level (out of ~6 s total cascade prove+verify), so the constant-factor option 2 win is unmeasurable in wall-clock. **Deferred from this loop iteration** — needs explicit user decision on whether to invest in option 1.
  - **Item (c) — L+1 shared B-commit for the k chunks**: the recon (Iter F-1.5 explore subagent) confirmed the current L+1 ships **k separate B-commitments** (`tiered_setup.rs:212-214` `TieredSetupCommitments.chunk_b_commitments: Vec<FlatRingVec<F>>`, `flow.rs:1747-1748` per-chunk transcript absorption). Realising the book's "T2 ratio ≈ 1" requires:
    1. Restructuring `build_tiered_handle_material` to commit the k chunks under ONE shared B-matrix (one logical commitment with k structured columns instead of k separate B mat-vecs).
    2. Restructuring `TieredSetupCommitments` on-wire to ship one shared B-commitment instead of `Vec` of k.
    3. Mirror the change in `derive_tiered_setup_material_for_verifier`.
    4. Update the L+1 prover's `prove_recursive_level_with_policy` and `prove_recursive_multi_fold_with_params` to ingest the shared-B representation as a single commitment.
    5. Update the L+1 verifier's `expand_tiered_setup_claims` similarly.
    6. After (1)-(5), revisit the prior loop's Item-3 change (k-drop in `planned_joint_w_ring_with_setup_group_tiered`) — it would then ALIGN with runtime reality and could land safely.
    Estimated scope: 800-1500 LOC across prover, verifier, types, and on-wire serialization. High risk of breaking the cascade tamper-reject tests if not done carefully. **Deferred from this loop iteration** — needs explicit user decision on scope.

- Iter F-2: Item (a) — chunk_lp + meta_lp B-role SIS rank shrink.
  - **Root cause of prior Item-2 failure surfaced**: `derive_chunk_sis_ranks_from_widths` modified `chunk_lp.a_key` (raising row_len from 1 to 2 for chunk widths that demanded it), but `GroupSpec` only carries `b_key`. The L0 commit used `chunk_lp.a_key.row_len=2` while the L+1 M-relation read `outer.a_key.row_len=1` for tier-marked groups — they disagreed and verify rejected with `InvalidProof@stage2-closing`. Pinpointed by instrumenting `prepare_m_eval` to dump LP shapes on both sides + per-check-site error tracing in `verify_setup_claim_reduction`.
  - **Fix**: Restrict `derive_chunk_sis_ranks_from_widths` to SHRINK-ONLY on B-role; never touch `a_key`, `d_key`, or `b_key.col_len`. Helper is now structurally incapable of desyncing the L0-commit / L+1-M-relation abstraction:
    - A-role and D-role at L+1 read from OUTER LP via the `GroupSpec` abstraction (which lacks `a_key`/`d_key`). Touching them would silently desync.
    - B-role is the only key carried per-`GroupSpec`, so shrinking it propagates consistently.
    - Growth is rejected: would indicate the OUTER LP is insecure for chunk widths (a different bug surfaced via `validate_stored_sis_ranks` rather than papered over here).
  - 4 call sites wired: `tiered_setup_group_lp`, `tiered_setup_group_lp_from_dims` (both in `akita-types::layout::proof_size`), `meta_lp_from_chunks` (prover `flow.rs`), `meta_lp_from_chunks` (verifier `levels.rs`).
  - 2 new lock-in tests in `akita-types::layout::proof_size`:
    - `derive_chunk_sis_ranks_only_touches_b_role_and_shrinks`: asserts `a_key`/`d_key`/`b_key.col_len` unchanged, `b_key.row_len` may shrink but never grows.
    - `derive_chunk_sis_ranks_returns_unchanged_when_b_collision_unpinned`: early-return contract for un-pinned collision buckets.
  - Pre-existing lint cleanups so workspace clippy is green: gated `SumcheckInstanceVerifier` / `multilinear_eval` in `akita-sumcheck/src/eq_weighted_table.rs` behind `cfg(any(test, feature = "test-helpers"))`; gated `padded_x_bits` / `boolean_point` in `akita-verifier/src/protocol/ring_switch.rs` likewise; added `PhantomData<Cfg>` placeholder in `akita-config::proof_optimized::cr_on_max_setup_matrix_size` (no-planner branch).
  - **E2E gate (release mode)**: `tiered_dense_prove_verify_small` (NV=19 dense cascade, 6.7s), `tiered_dense_prove_verify_mid_f4` (f=4 cascade, 6.9s), `tiered_dense_cascade_l0_l1_small` (NV=19 (2,2) cascade, 8.1s), `tiered_dense_cascade_l0_l1_fires`, `tiered_dense_default_cascade_fires`, `tiered_rejects_tampered_s_opening_value`, `tiered_rejects_tampered_next_w_commitment` ALL pass. Also `setup_claim_reduction_e2e` (5 tests, 2.6s), `multi_group_commit` (7 tests, 7.6s), and `akita-config::current_d*_*_schedule_stays_within_audited_sis_widths` (5 tests, instant) pass. `cargo clippy --workspace --lib -- -D warnings` clean.
