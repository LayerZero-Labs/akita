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
| Core idea: `k = f²` chunks of `2^{r_chunk} = 2^{r − log₂ f}` blocks; shared per-chunk matrices `D_chunk/B_chunk` at `1/f` baseline column width; tier-3 meta commitment | 686–700 | ALIGNED on shape, SCOPE-DEFERRED on cost savings — see **SCOPE-3** | `tiered_setup_group_lp` at `crates/akita-types/src/layout/proof_size.rs:706` shrinks chunk_lp by `log₂ f` on both axes; meta tier derived via `meta_lp_from_chunks` at `crates/akita-prover/src/protocol/flow.rs:521`. The **shared per-chunk matrices** (per-chunk D / B reused k× via block-diagonal MLE collapse with `O(log k)` overhead) is not yet exploited in the cost model — the planner sizes the k chunks as k independent commitment groups. See **SCOPE-3**. |
| Effect on setup matrix (S shrinks by ≥f) | 701–708 | ALIGNED | Per-chunk LP `b_key.col_len()` and `d_key.col_len()` are 1/f of the baseline by construction (chunk_lp = base_lp with `r_vars -= log₂ f`, `m_vars -= log₂ f`) |
| Modified relation: 10 check groups (per-chunk D/B + 1 eval + 1 fold + n_A Ajtai for the original poly; same 5 for meta) | 709–754 | ALIGNED with terminology drift — see **DRIFT-2** | `MRowLayout` at `crates/akita-types/src/layout/params.rs:282–325` has the 10 fields: `original_{d,b,eval,fold,a}` + `meta_{d,b,eval,fold,a}`, plus an additional 5-tuple `w_{d,b,eval,fold,a}` for the joint W+S case (the book describes the 10 groups for `[S, meta]`; the implementation extends to `[W, original, meta]` for the cascade case — book §5.6 line 940–953 alludes to this) |
| Per-chunk rows 1–2 have block-diagonal structure with SHARED `D_chunk/B_chunk` → MLE eval `O(|D_chunk|) + O(log k)`, independent of k | 751–754 | SCOPE-DEFERRED — see **SCOPE-3** | The implemented verifier evaluates the M-table per-chunk groups WITHOUT collapsing the shared per-chunk matrices yet (cost grows ~k×). The structured eval `setup_weight_table_at_point_grouped` at `crates/akita-verifier/src/protocol/ring_switch.rs:1831` does avoid the naive hypercube materialization (it's `O(log)` per group), but does not yet exploit the shared-matrix block-diagonal collapse the book promises. |
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
- **Book**: §5.6 Figure 12 Round 8 line 912–919 specifies `γ_range · s_claim + γ_rel · (λ · m̃_alg(r_x) + y_setup) = (batched sumcheck output)`. Two named coefficients.
- **Implementation**: `crates/akita-verifier/src/stages/stage2.rs:318–333` `expected_output_claim_with_m_setup` computes `batching_coeff * virtual_oracle + relation_oracle` (one named coefficient).
- **Why it's drift**: notational. Algebraically the two forms are isomorphic under the normalization `γ_rel = 1` and `γ_range = batching_coeff`. The implementation uses one challenge with the relation side having coefficient 1; the book uses two challenges, one for each batch. Either form is correct under the standard sumcheck-batching reduction.
- **Soundness impact**: NONE. The single-coefficient form is a strict special case of the two-coefficient form (set `γ_rel = 1` in book's expression). Both forms have the same per-round knowledge error.
- **Production blocker**: NO. Functional code is correct.
- **Recommended fix**: rename `batching_coeff` to `gamma_range_over_gamma_rel` (with a doc comment citing book line 912–919) OR refactor to sample two challenges and use both. The first option is one-line.

### DRIFT-2 — `MRowLayout` adds an additional W-tier row family vs the book's 10
- **Book**: §5.5 lines 709–754 enumerates 10 check groups: 5 for the original polynomial (per-chunk D, per-chunk B, eval, fold, Ajtai) + 5 for the tier-3 meta commitment (D_meta, B_meta, eval-like, fold, A_meta).
- **Implementation**: `crates/akita-types/src/layout/params.rs:282–325` `MRowLayout` has 15 fields: the 10 book-enumerated `original_*` + `meta_*` plus an additional 5 `w_*` (`w_d`, `w_b`, `w_eval`, `w_fold`, `w_a`) for the joint W+S cascade case.
- **Why it's drift**: the book §5.5 describes the 10-group layout for the original polynomial + meta tier in isolation. When the cascade applies (book §5.6 lines 940–953: "the next-level witness consists of (1) the standard folded witness w; (2) the shared-matrix polynomial S̃, entering unfolded"), the M-relation at level L+1 must bind BOTH W (the folded recursive witness) AND S (the routed setup polynomial chunks + meta). The implementation extends the 10-group layout to 15 by adding a W-tier alongside the original (chunks) and meta tiers.
- **Soundness impact**: NONE. The W-tier rows enforce the same Ajtai binding contract as the book's per-tier rows; they're a faithful extension of the §5.5 schema to the joint W+S case the book §5.6 mandates.
- **Production blocker**: NO.
- **Recommended fix**: document the extension explicitly. The current `MRowLayout` field doc comments already say "Book §5.4 original polynomial D rows" / "Book §5.4 meta D rows" / "Ordinary recursive-W D rows for production tiered batches" — clear enough. Could add a top-level comment on `MRowLayout` calling out the book-vs-impl row-family count.

### DRIFT-3 — Verifier `tiered_s_cache` pre-population happens at `setup_verifier` time, not at setup-table generation
- **Book**: §5.3 line 952 + Figure 12 line 817 specify `C_S` as a preprocessed verifier input, derived during setup.
- **Implementation**: `crates/akita-prover/src/api/setup.rs:109–127` `AkitaProverSetup::verifier_setup` pre-populates `ntt_shared_cache` via `ntt_shared_get_or_init::<D>()`. The `tiered_s_cache` is pre-populated lazily on first verify (`crates/akita-verifier/src/protocol/levels.rs:570` `tiered_setup_material_for_verifier` uses `tiered_s_cache_get_or_init`). The previous Ralph loop also added `prepopulate_tiered_s_cache` at `setup_verifier` for the singleton `(max_num_vars, max_num_vars, singleton)` shape (commit `8e87160`).
- **Why it's drift**: minor — the book conceptually treats `C_S` as a preprocessed artifact independent of the proof shape. The implementation pre-populates the singleton shape at `setup_verifier` time and defers other shapes to lazy on-first-use. For production where most verifies are singleton, this matches the book's "preprocessed" framing exactly; for non-singleton (multi-poly, multi-point, sub-NV) verifies, the first call pays the derivation cost.
- **Soundness impact**: NONE. The cache contract is `setup.expanded`-deterministic regardless of pre-pop timing.
- **Production blocker**: NO. Pre-pop at setup_verifier covers the common case.
- **Recommended fix**: thread the verifier-known shape (which subset of `(num_vars, batch)` cases will be verified) into `setup_verifier` so non-singleton pre-pop can also happen at setup. Or, accept the lazy-first-use behavior as the production design and document.

### DRIFT-4 — Production presets default `f = 2` tier instead of `f = 1` un-tiered
- **Book**: §5.4 line 793–796 says "the sweet spot is f = 8". The un-tiered (f = 1) case is the §5.3 baseline. The book does not say `f = 2` is the production default; it says `f = 8` should be the cascade default.
- **Implementation**: After audit B-1 (commit `c9d9904`), the production fp128 presets (`D128Full`, `D64OneHot`, …) default to `use_setup_claim_reduction = true` with `planner_setup_shrink_factor() = 2`. This is because at NV=15 (small NV used by the `setup_claim_reduction_e2e` tests) f=8 doesn't schedule, and f=1 doesn't pass the force-routing gate (audit S-8), so f=2 is the smallest tier shape that fires the cascade default while still being feasible at low NV.
- **Why it's drift**: the production preset's tier-shape default is `f=2`, not the book's recommended sweet spot `f=8`. Users opting into `f=8` use `TieredClaimReductionCfg<DenseCfg>`; users opting into the headline cascade `(f_L0=8, f_L1=4)` use `TieredCascadeCfg<DenseCfg>`. The bare default produces `routing=1 tiers=[2]` at NV=19.
- **Soundness impact**: NONE. Both `f=2` and `f=8` are sound tier shapes; the choice affects only proof-size and verifier-perf trade-offs.
- **Production blocker**: PARTIAL — the production default isn't the book's recommended `f=8` sweet spot. Whether this matters depends on intended use:
  - If the production preset will only be used at NV ≥ 32 (the book's measurement range), flip default to `f=8` (or even `(8, 4)`).
  - If the production preset must also work at NV < 32 (e.g., for small Jolt instances), the `f=2` default is the right trade-off.
- **Recommended fix**: either (a) keep `f=2` default and document why (book's `f=8` sweet spot fires only at NV ≥ ~22, smaller NVs need `f=2`), or (b) split the production presets into two tiers: `D128FullProduction` defaults to `f=8` (NV ≥ 22); `D128FullSmall` defaults to `f=2` (NV ≥ 19). Pick based on Jolt's actual NV range.

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
- **Book**: §5.8 Table 1141–1158 measures verifier op counts at NV ∈ {32, 38, 44} for the headline `(f_L0=8, f_L1=4)` cascade.
- **Implementation**: `tiered_dense_cascade_l0_l1_headline_small` runs at NV=22 (smallest schedulable on this 123 GiB host for dense `(8, 4)`); `tiered_onehot_cascade_l0_l1_headline_small` runs at NV=28 (smallest schedulable for onehot).
- **Why the gap**: hardware budget. Dense `(8, 4)` at NV ≥ 25 OOMs on 123 GiB; onehot `(8, 4)` at NV ≥ 30 likely OOMs. The protocol code is correct (schedules for NV ∈ {32, 38, 44} resolve cleanly via `Cfg::get_params_for_prove`); only the E2E prove + verify cannot complete in our hardware envelope.
- **Impact**: ASYMPTOTIC measurement gap. The implementation passes the cascade firing-and-correctness checks at our hardware-feasible NVs; the cascade speedup the book promises (16× / 35× / 265× at NV=32 / 38 / 44) cannot be empirically witnessed here.
- **Production readiness**: BLOCKER FOR EMPIRICAL VALIDATION; NOT FOR CORRECTNESS. The protocol-shape pipe is intact; just the speedup demonstration requires bigger hardware.
- **Recommended fix**: either (a) instrument an op-counter in the verifier (count field-mult invocations during each phase; report counts independent of wall-clock) and re-run measurements at our feasible NVs to verify the predicted scaling; or (b) acquire a host with ≥256 GB RAM and run the headline cascade at NV=32. Option (a) is cheap (~100 LOC instrumentation + 1 measurement test) and would let us extrapolate book-comparable op counts at NV ∈ {32, 38, 44} from our NV=22 / NV=28 measurements.

### GAP-3 — Cascade-discovery cost model under-credits shared-matrix savings (audit S-8 left this in place)
- **Book**: §5.5 lines 751–754 promises per-chunk MLE evaluation cost `O(|D_chunk|) + O(log k)`, independent of k. This makes the cascade `next_w_len` grow by the book's T2 ratio (1.0–3.0×), not by ~k×.
- **Implementation**: cost model after S-8 (commit `f2c7b9b`) accurately scores the cascade PENALTY but not the SAVINGS from shared per-chunk matrices. `planned_joint_w_ring_with_setup_group_tiered` in `crates/akita-types/src/layout/proof_size.rs` sizes the k chunks as k independent commitment groups (commented at lines 525–533 explicitly).
- **Why the gap**: tracked as Phase 5 work in the implementation chain. Cost model would discover the cascade naturally for the headline shape at NV ~= 32 only after this gap closes; currently the force-routing gate is required.
- **Impact**: PERF-ONLY (cost-model accuracy). The implemented protocol shape IS correct; this is purely a planner-side cost-model gap that makes the DP unable to discover the cascade naturally without the force gate.
- **Production readiness**: NICE-TO-HAVE. Force-routing gate is the canonical opt-in mechanism today; gap closure would let the DP discover the cascade unprompted. See **SCOPE-3** for the deeper refactor this gap entails.
- **Recommended fix**: see **SCOPE-3**.

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
- **Book**: `D_chunk` and `B_chunk` are shared across k chunks; per-chunk rows have block-diagonal structure with `O(|D_chunk|) + O(log k)` MLE evaluation cost.
- **Current state**: PARTIAL. `eval_setup_weight_at_point_grouped` (`crates/akita-verifier/src/protocol/ring_switch.rs:1831`) avoids the naive hypercube materialization (it's `O(log)` per group) but does not collapse the shared per-chunk matrices into a single `O(log k)`-amortized eval. The planner cost model treats each chunk as a full-rank independent commitment group.
- **Why deferred**: Phase 5 work. Recorded in:
  - `crates/akita-planner/src/schedule_params.rs:266–282` (comment chain explaining why the planner DP can't discover the cascade unprompted)
  - `audit.md` S-8 (the cost-model widening; commit `f2c7b9b` makes the cost-model PENALTY accurate but doesn't close the SAVINGS gap)
  - `specs/security_analysis.md` §10.4 (notes that per-chunk D/B are sized independently today)
- **Suggested next slice**: Phase 5 implementation has three parts:
  1. **Block-diagonal MLE collapse** in the verifier: extend `eval_setup_weight_at_point_grouped` to recognize block-diagonal structure and amortize the per-chunk eval at `O(log k)` cost.
  2. **Per-chunk SIS rank shrink** in the planner: when chunk widths drop by `1/f`, the SIS-secure rank often drops too (book §5.4 line 798–799 example: `n_A = 3 → 1` at NV=32 with f=8). Wire this into the planner's `sis_floor` lookup.
  3. **Cost-model crediting**: with (1) and (2), the planner's `next_w_len` for the cascade grows by the book's T2 ratio (≈1) instead of ~k×, and the DP discovers the cascade unprompted for the headline shape. The force-routing gate can then become an explicit opt-in or be retired.

Estimated scope: ~500–1000 LOC across types, planner, verifier; significant test work.

### SCOPE-4 — Verifier op count measurement at book-comparable NVs
- **Book**: §5.8 Table 1141–1158 measures verifier op counts at NV ∈ {32, 38, 44}.
- **Current state**: speedup measurement test reports verifier wall-clock at NV=22 dense / NV=28 onehot. The dense ceiling on this 123 GiB host is NV=22; onehot ceiling NV=28.
- **Why deferred**: hardware budget. See **GAP-2**.
- **Suggested next slice**: option (a) from **GAP-2**: instrument an op-counter, run at NV=22, extrapolate to NV=32 via the known scaling laws. Cheap (~100 LOC).

### SCOPE-5 — Production preset tier-shape choice
- **Book**: §5.4 line 793 "the sweet spot is f = 8".
- **Current state**: production presets default `f = 2` (see **DRIFT-4**).
- **Why deferred**: small-NV use cases (`setup_claim_reduction_e2e` tests at NV=12, 15) don't schedule under f=8.
- **Suggested next slice**: decide on Jolt's intended NV range. If Jolt's smallest NV is ≥ 22 in production, flip default to f=8.

---

## Recommended production refactor priorities

Ranked, with the drift/gap/scope IDs each step closes.

1. **Op-counter instrumentation + book-comparable measurement at NV=22**: closes **GAP-2** + **SCOPE-4**. The cheapest item with the highest production payoff — gives concrete evidence that the cascade matches the book's predicted asymptotic at the smallest NV where the planner schedules, without requiring bigger hardware. ~100 LOC + 1 measurement test.

2. **Tier-shape production-default policy decision**: closes **DRIFT-4** + **SCOPE-5**. Either flip `D128Full` / `D64OneHot` to default `f = 8`, or document why `f = 2` is the right small-NV default. One-line code change once decided. Optionally split into `*Production` vs `*Small` preset aliases.

3. **Round 8 batching coefficient naming**: closes **DRIFT-1**. Rename `batching_coeff` to `gamma_range_over_gamma_rel` (or refactor to two challenges). Defense-in-depth doc improvement.

4. **`MRowLayout` doc comment**: closes **DRIFT-2**. Add a top-level comment on the struct calling out the 10-vs-15 group extension for the joint W+S case.

5. **Phase 5: shared per-chunk matrix collapse**: closes **SCOPE-3** + **GAP-3**. Significant refactor (~500–1000 LOC + tests), but unlocks the cascade-discovery property the planner can't currently express, AND realises the book §5.4 line 798–799 setup-storage reduction (32.5 GB → 4.3 GB at NV=44).

6. **Non-singleton verifier setup pre-population**: closes **DRIFT-3**. Thread the verifier shape into `setup_verifier` so the first verify at non-singleton shape doesn't pay the cache-derivation cost.

7. **General sliced-tensor transducer**: closes **GAP-1**. Only if a future protocol variant needs non-offset slices; otherwise defer indefinitely.

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
