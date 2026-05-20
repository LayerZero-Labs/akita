# `feat/tensor-challenges` vs `main` — Deep §5 conformance audit

**Date**: 2026-05-20
**HEAD**: `dd5302587bf76291135c58e525b2763093514395` (audited code HEAD before this audit-doc iteration commit)
**Merge base**: `4b0b86a946dca5124ddc1c0197bda7b73284a137`
**Commits on top of main**: 155 total (`152` non-merge + `3` merge commits)
**Diff stat**: `133 files changed, 149999 insertions(+), 5604 deletions(-)`
**Book ref**: `/home/giuseppe/lattice-jolt/sections/akita/5_fourth_root_verifier.tex`
**Scope**: every meaningful change on top of main, classified against book §5 / Figure 12.
**Methodology**: commit-by-commit walkthrough + file-by-file diff + book line cross-reference.

> Iteration status: this is the iteration-3 scaffold plus first-pass evidence.
> Anything marked `ITERATION-TODO` is intentionally not yet accepted as audited.
> The completion promise is false until all TODO markers are removed and every
> commit row has a final class.

## Executive summary

- **Audit inventory updated**: audited code HEAD is `dd530258`, merge-base is `4b0b86a9`, with `155` commits and `133` changed files on top of `origin/main`.
- **Diff shape is heavily audit/spec/generated skewed**: `scripts/` accounts for `108,492` net LOC, `specs/` for `6,100` net LOC, and generated schedule/security data are a major part of the raw insertion count. These need classification but should not be mistaken for protocol code.
- **Primary protocol crates touched**: `akita-prover` (`+5,562` net), `akita-verifier` (`+4,742` net), `akita-types` (`+4,588` net), `akita-pcs` tests/benches (`+6,174` net), `akita-config` (`+1,799` net), `akita-planner` (`+1,043` net), `akita-challenges` (`+1,406` net), and `akita-algebra` (`+2,053` net).
- **Initial aligned evidence found**: tensor challenge left/right sampling and transcript digesting match §5.2 / Figure 12 rounds 2-4; setup claim-reduction now uses the book-shaped row/coeff degree-2 reducer with the dedicated transcript label; `MRowLayout` documents the book's 10 tier groups plus the §5.6 joint-W extension.
- **Initial gap candidate confirmed for revalidation**: the implementation has an offset-slice carry-DP evaluator, but no general sliced tensor transducer API matching §5.3 Definition/Algorithm. This is likely a non-blocking gap if all production calls use offset slices only.
- **New high-priority drift candidate from iteration 3**: after `81cceec` / `dd53025`, the claim reducer is closer to §5.4 lines 615-621, but the active schedule/test evidence no longer emits the book §5.8 headline `(f_L0=8, f_L1=4)` recursive cascade. `DenseCascadeCfg`'s sentinel at NV=22 asserts `routing_count == 0`, while the old positive regression for two routed tiers is ignored as "compact r_x-fixed S cannot be f^2-tiered".
- **Initial drift register is not final**: prior drift closures in `specs/section5_protocol_drift_audit.md` must be rechecked against HEAD and commit history, especially cascade natural discovery, force-route retirement, chunk aggregation transcript binding, and `setup_verifier` prepopulation timing. Iteration 3 re-opens the cascade closure as **DRIFT-1 candidate** until the §5.6-§5.8 sub-audit resolves whether another path realizes the book cascade.
- **Production-readiness verdict is deferred** until the full commit chronology, public API delta, type-shape delta, test delta, security delta, and cascade measurement walkthrough are complete.

## Diff overview by crate

| Crate / area | LOC added | LOC removed | Net | Files touched | §5 subsections this area appears to implement |
|---|---:|---:|---:|---:|---|
| `crates/akita-algebra` | 2,053 | 0 | 2,053 | 4 | §5.3 offset-slice tensor contraction; NTT/cache support for verifier setup material |
| `crates/akita-challenges` | 1,412 | 6 | 1,406 | 5 | §5.2 tensor stage-1 challenge sampling and challenge-family security support |
| `crates/akita-config` | 2,002 | 203 | 1,799 | 7 | §5.5 / §5.8 preset routing, cascade configs, production claim-reduction enablement |
| `crates/akita-field` | 224 | 0 | 224 | 6 | §5.8 measurement support; opt-in verifier op counter |
| `crates/akita-pcs` | 6,219 | 45 | 6,174 | 13 | E2E tests and benches for §5.2, §5.4, §5.5, §5.8 |
| `crates/akita-planner` | 1,177 | 134 | 1,043 | 4 | §5.5 / §5.8 cascade schedule search and cost objective |
| `crates/akita-prover` | 6,417 | 855 | 5,562 | 21 | Figure 12 prover flow, recursive S routing, tiered setup material, claim-reduction prover |
| `crates/akita-scheme` | 59 | 23 | 36 | 2 | Setup/verifier orchestration and cache prepopulation |
| `crates/akita-setup` | 2 | 6 | -4 | 1 | Setup API integration; likely incidental |
| `crates/akita-sumcheck` | 464 | 50 | 414 | 4 | §5.4 / §5.6 sumcheck helpers and test-gated eq table support |
| `crates/akita-transcript` | 43 | 0 | 43 | 1 | Figure 12 transcript label delta |
| `crates/akita-types` | 8,544 | 3,956 | 4,588 | 21 | Public proof/layout/schedule/type-shape deltas for §5.4-§5.6 |
| `crates/akita-verifier` | 5,016 | 274 | 4,742 | 9 | Figure 12 verifier replay, setup claim-reduction, grouped M-eval, caches |
| `scripts` | 108,493 | 1 | 108,492 | 18 | Security analysis and planner audit tooling; mostly §5.7 / §5.8 support |
| `.cursor` | 578 | 0 | 578 | 1 | Ralph scratchpad state for audit-loop reproducibility |
| `specs` | 6,109 | 9 | 6,100 | 14 | Prior audits, designs, and handoff docs to revalidate |
| root / other | 747 | 0 | 747 | 2 | `audit.md`, lockfile |

## Change classification

### ALIGNED changes (faithful book implementation)

- **Tensor left/right challenge sampling and Fiat-Shamir separation** (commits: `d346d38`, `6dca0c8`, `7b91d8f`, `f90a056` require per-commit blame refinement)
  - **Book**: §5.2 lines 94-111:
    > "Instead of sampling $2^r$ independent challenges ... define the challenge for block $(p,q)$ as a ring product ... $c_{p \| q} := \alpha_p \cdot \beta_q$ ... The verifier now derives and evaluates only $2 \times 2^{r/2}$ base challenges."
  - **Book**: §5.2 lines 122-136:
    > "Round 2 ... $\boldsymbol{\alpha} \gets \C^{2^{r/2}}$ ... Round 3 ... $\varnothing$ (empty) ... Round 4 ... $\boldsymbol{\beta} \gets \C^{2^{r/2}}$ ... In a Fiat-Shamir instantiation, $\boldsymbol{\beta} = H(\mathbf{v},\, \boldsymbol{\alpha})$."
  - **Implementation**: `crates/akita-challenges/src/stage1.rs:532-542` samples `CHALLENGE_STAGE1_FOLD_TENSOR_LEFT`, absorbs `ABSORB_STAGE1_TENSOR_LEFT`, then samples `CHALLENGE_STAGE1_FOLD_TENSOR_RIGHT`.
  - **Why aligned**: The verifier/prover derive left and right tensor halves in the book order. The explicit left digest absorb is an implementation domain-separation step for the book's empty Round 3; it binds the right-half challenge to the sampled left half without adding a prover message.

- **Setup-side claim-reduction verifier uses a degree-2 sumcheck and dedicated transcript label** (commits: `2a4df12`, `9451f22`, `aa01e37`, `81cceec` require per-commit blame refinement)
  - **Book**: §5.4 lines 599-626:
    > "Define the scaling factor ... $\lambda := w_{\mathsf{eval}} \cdot \wt{\alpha}(r_y)$ ... Subtracting the verifier-computable algebraic part ... the remaining claim is ... proved by a degree-$2$ sumcheck over $\lceil \log_2 m_{\mathsf{row}} \rceil + \log_2 d$ variables ... The critical design choice: carry the scaled claim ... rather than dividing by $\lambda$."
  - **Implementation**: `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:175-182` calls `verify_sumcheck_rounds_only(..., 2, payload.m_setup_eval, ..., CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND)`.
  - **Implementation**: `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:184-189` checks `weight_at_point * payload.s_opening_value == final_running_claim` before the optional cleartext-vs-recursive closing path.
  - **Why aligned**: The verifier performs the book's short setup-side sumcheck with degree 2 and carries a scaled setup claim instead of dividing by the possibly-zero scaling factor.

- **Book-shaped setup-claim reducer fixes the sumcheck variable set to row families plus coefficients** (commits: `81cceec`, `dd53025`)
  - **Book**: §5.4 lines 615-621:
    > "This is proved by a degree-$2$ sumcheck over $\lceil \log_2 m_{\mathsf{row}} \rceil + \log_2 d$ variables, reducing to a single point evaluation on the preprocessed shared-matrix commitment: $\lambda \cdot \mle{S}(r_i,\, r_x,\, r_k) = y_{\mathsf{setup}}$."
  - **Implementation**: `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:91-105` flattens only `row | coeff`; `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:108-118` sets rounds to `row_bits + coeff_bits`; `crates/akita-types/src/layout/proof_size.rs:511-528` uses the same planned round count.
  - **Implementation**: `crates/akita-prover/src/protocol/flow.rs:1309-1327` computes `claim_scale = w_eval * alpha(r_y)` and emits `SetupClaimReductionPayload { m_setup_eval: out.input_claim, s_opening_value, sumcheck }`; `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:253-280` replays the main stage-2 sumcheck, computes the same scale, and verifies the setup reducer.
  - **Why aligned**: The reducer no longer treats the setup claim as a full row/column/coeff MLE sumcheck; it fixes the stage-2 `r_x` point first and runs exactly the book's row-family plus coefficient sumcheck, carrying the scaled claim without division. This alignment creates a separate cascade drift candidate below because §5.5/§5.8 tiering still talks about routing/chunking the setup polynomial across recursive levels.

- **Transcript labels added for Figure 12 stage boundaries** (commits: `dd5889b`, `f18418f`, `f90a056` require per-commit blame refinement)
  - **Book**: §5.6 lines 837-854, 885-919:
    > "Round 2 ... sample $\boldsymbol{\alpha}$" / "Round 4 ... sample $\boldsymbol{\beta}$" / "Round 8 ... verify $\gamma_{\mathsf{range}} \cdot s_{\mathsf{claim}} + \gamma_{\mathsf{rel}} \cdot (...)$".
  - **Implementation**: `crates/akita-transcript/src/labels.rs:46-55` defines `CHALLENGE_SUMCHECK_BATCH` and `CHALLENGE_SUMCHECK_BATCH_REL`; `crates/akita-transcript/src/labels.rs:66-73` defines the tensor stage-1 labels; `crates/akita-transcript/src/labels.rs:87-103` defines setup claim-reduction and tiered chunk aggregation labels.
  - **Why aligned**: The new labels create explicit Fiat-Shamir domains for the Figure 12 challenge points and the later chunk-aggregation extension. `ITERATION-TODO`: verify `all_labels()` order against actual consumption order in prover and verifier, not just declaration order.

- **Tiered row-layout documentation accounts for book 10 groups plus §5.6 joint-W extension** (commit: `920086f`)
  - **Book**: §5.5 lines 709-754:
    > "The stage-2 relation operates on the combined witness with \textbf{10 check groups}---five from the original polynomial and five from the tier-3 meta-commitment."
  - **Book**: §5.6 lines 940-952:
    > "The next-level witness consists of ... the standard folded witness ... the shared-matrix polynomial $\mle{S}$ ... The two polynomials share folding challenges and a joint $D$-commitment but have separate $B$-commitments."
  - **Implementation**: `crates/akita-types/src/layout/params.rs:284-304` documents `w_{d,b,eval,fold,a}`, `original_{d,b,eval,fold,a}`, and `meta_{d,b,eval,fold,a}` before `pub struct MRowLayout`.
  - **Implementation**: `crates/akita-types/src/layout/params.rs:315-344` defines the 15 row-family fields.
  - **Why aligned**: The book's 10 groups cover the tiered S/meta relation; the implementation adds 5 W groups for the §5.6 next-level joint witness case and documents the extension boundary explicitly.

### DRIFT (implementation differs from the book in observable behaviour)

- **DRIFT-1: Headline `(f_L0=8, f_L1=4)` cascade is not emitted on the active book-shaped reducer path** (commits: `81cceec`, `dd53025`; prior related commits: `9ddf99b`, `9d33e27`, `fd0ddb3`, `4f90979`, `71d7eef`)
  - **Book**: §5.4 lines 627-632:
    > "The matrix polynomial $\mle{S}$ from level~$L$ is batched into level~$L{+}1$'s witness for joint PCS opening: it enters level~$L{+}1$ \emph{unfolded} as an additional polynomial alongside the folded witness."
  - **Book**: §5.8 lines 1171-1175:
    > "Technique~2 requires tiered commitments at $f = 8$ (L0) $+ f_{\mathsf{L1}} = 4$ (L1) to keep the T2 cascade ratio $\lesssim 1$ across two levels. Setup storage drops from $32.5\,\text{GB}$ to ${\approx}\, 4.3\,\text{GB}$ for $n_v = 44$."
  - **Implementation**: `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:91-118` and `crates/akita-types/src/layout/proof_size.rs:511-528` reduce the setup-side claim to a compact `row | coeff` polynomial with `r_x` already fixed.
  - **Implementation**: `crates/akita-pcs/tests/tiered_setup_e2e.rs:500-527` contains the old positive regression for at least two routed tiers `[8, 4]`, but it is `#[ignore = "documents book f=8 full-S cascade drift: compact r_x-fixed S cannot be f^2-tiered"]`.
  - **Implementation**: `crates/akita-pcs/tests/tiered_setup_e2e.rs:800-857` asserts `DenseCascadeCfg` at NV=22 has book tier policy values `(8,4,1...)` but `routing_count == 0`; `crates/akita-pcs/tests/tiered_setup_e2e.rs:741-789` similarly asserts the default dense CR-on preset cleartext-discharges with `routing_count == 0`.
  - **Nature of drift**: structural / schedule-selection / cascade-shape. The Round 8 reducer is aligned with the book's short row/coeff sumcheck, but the advertised §5.8 two-level recursive tier cascade is not active on the current schedule path.
  - **Soundness impact**: NONE for accepted proofs if cleartext discharge runs; ASYMPTOTIC / production-performance impact for the fourth-root verifier claim.
  - **Production blocker**: CONDITIONAL. It is a blocker for claiming the book §5.8 "T1+T2 @ L0+L1" cascade or production-ready fourth-root performance. It is not a blocker for the local soundness of the row/coeff claim-reduction sumcheck.
  - **Recommended fix**: decide the intended contract, then make it explicit. If the book contract is full recursive cascade, restore a schedule/proof path that routes the full preprocessed `S` commitment through `[8,4]` tiers and keep the row/coeff reducer as the local sumcheck view. If the new contract is compact `r_x`-fixed cleartext discharge, update §5.8-facing docs and do not market the branch as implementing the book cascade speedup.
  - **Cross-reference**: prior `section5_protocol_drift_audit.md` GAP-3 / SCOPE-3 claimed force-route and shared-matrix-collapse closure; iteration 3 refines that as re-opened for HEAD because current tests encode no routed headline cascade.

- **DRIFT-TODO: Cascade discovery and force-route retirement require fresh audit**
  - **Book**: §5.8 lines 1171-1175:
    > "Technique~2 requires tiered commitments at $f = 8$ (L0) $+ f_{\mathsf{L1}} = 4$ (L1) to keep the T2 cascade ratio $\lesssim 1$ across two levels. Setup storage drops from $32.5\,\text{GB}$ to ${\approx}\, 4.3\,\text{GB}$ for $n_v = 44$."
  - **Implementation**: `ITERATION-TODO`: audit `crates/akita-planner/src/schedule_params.rs` commits `d436922`, `9ddf99b`, `9d33e27`, `fd0ddb3`, `4f90979`, `71d7eef` for whether cascade is now forced or naturally objective-driven.
  - **Nature of drift**: cost-model / schedule selection, if any force-routing gate remains or if the objective differs from the book's cost model.
  - **Soundness impact**: likely NONE if only schedule-choice; production performance impact may be material.
  - **Production blocker**: `ITERATION-TODO`.
  - **Recommended fix**: `ITERATION-TODO`.
  - **Cross-reference**: prior `section5_protocol_drift_audit.md` GAP-3 / DRIFT-4; must be confirmed or refined.

### GAP (book describes something the implementation does NOT have)

- **GAP-1: General sliced tensor transducer API is absent; offset-slice specialization exists**
  - **Book**: §5.3 lines 289-335:
    > "A sliced tensor transducer for the block lengths $(s_1,\ldots,s_k)$ consists of ... a finite state set $Q$ ... transition sets $\Gamma_t \subseteq Q \times \{0,1\}^{s_t} \times Q \times \{0,1\}^{J_t} \times \F$ ..."
  - **Book**: §5.3 lines 342-365:
    > "Algorithm \textsc{ContractSlicedTensor} ... For $t=1,\ldots,k$ ... For all $(q,x,q',y,\eta)\in\Gamma_t$ ... return $\sum_{q\in Q} z(q)\cdot \omega_{\mathsf{fin}}(q)$."
  - **What's missing**: no generic `Q` / `Gamma_t` / final-weight transducer abstraction was found in the current implementation slice. The implementation has a specialized carry transition for contiguous offset slices.
  - **Implementation present**: `crates/akita-algebra/src/offset_eq.rs:25-28` defines `CarryTransition<F>` with two carry states; `crates/akita-algebra/src/offset_eq.rs:174-200` dispatches `eval_offset_eq_tensor` to aligned or carry-DP paths; `crates/akita-algebra/src/offset_eq.rs:328` starts the carry-DP implementation.
  - **Impact**: likely PERF-ONLY / extensibility, not current soundness, if all production calls are contiguous offset slices.
  - **Production readiness**: likely NICE-TO-HAVE; `ITERATION-TODO`: confirm every production call site is an offset slice.
  - **Recommended next step**: only implement a generic `SlicedTensorTransducer` trait if a non-offset slice caller exists or is planned; otherwise document this as a deliberate specialization.

### SCOPE-OUTSIDE-§5 changes

`ITERATION-TODO`: classify script-only security-analysis data, generated schedule table churn, bounded-L1 sampler internals, cleanup commits, clippy/test-housekeeping commits, and archived specs. These must be listed so reviewers do not evaluate them as direct §5 protocol deltas.

### SCAFFOLDING / DEAD CODE / BLOAT

`ITERATION-TODO`: re-open `audit.md` C-1..C-14 against current HEAD. Known candidates from the commit messages include cleanup commits `d8de222`, `0639189`, `f9d87e6`, and doc/test-helper cleanup around `defe58f`; no final disposition yet.

## Per-Figure-12-round walkthrough

| Round | Book line | Prover commit(s) + file:line | Verifier commit(s) + file:line | Transcript labels | Status |
|---|---|---|---|---|---|
| 1 | L826-L834: "For each block ... $\mathbf{v} := D ...$ ... send $\mathbf{v}$." | `ITERATION-TODO` | `ITERATION-TODO` | `ABSORB_PROVER_V` (`labels.rs`, existing/new status TODO) | `ITERATION-TODO` |
| 2 | L837-L842: "sample $\boldsymbol{\alpha} \gets \C^{2^{r/2}}$." | `d346d38` candidate; `crates/akita-challenges/src/stage1.rs:532-535` | same shared sampler; verifier call sites TODO | `CHALLENGE_STAGE1_FOLD_TENSOR_LEFT` | ALIGNED candidate |
| 3 | L844-L847: "No prover message. In the Fiat-Shamir transcript, Round 3 is implicit." | `d346d38` candidate; `crates/akita-challenges/src/stage1.rs:538-539` absorbs left digest | same shared sampler; verifier call sites TODO | `ABSORB_STAGE1_TENSOR_LEFT` | ALIGNED implementation extension candidate |
| 4 | L850-L862: "sample $\boldsymbol{\beta}$ ... compute $c_{p \| q} := \alpha_p \cdot \beta_q$." | `d346d38` candidate; `crates/akita-challenges/src/stage1.rs:540-546` | same shared sampler; verifier call sites TODO | `CHALLENGE_STAGE1_FOLD_TENSOR_RIGHT` | ALIGNED candidate |
| 5 | L865-L870: "Assemble $\mathbf{w}$ ... commit at the next level's parameters ... send $\mathbf{u}'$." | `ITERATION-TODO` | `ITERATION-TODO` | `ABSORB_SUMCHECK_W` candidate | `ITERATION-TODO` |
| 6 | L873-L882: "Ring switch ... sample $\alpha \gets \F_{q^k}$ ... evaluate at $X=\alpha$." | `ITERATION-TODO` | `ITERATION-TODO` | `CHALLENGE_RING_SWITCH` candidate | `ITERATION-TODO` |
| 7 | L885-L897: "sample $\tau_0,\tau_1$ ... batched sumcheck ... send $w_{\mathsf{eval}}$ and $s_{\mathsf{claim}}$." | `ITERATION-TODO` | `ITERATION-TODO` | `CHALLENGE_TAU0`, `CHALLENGE_TAU1`, `CHALLENGE_SUMCHECK_ROUND`, `ABSORB_SUMCHECK_S_CLAIM` candidates | `ITERATION-TODO` |
| 8 | L900-L919: "Compute $\lambda$ ... degree-$2$ sumcheck ... close deferred output equality..." | `ITERATION-TODO` | `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:124-147`; prover TODO | `CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND`, `CHALLENGE_SUMCHECK_BATCH_REL` | ALIGNED candidate; ordering TODO |

## Per-§5-subsection completeness matrix

| Subsection | Total items in book | ALIGNED | DRIFT | GAP | Coverage |
|---|---:|---:|---:|---:|---:|
| §5.1 Problem and setup | narrative | 0 | 0 | 0 | `ITERATION-TODO` |
| §5.2 Tensor stage-1 challenges | `ITERATION-TODO` | 1 seeded | 0 seeded | 0 seeded | `ITERATION-TODO` |
| §5.3 Automaton contraction | `ITERATION-TODO` | 1 seeded | 0 seeded | 1 seeded | `ITERATION-TODO` |
| §5.4 Claim-reduction sumcheck | `ITERATION-TODO` | 2 seeded | 0 seeded | 0 seeded | `ITERATION-TODO` |
| §5.5 Tiered commitment design | `ITERATION-TODO` | 1 seeded | 1 candidate | 0 seeded | `ITERATION-TODO` |
| §5.6 Combined protocol (Figure 12) | 8 rounds + output | 3 candidates | 0 seeded | 0 seeded | `ITERATION-TODO` |
| §5.7 Security analysis | `ITERATION-TODO` | 0 | 0 | 0 | `ITERATION-TODO` |
| §5.8 Concrete instantiation | `ITERATION-TODO` | 0 | 1 re-opened | 0 | `ITERATION-TODO` |

## Commit-by-commit walkthrough (chronological)

The branch has `155` commits on top of the merge-base: `152` non-merge commits plus `3` merge commits (`ebb7c93`, `e94edeb`, `6bbaaec`). Iteration 3 extended the complete spine below. Rows marked `PRELIMINARY` are not accepted final classifications until the relevant slice audit has attached file:line + book-line evidence.

| # | Hash | Date | Title | §5 subsection | Class | Notes |
|---:|---|---|---|---|---|---|
| 1 | `d7dd31e` | 2026-05-04 | Implemented bounded challenges | §5.2 / §5.7 | SCAFFOLDING | PRELIMINARY: sparse challenge-family support for challenge space `C`; needs entropy/norm evidence. |
| 2 | `e44fe69` | 2026-05-04 | chore(challenges): clean up unused sampler scaffolding | n/a | SCAFFOLDING | PRELIMINARY cleanup. |
| 3 | `657c864` | 2026-05-04 | test(challenges): move sparse-challenge integration tests to akita-challenges | §5.2 / §5.7 | TEST | PRELIMINARY test relocation. |
| 4 | `5052abb` | 2026-05-04 | refactor(challenges): own SparseChallenge / SparseChallengeConfig | §5.2 public API | SCAFFOLDING | PRELIMINARY API foundation; verify public surface. |
| 5 | `d6acaa3` | 2026-05-05 | Remove unused code | n/a | SCAFFOLDING | PRELIMINARY cleanup. |
| 6 | `ebb7c93` | 2026-05-05 | Merge remote-tracking branch 'origin/main' into feat/l1-bound-challenges | n/a | MERGE | Merge commit; no independent §5 classification. |
| 7 | `9c8e1ac` | 2026-05-05 | refactor(challenges): retire SplitRing, switch fp128 D=64 to ExactShell | §5.7 / §5.8 | SCAFFOLDING | PRELIMINARY security-sensitive challenge-family cutover; compare book lines 233-240 and 1045-1057. |
| 8 | `fd9cc9d` | 2026-05-05 | refactor(challenges): drop dead public API, shrink akita-challenges surface | n/a | SCAFFOLDING | PRELIMINARY API cleanup. |
| 9 | `44398ee` | 2026-05-05 | Fix clippy warning | n/a | SCAFFOLDING | Mechanical. |
| 10 | `e6f3c6a` | 2026-05-05 | refactor(challenges): split akita-challenges by audience (type / config / sampler) | §5.2 API | SCAFFOLDING | PRELIMINARY module/API shape. |
| 11 | `cb80a65` | 2026-05-05 | refactor(challenges): hard-code BoundedL1Ball sampler to (D=32, M=8, B=121) | §5.7 | SCAFFOLDING | PRELIMINARY challenge-family preset support. |
| 12 | `8325bce` | 2026-05-05 | refactor(challenges): drop Wide, store WAYS as u128 with one fewer row | §5.2 support | SCAFFOLDING | PRELIMINARY sampler implementation detail. |
| 13 | `e1f0c0a` | 2026-05-05 | refactor(challenges): use i8 for SparseChallenge coefficients and flatten unrank scan | §5.2 support | SCAFFOLDING | PRELIMINARY sparse-challenge representation/API delta. |
| 14 | `75a9abc` | 2026-05-05 | refactor(challenges): clean up bounded-L1 sampler internals | n/a | SCAFFOLDING | PRELIMINARY cleanup. |
| 15 | `daefb21` | 2026-05-05 | docs(challenges): fix rustdoc intra-doc link warnings | n/a | DOCS | Rustdoc cleanup. |
| 16 | `f292c65` | 2026-05-05 | refactor(challenges): rename samplers to `_challenge` and tighten bounded-L1 docs | §5.2 API/docs | SCAFFOLDING | PRELIMINARY public API/name delta. |
| 17 | `6f6e211` | 2026-05-05 | test(challenges): pin bounded-L1 DP recurrence and decode-rank injectivity | §5.2 / §5.7 | TEST | PRELIMINARY sampler invariant coverage. |
| 18 | `69cad1f` | 2026-05-05 | perf(challenges): optimize bounded-L1 sparse sampling | §5.2 support | SCAFFOLDING | PRELIMINARY performance-only if semantically equivalent. |
| 19 | `e3af552` | 2026-05-05 | refactor(challenges): make bounded-L1 config a fixed preset | §5.2 / §5.7 | SCAFFOLDING | PRELIMINARY challenge-space config shape. |
| 20 | `3d77a42` | 2026-05-05 | refactor(challenges): tighten bounded-L1 preset API | §5.2 API | SCAFFOLDING | PRELIMINARY public API delta. |
| 21 | `1ade67b` | 2026-05-05 | Update spec | n/a | DOCS | Prior spec update; content still to classify. |
| 22 | `d3321d9` | 2026-05-05 | fix(config): cover grouped setup envelopes | §5.4 / §5.5 | GAP-CLOSING | PRELIMINARY setup envelope sizing/type-shape impact. |
| 23 | `dd5889b` | 2026-05-05 | refactor(transcript): cut over akita byte domains | Figure 12 transcript | SCAFFOLDING | PRELIMINARY transcript domain delta; verify byte-boundary order. |
| 24 | `5303938` | 2026-05-05 | perf(transcript): reduce label absorb overhead | transcript support | SCAFFOLDING | PRELIMINARY performance-only if transcript bytes unchanged. |
| 25 | `1ef0042` | 2026-05-05 | test(challenges): refresh bounded l1 vector | §5.2 / §5.7 | TEST | PRELIMINARY test vector delta. |
| 26 | `2dee391` | 2026-05-06 | perf(prover): use rotated path for dense fold challenges | §5.2 support | SCAFFOLDING | PRELIMINARY; verify challenge semantics unchanged. |
| 27 | `6dca0c8` | 2026-05-06 | fix(protocol): harden challenge label and sparse eval handling | §5.2 / transcript | GAP-CLOSING | PRELIMINARY challenge-label hardening. |
| 28 | `d346d38` | 2026-05-06 | feat(protocol): add tensor challenge plumbing | §5.2 / Fig. 12 R2-R4 | ALIGNED | PRELIMINARY; seeded evidence in `sample_stage1_challenges`, lines 522-552. |
| 29 | `8fffbb1` | 2026-05-06 | docs(spec): plan tensor aggregate evaluator | §5.2 | DOCS | Planning doc; scope outside runtime. |
| 30 | `09bd2b4` | 2026-05-06 | feat(challenges): add exact tensor aggregate eval | §5.2 / §5.3 | ALIGNED | PRELIMINARY; exact aggregate evaluator for tensor challenge summaries. |
| 31 | `247af39` | 2026-05-06 | feat(verifier): store tensor challenge evals compactly | §5.2 | ALIGNED | PRELIMINARY verifier compact tensor eval storage. |
| 32 | `34b7198` | 2026-05-06 | feat(verifier): decompose tensor challenge summaries | §5.2 | ALIGNED | PRELIMINARY tensor-summary decomposition. |
| 33 | `bfc7597` | 2026-05-06 | feat(verifier): use tensor aggregate summaries in m-eval | §5.2 / §5.3 | ALIGNED | PRELIMINARY M-eval use of tensor summaries. |
| 34 | `1529152` | 2026-05-06 | test(pcs): cover tensor stage1 e2e | §5.2 / Fig. 12 R2-R4 | TEST | Tensor stage-1 E2E coverage candidate. |
| 35 | `41eb5a9` | 2026-05-06 | fix(schedule): use effective challenge mass for batched roots | §5.7 | GAP-CLOSING | PRELIMINARY tensor extraction/norm scheduling. |
| 36 | `bba4da6` | 2026-05-06 | bench(challenges): compare tensor aggregate evaluators | §5.2 | SCOPE-OUTSIDE | Bench-only evidence. |
| 37 | `c052fc1` | 2026-05-06 | docs(spec): record tensor aggregate validation | §5.2 | DOCS | Validation notes. |
| 38 | `7b91d8f` | 2026-05-06 | fix(protocol): harden tensor stage1 gating | §5.2 | GAP-CLOSING | PRELIMINARY gate/hardening. |
| 39 | `de81b3c` | 2026-05-06 | fix(types): account for tensor extraction margins | §5.7 | GAP-CLOSING | PRELIMINARY MSIS norm margin support. |
| 40 | `e94edeb` | 2026-05-06 | Merge origin/main into feat/tensor-challenges | n/a | MERGE | Merge commit; no independent §5 classification. |
| 41 | `6ae3fbb` | 2026-05-07 | fix(protocol): harden tensor stage1 production gates | §5.2 / §5.8 | GAP-CLOSING | PRELIMINARY production gating. |
| 42 | `6195c76` | 2026-05-07 | chore(planner): report tensor extraction buckets | §5.7 | SCAFFOLDING | Planner/security reporting support. |
| 43 | `51a875c` | 2026-05-07 | fix(config): gate generated tensor schedules | §5.8 | GAP-CLOSING | PRELIMINARY generated schedule gating. |
| 44 | `0e44b61` | 2026-05-07 | test(protocol): cover dense tensor stage1 e2e | §5.2 | TEST | Dense tensor E2E coverage. |
| 45 | `5efaa5d` | 2026-05-07 | test(protocol): reject tampered tensor stage1 claim | §5.2 / §5.7 | TEST | Tamper rejection coverage. |
| 46 | `b6b777c` | 2026-05-07 | fix(prover): remove redundant headroom check | n/a | SCAFFOLDING | Cleanup. |
| 47 | `5ca5692` | 2026-05-07 | perf(challenges): stack allocate tensor aggregate buffers | §5.2 | SCAFFOLDING | Perf-only if equivalent. |
| 48 | `3578242` | 2026-05-07 | perf(verifier): reuse tensor carry weight buffers | §5.3 | SCAFFOLDING | Perf support for tensor carry evaluation. |
| 49 | `6340658` | 2026-05-07 | perf(verifier): share tensor carry summaries across claims | §5.3 | SCAFFOLDING | Perf support for tensor carry evaluation. |
| 50 | `6bbaaec` | 2026-05-07 | Merge origin/main into feat/tensor-challenges | n/a | MERGE | Merge commit; no independent §5 classification. |
| 51 | `7d267bc` | 2026-05-07 | test(pcs): forward tensor config field roles | §5.2 / §5.8 | TEST | Config/test coverage. |
| 52 | `b4dbbd5` | 2026-05-07 | bench(pcs): compare flat and tensor verifier replay | §5.8 | SCOPE-OUTSIDE | Bench-only. |
| 53 | `0cbd944` | 2026-05-07 | bench(pcs): expand stage1 verifier matrix | §5.8 | SCOPE-OUTSIDE | Bench-only. |
| 54 | `dca90e8` | 2026-05-07 | bench(pcs): add larger tensor verifier cases | §5.8 | SCOPE-OUTSIDE | Bench-only. |
| 55 | `17495f5` | 2026-05-07 | bench(pcs): retime tensor stage1 schedules | §5.8 | SCOPE-OUTSIDE | Bench-only. |
| 56 | `1168e74` | 2026-05-07 | bench(pcs): print verifier proof metadata | §5.8 | SCOPE-OUTSIDE | Bench-only. |
| 57 | `e3f4ade` | 2026-05-07 | docs(spec): add fourth-root verifier audit reports | n/a | DOCS | Prior audits; revalidated separately. |
| 58 | `1251424` | 2026-05-07 | feat(verifier): split prepared M eval by setup dependency | §5.4 | ALIGNED | PRELIMINARY M-table decomposition. |
| 59 | `997f5ab` | 2026-05-07 | feat(types): expose setup matrix polynomial view | §5.4 | ALIGNED | PRELIMINARY enveloping/setup polynomial view. |
| 60 | `2a4df12` | 2026-05-07 | feat(sumcheck): add setup claim reduction prototype | §5.4 / Fig. 12 R8 | ALIGNED | PRELIMINARY degree-2 setup claim sumcheck. |
| 61 | `cc552d5` | 2026-05-07 | test(verifier): bridge M eval split to setup claim proof | §5.4 | TEST | Unit bridge coverage. |
| 62 | `fdff19c` | 2026-05-07 | feat(types): add optional setup claim proof payload | §5.4 / wire type | ALIGNED | PRELIMINARY proof type-shape delta. |
| 63 | `162747f` | 2026-05-07 | feat(config): add setup claim reduction opt-in | §5.4 / §5.8 | ALIGNED | PRELIMINARY config opt-in. |
| 64 | `6132284` | 2026-05-07 | feat(prover): widen centered fold accumulators | n/a | SCOPE-OUTSIDE | Arithmetic robustness; not directly §5 unless needed by tiered rows. |
| 65 | `0d2c294` | 2026-05-07 | feat(config): cut generated schedules to tensor stage1 | §5.2 / §5.8 | ALIGNED | PRELIMINARY generated schedule update. |
| 66 | `692cc8f` | 2026-05-07 | feat(verifier): reduce setup claims over setup coordinates | §5.4 | ALIGNED | PRELIMINARY setup-coordinate reduction. |
| 67 | `d2735ec` | 2026-05-07 | fix(config): validate tensor schedules across setup capacities | §5.8 | GAP-CLOSING | PRELIMINARY schedule-capacity validation. |
| 68 | `49e11c8` | 2026-05-08 | Tune tensor stage1 schedules for prover cost | §5.8 | SCOPE-OUTSIDE | Cost tuning; verify no protocol drift. |
| 69 | `9451f22` | 2026-05-12 | feat(prover,verifier): scaffold setup claim-reduction sumcheck | §5.4 / Fig. 12 R8 | ALIGNED | PRELIMINARY setup-claim scaffold. |
| 70 | `aa01e37` | 2026-05-12 | feat(prover,verifier): wire setup claim-reduction into stage-2 | §5.4 / Fig. 12 R7-R8 | ALIGNED | PRELIMINARY stage-2 integration. |
| 71 | `b1726ad` | 2026-05-12 | test(pcs): cover claim reduction at recursive fold levels | §5.4 / §5.6 | TEST | Recursive CR coverage. |
| 72 | `11ce32e` | 2026-05-12 | test(types): initialize `use_setup_claim_reduction` in unit-test fixtures | §5.4 | TEST | Fixture maintenance. |
| 73 | `8560faa` | 2026-05-12 | perf(verifier): algebraic-only m(r_x) path + claim-reduction benches | §5.4 | ALIGNED | PRELIMINARY algebraic/setup decomposition. |
| 74 | `655990a` | 2026-05-12 | docs(specs): tensor-everywhere implementation plan | n/a | DOCS | Planning doc. |
| 75 | `75a221a` | 2026-05-12 | perf(verifier): cache batched_verify schedules on AkitaVerifierSetup | §5.8 support | SCAFFOLDING | Perf/cache support; cache-key audit needed. |
| 76 | `c836302` | 2026-05-12 | perf(verifier): structured w_setup evaluator for claim-reduction sumcheck | §5.4 | ALIGNED | PRELIMINARY structured evaluator. |
| 77 | `d60f17f` | 2026-05-12 | perf(types): cache eq tables and bound by live prefix in setup MLE | §5.4 support | SCAFFOLDING | Perf support. |
| 78 | `bb80bd8` | 2026-05-12 | docs(specs): record Phase A/C/D-light verifier bench results | n/a | DOCS | Bench notes. |
| 79 | `3c03822` | 2026-05-12 | docs(specs): record prover/verifier comparison vs main baseline | n/a | DOCS | Bench notes. |
| 80 | `ab41530` | 2026-05-12 | docs(specs): add recursive-S opening plan for fourth-root verifier | §5.6 | DOCS | Design plan for book lines 940-953. |
| 81 | `52fd7f3` | 2026-05-12 | docs(specs): pivot recursive-S plan to batched-MLE evaluation | §5.6 | DOCS | Design pivot. |
| 82 | `c02bef4` | 2026-05-12 | docs(specs): Phase G.0 negative result — naive batched MLE is slower | §5.8 | DOCS | Negative benchmark/design record. |
| 83 | `85efcaa` | 2026-05-12 | docs(specs): pivot to Phase K hybrid per-level stage-1 shape | §5.8 | DOCS | Design pivot. |
| 84 | `6579d33` | 2026-05-12 | feat(types,pcs): Phase K.0 — hand-built mixed-shape stage-1 schedule works E2E | §5.2 / §5.8 | SCAFFOLDING | PRELIMINARY hybrid schedule experiment. |
| 85 | `9bd2d0c` | 2026-05-12 | feat(planner): Phase K.1 — per-level stage-1 shape search | §5.8 | SCAFFOLDING | Planner search experiment; likely outside final §5 path. |
| 86 | `9ddb939` | 2026-05-12 | feat(pcs): Phase K.4 — hybrid planner-search bench + cascade-bug docs | §5.8 | SCOPE-OUTSIDE | Bench/docs. |
| 87 | `acb5dad` | 2026-05-13 | fix(planner,config): Phase K.1 fix-up — shape-aware recursive layout | §5.8 | GAP-CLOSING | PRELIMINARY schedule correctness. |
| 88 | `9f332e5` | 2026-05-13 | feat(planner): Phase K.7 — HACHI_PLANNER_S1_WEIGHT env-var knob for tuning | n/a | SCAFFOLDING | Prior audit S-7; verify current status/staleness. |
| 89 | `47a5385` | 2026-05-13 | docs(specs),test(pcs): Phase K.5 — Fiat-Shamir audit for mixed shapes | §5.7 | DOCS | Security/test support. |
| 90 | `b8bf437` | 2026-05-13 | docs(specs): Phase K.6 — apples-to-apples vs main baseline | §5.8 | DOCS | Bench notes. |
| 91 | `9089d66` | 2026-05-13 | fix(planner,config,types): restore 128-bit security baseline | §5.7 | GAP-CLOSING | Security baseline restoration; verify against `security_analysis.md` §§1-9. |
| 92 | `95e79c5` | 2026-05-13 | docs(specs): Phase D-full design — recursive S opening + tiered commitments | §5.4-§5.6 | DOCS | Design doc. |
| 93 | `ccbbb8e` | 2026-05-13 | Phase D-full v2 foundations (slices A through C.2.c partial) | §5.4-§5.6 | ALIGNED | PRELIMINARY tiered/setup/multiclaim foundation. |
| 94 | `a669f8b` | 2026-05-13 | Slice D: multi-group batched Hachi commit kernel + LP shape | §5.4-§5.6 | ALIGNED | PRELIMINARY split/joint commitment support. |
| 95 | `454409f` | 2026-05-13 | Slice E: per-handle / per-claim LevelParams plumbing | §5.6 | ALIGNED | PRELIMINARY mixed next-level witness plumbing. |
| 96 | `ce8ecf0` | 2026-05-13 | Slice F.1: routes_recursively flag on verify_setup_claim_reduction | §5.6 output | ALIGNED | PRELIMINARY recursive S routing control. |
| 97 | `2091fee` | 2026-05-13 | docs(phase-d-full): handoff status after slices D + E + F.1 | n/a | DOCS | Handoff doc. |
| 98 | `c6a5524` | 2026-05-14 | Slice F.2-F.5 + Slice G infrastructure on feat/tensor-challenges | §5.4-§5.6 | ALIGNED | PRELIMINARY infrastructure; broad diff needs decomposition. |
| 99 | `64af8ab` | 2026-05-14 | Slice G unit tests: tier-aware proof_size helpers | §5.5 / §5.8 | TEST | Proof-size helper tests. |
| 100 | `1d85169` | 2026-05-14 | Slice G rustdoc: tiered helper examples on the public surface | §5.5 | DOCS | Public surface docs. |
| 101 | `05f79bb` | 2026-05-14 | Slice G prep: PreparedMEval tier_setup_params + FlatMatrix chunk view | §5.4 / §5.5 | ALIGNED | PRELIMINARY type/view support. |
| 102 | `5ccde2a` | 2026-05-15 | Slice H staging: tiered routed-S architecture (Phases 1-4 + 6, partial 5) | §5.5 / §5.6 | ALIGNED | PRELIMINARY routed-S architecture; partial status important. |
| 103 | `22bf830` | 2026-05-15 | Task A: block-diagonal D_chunk/B_chunk MLE collapse in verifier eval | §5.5 | GAP-CLOSING | PRELIMINARY shared per-chunk MLE collapse. |
| 104 | `e84298f` | 2026-05-15 | Task B+C: remove phantom meta_* rows; the meta tier is a regular group | §5.5 | GAP-CLOSING | PRELIMINARY 10-group cleanup. |
| 105 | `896694c` | 2026-05-15 | verifier: align num_eval_rows with prover's per-GROUP y_ring count | §5.5 / §5.6 | GAP-CLOSING | PRELIMINARY prover/verifier row-shape alignment. |
| 106 | `bf3d84c` | 2026-05-15 | verifier: per-chunk MLE openings, w_ring count fix, invariant tests | §5.5 / §5.6 | GAP-CLOSING | PRELIMINARY invariant/test support. |
| 107 | `cb36143` | 2026-05-15 | prover: tier-aware D-row quotient + cross-check tests | §5.5 | GAP-CLOSING | PRELIMINARY D-row quotient alignment. |
| 108 | `d8de222` | 2026-05-18 | WIP: surgical cleanup of bloat/scaffolding outside 4th-root scope | n/a | SCAFFOLDING | Cleanup; cross-reference `audit.md` C-items. |
| 109 | `586d763` | 2026-05-18 | verifier: structured grouped m_setup eval + per-level cascade tier | §5.4 / §5.5 | ALIGNED | PRELIMINARY grouped setup evaluator. |
| 110 | `831ccfc` | 2026-05-19 | verifier: cache + NTT-accelerate preprocessed C_S per book Fig. 12 | §5.6 / §5.8 | ALIGNED | PRELIMINARY C_S preprocessed input/perf. |
| 111 | `d436922` | 2026-05-19 | planner: force-route cascade L1 per book §5.8 line 1170 | §5.8 | DRIFT | PRELIMINARY cost-model/schedule-selection drift candidate; later commits may close. |
| 112 | `0d8b44e` | 2026-05-19 | planner: model tiered M-table 3-group layout in setup field length | §5.5 / §5.8 | GAP-CLOSING | PRELIMINARY cost model alignment. |
| 113 | `e7c66d6` | 2026-05-19 | test housekeeping: fix pre-existing clippy/build errors for --all-targets | n/a | SCAFFOLDING | Test/build housekeeping. |
| 114 | `877e145` | 2026-05-19 | test: measure cascade verifier speedup at NV=22 dense (book Table 1141-1158) | §5.8 | TEST | Performance measurement. |
| 115 | `48cd8e9` | 2026-05-19 | verifier: share NttSlotCache across cascade derivations (5.2x verify speedup) | §5.8 | SCAFFOLDING | Perf/cache support. |
| 116 | `7c2bef8` | 2026-05-19 | test: measure amortized verify per book Fig. 12 (cold + cache-hit split) | §5.8 | TEST | Performance measurement. |
| 117 | `4a4c40b` | 2026-05-19 | verifier: pre-populate NTT slot cache at setup_verifier per book Fig. 12 | §5.6 / §5.8 | ALIGNED | PRELIMINARY preprocessed verifier input timing. |
| 118 | `0c47316` | 2026-05-19 | test: onehot cascade speedup measurement (D=64) at NV=28 | §5.8 | TEST | Performance measurement. |
| 119 | `8e87160` | 2026-05-19 | verifier: pre-populate tiered_s_cache at setup time (final Fig. 12 alignment) | §5.6 | GAP-CLOSING | Prior DRIFT-3 closure candidate. |
| 120 | `0639189` | 2026-05-19 | prover: remove diagnostic level==1 harness + stale rejection doc (C-1, C-11) | n/a | SCAFFOLDING | Bloat cleanup. |
| 121 | `f17b0dc` | 2026-05-19 | verifier: schedule-vs-proof + routes-recursively defense-in-depth (S-1, S-5, C-14) | §5.6 / §5.7 | GAP-CLOSING | Defense-in-depth closure candidate. |
| 122 | `f9d87e6` | 2026-05-19 | test-helpers: gate eq_weighted_table siblings + split_eval_table (C-5, C-7) | n/a | SCAFFOLDING | Bloat/test-helper cleanup. |
| 123 | `30ed738` | 2026-05-19 | test: add tiered_rejects_tampered_next_w_commitment (B-3 / S-3) | §5.5 / §5.7 | TEST | Tamper rejection coverage. |
| 124 | `defe58f` | 2026-05-19 | prover: document tiered handle material's shared structure (C-2) | §5.5 | DOCS | Source docs for tiered material structure. |
| 125 | `c9d9904` | 2026-05-19 | config: flip production fp128 presets to claim-reduction on (B-1) | §5.4 / §5.8 | ALIGNED | PRELIMINARY production cutover; security/perf audit required. |
| 126 | `d7820f6` | 2026-05-19 | docs: refresh security analysis with §5 post-Phase-D-full v2 re-audit | §5.7 | DOCS | Security analysis §10. |
| 127 | `13812f8` | 2026-05-19 | test: extend tiered_grouped_m_rows to cover A-rows per book §5.4 (C-13) | §5.5 | TEST | A-row coverage. |
| 128 | `f2c7b9b` | 2026-05-19 | planner: widen level_proof_bytes to shape-aware CR + tiered cost model (S-8) | §5.5 / §5.8 | GAP-CLOSING | Cost model proof-size alignment. |
| 129 | `5cb0e47` | 2026-05-19 | docs: book §5 vs implementation protocol drift audit | n/a | DOCS | Prior structural audit. |
| 130 | `920086f` | 2026-05-19 | types: top-level MRowLayout 10-vs-15 group doc (DRIFT-2) | §5.5 / §5.6 | GAP-CLOSING | Prior DRIFT-2 closure candidate. |
| 131 | `d0ea827` | 2026-05-19 | config: document tier-shape policy default f=2 (DRIFT-4 / SCOPE-5) | §5.5 / §5.8 | DRIFT | PRELIMINARY intentional production-default drift vs book f=8; documented. |
| 132 | `f18418f` | 2026-05-19 | stage2: sample two batching coefficients γ_range, γ_rel (DRIFT-1) | §5.6 R8 | GAP-CLOSING | Book Round 8 two-coefficient form. |
| 133 | `ce01879` | 2026-05-19 | field: opt-in verifier op-counter (audit GAP-2 / SCOPE-4) | §5.8 | SCOPE-OUTSIDE | Measurement instrumentation, not protocol. |
| 134 | `b4b02c7` | 2026-05-19 | phase5/item1: shared per-chunk matrix collapse in setup col envelope | §5.5 | GAP-CLOSING | PRELIMINARY shared matrix collapse support. |
| 135 | `0cf214d` | 2026-05-19 | phase5/(a): chunk_lp + meta_lp B-role SIS rank shrink (release-mode gated) | §5.5 / §5.7 | GAP-CLOSING | PRELIMINARY SIS/rank/cascade alignment. |
| 136 | `012d172` | 2026-05-19 | phase5: document Item (b) and Item (c) scope blockers post-Item-(a) | §5.5 | DOCS | Scope blockers documented. |
| 137 | `6c9c38f` | 2026-05-19 | phase5/(b): chunk-axis amortised W+T blocks via eval_offset_eq_tensor | §5.3 / §5.5 | GAP-CLOSING | PRELIMINARY automaton/cascade amortisation. |
| 138 | `693a649` | 2026-05-19 | phase5/drift1: shallow batched chunk mat-vec (one A-step over k blocks) | §5.5 / §5.8 | GAP-CLOSING | PRELIMINARY chunk mat-vec perf/cost alignment. |
| 139 | `f90a056` | 2026-05-19 | phase5/drift3: add CHALLENGE_TIERED_CHUNK_AGGREGATION transcript label | §5.5 / transcript | ALIGNED | PRELIMINARY extension; verify ordering. |
| 140 | `f5e3ee3` | 2026-05-19 | phase5/drift3: γ-fold k chunks claims to ONE aggregated chunks claim | §5.5 / §5.7 | GAP-CLOSING | PRELIMINARY shared chunk claim aggregation; security §11. |
| 141 | `883993d` | 2026-05-19 | phase5/drift4: investigate force-routing retirement; defer to wire-cost gap closure | §5.8 | DRIFT | Documents cost-model drift; superseded by later objective commits. |
| 142 | `2866076` | 2026-05-19 | specs: close SCOPE-3 + update GAP-3 for Drift 3 + Drift 4 | n/a | DOCS | Prior audit update. |
| 143 | `5d0d1b0` | 2026-05-19 | specs/drift4: expand GAP-3 analysis with cleartext-cost asymmetry finding | n/a | DOCS | Prior audit update. |
| 144 | `9ddf99b` | 2026-05-19 | phase5/drift4-step1: add setup storage objective | §5.8 | GAP-CLOSING | PRELIMINARY natural cascade discovery cost-model closure. |
| 145 | `9d33e27` | 2026-05-19 | phase5/drift4-step2: charge cleartext setup discharge | §5.8 | GAP-CLOSING | PRELIMINARY symmetric cleartext cost. |
| 146 | `fd0ddb3` | 2026-05-19 | phase5/drift4-step4: retire force-routing gates | §5.8 | GAP-CLOSING | PRELIMINARY closes force-route drift if tests/probe confirm. |
| 147 | `4f90979` | 2026-05-19 | phase5/drift4-step3: allow objective-driven batched root params | §5.8 | GAP-CLOSING | PRELIMINARY objective-driven root scheduling. |
| 148 | `71d7eef` | 2026-05-19 | phase5/drift4-step5: fix batched root verification | §5.8 | GAP-CLOSING | PRELIMINARY schedule/proof alignment. |
| 149 | `26df279` | 2026-05-19 | phase5/polish-drift1: stage2 module doc + test naming follow γ_range/γ_rel form | §5.6 R8 | DOCS | DRIFT-1 polish. |
| 150 | `e2fc865` | 2026-05-19 | specs/audit: mark DRIFT-1/2/4 + GAP-2 + SCOPE-4/5 CLOSED with closure register | n/a | DOCS | Prior audit closure register. |
| 151 | `7c846fb` | 2026-05-19 | specs/audit: close DRIFT-3 via option (b) + document option (a) upgrade path | n/a | DOCS | Prior audit closure register. |
| 152 | `5106c35` | 2026-05-19 | specs/audit: seed full §5 diff audit loop | n/a | DOCS | Creates this audit doc/scratchpad; not protocol code. |
| 153 | `81cceec` | 2026-05-19 | phase5/setup-claim: align reducer with book shape | §5.4 / Fig. 12 R8 | GAP-CLOSING | NEW in iteration 2; must audit against book lines 599-626 and setup reducer code. |
| 154 | `16595eb` | 2026-05-19 | specs/audit: expand full section 5 diff spine | n/a | DOCS | Iteration-2 audit-doc expansion; not protocol code. |
| 155 | `dd53025` | 2026-05-20 | phase5/setup-claim: fix routed schedule gates | §5.4 / §5.8 | DRIFT | Iteration-3 evidence: fixes gates for compact row/coeff setup-claim routing, but current cascade sentinels assert `routing_count == 0` for the headline `(8,4)` path; see DRIFT-1 candidate. |

## Public API surface delta

`ITERATION-TODO`: seed with `git diff main..HEAD -- '**/lib.rs'` and `git diff main..HEAD -- '**/mod.rs'`, then verify every `pub` symbol at its definition. Known candidate areas: `SparseChallenge`, `Stage1ChallengeShape::Tensor`, `MRowLayout`, `SetupClaimReductionPayload`, `TieredSetup*`, `RecursivePolyHandle`, planner schedule types, and verifier prepopulation APIs.

## Transcript label delta

Initial label deltas to verify:

| Label | File:line | Figure 12 mapping | Status |
|---|---|---|---|
| `CHALLENGE_SUMCHECK_BATCH_REL` | `crates/akita-transcript/src/labels.rs:55` | Figure 12 Round 8, `γ_rel` in lines 912-919 | ALIGNED candidate |
| `CHALLENGE_STAGE1_FOLD_TENSOR_LEFT` | `crates/akita-transcript/src/labels.rs:68` | Figure 12 Round 2, lines 837-842 | ALIGNED candidate |
| `ABSORB_STAGE1_TENSOR_LEFT` | `crates/akita-transcript/src/labels.rs:71` | Figure 12 implicit Round 3, lines 844-847 | ALIGNED extension candidate |
| `CHALLENGE_STAGE1_FOLD_TENSOR_RIGHT` | `crates/akita-transcript/src/labels.rs:73` | Figure 12 Round 4, lines 850-854 | ALIGNED candidate |
| `CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND` | `crates/akita-transcript/src/labels.rs:88` | Figure 12 Round 8, lines 907-911 | ALIGNED candidate |
| `CHALLENGE_TIERED_CHUNK_AGGREGATION` | `crates/akita-transcript/src/labels.rs:103` | §5.5 chunk aggregation extension; not explicit in Figure 12 | `ITERATION-TODO`: classify as ALIGNED extension or DRIFT |

`ITERATION-TODO`: verify `all_labels()` order at `crates/akita-transcript/src/labels.rs:106-140` against actual consumption order in the prover, not just declaration order.

## Type-shape delta

`ITERATION-TODO`: document every protocol-relevant struct/enum field or variant delta. Seed candidates:

| Type | Crate | Field change | Book justification | Soundness impact |
|---|---|---|---|---|
| `MRowLayout` | `akita-types` | 15 tiered row-family fields: `w_*`, `original_*`, `meta_*` | §5.5 lines 709-754 plus §5.6 lines 940-952 | likely NONE if documented and consistently used |
| `SetupClaimReductionPayload` | `akita-types` | `ITERATION-TODO` | §5.4 lines 599-626 | `ITERATION-TODO` |
| `LevelParams` | `akita-types` | `ITERATION-TODO` | §5.4 split commitment / §5.5 tiered shape | `ITERATION-TODO` |
| `TieredSetupParams` / `TieredSetupCommitments` / `TieredSetupCacheKey` / `TieredSetupProverExtras` | mixed | `ITERATION-TODO` | §5.5 lines 686-754 | `ITERATION-TODO` |
| `RecursivePolyHandle` / `RecursiveHandlePoly` | mixed | `ITERATION-TODO` | §5.6 lines 940-953 | `ITERATION-TODO` |

## Test coverage delta

`ITERATION-TODO`: enumerate every new test file and function. Seed files from changed-file inventory:

| Test file | What it likely verifies | §5 element |
|---|---|---|
| `crates/akita-pcs/tests/tensor_stage1_e2e.rs` | tensor stage-1 E2E and tamper behavior | §5.2 / Figure 12 rounds 2-4 |
| `crates/akita-pcs/tests/setup_claim_reduction_e2e.rs` | setup claim-reduction E2E | §5.4 / Figure 12 round 8 |
| `crates/akita-pcs/tests/tiered_setup_e2e.rs` | tiered/cascade proof and speed measurements | §5.5 / §5.8 |
| `crates/akita-pcs/tests/multi_group_commit.rs` | multi-group commitment kernel | §5.4 / §5.5 |
| `crates/akita-pcs/tests/recursive_multi_claim.rs` | recursive multi-claim opening | §5.6 output recursion |
| `crates/akita-challenges/tests/sparse_challenge.rs` | challenge sampler vectors/invariants | §5.2 / §5.7 support |

## Security delta vs `specs/security_analysis.md`

`ITERATION-TODO`: read all of §§1-9 and §10-§11 against HEAD. Initial security-sensitive deltas to audit:

- Challenge-family entropy and MSIS norm penalty versus book §5.2 lines 200-251 and §5.7 lines 1045-1062.
- Claim-reduction sumcheck knowledge error versus book §5.7 lines 975-986 and 1026-1043.
- Tiered chunk aggregation (`CHALLENGE_TIERED_CHUNK_AGGREGATION`) and any extra batching error not in the original book.
- Planner/objective changes that touch cascade security only indirectly through shape selection.

## Cascade discovery walkthrough

Iteration-3 status: **re-opened**. The current HEAD has two different stories that must be reconciled before this section can be marked complete:

- The planner objective code still enumerates both `routes_setup_recursively = false` and `true`, and charges setup storage / cleartext discharge objective terms in `crates/akita-planner/src/schedule_params.rs:537-627` and `crates/akita-planner/src/schedule_params.rs:998-1091`.
- The current book-shaped setup reducer fixes `r_x` before the setup-side sumcheck, so the routed polynomial is compact row/coeff data (`crates/akita-verifier/src/protocol/setup_claim_reduction.rs:91-118`) rather than the older full row/col/coeff setup matrix.
- Test evidence says the headline cascade is not currently emitted: `crates/akita-pcs/tests/tiered_setup_e2e.rs:800-857` asserts `DenseCascadeCfg` has `(f_L0,f_L1)=(8,4)` policy but `routing_count == 0`; `crates/akita-pcs/tests/tiered_setup_e2e.rs:500-527` keeps the old positive `[8,4]` routed assertion ignored as a documented drift.

Current answers, pending the §5.6-§5.8 sub-audit:

1. Does the planner emit `(f_L0=8, f_L1=4)` for `DenseCascadeCfg`? **It exposes the per-level tier policy, but active schedule assertions at NV=22 expect zero recursive setup routes.**
2. Forced or natural? **Neither for the active sentinel path; it cleartext-discharges.** The code no longer appears to force the old full-S route, but stale comments in `crates/akita-config/src/proof_optimized.rs:923-949` still mention force-routing and must be updated or reconciled.
3. Smallest NV? **TODO** via `probe_cascade_schedules_extended`; current non-ignored sentinels mention dense default NV=19 and headline schedule-only NV=22, both with `routing_count == 0`.
4. Wall-clock cost? **TODO**; existing measurement tests need reclassification because a zero-route compact reducer is not the book's L0+L1 cascade.
5. Book Table 1141-1158 prediction? **Still 16x / 35x / 265x for T1+T2 @ L0+L1, but current HEAD has not yet shown that path active.**

## Production readiness verdict

1. **Cryptographically sound**: `ITERATION-TODO`.
2. **Protocol-aligned to book §5**: `ITERATION-TODO`.
3. **Production-ready performance**: `ITERATION-TODO`.

Top production blockers are not final. Current candidate blockers to revalidate: natural cascade discovery / force routing, general-vs-offset transducer scope, non-singleton verifier setup prepopulation timing, and security-analysis freshness after Phase 5 chunk aggregation.

## Methodology + reproducibility

Commands and evidence used in iteration 1:

- `git status --short --branch`
- `git rev-parse --abbrev-ref HEAD && git rev-parse HEAD`
- `git merge-base feat/tensor-challenges origin/main`
- `git rev-list --count <merge-base>..HEAD`
- `git diff --shortstat <merge-base>..HEAD`
- `git diff --dirstat=files,0 <merge-base>..HEAD`
- `git diff --numstat <merge-base>..HEAD`
- `git diff --name-status <merge-base>..HEAD`
- `git log --no-merges --pretty='%h%x09%ad%x09%s' --date=short <merge-base>..HEAD --reverse`
- `git log --merges --pretty='%h%x09%ad%x09%s' --date=short <merge-base>..HEAD --reverse`
- `ReadFile` on the book and prior audits.
- `rg` on tensor labels, `MRowLayout`, setup claim-reduction, and offset-eq evidence.

Additional commands/evidence used in iteration 3:

- `git status --short --branch`
- `git branch --show-current`
- `git rev-parse HEAD`
- `git merge-base feat/tensor-challenges origin/main`
- `git rev-list --count $(git merge-base feat/tensor-challenges origin/main)..HEAD`
- `git diff --shortstat $(git merge-base feat/tensor-challenges origin/main)..HEAD`
- `git log --pretty='%h %ad %s' --date=short $(git merge-base feat/tensor-challenges origin/main)..HEAD --reverse`
- `git diff --numstat $(git merge-base feat/tensor-challenges origin/main)..HEAD`
- `ReadFile` on `setup_claim_reduction.rs`, `stage2.rs`, `flow.rs`, `levels.rs`, `schedule_params.rs`, `proof_size.rs`, and `tiered_setup_e2e.rs`.
- Three read-only subagents were dispatched for §5.2-§5.3, §5.4-§5.5, and §5.6-§5.8. Their results are not merged yet in this doc revision.

Files not yet fully audited:

- All generated schedule tables under `crates/akita-types/src/generated/` are included in the diff count but have not yet been line-audited. They should be audited as planner/security outputs, not hand-written protocol code.
- `scripts/security_analysis/params.json` and `quadruples.json` dominate raw LOC and need reproducibility classification rather than code-path conformance.
- `akita-field` primitive changes have not yet been audited against a §5 contract except the op-counter candidate; most are likely measurement support or field plumbing.

With more time in later iterations, the commit table should be expanded first, then used as the spine for public API, type-shape, tests, transcript, and security deltas.
