# `feat/tensor-challenges` vs `main` — Deep §5 conformance audit

**Date**: 2026-05-19
**HEAD**: `7c846fb6cb5f2cf5483d7b6554dfe3720efaf48d`
**Merge base**: `4b0b86a946dca5124ddc1c0197bda7b73284a137`
**Commits on top of main**: 151 total (`148` non-merge + `3` merge commits)
**Diff stat**: `131 files changed, 148135 insertions(+), 5562 deletions(-)`
**Book ref**: `/home/giuseppe/lattice-jolt/sections/akita/5_fourth_root_verifier.tex`
**Scope**: every meaningful change on top of main, classified against book §5 / Figure 12.
**Methodology**: commit-by-commit walkthrough + file-by-file diff + book line cross-reference.

> Iteration status: this is the iteration-1 scaffold plus first-pass evidence.
> Anything marked `ITERATION-TODO` is intentionally not yet accepted as audited.
> The completion promise is false until all TODO markers are removed and every
> commit row has a final class.

## Executive summary

- **Audit inventory established**: branch HEAD is `7c846fb6`, merge-base is `4b0b86a9`, with `151` commits and `131` changed files on top of `origin/main`.
- **Diff shape is heavily audit/spec/generated skewed**: `scripts/` accounts for `108,492` net LOC, `specs/` for `5,830` net LOC, and generated schedule/security data are a major part of the raw insertion count. These need classification but should not be mistaken for protocol code.
- **Primary protocol crates touched**: `akita-prover` (`+5,653` net), `akita-verifier` (`+4,267` net), `akita-types` (`+4,591` net), `akita-pcs` tests (`+5,993` net), `akita-config` (`+1,784` net), `akita-planner` (`+1,044` net), `akita-challenges` (`+1,406` net), and `akita-algebra` (`+2,053` net).
- **Initial aligned evidence found**: tensor challenge left/right sampling and transcript digesting match §5.2 / Figure 12 rounds 2-4; setup claim-reduction uses a degree-2 sumcheck with the dedicated transcript label; `MRowLayout` documents the book's 10 tier groups plus the §5.6 joint-W extension.
- **Initial gap candidate confirmed for revalidation**: the implementation has an offset-slice carry-DP evaluator, but no general sliced tensor transducer API matching §5.3 Definition/Algorithm. This is likely a non-blocking gap if all production calls use offset slices only.
- **Initial drift register is not final**: prior drift closures in `specs/section5_protocol_drift_audit.md` must be rechecked against HEAD and commit history, especially cascade natural discovery, force-route retirement, chunk aggregation transcript binding, and `setup_verifier` prepopulation timing.
- **Production-readiness verdict is deferred** until the full commit chronology, public API delta, type-shape delta, test delta, security delta, and cascade measurement walkthrough are complete.

## Diff overview by crate

| Crate / area | LOC added | LOC removed | Net | Files touched | §5 subsections this area appears to implement |
|---|---:|---:|---:|---:|---|
| `crates/akita-algebra` | 2,053 | 0 | 2,053 | 4 | §5.3 offset-slice tensor contraction; NTT/cache support for verifier setup material |
| `crates/akita-challenges` | 1,412 | 6 | 1,406 | 5 | §5.2 tensor stage-1 challenge sampling and challenge-family security support |
| `crates/akita-config` | 1,987 | 203 | 1,784 | 7 | §5.5 / §5.8 preset routing, cascade configs, production claim-reduction enablement |
| `crates/akita-field` | 224 | 0 | 224 | 6 | §5.8 measurement support; opt-in verifier op counter |
| `crates/akita-pcs` | 6,040 | 47 | 5,993 | 13 | E2E tests and benches for §5.2, §5.4, §5.5, §5.8 |
| `crates/akita-planner` | 1,178 | 134 | 1,044 | 4 | §5.5 / §5.8 cascade schedule search and cost objective |
| `crates/akita-prover` | 6,505 | 852 | 5,653 | 21 | Figure 12 prover flow, recursive S routing, tiered setup material, claim-reduction prover |
| `crates/akita-scheme` | 59 | 23 | 36 | 2 | Setup/verifier orchestration and cache prepopulation |
| `crates/akita-setup` | 2 | 6 | -4 | 1 | Setup API integration; likely incidental |
| `crates/akita-sumcheck` | 464 | 50 | 414 | 4 | §5.4 / §5.6 sumcheck helpers and test-gated eq table support |
| `crates/akita-transcript` | 43 | 0 | 43 | 1 | Figure 12 transcript label delta |
| `crates/akita-types` | 8,547 | 3,956 | 4,591 | 21 | Public proof/layout/schedule/type-shape deltas for §5.4-§5.6 |
| `crates/akita-verifier` | 4,542 | 275 | 4,267 | 9 | Figure 12 verifier replay, setup claim-reduction, grouped M-eval, caches |
| `scripts` | 108,493 | 1 | 108,492 | 18 | Security analysis and planner audit tooling; mostly §5.7 / §5.8 support |
| `specs` | 5,839 | 9 | 5,830 | 13 | Prior audits, designs, and handoff docs to revalidate |
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

- **Setup-side claim-reduction verifier uses a degree-2 sumcheck and dedicated transcript label** (commits: `2a4df12`, `9451f22`, `aa01e37` require per-commit blame refinement)
  - **Book**: §5.4 lines 599-626:
    > "Define the scaling factor ... $\lambda := w_{\mathsf{eval}} \cdot \wt{\alpha}(r_y)$ ... Subtracting the verifier-computable algebraic part ... the remaining claim is ... proved by a degree-$2$ sumcheck over $\lceil \log_2 m_{\mathsf{row}} \rceil + \log_2 d$ variables ... The critical design choice: carry the scaled claim ... rather than dividing by $\lambda$."
  - **Implementation**: `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:124-147` calls `verify_sumcheck_rounds_only(..., 2, payload.m_setup_eval, ..., CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND)`.
  - **Implementation**: `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:149-156` checks `weight_at_point * payload.s_opening_value == payload.m_setup_eval` before the optional cleartext-vs-recursive closing path.
  - **Why aligned**: The verifier performs the book's short setup-side sumcheck with degree 2 and carries a scaled setup claim instead of dividing by the possibly-zero scaling factor.

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
| §5.4 Claim-reduction sumcheck | `ITERATION-TODO` | 1 seeded | 0 seeded | 0 seeded | `ITERATION-TODO` |
| §5.5 Tiered commitment design | `ITERATION-TODO` | 1 seeded | 1 candidate | 0 seeded | `ITERATION-TODO` |
| §5.6 Combined protocol (Figure 12) | 8 rounds + output | 3 candidates | 0 seeded | 0 seeded | `ITERATION-TODO` |
| §5.7 Security analysis | `ITERATION-TODO` | 0 | 0 | 0 | `ITERATION-TODO` |
| §5.8 Concrete instantiation | `ITERATION-TODO` | 0 | 1 candidate | 0 | `ITERATION-TODO` |

## Commit-by-commit walkthrough (chronological)

The branch has `151` commits on top of the merge-base: `148` non-merge commits plus `3` merge commits (`ebb7c93`, `e94edeb`, `6bbaaec`). This table is seeded by chronology but not final.

| # | Hash | Date | Title | §5 subsection | Class | Notes |
|---:|---|---|---|---|---|---|
| 1 | `d7dd31e` | 2026-05-04 | Implemented bounded challenges | §5.2 / §5.7 candidate | UNREVIEWED | Requires sparse challenge family audit against book challenge-space and security assumptions. |
| 2 | `e44fe69` | 2026-05-04 | chore(challenges): clean up unused sampler scaffolding | n/a candidate | UNREVIEWED | Likely SCAFFOLDING cleanup; verify diff. |
| 3 | `657c864` | 2026-05-04 | test(challenges): move sparse-challenge integration tests to akita-challenges | §5.2 / test candidate | UNREVIEWED | Test-only candidate. |
| 4 | `5052abb` | 2026-05-04 | refactor(challenges): own SparseChallenge / SparseChallengeConfig | §5.2 public API candidate | UNREVIEWED | Public API delta likely; verify symbols. |
| 5 | `d6acaa3` | 2026-05-05 | Remove unused code | n/a candidate | UNREVIEWED | Cleanup; verify no protocol removal. |
| 6 | `9c8e1ac` | 2026-05-05 | refactor(challenges): retire SplitRing, switch fp128 D=64 to ExactShell | §5.7 / §5.8 candidate | UNREVIEWED | Security-sensitive challenge-family cutover; compare to book L233-L240 and L1045-L1057. |
| 7 | `fd9cc9d` | 2026-05-05 | refactor(challenges): drop dead public API, shrink akita-challenges surface | n/a candidate | UNREVIEWED | Public API removal; classify for API delta. |
| 8 | `44398ee` | 2026-05-05 | Fix clippy warning | n/a | UNREVIEWED | likely TEST/SCAFFOLDING or mechanical. |
| 9 | `e6f3c6a` | 2026-05-05 | refactor(challenges): split akita-challenges by audience (type / config / sampler) | §5.2 API candidate | UNREVIEWED | Public API/module shape delta. |
| 10 | `cb80a65` | 2026-05-05 | refactor(challenges): hard-code BoundedL1Ball sampler to (D=32, M=8, B=121) | §5.7 candidate | UNREVIEWED | Challenge entropy / norm assumptions. |
| 11 | `8325bce` | 2026-05-05 | refactor(challenges): drop Wide, store WAYS as u128 with one fewer row | §5.2 support candidate | UNREVIEWED | Sparse sampler implementation detail. |
| 12 | `e1f0c0a` | 2026-05-05 | refactor(challenges): use i8 for SparseChallenge coefficients and flatten unrank scan | §5.2 support candidate | UNREVIEWED | Sparse challenge representation and API delta. |
| 13 | `75a9abc` | 2026-05-05 | refactor(challenges): clean up bounded-L1 sampler internals | n/a candidate | UNREVIEWED | likely scaffolding cleanup. |
| 14 | `daefb21` | 2026-05-05 | docs(challenges): fix rustdoc intra-doc link warnings | docs | UNREVIEWED | likely DOCS. |
| 15 | `f292c65` | 2026-05-05 | refactor(challenges): rename samplers to `_challenge` and tighten bounded-L1 docs | §5.2 API/docs candidate | UNREVIEWED | Public API/name delta. |
| 16 | `6f6e211` | 2026-05-05 | test(challenges): pin bounded-L1 DP recurrence and decode-rank injectivity | §5.2 / test candidate | UNREVIEWED | Test coverage delta. |
| 17 | `69cad1f` | 2026-05-05 | perf(challenges): optimize bounded-L1 sparse sampling | §5.2 support candidate | UNREVIEWED | Performance-only if semantically equivalent. |
| 18 | `e3af552` | 2026-05-05 | refactor(challenges): make bounded-L1 config a fixed preset | §5.2 / §5.7 candidate | UNREVIEWED | Challenge-space config shape. |
| 19 | `3d77a42` | 2026-05-05 | refactor(challenges): tighten bounded-L1 preset API | §5.2 API candidate | UNREVIEWED | Public API delta. |
| 20 | `1ade67b` | 2026-05-05 | Update spec | docs | UNREVIEWED | Prior spec update; read and classify. |
| 21 | `d3321d9` | 2026-05-05 | fix(config): cover grouped setup envelopes | §5.4 / §5.5 candidate | UNREVIEWED | Setup envelope sizing; type-shape impact. |
| 22 | `dd5889b` | 2026-05-05 | refactor(transcript): cut over akita byte domains | Figure 12 transcript candidate | UNREVIEWED | Transcript label/domain delta. |
| 23 | `5303938` | 2026-05-05 | perf(transcript): reduce label absorb overhead | transcript support candidate | UNREVIEWED | Verify no byte-boundary drift. |
| 24 | `1ef0042` | 2026-05-05 | test(challenges): refresh bounded l1 vector | §5.2 / test candidate | UNREVIEWED | Test vector delta. |
| 25 | `2dee391` | 2026-05-06 | perf(prover): use rotated path for dense fold challenges | §5.2 support candidate | UNREVIEWED | Verify challenge semantics unchanged. |
| 26-151 | `ITERATION-TODO` | 2026-05-06..2026-05-19 | Remaining 126 commits | mixed | UNREVIEWED | Next iterations must expand this into one row per commit. |

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

`ITERATION-TODO`: run or inspect `probe_cascade_schedules_extended` and classify the current state after commits `9ddf99b`, `9d33e27`, `fd0ddb3`, `4f90979`, `71d7eef`.

Open questions for the next slice:

1. Does the planner emit `(f_L0=8, f_L1=4)` for `DenseCascadeCfg` at the configured NVs?
2. Is that emission forced or naturally objective-driven?
3. What is the smallest NV where each cascade config emits the cascade?
4. What are cold and amortized verifier costs at NV=22 dense and NV=28 onehot?
5. How do these extrapolate against book Table lines 1141-1158?

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

Files not yet fully audited:

- All generated schedule tables under `crates/akita-types/src/generated/` are included in the diff count but have not yet been line-audited. They should be audited as planner/security outputs, not hand-written protocol code.
- `scripts/security_analysis/params.json` and `quadruples.json` dominate raw LOC and need reproducibility classification rather than code-path conformance.
- `akita-field` primitive changes have not yet been audited against a §5 contract except the op-counter candidate; most are likely measurement support or field plumbing.

With more time in later iterations, the commit table should be expanded first, then used as the spine for public API, type-shape, tests, transcript, and security deltas.
