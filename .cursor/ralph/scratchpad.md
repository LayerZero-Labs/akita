---
iteration: 2
max_iterations: 8
completion_promise: "Deep §5 conformance audit completed on `feat/tensor-challenges`. `specs/section5_full_diff_audit.md` exists at HEAD and contains: (1) commit-by-commit walkthrough of every commit between `git merge-base origin/main HEAD` and HEAD, each classified as ALIGNED / DRIFT / GAP-CLOSING / SCOPE-OUTSIDE / SCAFFOLDING / TEST / DOCS / REVERT with §5 subsection cross-reference; (2) DRIFT register with file:line + exact book line citation + soundness-impact + production-blocker status for every drift; (3) GAP register with same fidelity for every §5 element absent from the implementation; (4) per-Figure-12-round implementation map (rounds 1-8) with commit hashes + prover/verifier file:line + transcript labels; (5) per-§5-subsection completeness matrix showing ALIGNED/DRIFT/GAP counts per subsection; (6) public API surface delta (every new pub symbol vs main); (7) transcript label delta (every new label cross-referenced to a Figure 12 round); (8) type-shape delta covering LevelParams / MRowLayout / proof shapes / cache types / tiered types / claim-reduction types; (9) test coverage delta listing every new test + gaps; (10) security delta vs security_analysis.md §10; (11) cascade discovery walkthrough; (12) three-tier production-readiness verdict. Every ALIGNED claim has a file:line + book line citation; every DRIFT has a fix recommendation; every GAP has a next-step. The drift register cross-references and either CONFIRMS or REFINES every entry in the prior `specs/section5_protocol_drift_audit.md`. No source code changed (audit doc + scratchpad only). No commits other than the audit doc + scratchpad. No push. cargo state unchanged from loop start."
---

Deep review of every change on `feat/tensor-challenges` vs `main` for §5 protocol conformance.

The branch has accumulated 100+ commits implementing book §5 (fourth-root verifier). Before any production cutover, do a clean-room re-audit of EVERY logical change against the book to surface drift, semantic differences, and missing components. This is a deeper audit than `specs/section5_protocol_drift_audit.md` (which was structural, per-subsection) — the goal here is to enumerate the FULL DIFF vs `main`, classify each meaningful change, and confirm or surface drift on each one.

# Source of truth (do not drift)

- Book: `/home/giuseppe/lattice-jolt/sections/akita/5_fourth_root_verifier.tex` — every §5 subsection, Figure 12, theorems, lemmas, remarks.
- Prior audits (read but DO NOT trust uncritically; re-validate each finding):
  - `/home/giuseppe/akita/specs/section5_protocol_drift_audit.md` (structural drift audit).
  - `/home/giuseppe/akita/specs/security_analysis.md` §10 (post-Phase-D-full security walkthrough).
  - `/home/giuseppe/akita/audit.md` (prior S/B/C-numbered findings, some now stale).
- Repo rules: `.cursor/rules/code_changes.mdc`, `.cursor/rules/completion_check.mdc`, `.cursor/rules/blockers.mdc`.

# Scope

- Compare `feat/tensor-challenges` HEAD vs `origin/main` (or wherever the branch diverged — confirm via `git merge-base feat/tensor-challenges origin/main`).
- EVERY commit, EVERY file change, EVERY new public API or trait method.
- Cross-reference each non-trivial change against the specific book §5 passage it implements.
- Outcome categories per change: ALIGNED / DRIFT / GAP / SCOPE-OUTSIDE-§5 / SCAFFOLDING.

# Deliverable

Write a new file `/home/giuseppe/akita/specs/section5_full_diff_audit.md`. Structure:

```markdown
# `feat/tensor-challenges` vs `main` — Deep §5 conformance audit

**Date**: <YYYY-MM-DD>
**HEAD**: `<git rev-parse HEAD>`
**Merge base**: `<git merge-base feat/tensor-challenges origin/main>`
**Commits on top of main**: <git rev-list --count merge-base..HEAD>
**Diff stat**: <git diff --shortstat merge-base..HEAD>
**Book ref**: `/home/giuseppe/lattice-jolt/sections/akita/5_fourth_root_verifier.tex`
**Scope**: every meaningful change on top of main, classified against book §5 / Figure 12.
**Methodology**: commit-by-commit walkthrough + file-by-file diff + book line cross-reference.

## Executive summary

(5-8 bullets. Headline counts. Top blockers if any. Most surprising findings. Overall production-readiness verdict.)

## Diff overview by crate

Table: crate | LOC added | LOC removed | net | files touched | §5 subsections this crate implements.

## Change classification

### ALIGNED changes (faithful book implementation)

For each meaningful change, one entry:

- **<short title>** (commits: `<hash1>`, `<hash2>`, ...)
  - **Book**: §5.X lines L-M, exact passage quoted.
  - **Implementation**: file:line(s).
  - **Why aligned**: 1-3 sentences explaining the implementation faithfully realizes the book contract.

### DRIFT (implementation differs from the book in observable behaviour)

For each drift, one entry:

- **DRIFT-N: <short title>** (commits: `<hash>`)
  - **Book**: §5.X lines L-M.
  - **Implementation**: file:line.
  - **Nature of drift**: notational / algebraic / structural / transcript-order / cost-model / etc.
  - **Soundness impact**: NONE / DEFENSE-IN-DEPTH / LATENT / IMMEDIATE.
  - **Production blocker**: YES / NO / CONDITIONAL.
  - **Recommended fix**: smallest coherent change that closes the drift.
  - **Cross-reference**: which prior audit (`section5_protocol_drift_audit.md` DRIFT-N or `audit.md` S/B/C-N or `security_analysis.md` §X) flagged this, if any.

### GAP (book describes something the implementation does NOT have)

For each gap, one entry:

- **GAP-N: <short title>**
  - **Book**: §5.X lines L-M.
  - **What's missing**: precise description + where it would land in the impl.
  - **Impact**: PERF-ONLY / ASYMPTOTIC / SOUNDNESS.
  - **Production readiness**: BLOCKER / NICE-TO-HAVE / COSMETIC.
  - **Recommended next step**: concrete code-level direction.

### SCOPE-OUTSIDE-§5 changes

Changes the branch makes that are not §5-related but are bundled into the same branch. List them so reviewers know NOT to evaluate against §5.

### SCAFFOLDING / DEAD CODE / BLOAT

Any change that landed but is not actually exercised in production paths. List concrete file:line examples with disposition recommendation. Cross-reference `audit.md` C-1..C-14 status.

## Per-Figure-12-round walkthrough

For each round 1-8 of Figure 12:

| Round | Book line | Prover commit(s) + file:line | Verifier commit(s) + file:line | Transcript labels | Status |

Same table as in `section5_protocol_drift_audit.md` but enriched with the commit hashes that introduced each piece + the EXACT round-step ordering verification (i.e., absorb-then-sample, or sample-then-absorb, with the same byte boundaries the book implies).

## Per-§5-subsection completeness matrix

A matrix that says, per subsection, what fraction of the book's prose / theorems / lemmas / algorithms are realized in code:

| Subsection | Total items in book | ALIGNED | DRIFT | GAP | Coverage |
|---|---|---|---|---|---|
| §5.1 Problem and setup | n/a (narrative) | — | — | — | — |
| §5.2 Tensor stage-1 challenges | <count> | <n> | <n> | <n> | <%> |
| §5.3 Automaton contraction | <count> | <n> | <n> | <n> | <%> |
| §5.4 Claim-reduction sumcheck | <count> | <n> | <n> | <n> | <%> |
| §5.5 Tiered commitment design | <count> | <n> | <n> | <n> | <%> |
| §5.6 Combined protocol (Figure 12) | 8 rounds + output | <n> | <n> | <n> | <%> |
| §5.7 Security analysis | <count> | <n> | <n> | <n> | <%> |
| §5.8 Concrete instantiation | <count> | <n> | <n> | <n> | <%> |

## Commit-by-commit walkthrough (chronological)

For EVERY commit between merge-base and HEAD, one line:

| # | Hash | Date | Title | §5 subsection | Class | Notes |

Pull the chronology from `git log --merges --no-merges --pretty='%h %ad %s' merge-base..HEAD --reverse`. For each commit:
- **§5 subsection**: which subsection the commit primarily implements (or "n/a" if scope-outside).
- **Class**: ALIGNED / DRIFT / GAP-CLOSING / SCOPE-OUTSIDE / SCAFFOLDING / TEST / DOCS / REVERT.
- **Notes**: any prior-audit ID this commit closes, any drift it introduces, any new public API it adds.

Don't summarize multiple commits into one line — every commit gets its own row. If a commit is a pure mechanical rename or fmt-only, mark it as such and skip the §5 cross-reference.

## Public API surface delta

List EVERY new public symbol introduced on the branch vs main:

| Symbol | Crate | File:line | Introduced in | Purpose | Stability |

- `Symbol`: pub type / fn / trait / const.
- `Stability`: stable-for-production / experimental / test-only.
- `Purpose`: what §5 element it serves, or scope-outside.

Includes new trait methods on existing traits (which are also a public API delta). Use `git diff main..HEAD -- '**/lib.rs'` and `git diff main..HEAD -- '**/mod.rs'` to seed the list, then verify against the actual symbol definitions.

## Transcript label delta

List EVERY change to `crates/akita-transcript/src/labels.rs` and cross-reference each new label to a Figure 12 round (or scope-outside). Verify the order in `pub fn all_labels()` matches the order of consumption in the prover.

## Type-shape delta

For each protocol-relevant struct/enum that gained or lost fields/variants, document the change:

| Type | Crate | Field change | Book justification | Soundness impact |

Particular attention to:
- `LevelParams` (every new field is a potential protocol-shape change).
- `MRowLayout` (the 10-vs-15 group enumeration).
- `AkitaBatchedProof`, `AkitaLevelProof`, `AkitaStage2Proof` (wire format — any change here is a transcript change).
- `AkitaScheduleLookupKey` (cache invariant).
- `LevelStep`, `FoldStep`, `DirectStep`, `Schedule` (planner emission shape).
- `RecursivePolyHandle`, `RecursiveHandlePoly` (mixed-witness plumbing per book §5.6 lines 940-953).
- `TieredSetupParams`, `TieredSetupCommitments`, `TieredSetupCacheKey`, `TieredSetupProverExtras` (tiered routing per §5.5).
- `PreparedMEval`, `SetupClaimReductionPayload` (claim-reduction per §5.4).

## Test coverage delta

List EVERY new test file or test function on the branch and classify each:

| Test | File:line | What it verifies | §5 element |

Coverage gaps:
- Any §5 element with NO test coverage at all.
- Any §5 element with ONLY E2E coverage (no unit-level pin).
- Any §5 element whose test is `#[ignore]`-gated (note WHY).

## Security delta vs `specs/security_analysis.md`

Read `specs/security_analysis.md` §§1-9 (pre-Phase-D-full baseline) and §10 (post-Phase-D-full re-audit). For each Phase-D-full change on the branch:
- Does §10 already cover it?
- If not, what additional security reasoning is needed?
- If yes, is the §10 reasoning still current relative to HEAD?

Surface any change that touches MSIS / CWSS / ring-switch / sumcheck soundness that is NOT yet documented in `security_analysis.md`.

## Cascade discovery walkthrough

Specifically for the §5.5 + §5.8 cascade:
1. Does the schedule planner emit the cascade `(f_L0=8, f_L1=4)` for `DenseCascadeCfg`?
2. Does it emit it FORCED (via the force-routing gate) or NATURALLY (via cost-model crediting)?
3. What is the smallest NV the planner emits cascade for, for each cascade config? Document via `probe_cascade_schedules_extended`.
4. What is the verifier wall-clock cost (cold + amortized) for the cascade at our hardware-feasible NVs (NV=22 dense / NV=28 onehot)?
5. What does book Table 1141-1158 predict at NV=32 / 38 / 44? How does our trend extrapolate?

## Production readiness verdict

Three-tier verdict:
1. **Cryptographically sound**: yes / no / conditional. Cite security_analysis.md sections.
2. **Protocol-aligned to book §5**: yes / partial / no. Cite the drift register.
3. **Production-ready performance**: yes / no / hardware-limited. Cite the cascade walkthrough.

Top 5 production blockers (if any), ranked, with the drift/gap entry each closes.

## Methodology + reproducibility

- How you walked the diff (`git log`, `git diff` commands used).
- How you mapped commits to §5 subsections.
- Files NOT audited (be explicit about scope limits — e.g., did you audit `akita-field`'s NTT primitive changes against any §5 contract? probably not, justify why).
- Tools used: `Grep`, `Read`, `Shell` for git commands.
- What you'd do differently with infinite time.
```

# Per-iteration discipline

This audit is INHERENTLY iterative — each iteration of the Ralph loop should:
1. Pick a slice of work (e.g., "audit commits abc1..def4" or "walk §5.4 implementations") and complete it cleanly.
2. Append to the audit doc.
3. Commit the doc (the audit itself is a working tree artifact, not a code change).
4. Hand off to the next iteration with the doc-state recorded in the scratchpad.

Do NOT try to do everything in one pass. The branch has 100+ commits; budget 4-6 iterations to walk them all carefully. Use the per-§5-subsection completeness matrix as the convergence signal — when every cell has a count (not "TBD") and every commit in the chronology has a class assignment, the audit is complete.

# Constraints (audit-time only — no code changes)

- This is a READ-ONLY audit (one new file write only — the audit doc itself).
- DO NOT edit any source file. DO NOT close any drift item by code change in this loop. If an actionable fix is identified, document it in the audit doc and queue it for a SEPARATE Ralph loop.
- DO NOT commit anything except the audit doc + scratchpad updates.
- DO NOT push.
- All findings must have file:line evidence and exact book line citations. Paraphrased citations are not acceptable.
- DO NOT soften findings. If a drift exists, name it. If a §5 element is absent, name the gap.

# Lessons learned (apply throughout)

- Use `best-of-n-runner` for independent audit slices (e.g., one subagent audits §5.2 + §5.3, another audits §5.4 + §5.5). Each gets its own git worktree so concurrent file reads don't conflict (the audit doc itself needs serial writes — the parent owns the merge).
- When dispatching subagents, omit the `model` field so they inherit Claude.
- Don't accept a "this seems fine" classification from a subagent — require a file:line + book-line citation for EVERY ALIGNED claim.
- Read the book actively — don't skim. The book has 1181 lines of §5; budget at least 2-3 hours of reading-equivalent across the audit, distributed across iterations.
- The §5.3 automaton contraction is a known partial implementation (only offset-slice specialization). Confirm this still holds vs `main` (it should — that's where the spec started).
- The §5.5 shared per-chunk matrix collapse (Phase 5) is documented SCOPE-DEFERRED in `section5_protocol_drift_audit.md` SCOPE-3. Re-verify it's still deferred vs the latest HEAD.

# Iteration 1 plan

1. Establish reproducibility facts: branch, HEAD, merge-base, commit count, diff stat, changed-file inventory.
2. Read the book and prior audits enough to seed the audit taxonomy and first known drift/gap candidates.
3. Create `specs/section5_full_diff_audit.md` with the required structure, metadata, methodology, diff overview, and TODO markers for later iterations.
4. Start the commit-by-commit table with the full chronological commit list and classify the first slice where evidence is available.
5. Commit only `specs/section5_full_diff_audit.md` and `.cursor/ralph/scratchpad.md`, then update this scratchpad with the next slice handoff.

# Iteration 1 handoff state

- Created `specs/section5_full_diff_audit.md`.
- Filled immutable metadata:
  - HEAD `7c846fb6cb5f2cf5483d7b6554dfe3720efaf48d`
  - merge-base `4b0b86a946dca5124ddc1c0197bda7b73284a137`
  - `151` commits total on top of main (`148` non-merge + `3` merge commits)
  - diff stat `131 files changed, 148135 insertions(+), 5562 deletions(-)`
- Filled diff overview by crate/area from `git diff --numstat`.
- Read the full book §5 once and seeded evidence-backed entries for:
  - §5.2 tensor left/right challenge sampling and transcript digesting.
  - §5.4 setup claim-reduction verifier degree-2 sumcheck.
  - Figure 12 transcript label deltas.
  - §5.5 `MRowLayout` 10-vs-15 group extension.
  - §5.3 general transducer GAP candidate versus implemented offset-slice carry DP.
- Spawned read-only isolated audit subagents for:
  - §5.2 + §5.3
  - §5.4 + §5.5
  - §5.6 Figure 12 + transcript labels
- The audit doc is intentionally incomplete and contains `ITERATION-TODO` markers. The completion promise is still false.

# Iteration 2 recommended slice

1. Expand the commit-by-commit table from rows 26-151 into one row per commit, including the 3 merge commits.
2. Incorporate completed subagent evidence only if each claim has exact book-line quote + file:line evidence.
3. Audit the first chronological slice in depth: commits `d7dd31e` through `1ef0042` (challenge-family / transcript domain setup), including public API deltas in `akita-challenges` and `akita-transcript`.
4. Start the Public API surface delta from `lib.rs` / `mod.rs` diffs and verify actual symbol definitions.
5. Preserve read-only discipline: edit only `specs/section5_full_diff_audit.md` and this scratchpad; commit only those files; do not push.

# Iteration 2 handoff state

- Audited code HEAD advanced to `81cceecf6206123170eab38d388768a320b16f00`; merge-base remains `4b0b86a946dca5124ddc1c0197bda7b73284a137`.
- Updated `specs/section5_full_diff_audit.md` metadata to `153` commits total (`150` non-merge + `3` merge commits) and diff stat `133 files changed, 149559 insertions(+), 5562 deletions(-)`.
- Refreshed the diff overview by crate/area from current `git diff --numstat`.
- Expanded the commit-by-commit walkthrough spine to one row for every commit `d7dd31e` through `81cceec`, including merge commits. Classifications are explicitly `PRELIMINARY` in notes and must not be treated as final until a slice audit attaches file:line + book-line evidence.
- Re-read current source evidence for:
  - `crates/akita-challenges/src/stage1.rs:522-552` tensor left/right sampling and left digest absorb.
  - `crates/akita-algebra/src/offset_eq.rs:20-200` offset-slice carry DP and aligned fast path.
  - `crates/akita-verifier/src/protocol/setup_claim_reduction.rs:108-188` setup claim-reduction rounds + degree-2 verifier check.
  - `crates/akita-types/src/layout/params.rs:273-347` `MRowLayout` 10-vs-15 documentation and fields.
  - `crates/akita-transcript/src/labels.rs:42-140` transcript labels and `all_labels()`.
- Spawned four read-only `best-of-n-runner` subagents for independent slices:
  - §5.2 + §5.3 tensor/automaton.
  - §5.4 + §5.5 claim-reduction/tiered.
  - §5.6 Figure 12 transcript ordering.
  - API/type/test/security/commit-classification support.
- No source files were edited. Only `specs/section5_full_diff_audit.md` and this scratchpad changed.

# Iteration 3 recommended slice

1. Wait for and incorporate subagent evidence, rejecting any ALIGNED claim without exact book lines and file:line evidence.
2. Replace PRELIMINARY commit classes for at least commits `d7dd31e..de81b3c` with final classifications.
3. Complete the transcript label delta: verify every new label's consumption order in prover and verifier, especially `CHALLENGE_TIERED_CHUNK_AGGREGATION`.
4. Begin type-shape delta for `LevelParams`, `MRowLayout`, `SetupClaimReductionPayload`, and `TieredSetup*`.
5. Commit only the audit doc + scratchpad again; do not push.
---
iteration: 1
max_iterations: 25
completion_promise: "Phase 5 Drift 4 closure on feat/tensor-challenges: (1) the planner's objective in level_proof_bytes / find_optimal_schedule_with_max / derive_optimal_suffix_schedule includes a setup-precompute storage term and a symmetric cleartext-discharge term; (2) ALL force-routing gates in derive_optimal_suffix_schedule and find_optimal_schedule_with_max are RETIRED (no &[true] forcing, no best=∞ for tiered, no post-DP sanity check for tier_shrink > 1); (3) probe_cascade_schedules_extended (--ignored) confirms DenseCascadeCfg(8,4) and OneHotCascadeCfg(8,4) naturally produce routing=2 tiers=[8,4] at NV ≥ 32 (the headline cascade per book §5.8 line 1170); (4) cargo test --release -p akita-pcs --test tiered_setup_e2e passes the full dense + cascade + tamper-reject suite (8/8 expected at the post-Drift-4 schedule shape); (5) cargo clippy --workspace --lib -- -D warnings clean; (6) specs/section5_protocol_drift_audit.md GAP-3 marked CLOSED with the chosen weight documented in the audit table; specs/security_analysis.md §10.4 refreshed; (7) no new force-routing gates added as fallback."
---

# Handoff prompt — Drift 4 (planner natural cascade discovery) closure

## Context

You are continuing Phase 5 book alignment work on `feat/tensor-challenges` in the Akita repository (lattice-based polynomial commitment scheme; see `AGENTS.md`). The current HEAD is at commit `5d0d1b0`. Drifts 1 and 3 are CLOSED; Drift 4 is OPEN and is your task.

The book (Hachi §5.4 line 793 "sweet spot is f = 8", §5.4 line 798-799 "32.5 GB → 4.3 GB setup storage at NV = 44", §5.8 line 1170 headline cascade `(f_{L0}=8, f_{L1}=4)`, §5.8 Table 1141-1158 measured speedups at NV ∈ {32, 38, 44}) describes the cascade as the schedule a cost-aware planner naturally picks for large-NV polynomials. The current implementation uses force-routing gates to materialise the cascade; the DP's own cost model does not select it.

## The issue

The planner in `crates/akita-planner/src/schedule_params.rs` has FORCE-ROUTING gates at two sites that bypass the DP's cost comparison whenever the config prescribes a tiered shrink:

1. `derive_optimal_suffix_schedule` (~line 478): `route_choices = if level_tier.is_tiered() { &[true] } else { &[false, true] }`. Plus a parallel guard at the direct-baseline that sets `best = ∞` when `level_tier.is_tiered()` (~line 495).

2. `find_optimal_schedule_with_max` at the root (~line 889): identical gate on `tier_shrink > 1`, plus a post-DP sanity check (~line 1041) that errors if the chosen schedule contains no routed tiered fold step.

When these gates are removed, empirical probe (`probe_cascade_schedules_extended --ignored`) at NV ∈ {19, 22, 25, 28, 32, 35, 38, 41, 44, 47, 50} across all four cascade configs (`DenseCascadeCfg(8,4)`, `OneHotCascadeCfg(8,4)`, `DenseCascadeSmallCfg(2,2)`, `OneHotCascadeSmallCfg(2,2)`) shows the DP picks `routing=0 tiers=[]` at EVERY NV. The cascade loses to the no-cascade alternative by ~3-15 KB of on-wire proof bytes per level.

The drift is therefore not in the protocol shape (which is correct via the gate) but in the planner's COST MODEL: it can't justify the cascade on its current objective.

## Why the cascade loses on the current objective

The current objective in `find_optimal_schedule_with_max` is `root_proof_size + stage1_prover_penalty + suffix.objective_cost`, summed over levels. `level_proof_bytes` (in `crates/akita-types/src/layout/proof_size.rs`) only counts ON-WIRE proof bytes per level.

The cascade's actual benefit per book §5.4 line 798-799 is **verifier-side setup-precompute storage reduction** (32.5 GB → 4.3 GB at NV = 44 = a 7.5× reduction) plus **verifier wall-clock time savings**. Neither is in the per-proof wire-byte objective. So on the current objective, cascade and non-cascade schedules produce roughly the same proof size, and the cascade often loses by a few KB because each cascade level adds a fold-proof transcript.

## What was tried in the previous loop (commits `883993d`, `5d0d1b0`)

**Attempt 1 — just remove the gates.** Reverted. The DP picks `routing=0` at every NV. Logged in `specs/section5_protocol_drift_audit.md` GAP-3.

**Attempt 2 — add a notional cleartext-discharge cost.** Added `cleartext_mle_discharge_cost = s_field_len_in * field_bytes` at the suffix's direct baseline so a level that terminates direct with incoming S pays a proxy for the verifier's MLE evaluation work. Reverted. Failure mode:

- `routes=false` at root: root's CR sumcheck discharges `S_root` cleartext at the root verifier (cost ~ `|S_root|` field ops).
- `routes=true` with terminal direct: cascade folds S once or more, then suffix terminates with cleartext discharge of `S_remaining` (smaller).

Both paths pay roughly the same total cleartext work, just distributed across different levels. Adding the cost only at the suffix's terminal direct (where `s_field_len_in > 0`) under-counts the `routes=false` case, which pays the same cost at the root's CR sumcheck. The asymmetric addition doesn't reflect the true cost difference, so it doesn't flip the DP. The asymmetry analysis is in GAP-3 of `specs/section5_protocol_drift_audit.md`.

## Current state in repo

- Force-routing gates are RESTORED (the originals from `6c9c38f`; this loop did not add new ones). Doc comments on both gate sites point at `specs/section5_protocol_drift_audit.md` GAP-3 as the audit-closure trail.
- `cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small` passes (4.1s).
- Broader dense + cascade + tamper-reject suite (8/8) passes (~41s release).
- `cargo clippy --workspace --lib -- -D warnings` clean.
- Pre-existing failures in `stage2_fused_round2_*_transition` (akita-prover --lib) are unrelated.

## What needs to be done for FULL closure per the book

The closure has six concrete pieces. Land them in this order, gating each commit with the release E2E suite.

### Step 1 — Extend the planner objective with a setup-precompute storage cost term

Add a new objective component to `level_proof_bytes` (or a sibling `level_cost`) that includes per-level **verifier setup storage** cost, alongside the existing proof-bytes term. The cost should be expressed in bytes-equivalent units so it can be summed with proof bytes:

- For an un-tiered level: the verifier-stored setup material at that level is roughly `setup_field_len * field_bytes` (full S matrix at the level's shape).
- For a tiered level (chunks + meta): the verifier-stored setup material is `(num_chunks * chunk_b_commit_size + meta_b_commit_size) * field_bytes` (the precomputed chunk + meta B-commitments; book §5.4 line 798-799 quantifies this as 4.3 GB vs 32.5 GB un-tiered at NV=44).

The relative weight between proof bytes and storage bytes is a deployment policy. Two reasonable choices:

- **Amortised**: weight `setup_storage / N` where `N` is the expected number of proofs per setup (e.g., 1000). For Jolt-style integrations `N` is large, so storage is heavily amortised.
- **Per-deployment**: weight `setup_storage * w_storage` where `w_storage` is a config-tunable constant (default 1.0).

Pick one, document the choice in `LevelParams` or `PlannerConfig`, and add a default. The Jolt-amortised default is the natural production target; the per-deployment knob lets the planner be re-tuned without a code change.

### Step 2 — Account for cleartext-discharge cost SYMMETRICALLY

Add the verifier wall-clock cost for the cleartext MLE discharge of S at:

- The `routes=false` root path: when the root chooses not to recursively route S, the CR sumcheck at the root closes via cleartext discharge of `S_root`. Charge `|S_root| * field_bytes` (or the chosen weight unit) at the root's CR component.
- The `routes=true` terminal direct path: when the cascade terminates with a direct step at level `L+k` carrying `s_field_len_in > 0`, the suffix discharges `S_remaining` cleartext. Charge `|S_remaining| * field_bytes` at the suffix direct baseline.

The asymmetric addition I tried failed because it only added the cost on the `routes=true` side. Symmetric accounting is required for the cost comparison to be fair.

### Step 3 — Validate that non-cascade configs aren't broken

After Steps 1 + 2, the new objective changes the relative costs of EVERY schedule, not just cascade vs no-cascade. Run the schedule-correctness gates across every preset:

- `cargo test --release --lib --workspace` (validates `validate_stored_sis_ranks` and unit tests).
- `cargo test --release -p akita-pcs` (E2E integration tests).
- Specifically: `DenseCfg`, `OneHotCfg`, `D32Full`, `D64Full`, `D128Full`, `D32OneHot`, `D64OneHot`, `D128OneHot`, `BareCfg`, all `ClaimReductionCfg<*>` variants, and the `*Static` test wrappers.

If any preset's schedule changes shape (different number of fold levels, different log_basis selection), check that:

- The new schedule still passes `validate_stored_sis_ranks` (no SIS rank regression).
- The proof bytes don't regress beyond the noise floor.
- The cascade configs now naturally produce the prescribed shape at NV ≥ 32.

### Step 4 — Retire the force-routing gates

Once Steps 1-3 are validated, remove the gates at:

- `derive_optimal_suffix_schedule` ~line 478 (`route_choices = if level_tier.is_tiered() { &[true] } else { &[false, true] }`) → unconditionally `&[false, true]`.
- `derive_optimal_suffix_schedule` ~line 495 (direct-baseline-set-to-MAX-when-tiered guard) → unconditional direct baseline.
- `find_optimal_schedule_with_max` ~line 889 (root-direct-set-to-MAX-when-tiered guard) → unconditional direct baseline.
- `find_optimal_schedule_with_max` ~line 924 (root `route_choices` gate) → unconditional `&[false, true]`.
- `find_optimal_schedule_with_max` ~line 1041 (post-DP "must contain routed tiered fold" sanity check) → removed; small-NV schedules that legitimately want direct are now correct.

Drop the now-unused `level_tier` / `root_tier` / `tier_shrink` locals if they have no other use.

### Step 5 — Verify natural-discovery probe at NV ≥ 32

Run `cargo test --release -p akita-pcs --test tiered_setup_e2e probe_cascade_schedules_extended -- --ignored --nocapture` and verify:

- `DenseCascadeCfg(8,4)`: schedule at NV ≥ 32 has `routing=2 tiers=[8, 4]` (or `routing≥2` carrying `[8, 4]` at the first two routing folds).
- `OneHotCascadeCfg(8,4)`: same at NV ≥ 32.
- `DenseCascadeSmallCfg(2,2)` / `OneHotCascadeSmallCfg(2,2)`: cascade fires at the smallest NV the planner schedules (typically NV ≥ 19).
- Small NV (≤ 28) for headline `(8,4)`: DP may correctly pick a smaller cascade or direct; document the cutoff.

If the probe doesn't show natural discovery at NV ≥ 32, the Step 1-2 weights need tuning. Iterate.

### Step 6 — Update tests and specs

- Convert `tiered_dense_cascade_l0_l1_fires` (NV=22, schedule-only), `tiered_dense_cascade_l0_l1_small` (NV=19, E2E), and `tiered_dense_default_cascade_fires` (NV=19, schedule-only) to either:
  - Assert the new natural-discovery behaviour (cascade fires unprompted at the right NV), OR
  - Allow either cascade or direct at small NV (the per-level tier assertion stays — any cascade fold the planner emits must use the prescribed `f` policy).
- Add a NEW test `tiered_dense_cascade_dp_discovers_headline_at_nv32` (or similar) that exercises NV ≥ 32 and asserts the headline `(8, 4)` cascade emerges naturally from the DP under `DenseCascadeCfg`.
- Update `specs/section5_protocol_drift_audit.md`: mark GAP-3 fully CLOSED, document the chosen weight in the audit table.
- Update `specs/security_analysis.md` §10.4: refresh the cascade cost-model description to reference the new objective.
- Confirm `specs/security_analysis.md` §11 (γ-folding soundness from Drift 3) needs no changes — it's independent of the cost-model.

## Discipline rules (NON-NEGOTIABLE, inherited from prior loops)

- `cargo test --release` for ALL E2E. Debug profile is BANNED (10-50× slower; prior loops lost time on this).
- Gate EVERY commit with `cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small` (NV=19 dense cascade, ~4-6s release).
- After Step 3, also run the broader gate: `cargo test --release -p akita-pcs --test tiered_setup_e2e -- --test-threads=1 tiered_rejects_tampered tiered_dense_prove_verify tiered_dense_cascade tiered_dense_default` (~41s release, exercises the full dense + tamper-reject + cascade suite serially to avoid OOM).
- `cargo clippy --workspace --lib -- -D warnings` clean before each commit.
- `cargo fmt -q` before each commit.
- No `eprintln!` / debug spam in production paths. Any tracing-level diagnostics use `tracing::debug!` with named fields.
- Each meaningful step (Steps 1-6 above) is its own commit on `feat/tensor-challenges` with a `phase5/drift4-stepN: ...` prefix.
- Document each iteration in `.cursor/ralph/scratchpad.md` (this file is gitignored; lives only locally).
- Do NOT add new force-routing gates as a fallback. If a step regresses tests, fix the cost model or roll back; do not paper over with gates. The previous loop's mistake was retrying the same gate-on/gate-off split multiple times — converge on the cost-model fix instead.

## Pre-flight checklist

Before starting, confirm:

```bash
git rev-parse HEAD                  # should be 5d0d1b0 or a successor
cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small  # passes ~4s
cargo clippy --workspace --lib -- -D warnings  # clean
git diff 6c9c38f -- crates/akita-planner/src/schedule_params.rs  # only doc comment updates from prior loop
```

Read these files first (in order):

1. `specs/section5_protocol_drift_audit.md` §SCOPE-3 + §GAP-3 (audit context).
2. `specs/security_analysis.md` §11 (γ-folding soundness from Drift 3; informs cost-model invariants).
3. `crates/akita-planner/src/schedule_params.rs` lines ~440-540 (`derive_optimal_suffix_schedule`) and ~830-1075 (`find_optimal_schedule_with_max`).
4. `crates/akita-types/src/layout/proof_size.rs::level_proof_bytes` and `planned_joint_w_ring_with_setup_group_tiered` (the cost-model surface to extend).
5. `crates/akita-pcs/tests/tiered_setup_e2e.rs::probe_cascade_schedules_extended` (the natural-discovery probe).

## Completion promise (for the loop closure)

```text
Phase 5 Drift 4 closure on feat/tensor-challenges: (1) the planner's
objective in level_proof_bytes / find_optimal_schedule_with_max /
derive_optimal_suffix_schedule includes a setup-precompute storage
term and a symmetric cleartext-discharge term; (2) ALL force-routing
gates in derive_optimal_suffix_schedule and find_optimal_schedule_with_max
are RETIRED (no &[true] forcing, no best=∞ for tiered, no post-DP
sanity check for tier_shrink > 1); (3) probe_cascade_schedules_extended
(--ignored) confirms DenseCascadeCfg(8,4) and OneHotCascadeCfg(8,4)
naturally produce routing=2 tiers=[8,4] at NV ≥ 32 (the headline
cascade per book §5.8 line 1170); (4) cargo test --release -p akita-pcs
--test tiered_setup_e2e passes the full dense + cascade + tamper-reject
suite (8/8 expected at the post-Drift-4 schedule shape); (5) cargo
clippy --workspace --lib -- -D warnings clean; (6) specs/section5_protocol_drift_audit.md
GAP-3 marked CLOSED with the chosen weight documented in the audit table;
specs/security_analysis.md §10.4 refreshed; (7) no new force-routing
gates added as fallback.
```

## Iteration log

### Iteration 1

- Initialized the Ralph loop state for Drift 4 closure.
- Starting with Step 1: pre-flight checks and reading the cost-model/audit surfaces before editing.
- Pre-flight confirmed HEAD `5d0d1b0`, small release E2E green, and workspace lib clippy clean.
- Step 1 implemented as a sibling planner objective term: `level_proof_bytes` remains wire bytes, while the DP adds amortized verifier setup-precompute storage via `planned_verifier_setup_storage_field_len`.
- Step 1 gate: `cargo fmt -q && cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small && cargo clippy --workspace --lib -- -D warnings` passed.
- Step 1 committed: `9ddf99b phase5/drift4-step1: add setup storage objective`.
- Step 2 implemented symmetric cleartext-discharge objective charges at root `routes=false` and suffix terminal direct with incoming `S`.
- A first Step 2 attempt overcharged every recursive non-routing fold and broke the NV=19 f=2 gate; narrowed back to the two handoff-specified discharge sites.
- Step 2 also corrected the storage helper to derive tiered chunk/meta storage against the receiving `next_params`, matching runtime.
- Step 2 gate: `cargo fmt -q && cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small && cargo clippy --workspace --lib -- -D warnings` passed.
- Step 2 committed: `9d33e27 phase5/drift4-step2: charge cleartext setup discharge`.
- Step 3 initial validation found `akita-config::schedule_policy::batched_root_next_w_len_requires_group_and_point_counts` was asserting exact root params; the new objective legitimately picked a different root basis while preserving the group/point-count invariant. Adjusted that test locally; workspace lib now only fails the known pre-existing `stage2_fused_round2_*` prover tests.
- `cargo test --release -p akita-pcs` still failed before Step 4 because small-NV CR-on defaults prefer direct under the new objective, but the old force-routing gate converted that into an error.
- Step 4 removed the force-routing gates and post-DP tier sanity check from `find_optimal_schedule_with_max` / `derive_optimal_suffix_schedule`.
- Step 4 gate: `cargo fmt -q && cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small && cargo clippy --workspace --lib -- -D warnings` passed.
- Step 4 committed: `fd0ddb3 phase5/drift4-step4: retire force-routing gates`.
- Step 5 probe passed: `DenseCascadeCfg(8,4)` and `OneHotCascadeCfg(8,4)` naturally emit `routing=2 tiers=[8,4]` for every NV >= 32. Dense reaches `[8,4]` at NV=28; onehot at NV=32. Small `(2,2)` emits both tiers by NV=25 dense / NV=35 onehot.
- Step 3 test invariant committed: `4f90979 phase5/drift4-step3: allow objective-driven batched root params`.
- Current blocker: `cargo test --release -p akita-pcs --test akita_e2e batched_onehot_4x30_keeps_folding_past_oversized_tail -- --test-threads=1` fails with `scheduled recursive level did not match runtime state: step.current_w_len=17950976, inputs.current_w_len=50981120, step.log_basis=4, current_log_basis=4`.
- Tried and reverted two non-fixes: (1) making suffix DP state add `s_field_len_emitted` to `next_w_len` (broke `tiered_dense_prove_verify_small`, proving normal tiered S is not part of `handle[0].w`); (2) bypassing generated tables for CR-on configs / singleton-only root storage objective (did not change the batched onehot failure).
- Next iteration should inspect why the batched onehot proof path's root runtime output is `50981120` while the selected schedule's root handoff is `17950976`; likely a root batched witness-size/schedule-selection mismatch exposed by the new objective.
---
iteration: 8
max_iterations: 25
completion_promise: "Phase 5 is end-to-end book-aligned on feat/tensor-challenges: (1) build_tiered_handle_material runs ONE batched shared-B mat-vec across the k chunks (Drift 1 shallow); (2) the L+1 prover emits a SINGLE aggregated chunks claim via γ-folding (book §5.4-§5.5), matching the book's Growth ≈ 1.0-3.0× promise, with the planner k-drop re-landed and runtime-faithful (Drift 3); (3) force-routing gates retired because the DP discovers the headline (8, 4) cascade naturally at NV ≥ 32 for DenseCascadeCfg (Drift 4); cargo test --release -p akita-pcs --test tiered_setup_e2e passes the full cascade + tamper-reject suite in release; clippy --workspace --lib -- -D warnings clean; specs/security_analysis.md §11 refreshed with γ-folding soundness analysis (Schwartz-Zippel knowledge error 2k / |F_{q^k}| absorbed into composed budget, ≥128-bit confirmed); specs/section5_protocol_drift_audit.md SCOPE-3 + GAP-3 fully CLOSED."
---

# Phase 5 book-alignment ralph loop (continuation)

Starting from commit `6c9c38f` on `feat/tensor-challenges`. Three drifts remain after the prior loop closed Drift 2.

## Drift map

### DRIFT 1 — shallow batched chunk mat-vec (no protocol bytes change)
- `build_tiered_handle_material` in `crates/akita-prover/src/protocol/flow.rs` currently calls `commit_dense_s_handle_direct` k times serially on the SAME shared B matrix.
- Replace with ONE batched call: one A-step over `k * num_blocks_chunk` blocks via `mat_vec_mul_ntt_i8_dense` (already takes `blocks: &[&[CyclotomicRing]]`), one B-step over the larger digit-column count via `mat_vec_mul_ntt_single_i8`, output partitioned into k chunks of `n_B_chunk` ring elements.
- Mirror in `crates/akita-verifier/src/protocol/levels.rs::derive_tiered_setup_material_for_verifier_uncached`.
- Wire shape unchanged (still k u_j entries).
- Scope: ~50-150 LOC.
- Gate: `cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small`.

### DRIFT 3 — deep γ-folding to make L+1 chunks claim a SINGLE aggregated claim
- After L0 chunk commits absorb, sample γ challenges from transcript (verify book §5.4-§5.5 for exact form: per-chunk γ_j vs tensored γ).
- Prover computes aggregated chunk poly = Σ_j γ_j · chunk_j and aggregated u = Σ_j γ_j · u_j. By Ajtai linearity u = B_chunk · t̂_agg.
- L+1 ingests ONE aggregated chunk claim (claim_count = 1) at the SHARED opening point (book line 949 "share folding challenges").
- Meta tier still commits the k per-chunk u_j (the verifier's reconstruction mechanism).
- Verifier-side γ replay + aggregated-claim verification.
- New transcript label: `CHALLENGE_TIERED_CHUNK_AGGREGATION` after the meta-tier commit absorption.
- Soundness: γ ∈ F_{q^k} gives knowledge-soundness via Schwartz-Zippel (probability 2k / |F_{q^k}|).
- After landing: RE-LAND the prior loop's k-drop in `planned_joint_w_ring_with_setup_group_tiered` (`crates/akita-types/src/layout/proof_size.rs`).
- Tamper-reject tests must still pass.
- Scope: 600-1000 LOC across prover, verifier, types, Phase-5 merge logic.

### DRIFT 4 — force-routing gate retirement
- After Drift 3, planner's natural-discovery probe (`tiered_dense_cascade_dp_discovers_headline*`) picks the headline (f_L0=8, f_L1=4) cascade unprompted at NV ≥ 32.
- Remove `&[true]` forcing in `derive_optimal_suffix_schedule` and `find_optimal_schedule_with_max` in `crates/akita-planner/src/schedule_params.rs`.
- Verify via `probe_cascade_schedules_extended` (`--ignored`).

## Discipline (NON-NEGOTIABLE)
- `cargo test --release` for ALL E2E. Debug profile is BANNED.
- Gate every commit with `cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small` (NV=19 dense cascade, ~6s release).
- `cargo clippy --workspace --lib -- -D warnings` clean.
- No `eprintln!` / debug spam in production paths.
- Each meaningful step its own commit on `feat/tensor-challenges`.
- Document each iteration here.
- Update `specs/section5_protocol_drift_audit.md` (mark SCOPE-3 + GAP-3 CLOSED only when DP discovers cascade naturally) and `specs/security_analysis.md` §11 (γ-folding soundness) when work lands.

## Iteration log

### Iter 1 — start
- Resetting scratchpad to reflect the continuation prompt.
- First task: locate Drift 1 work surfaces and assess the shallow batched mat-vec restructure.

### Iter 2 — Drift 1 landed (`693a649`)
- `build_tiered_handle_material` replaced k× serial `commit_dense_s_handle_direct` with ONE batched A-step (`mat_vec_mul_ntt_i8_dense` over k*num_blocks_chunk blocks) + parallel per-chunk B-step.
- Verifier mirror added (`derive_batched_chunk_commitments`).
- Wire shape unchanged. Compute schedule changes only.
- Gate: 4.7s release.

### Iter 3 — γ-folding label (`f90a056`)
- Added `CHALLENGE_TIERED_CHUNK_AGGREGATION = b"ak/c/tca"` for the upcoming γ aggregation.
- Doc'd k-independent F-scalar form (1/|F| collision per chunks group; mirrors existing CHALLENGE_EVAL_BATCH).

### Iter 4 — Drift 3 γ-folding landed (`f5e3ee3`)
- Prover `prove_recursive_level_with_policy`: absorbs k chunk u + meta u, samples γ, computes chunk_poly_agg = Σ γ_j chunk_polys[j], re-commits via standard chunk_lp chain → (u_agg_fresh, hint_agg). Pushes ONE chunks_agg handle + 1 meta handle (instead of k+1).
- Verifier `expand_tiered_setup_claims`: symmetric. Absorbs k chunk_b + meta_b, samples γ, derives u_agg_fresh from public S + γ, pushes 1 chunks_agg claim + 1 meta claim.
- Structural cascade updates: `MRowLayout::m_row_layout`, `total_b_row_count`, `total_d_row_count`, `group_b_row_count`, `group_d_row_count`, `compute_r_split_eq`-tier-aware weights, `planned_joint_w_ring_with_setup_group_tiered` all drop the `tier.num_chunks` multiplier — each tier-marked group now contributes 1 row's worth (claim_count = 1 after γ-aggregation).
- The planner's k-drop from the prior loop's Iter 6 revert is RE-LANDED via `planned_joint_w_ring_with_setup_group_tiered` (w_hat_s, t_hat_s, total_b, total_d all use 1× chunks contribution).
- Tamper-reject tests pass — γ-folding's M-relation linearity preserves the standard sumcheck soundness for chunks.
- Gate: full dense + tamper-reject suite passes in release (7/7, ~41s).
- Clippy clean.
- Pre-existing failures in `stage2_fused_round2_*` (akita-prover --lib) are unrelated to this change (verified by `git stash` reproduction at `f90a056`).

### Iter 5 — Drift 4 force-routing gate retirement INVESTIGATED + DEFERRED (`883993d`)
- Removed `route_choices = &[true]` gates in `derive_optimal_suffix_schedule` and `find_optimal_schedule_with_max`. Removed the post-DP "must contain routed tiered fold" sanity check.
- Ran `probe_cascade_schedules_extended` (--ignored) at NV ∈ {19, 22, 25, 28, 32, 35, 38, 41, 44, 47, 50} for all 4 cascade configs (DenseCascadeSmallCfg, OneHotCascadeSmallCfg, DenseCascadeCfg(8,4), OneHotCascadeCfg(8,4)).
- **Finding**: DP picks `routing=0 tiers=[]` (direct schedule) at EVERY NV across ALL 4 configs. Cascade is consistently 3-15 KB more expensive than direct in the planner's wire-byte objective.
- **Root cause**: the planner's `level_proof_bytes` only counts on-wire bytes. The cleartext MLE discharge of a terminating L+1 direct step has NO wire cost (setup material is verifier-derivable from public S). The cascade's real benefit is verifier wall-clock + setup-precompute storage, neither of which the planner objective captures.
- **Decision**: restored force-routing gates and post-DP sanity check. Documented the gap on both gate sites and in `specs/section5_protocol_drift_audit.md` GAP-3 with two options for future closure: (a) extend objective to charge a notional verifier-wall-clock cost for cleartext MLE discharge; (b) leave force-routing as the canonical opt-in (current state).
- **Drift 4 net**: NOT closed this loop. The cost-model gap is real but larger than this loop's scope.

### Iter 6 — Specs updated (`2866076`)
- `specs/security_analysis.md` §11 added: γ-folding soundness (k F-scalars, M-table linearity reduction, Schwartz-Zippel knowledge error `2^-128` per chunks group per cascade level, composed budget impact negligible).
- `specs/section5_protocol_drift_audit.md` SCOPE-3 marked CLOSED post-`f5e3ee3` (Drift 3 γ-aggregation runtime + cost-model fully landed). GAP-3 updated to PARTIAL post-`883993d` (Drift 4 investigation documented).

## Final status

- **Drift 1** (shallow batched chunk mat-vec): CLOSED (`693a649`). One A-step over k*num_blocks_chunk blocks via mat_vec_mul_ntt_i8_dense, per-chunk parallel B-step. Verifier mirror.
- **Drift 3** (deep γ-folding to ONE aggregated chunks claim): CLOSED (`f5e3ee3`). γ-folded k chunks → 1 chunks_agg claim. Planner k-drop re-landed in `planned_joint_w_ring_with_setup_group_tiered`, `m_row_layout`, `total_b_row_count`, `total_d_row_count`. Tamper-reject + dense cascade + E2E suite (8/8) green in release.
- **Drift 4** (force-routing gate retirement): NOT CLOSED, two investigation attempts (`883993d`, `5d0d1b0`) documented. The cost-model gap is real and larger than this loop's scope:
  - First attempt: removed gates without other changes → DP picks direct everywhere (probe shows routing=0 at NV ∈ {19..50} across all 4 cascade configs).
  - Second attempt: added cleartext_mle_discharge_cost = s_field_len_in * field_bytes at suffix direct baseline → still doesn't flip the root choice because the cleartext cost is paid SYMMETRICALLY in both routes=false (root CR sumcheck) and routes=true (suffix terminal direct) cases.
  - Real fix requires modelling per-level setup-precompute storage costs (book §5.4 line 798-799: 32.5 GB → 4.3 GB at NV=44 is the cascade's actual benefit). Out of scope for this loop.

The completion promise CANNOT be fully met as worded: Drift 4 natural cascade discovery requires a planner-objective extension (per-level setup storage cost) larger than this loop's scope. Drifts 1 and 3 are fully landed and runtime-faithful with a clean tamper-reject gate. specs/security_analysis.md §11 + specs/section5_protocol_drift_audit.md SCOPE-3 (CLOSED) + GAP-3 (PARTIAL, with explicit option (a) recommendation) are updated.

Final commit chain on `feat/tensor-challenges`:
- `693a649` phase5/drift1
- `f90a056` phase5/drift3 transcript label
- `f5e3ee3` phase5/drift3 γ-folding
- `883993d` phase5/drift4 investigation #1
- `2866076` specs SCOPE-3 + GAP-3 update
- `5d0d1b0` specs GAP-3 deeper analysis (Drift 4 investigation #2)

Final gate: `cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small` passes (4.1s); broader dense+tamper+cascade suite (8/8) passes (41s); `cargo clippy --workspace --lib -- -D warnings` clean.
