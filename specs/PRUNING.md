# Spec lifecycle and pruning

Akita accumulates design specs in `specs/`. Without pruning, the directory drifts
into a mix of live design, shipped-and-forgotten records, and contradictory
historical snapshots.

**Canonical policy:** [`docs/documentation.md`](../docs/documentation.md) (per-PR
obligations, hard CI checks, blast-radius PR comments).

**Narrative home:** the [Akita Book](../book/README.md). Once durable content is
folded into a book chapter, the spec is reference-only and must be archived.

## Three layers (no duplication)

| Layer | Role | Update when |
|-------|------|-------------|
| **Book** | Explanations readers consume | Behavior or architecture is stable enough to teach |
| **Specs** | Design records + acceptance criteria | Designing, implementing, or auditing a change |
| **AGENTS.md / docs/** | Agent contracts, graphs, generated tables | Verifier-reachable contracts or repo structure changes |

Do not maintain the same fact in two places. The book wins for narrative; specs win
for in-flight acceptance criteria until fold.

## Status vocabulary (exactly one per spec)

Every spec header uses **one** of these values (see `specs/TEMPLATE.md`):

| Status | Meaning | Location | Next step |
|--------|---------|----------|-----------|
| `proposed` | Not approved | `specs/` | Review or delete |
| `approved` | `spec-approved`, not started | `specs/` | Implement |
| `active` | Implementation in flight | `specs/` | Land PRs; check acceptance criteria |
| `implemented` | Shipped; still useful as reference | `specs/` | Fold into book, then archive |
| `superseded` | Replaced (`Superseded-by:` set) | `specs/archive/` | Do not edit for current behavior |
| `historical` | Retrospective only | `specs/archive/` | Do not edit |
| `archived` | Folded into book | `specs/archive/` | Edit book chapter instead |

**Ambiguity removed:**

- `implemented` **≠** `archived`. Shipped work stays in `specs/` until its
  durable content is folded into the book (or explicitly marked reference-only
  with no fold planned).
- `active` and `approved` must not remain on merged work. Update the header in
  the implementation PR.
- `proposed` on a fully checked acceptance list is a **process violation** (CI
  blast-radius + reviewer duty).

Target steady state: **≤15** specs in `specs/` root with status
`proposed` / `approved` / `active` / `implemented`. Everything else is archived.

## Status transitions (required actions)

| Event | Author must |
|-------|-------------|
| Spec approved for implementation | `Status: approved` (or `active` when work starts) |
| Implementation PR merges | `Status: implemented`, `PR:` set, acceptance boxes checked |
| Durable content folded into book | `Book-chapter:` set to real path; `git mv` to `specs/archive/<quarter>/`; row in `specs/archive/README.md` |
| New spec replaces old | Old: `Status: superseded`, `Superseded-by:`; new: `Supersedes:` |
| Spec wrong but historically useful | `Status: historical`; archive without book fold |

## Staleness signals

1. **Status drift** — header disagrees with merged reality.
2. **Dead symbols** — cites removed crates/APIs (`akita-scheme`, `PlannerConfig`,
   `schedule_policy.rs`, `_with_policy`, …). CI scans **live specs** via
   `scripts/check-spec-references.sh` (see script for the current live list).
3. **Contradiction with `AGENTS.md`** — architecture index wins for current structure.
4. **Superseded** — newer spec covers the same ground (link both directions).
5. **Folded** — `Book-chapter:` set and chapter prose landed → archive the spec.

Run `scripts/check-spec-references.sh --all` quarterly on the full non-archive tree.

### Live specs excluded from CI symbol scan (known stale refs)

These remain **live design** but still mention removed names; scrub before adding
back to the CI live list in `check-spec-references.sh`:

- `specs/akita-compute-backend-metal.md` (`akita-scheme`, `_with_policy`)
- `specs/transcript-immediate-fixes.md` (`akita-scheme`)

## Cadence

| When | What |
|------|------|
| **Every PR** | Update spec headers if applicable; review blast-radius comment (`<!-- akita-doc-blast-radius -->`); keep hard checks green |
| **Monthly (~15 min)** | Run `./scripts/check-doc-guardrails.sh`; run `check-spec-references.sh --all`; triage false negatives in `docs/doc-blast-radius.json` |
| **Quarterly** | Execute an audit slice below; fold + archive; refresh `book/src/foundations/spec-index.md` |

## Archive layout

```
specs/archive/
  README.md          # index: filename | final status | book chapter | date
  2026-Q2/
    fp16-small-field-support.md
    ...
```

Archiving = `git mv` + archive index row + fix inbound links + update book spec index.

## Folding into the book

1. Extract durable concepts (invariants, diagrams, formulas, contracts). Omit PR
   narration and execution checklists unless they are the contract.
2. Land book prose (or stub refresh with accurate sources) in the owning chapter.
3. Set `Book-chapter:` to a path under `book/src/` that **exists** (CI checks this).
4. Archive the spec in the same PR or the immediately stacked follow-up.

### Book chapter paths (consolidated outline)

Use these targets (not the pre-consolidation folder paths):

| Spec topic | Book chapter |
|------------|--------------|
| PCS decomposition / crate map | `book/src/how/architecture.md` |
| Optimized verifier | `book/src/how/verification.md` |
| Extension opening batching | `book/src/how/proving/extension-opening-reduction.md` |
| Tensor / sparse challenges | `book/src/how/proving/root-fold-ring-switch.md` |
| Terminal fold | `book/src/how/recursion.md` |
| Weak binding / norm fix | `book/src/how/security.md` |
| SIS consolidation | `book/src/how/security.md` |
| Planner refactor | `book/src/how/configuration.md` |
| Transcript hardening | `book/src/how/transcript.md` |
| Security hardening / no-panic | `book/src/how/verification.md` |
| remove-fp16 | `book/src/foundations/rings-and-fields.md` |
| CRT accumulation | `book/src/how/optimizations.md` |
| SIMD / fp31 | `book/src/how/optimizations.md` |
| ZK hiding specs | `book/src/foundations/zero-knowledge.md` |
| Profiling / CI timing | `book/src/usage/profiling.md` |
| w-to-e notation | `book/src/foundations/glossary.md` |
| Setup product sumcheck | `book/src/how/proving/sumcheck-stages.md` |

## 2026-Q2 audit (archive pass — partial)

Classification from the initial book scaffold. Five specs moved to
`specs/archive/2026-Q2/` on the book scaffold PR; the remaining rows below are
for a stacked follow-up.

### Fold into book, then archive

| Spec | Book chapter |
|------|--------------|
| `akita-pcs-crate-decomposition.md` | `how/architecture.md` |
| `extension-field-opening-batching.md` | `how/proving/extension-opening-reduction.md` |
| `tensor-structured-folding-challenges.md`, `archive/bounded-l1-sparse-challenge.md` | `how/proving/root-fold-ring-switch.md` |
| `terminal-fold-cutover.md` | `how/recursion.md` |
| `weak-binding-norm-fix.md` (committed-fold section) | `how/security.md` |
| `akita-sis-consolidation.md` | `how/security.md` |
| `planner-refactor.md`, `planner-owns-schedule-expansion.md` | `how/configuration.md` |
| `transcript-hardening.md` | `how/transcript.md` |
| `security-hardening.md` | `how/verification.md` |
| `remove-fp16.md` | `foundations/rings-and-fields.md` |
| `crt-ntt-accumulation-safety.md` | `how/optimizations.md` |
| `avx-simd-port.md`, `fp31-field-optimization-retrospective.md` | `how/optimizations.md` |
| `akita-zk-commitment-hiding.md`, `akita-zk-v-hiding.md`, `akita-zk-sumcheck-hiding-plain.md` | `foundations/zero-knowledge.md` |
| `profile-bench-coverage-matrix.md` (Active Matrix) | `usage/profiling.md` |
| `ci-test-timing.md` | `usage/profiling.md` |
| `w-to-e-notation.md` | `foundations/glossary.md` |
| `setup-product-sumcheck.md` | `how/proving/sumcheck-stages.md` |

### Archive directly (little or no fold)

| Spec | Reason |
|------|--------|
| `fp16-small-field-support.md` | superseded by `remove-fp16.md` |
| `simd-ring-subfield-fp8.md` | consumer removed |
| `planner-config-consolidation.md` | superseded; proposes non-existent crates |
| `extension-field-trace-cutover.md` | superseded by opening batching spec |
| `general-field-support.md` | historical |
| `extension-claim-incidence-cutover.md` | landed (PR #69) |
| `small-field-prover-opening-optimization.md` | retrospective |
| `akita-crate-followup-jolt-integration.md` | retrospective |
| `core-protocol-naming-cleanup.md` | superseded by `w-to-e-notation.md` |
| `rust-file-line-cap.md` | policy in CI + CONTRIBUTING |

### Keep as live specs

`setup-layout-repack.md`, `setup-offloading-planner.md`,
`eor-streamed-prover.md`, `packed-sumcheck.md`,
`planner-incidence-generalization.md`, `akita-field-refactor.md`,
`akita-compute-backend-metal.md`, `crt-ntt-prime-profiles.md`,
`transcript-immediate-fixes.md`, `eor-sumcheck-prover-acceleration.md`,
`cross-repo-field-microbench.md`,
`sis-quantum128-scalar-n-table.md`, plus `TEMPLATE.md`,
`SPEC_REVIEW.md`, and this file.

## Never commit / never fold

Root-level `*-NEVER-COMMIT.md` scratch files are local-only.
