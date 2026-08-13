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
    planner-refactor.md
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
| Sparse challenges | `book/src/how/proving/root-fold-ring-switch.md` |
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

## 2026-Q3 stale-spec removal (deleted, not archived)

The Q2 audit classified a large backlog for a stacked follow-up that never
landed, so the specs kept accumulating dead references. This pass **deleted**
21 specs outright rather than archiving them: each was either superseded by a
spec that already owns the content, a retrospective of shipped work, or a
shipped change whose header still read `proposed` / `draft` / `in review`. None
carried durable content that the book or a surviving spec did not already own.

Recovery is via git history (`git log --diff-filter=D -- specs/`), not the
archive.

### Superseded or abandoned (successor owns the content)

| Deleted spec | Content now owned by |
|--------------|----------------------|
| `distributed-verifier-row-eval.md` | `digit-innermost-layout.md` (PR #296 closed unlanded) |
| `akita-sumcheck-unification.md` | `archive/2026-Q3/digit-range-pipeline-refactor.md` |
| `schedule-catalog-ownership.md` | `heterogeneous-group-source-contracts.md` |
| `transcript-immediate-fixes.md` | `book/src/how/transcript.md` |
| `batched-stage3-setup-opening.md` | `archive/2026-Q3/group-local-opening-points.md` |
| `extension-field-trace-cutover.md` | `extension-field-opening-batching.md` |
| `fp16-small-field-support.md` | `remove-fp16.md` |
| `crt-ntt-prime-profiles.md` | `book/src/foundations/ntt-crt.md` |

### Retrospectives of shipped work (no forward value)

`fp31-field-optimization-retrospective.md`,
`small-field-prover-opening-optimization.md`,
`akita-crate-followup-jolt-integration.md`,
`core-protocol-naming-cleanup.md` (superseded by archived `w-to-e-notation.md`),
`general-field-support.md`, `extension-claim-incidence-cutover.md`,
`simd-ring-subfield-fp8.md`.

### Status drift (PR merged; header still open)

| Deleted spec | Header said | Shipped in |
|--------------|-------------|------------|
| `shared-opening-claims-api.md` | `proposed` | landed; `OpeningClaimsLayout` / `PolynomialGroupLayout` are the live types |
| `transcript-hardening.md` | `DRAFT` | PR #90 |
| `y-ring-trace-internalization.md` | `in review` | PR #154 |
| `ring-dim-challenge-cutover.md` | `draft` | PR #268 |
| `sis-infinity-estimator-crate.md` | `proposed` | `crates/akita-sis-estimator/` |
| `single-point-opening-batch.md` | landed PR #186 | superseded by archived `group-local-opening-points.md` |

### Still owed: fold into book, then archive

Shipped records kept only because their book chapters are still thin stubs.
Fold each, then `git mv` to `specs/archive/`:

| Spec | Book chapter |
|------|--------------|
| `akita-pcs-crate-decomposition.md` | `how/architecture.md` |
| `extension-field-opening-batching.md` | `how/proving/extension-opening-reduction.md` |
| `terminal-fold-cutover.md` | `how/recursion.md` |
| `weak-binding-norm-fix.md` (committed-fold section) | `how/security.md` |
| `security-hardening.md` | `how/verification.md` |
| `remove-fp16.md` | `foundations/rings-and-fields.md` |
| `crt-ntt-accumulation-safety.md`, `avx-simd-port.md` | `how/optimizations.md` |
| `ci-test-timing.md` | `usage/profiling.md` |
| `setup-product-sumcheck.md` | `how/proving/sumcheck-stages.md` |

### Keep as live specs

`flat-public-matrix-and-exact-ntt-cache.md`,
`role-native-projected-digit-layout.md`, `setup-layout-repack.md`,
`setup-offloading-planner.md`,
`eor-streamed-prover.md`, `packed-sumcheck.md`,
`planner-incidence-generalization.md`, `akita-field-refactor.md`,
`akita-compute-backend-metal.md`,
`large-digit-ntt-infrastructure.md`,
`eor-sumcheck-prover-acceleration.md`,
`cross-repo-field-microbench.md`,
`sis-quantum128-scalar-n-table.md`, plus `TEMPLATE.md`,
`SPEC_REVIEW.md`, and this file.

## Never commit / never fold

Root-level `*-NEVER-COMMIT.md` scratch files are local-only.
