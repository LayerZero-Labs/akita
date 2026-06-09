# Spec Lifecycle and Pruning

Akita accumulates design specs in `specs/`. Without pruning, the directory drifts
into a mix of live design, shipped-and-forgotten records, and contradictory
historical snapshots. This document defines how specs age, when their durable
content moves into the [Akita Book](../book/README.md), and how stale specs are
archived or removed.

The companion to this process is the book: **the book is the single canonical
narrative; `specs/` is the design-record library.** Once a spec's durable
content is folded into a book chapter, the spec is reference-only and should be
archived.

## Lifecycle

Every spec carries a `Status` header (see `specs/TEMPLATE.md`):

| Status | Meaning | Lives in |
|--------|---------|----------|
| `proposed` | Not approved / not started | `specs/` |
| `approved` | `spec-approved`, awaiting implementation | `specs/` |
| `active` | Approved, implementation in flight | `specs/` |
| `implemented` | Shipped; durable reference value | `specs/` until folded, then `specs/archive/` |
| `superseded` | Replaced by another spec (`Superseded-by:`) | `specs/archive/` |
| `historical` | Retrospective/log of completed work | `specs/archive/` |
| `archived` | Durable content folded into the book | `specs/archive/` |

Target steady state: **at most ~10–15 `active`/`approved`/`proposed` specs** in
`specs/` root. Everything else is `implemented` awaiting fold, or archived.

## Staleness signals

A spec is a pruning candidate when any of these hold:

1. **Status drift** — header says `proposed`/`active` but all acceptance
   criteria are checked and the work shipped.
2. **Dead symbols** — the spec cites modules/types/crates that no longer exist
   (e.g. `akita-scheme`, `akita-cfg`, `sis_offline`, `Fp16`/`Q16`,
   `PlannerConfig`, `ScheduleProvider`, `_with_policy`). `scripts/check-spec-references.sh`
   greps for a known dead-symbol list.
3. **Contradiction with `AGENTS.md`** — `AGENTS.md` is the live architecture
   index; when a spec disagrees with it about current structure, the spec is stale.
4. **Superseded** — a newer spec covers the same ground (link both directions via
   `Supersedes:` / `Superseded-by:`).
5. **Folded into the book** — once `Book-chapter:` is set and the chapter is
   written, the spec is reference-only.

## Cadence

- **Per PR (required):** when an implementation PR lands, the author updates the
  spec `Status` (and `PR`), or opens an archive PR. Do not leave aspirational
  architecture in a spec marked `implemented`; delete or rewrite contradicting
  sections.
- **Monthly (light):** a 15-minute index pass — does any spec `Status` disagree
  with the code? Run `scripts/check-spec-references.sh`.
- **Quarterly (full):** a classification audit (like the 2026-Q2 audit below),
  tied to a book-chapter milestone. Fold durable content, then archive.

## Archive layout

```
specs/archive/
  README.md          # index: filename | final status | book chapter | date
  2026-Q2/
    fp16-small-field-support.md
    ...
```

Archiving is a `git mv` into `specs/archive/<quarter>/` plus an entry in
`specs/archive/README.md`. Update inbound links (other specs, `docs/`,
`AGENTS.md`) in the same PR. The book's [Spec index](../book/src/foundations/spec-index.md)
should reflect the archive.

## Folding into the book

1. Extract durable concepts (invariants, diagrams, formulas, contracts) — not
   execution checklists or PR-era narration.
2. Leave **one** canonical spec per ongoing design area.
3. Set the spec's `Book-chapter:` header to the owning page.
4. Archive the source spec. Never maintain the same content in two places.

## 2026-Q2 audit (first pass — execute in the stacked archive PR)

This classification comes from the council docs/specs audit that accompanied the
initial book scaffold. The **archive moves are intentionally deferred to a
follow-up PR** stacked on the scaffold, so the scaffold PR stays review-light.

### Fold into the book, then archive (durable content)

| Spec | Book chapter target |
|------|---------------------|
| `akita-pcs-crate-decomposition.md` | `how/architecture/architecture.md` |
| `optimized_verifier.md` | `how/verification/per-level-replay.md` |
| `extension-field-opening-batching.md` | `how/proving/extension-opening-reduction.md` (trim stale `akita-scheme`/`sis_policy.rs` refs) |
| `tensor-structured-folding-challenges.md`, `bounded-l1-sparse-challenge.md` | `how/proving/tensor-challenges.md` |
| `terminal-fold-cutover.md` | `how/recursion/intermediate-vs-terminal.md` |
| `weak-binding-norm-fix.md` (committed-fold section only) | `how/security/norm-bounds-weak-binding.md` |
| `akita-sis-consolidation.md` | `how/security/sis-msis-sizing.md` |
| `planner-refactor.md`, `planner-owns-schedule-expansion.md` | `how/config/planner-and-proof-size.md` |
| `transcript-hardening.md` | `how/transcript/transcript.md` (keep deferred-items table as live spec) |
| `security-hardening.md` | `how/verification/no-panic-contract.md` |
| `remove-fp16.md` | `foundations/fields.md` |
| `crt-ntt-accumulation-safety.md` | `how/optimizations/simd-and-packing.md` |
| `avx-simd-port.md`, `fp31-field-optimization-retrospective.md` | `how/optimizations/simd-and-packing.md` |
| `akita-zk-commitment-hiding.md`, `akita-zk-v-hiding.md`, `akita-zk-sumcheck-hiding-plain.md` | `foundations/zero-knowledge.md` |
| `profile-bench-coverage-matrix.md` (Active Matrix section) | `usage/profiling.md` (keep matrix as live spec) |
| `ci-test-timing.md` | `usage/profiling.md` |
| `w-to-e-notation.md` | `foundations/notation.md` + `foundations/glossary.md` |
| `setup-product-sumcheck.md` | `how/proving/sumcheck-stages.md` |

### Archive directly (historical/superseded; little to fold)

| Spec | Reason / superseded by |
|------|------------------------|
| `fp16-small-field-support.md` | superseded by `remove-fp16.md` |
| `simd-ring-subfield-fp8.md` | primary consumer (Fp16) removed |
| `planner-config-consolidation.md` | superseded by landed planner refactor + `AGENTS.md` (proposes `akita-cfg`/`akita-scheme`/triad that do not exist) |
| `extension-field-trace-cutover.md` | superseded by `extension-field-opening-batching.md` |
| `general-field-support.md` | extension-opening chain; self-labelled historical |
| `extension-claim-incidence-cutover.md` | landed (PR #69) |
| `small-field-prover-opening-optimization.md` | retrospective (PR #85) |
| `akita-crate-followup-jolt-integration.md` | retrospective (PR #65) |
| `core-protocol-naming-cleanup.md` | naming superseded by `w-to-e-notation.md` |
| `rust-file-line-cap.md` | policy now in CI + CONTRIBUTING |

### Keep as live specs (active design frontier)

`l2-msis-opnorm-folded-witness.md`, `setup-layout-repack.md`,
`eor-streamed-prover.md`, `packed-sumcheck.md`,
`planner-incidence-generalization.md`, `akita-field-refactor.md`,
`akita-compute-backend-metal.md` (Metal tail), `crt-ntt-prime-profiles.md`
(revise Q16 sections), `transcript-immediate-fixes.md`,
`eor-sumcheck-prover-acceleration.md`, `cross-repo-field-microbench.md`,
plus `TEMPLATE.md`, `SPEC_REVIEW.md`, and this file.

> Status headers on many `implemented` specs currently say `proposed`/`in progress`.
> The stacked archive PR should also correct those headers.

## Never fold

Root-level `*-NEVER-COMMIT.md` planning scratch is local-only and must never be
committed or folded into the book.
