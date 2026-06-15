# Documentation policy

Akita has three documentation layers. They have different jobs and different
staleness costs.

| Layer | Location | Role | Canonical for |
|-------|----------|------|----------------|
| **Book** | `book/` | Curated narrative (usage, protocol, foundations, roadmap) | Explanations a newcomer or integrator reads end to end |
| **Specs** | `specs/` | Design records with acceptance criteria and review history | In-flight design, contracts under review, audit trail until folded |
| **Runbook / ops** | `AGENTS.md`, `docs/` | Maintainer contracts, generated tables, historical snapshots | Agent/CI contracts, crate graph, audits |

**Rule:** one durable fact lives in one place. The book owns narrative truth once
a chapter is written. Specs are archived after fold. `AGENTS.md` mirrors
verifier-reachable contracts and commands; it is not a second book.

See also [`specs/PRUNING.md`](../specs/PRUNING.md) for spec lifecycle.

## Per-PR obligations

Every implementation PR must do **all** that apply:

1. **Spec header** — if the PR completes or supersedes a spec, update `Status`,
   `PR`, and acceptance checkboxes in the same PR (never leave shipped work at
   `proposed` / `active`).
2. **Book stub** — if behavior is user-visible or architecturally load-bearing,
   add or refresh the owning book page (stubs may stay stubs, but "Sources to
   fold in" must cite real paths).
3. **`AGENTS.md`** — update when verifier-reachable contracts, crate boundaries,
   commands, or feature flags change.
4. **`docs/crate-graph.md`** — update when `Cargo.toml` workspace edges change
   (or run the quarterly audit that keeps it in sync).
5. **Archive** — when a spec's durable content is folded into the book, `git mv`
   it to `specs/archive/<quarter>/` in the same PR that lands the book prose (or
   the immediately stacked follow-up). Set `Book-chapter:` first.

Direct doc-only PRs skip (1) when no spec exists. Trivial bugfixes with no
API/contract change may skip (2)–(4) when the PR touches no paths in
`docs/doc-blast-radius.json` and does not change public API or verifier contracts.

`Book-chapter` paths use `book/src/how/foo.md` or bare `how/foo.md` under
`book/src/`. Do not write `src/how/foo.md`.

## Hard checks (CI, blocking)

Run locally: `./scripts/check-doc-guardrails.sh`

| Check | Script | What it catches |
|-------|--------|-----------------|
| Dead symbols in live specs | `check-spec-references.sh` | References to removed crates/types (`akita-scheme`, `PlannerConfig`, …) in `specs/` outside `archive/` |
| Dead symbols in `docs/` | `check-doc-dead-symbols.sh` | Removed crates/types in non-historical `docs/*.md` (`README`/`AGENTS` by review) |
| `Book-chapter:` paths exist | `check-book-chapter-paths.sh` | Spec headers pointing at missing book pages |
| Book source paths exist | `check-book-source-paths.sh` | Stale `crates/` / `specs/` / `docs/` citations in `book/src/` |
| Book builds | `mdbook build` (in CI) | Broken internal links, preprocessor errors |

Add a symbol to the dead-pattern list in **both** check scripts when a rename or
cutover removes it from the codebase.

### Future hard checks (not yet implemented)

- `Book-chapter:` required when `Status: implemented` and spec is not tagged
  `reference-only` in PRUNING live list.
- Diff-based warning when `crates/<X>/` changes but no file in that crate's blast
  radius was touched (opt-in strict mode).
- Auto-regenerate `docs/crt-ntt-capacity-profile.md` and fail if dirty.

## Soft checks (PR comment, non-blocking)

On every PR, CI posts a comment (marker `<!-- akita-doc-blast-radius -->`) listing
**documentation regions** that may need updates based on changed paths.

Source of truth: [`docs/doc-blast-radius.json`](doc-blast-radius.json), maintained
by humans. Regions are **inexact by design**: a change to `akita-prover` protocol
code should remind authors to look at book proving pages, related specs, and
`AGENTS.md`, not prove that prose was updated.

Regenerate locally:

```bash
python3 scripts/doc_blast_radius.py --base origin/main --head HEAD
```

The comment is advisory. Reviewers use it as a checklist, not a merge gate.
Fork PRs do not receive blast-radius comments (read-only `GITHUB_TOKEN`).

## When to update what

| Change type | Spec | Book | AGENTS | docs/ |
|-------------|------|------|--------|-------|
| New feature (large) | Required up front | Stub or chapter after ship | If contract changes | Rare |
| API / proof shape change | Update or new spec | Owning chapter | Yes | crate-graph if deps change |
| Internal refactor, same API | Optional note | Only if narrative wrong | If hooks move | No |
| Preset / schedule table | planner specs | `how/configuration.md` | Profiling section | No |
| Security / SIS sizing | `l2-msis-*`, `akita-sis-*` | `how/security.md` | If verifier-reachable | No |
| Doc-only PR | Archive/fold as needed | Yes | If commands change | Yes |

## Folding and pruning cadence

- **Per PR (enforced):** spec headers, hard checks green, blast-radius comment
  reviewed.
- **Monthly (15 min):** run `check-doc-guardrails.sh`; scan `specs/` statuses vs
  merged PRs; triage blast-radius false negatives.
- **Quarterly:** execute a PRUNING audit slice (classify, fold, archive); refresh
  `book/src/foundations/spec-index.md` and `specs/archive/README.md`.

## Relationship to the paper

The Akita paper (`lattice-jolt` repo) is upstream narrative for **Foundations**
and parts of **How it works**. Book chapters cite paper sections in their stubs;
when paper and code diverge, **code + specs win** until the paper is updated.
Do not fork long proofs into the book; summarize and link.
