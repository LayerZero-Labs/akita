# Documentation policy

Akita has three documentation layers. They have different jobs and different
staleness costs.

| Layer | Location | Role | Canonical for |
|-------|----------|------|----------------|
| **Book** | `book/` | Curated narrative (usage, protocol, foundations, roadmap) | Explanations a newcomer or integrator reads end to end |
| **Specs** | `specs/` | Design records with acceptance criteria and review history | In-flight design, contracts under review, audit trail until folded |
| **Runbook / ops** | `AGENTS.md`, `docs/` | Maintainer contracts, generated tables, historical snapshots | Agent commands, verifier-contract summary, pointer hub |

**Rule:** one durable fact lives in one place. The book owns narrative truth once
a chapter is written. Specs are archived after fold. `AGENTS.md` holds essential
commands, a short verifier-contract summary, and pointers; it is not a crate
encyclopedia or second book. Crate maps, profiling runbooks, and full contracts
live in the book and `docs/`.

See also [`specs/PRUNING.md`](../specs/PRUNING.md) for spec lifecycle.

## Audience and writing standard

The Book is a self-contained introduction to Akita. It serves several readers
who enter the project with different questions.

| Reader | What the Book must provide |
| --- | --- |
| Programmer new to proof systems | The purpose of each mechanism, the terms needed to understand it, and small examples before general formulas |
| Application integrator | The public entry points, configuration choices, accepted inputs, errors, and operational limits |
| Akita contributor | The path that data follows through the crates, the owner of each decision, and the tests that protect it |
| Cryptographer or security reviewer | A clear account of the protocol chosen by the schedule and the code that enforces each claim |
| Performance engineer | The physical data layout, storage types, dispatch conditions, fast paths, and scalar reference behavior |
| Maintainer or reviewer | The source of truth for each fact and the other documentation that must change when code moves |

These are reading goals, not separate editions of the Book. A chapter should
build one shared explanation from first principles, then provide clear paths
into the public interface, implementation, security checks, and performance
details that apply to its topic.

A reader should not need to read source code before starting a Book chapter.
The chapter may link to the source for inspection, but it must first explain
the ideas needed to follow its own text.

The Akita paper is unpublished and still changing. Book prose must not cite,
mention, or depend on it. If the draft contains an idea that readers need, the
Book must explain that idea in full and verify it against current code, live
specifications, and tests. Do not describe a design that appears only in the
draft as current or planned behavior. This restriction remains in force until
maintainers explicitly mark the paper as published and stable.

When a chapter introduces a technical mechanism, it should proceed in this
order when the topic allows it:

1. State the problem that the mechanism solves.
2. Define each new term in plain language before using it in an equation.
3. Give one concrete example with small values.
4. Introduce the general notation and explain what each symbol means.
5. Explain where the mechanism appears in Akita's protocol.
6. Show the main path through the code.
7. Distinguish the protocol rule from choices made by the current
   implementation.
8. Identify the checks and tests that support security or compatibility claims.

Do not replace an explanation with a list of source files. Do not assume that
the reader already knows Akita names such as a fold, an opening, a committed
source, or a setup prefix. Define these terms when they first appear in a
reader path. If a common cryptography term is necessary, explain it in the
same paragraph.

Accuracy still comes from current code and live specifications. Extra context
must explain the implementation that exists. It must not preserve an older
design or describe planned work as current behavior.

A source map is not an audit map. When a chapter covers a security relevant
mechanism, it should state what property each important source path enforces,
where public parameters enter, and what evidence a reviewer can use to check
the correspondence. Tests are regression evidence. They do not replace the
protocol argument or a review of the enforcing code.

## Per-PR obligations

Every implementation PR must do **all** that apply:

1. **Spec header** — if the PR completes or supersedes a spec, update `Status`,
   `PR`, and acceptance checkboxes in the same PR (never leave shipped work at
   `proposed` / `active`).
2. **Book stub** — if behavior is user-visible or architecturally load-bearing,
   add or refresh the owning book page (stubs may stay stubs, but "Sources to
   fold in" must cite real paths).
3. **`AGENTS.md`** — update when verifier-reachable contract summary, essential
   commands, or feature-flag pointers change (detail goes in the book / `docs/`).
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
| Dead symbols in live docs | `check-doc-dead-symbols.sh` | Removed crates/types in non-historical `docs/*.md`, plus deleted public API names in `book/src`, `docs/`, and crate READMEs (`README`/`AGENTS` by review) |
| `Book-chapter:` paths exist | `check-book-chapter-paths.sh` | Spec headers pointing at missing book pages |
| Book source paths exist | `check-book-source-paths.sh` | Stale `crates/` / `specs/` / `docs/` citations in `book/src/` |
| Book builds | `mdbook build` (in CI) | Broken internal links, preprocessor errors |

Add a symbol to the dead-pattern list in **both** check scripts when a rename or
cutover removes it from the codebase. For public API type removals, add it to
the deleted-API pattern lists so live book pages and crate READMEs are covered
without scanning archived or generated output.

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
| Preset / schedule table | planner specs | `how/configuration.md` | Pointer only | `usage/profiling.md` if modes change |
| Security / SIS sizing | `sis-quantum128-scalar-n-table.md`, `fold-linf-rejection.md`, `akita-sis-*` | `how/security.md` | If verifier-reachable | No |
| Doc-only PR | Archive/fold as needed | Yes | If commands change | Yes |

## Folding and pruning cadence

- **Per PR (enforced):** spec headers, hard checks green, blast-radius comment
  reviewed.
- **Monthly (15 min):** run `check-doc-guardrails.sh`; scan `specs/` statuses vs
  merged PRs; triage blast-radius false negatives.
- **Quarterly:** execute a PRUNING audit slice (classify, fold, archive); refresh
  `book/src/foundations/spec-index.md` and `specs/archive/README.md`.

## Unpublished design material

Draft research material can help maintainers discover a concept, but it is not
a Book source and must not appear as a Book citation. Before adding the concept,
check it against current code, live specifications, and tests. Then explain it
in full inside the Book. Leave out designs that the repository does not support
or plan to support.
