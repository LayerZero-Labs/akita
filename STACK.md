# Setup Claim Offloading Stack

This document is the durable workflow for the setup-claim-offloading series.
The stack is semantic, not path-generated: each branch is a real review branch
whose diff should compile and make sense against its parent.

## Invariants

- `main` is the base for the first stack branch.
- Each later branch is based on the previous open stack branch.
- Stack branches are source-of-truth branches, not disposable materializations
  from a larger integration branch.
- A scratch integration branch may exist for smoke testing the full tip, but it
  must not drive PR branch generation by path restoration.
- Each branch should remain reviewable, locally coherent, and green for its
  focused checks.
- Force-pushes are allowed while PRs are drafts, but only with
  `--force-with-lease`.
- After changing a parent branch, rebase children in order and inspect
  `git range-diff` before pushing.
- Generated schedule tables, lockfiles, and benchmark updates should be isolated
  when they would obscure protocol-review diffs.

## Planned Branches

| # | Branch | Base | Scope |
|---|---|---|---|
| 01 | `quang/setup-layout-repack` | `main` | Complete setup-layout cutover: remove the global setup stride, pack base A/B/D setup views, split ZK B/D blinding tails onto a small separate setup seed/domain, and update fused setup paths. |
| 02 | `quang/setup-claim-packed-inner-product` | `quang/setup-layout-repack` | Express the base setup contribution as an inner product over raw packed setup indices; add explicit weight-builder equivalence tests. |
| 03 | `quang/setup-weight-evaluator` | `quang/setup-claim-packed-inner-product` | Add the succinct random-point evaluator for the base setup weight polynomial. |
| 04 | `quang/setup-claim-offloading` | `quang/setup-weight-evaluator` | Wire the matrix-claim sumcheck that delegates the raw base setup matrix claim. |
| 05 | `quang/setup-offload-tables-tests` | `quang/setup-claim-offloading` | Regenerated tables, broader tests, benchmarks, and cleanup if these are too noisy for earlier PRs. |

The later branches may be split further if the implementation reveals a cleaner
review boundary. The first branch should contain all setup-layout changes, not
a layout sub-stack. It should not introduce the offloading proof.

## Why Not a Jolt-Style Splitter

The Jolt refactor-audit stack uses pathspec ownership to regenerate disposable
PR branches from one source branch. That is a good fit when slices are mostly
crate- or directory-shaped.

Akita setup offloading cuts through shared protocol invariants:

- setup seed serialization and descriptor identity,
- setup sizing policy,
- `FlatMatrix` role views,
- prover commitment paths,
- verifier direct-recommit paths,
- fused setup contribution replay,
- optional ZK blinding paths, which deliberately use a separate small setup
  seed/domain instead of the base setup matrix,
- generated schedule/table policy.

Those changes often touch the same files in different semantic phases. A
path-restoration splitter would make invalid intermediate branches too easy.
Use manual semantic branches, with optional helper scripts only for bookkeeping.

## Useful Local Commands

Create the next branch in the stack:

```bash
git switch <parent-branch>
git pull --ff-only
git switch -c quang/<next-branch>
```

Rebase one child after changing its parent:

```bash
git switch <child-branch>
git fetch layerzero main
git rebase <new-parent-branch>
git range-diff <old-parent-branch>..<old-child-branch> <new-parent-branch>..HEAD
git push --force-with-lease
```

Check stack ancestry:

```bash
git log --oneline --decorate --graph main..quang/setup-offload-tables-tests
```

Focused checks for implementation branches:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

Use narrower package tests while developing, but the branch should document any
checks that were skipped before it is made ready for review.

## PR Discipline

- Open each PR as a draft until its parent is stable.
- Set each PR base to the previous branch in the stack, except PR 01, which
  targets `main`.
- The PR title should describe the project change directly.
- PR bodies should state the parent branch, focused scope, non-goals, and checks.
- Do not mix implementation and generated-table churn unless the generated files
  are the point of that branch.
- Do not add backward-compatibility shims for old setup layouts unless a reviewer
  explicitly asks for a temporary comparison harness. Akita makes no backward
  compatibility guarantees.

## Current Review Frontier

PR 01 is specified in `specs/setup-layout-repack.md`.
