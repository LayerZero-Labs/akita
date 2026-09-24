# Branches: `main` and `dev`

Akita develops on two long-lived branches:

- **`main`** carries the current Akita protocol. It receives bug fixes,
  refactors, performance work, CI, and documentation that leave that protocol
  unchanged, plus protocol changes that the maintainer team approves.
- **`dev`** is `main` plus opt-in protocol extensions and compute backends,
  such as LaBinius binary openings, zero knowledge, and the Metal prover
  backend. When no extension is selected, `dev` runs exactly the protocol on
  `main`.

Changes flow from `main` into `dev` continuously. They flow from `dev` into
`main` only as focused pull requests that maintainers choose to review.

The key words "MUST", "MUST NOT", "SHOULD", "SHOULD NOT", and "MAY" in this
document are to be interpreted as described in BCP 14
([RFC 2119](https://www.rfc-editor.org/rfc/rfc2119),
[RFC 8174](https://www.rfc-editor.org/rfc/rfc8174)) when, and only when, they
appear in all capitals.

## Choose a base branch

| Change | Base branch |
| --- | --- |
| Bug fix, refactor, performance, CI, or documentation that keeps the protocol unchanged | `main` |
| Behavior-preserving refactor that an extension needs (an extension seam) | `main` |
| Change to the current protocol | `main`, with a spec per [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| New protocol extension, opt-in feature, or compute backend | `dev` |
| Fix to code that exists only on `dev` | `dev` |

Fix a bug in code that both branches share on `main`. `dev` receives the fix
through the next sync.

## Review and merge rules

`main` is protected by repository rulesets. A pull request into `main` needs an
approving review from a member of the maintainer team, given after the most
recent push, and passing Socket Security checks. `main` keeps a linear history,
so pull requests merge by squash or rebase. Any contributor with write access
can merge once these conditions hold.

`dev` accepts changes only through pull requests with passing CI, and it
rejects force pushes and deletion. It allows squash merges and merge commits.
Squash-merge every pull request into `dev` except a sync from `main`, which
needs a merge commit (see [Sync `main` into `dev`](#sync-main-into-dev)). `dev`
does not require maintainer-team approval. Any contributor with write access
can merge into `dev`. Ask another `dev`
contributor to review changes that modify files from `main` or verifier-reachable
code.

CI runs every pull-request workflow for pull requests into any branch,
including `dev`. Workflows triggered by pushes or schedules, including the
weekly fuzz and security runs, run only on `main`.

## Extension contract

An *existing configuration* is a schedule in the trusted catalog on `main`
(`artifacts/schedules/`, indexed by `artifacts/schedule-catalog.tsv`), used
through a public API that `main` provides. A caller *selects* an extension by
choosing it explicitly at runtime, for example through a configuration type,
a schedule, or a backend handle.

1. For every existing configuration, `dev` built with default Cargo features
   MUST produce the same setup, commitment, and proof bytes, and the same
   transcript state, as `main` at the merge base.
2. Rule 1 MUST also hold when every extension Cargo feature is enabled and no
   extension is selected. Cargo unifies features across a build, so a feature
   that one dependent enables is enabled for every crate in that build.
3. Enabling a Cargo feature MUST NOT select an extension.
4. For every existing configuration, the `dev` verifier MUST accept exactly the
   proofs that the `main` verifier accepts.
5. A compute backend MUST produce the same proof bytes as the CPU backend for
   every configuration it supports.
6. `dev` MUST NOT modify an existing trusted schedule artifact. It MAY add new
   artifacts.
7. Verifier-reachable extension code MUST follow the
   [verifier no-panic contract](verifier-contract.md).

Together these rules make every extension conservative: restricted to existing
configurations, `dev` is `main`.

Rules 1, 2, and 5 compare bytes. The comparison is meaningful because the
honest prover is deterministic: fold grinding takes the first accepted nonce
(`first_jointly_accepted_nonce` in
`crates/akita-prover/src/protocol/fold_grind.rs`). An extension that adds
prover randomness, such as zero knowledge, affects only the configurations
that select it.

Rule 4 cannot be tested exhaustively. Rules 1 and 2 cover honest proofs. The
tamper and soundness suites in `crates/akita-pcs/tests/`
(`protocol_soundness.rs`, `transcript_hardening.rs`, and
`transcript_hardening_proptest.rs`) cover rejection. Review covers the rest.

**Enforcement gap.** No test on `main` pins proof bytes today. Until one does,
the author of a `dev` pull request that changes files from `main` MUST compare
proof bytes against `main` for the affected configurations and record the
result in the pull request.

## Keep `dev` close to `main`

A merge conflict between the branches occurs where both branches change the
same lines. Keep the set of `main` lines that `dev` changes small, and change
them in predictable places.

### Put extension code in new files

- Put extension code in new crates or new modules.
- Gate it behind a Cargo feature that is off by default.
- Change files from `main` only to register the extension at a declared seam:
  a module declaration, a trait implementation, an enum variant, a workspace
  member, or a book `SUMMARY.md` entry.

### Land seams on `main` first

When an extension needs a file from `main` to change beyond a registration,
first land a behavior-preserving refactor on `main` that creates the seam.

1. Open a pull request against `main` that restructures the existing code so
   the extension can attach to it. Include no extension code. Link the
   extension's tracking issue, and state that the protocol does not change.
2. Mark the seam with a doc comment that starts with `Extension seam:` and
   names the tracking issue. Maintainers can then find seams before they
   refactor around them: `rg "Extension seam:"`.
3. Build the extension on `dev` against the seam.

For example, a zero-knowledge fold adds masking to the existing fold. Instead
of copying the fold orchestration into a zero-knowledge module, a seam pull
request factors the orchestration so that the transparent fold and the
zero-knowledge fold both call it.

Prefer a seam that `main` uses itself, such as a function that two existing
call sites share. A trait whose only implementation on `main` is trivial is the
weakest kind of seam, and its tracking-issue link is its justification.
Declared seams are intentional boundaries under the single-source-of-truth
rules in [`AGENTS.md`](../AGENTS.md).

### Carry a seam while its `main` pull request is in review

Maintainer review can take weeks. To build on a seam before it lands on `main`:

1. Sync `main` into `dev`, so that the next pull request contains only the
   seam.
2. Open a pull request from the seam branch into `dev`, and squash-merge it.
3. List the carried seam in the extension's tracking issue.
4. If `main` merges a different version of the seam, take the `main` version
   during the next sync and adapt the extension to it.

`main` squash- or rebase-merges the seam, so the commit on `main` never shares
history with the copy on `dev`. The later sync merges cleanly wherever the two
copies are identical.

### Sync `main` into `dev`

1. After each merge into `main`, open a pull request from `main` into `dev`.
2. Merge it with **Create a merge commit**, not a squash merge. Check the
   selected merge method before you merge. A merge commit makes `main` a parent
   of `dev` and moves their merge base forward. After a squash merge, the next
   sync starts from the old merge base and conflicts on lines that `main` has
   changed since, and the footprint command below lists changes from `main`.
3. Resolve conflicts in favor of the `main` code, then reattach the extension at
   its seam.
4. Regenerate generated files instead of merging them by hand. Run
   `scripts/generate-schedule-artifacts.sh` for schedule artifacts. For a
   `Cargo.lock` conflict, start from the `main` version and let a Cargo build
   add the `dev` entries. Do not run `cargo update`.

Never rebase, reset, or force-push `dev`.

### Measure the footprint

List the files from `main` that `dev` modifies, deletes, or renames:

```bash
git fetch origin
git diff --stat --diff-filter=MDR origin/main...origin/dev
```

Each listed file SHOULD be a seam registration, a generated file or lockfile, or
a carried seam with an open pull request against `main`. Move any other change
behind a seam. Files that exist only on `dev` do not cause conflicts and are
not part of the footprint.

## Move an extension to `main`

`dev` never merges into `main` as a whole. `main` requires a linear history,
and a single squashed commit of `dev` cannot be reviewed. When maintainers
decide to adopt an extension:

1. Create a branch from the current `main`.
2. Bring over that extension's files and seam registrations only.
3. Run `git diff origin/main...HEAD` and confirm that no other change came
   along.
4. Open a pull request against `main`. Follow the spec workflow in
   [`CONTRIBUTING.md`](../CONTRIBUTING.md) for protocol changes.

After the pull request merges, the next sync brings the change into `dev`.
Resolve any conflict in favor of the `main` version.
