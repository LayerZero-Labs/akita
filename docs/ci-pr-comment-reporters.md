# Privileged CI pull-request comment reporters

This page explains the trust model and state transitions used by Akita's
benchmark and test-timing comment workflows. It is an architecture explanation
for maintainers changing these workflows, not a claim that benchmark results
from an untrusted fork are independently verified.

The implementation lives in:

- `.github/workflows/profile-bench.yml` and `.github/workflows/ci.yml`, which run
  pull-request code and produce structured artifacts;
- `.github/workflows/profile-bench-comment.yml` and
  `.github/workflows/test-timing-comment.yml`, which consume those artifacts and
  write comments;
- `scripts/ci_comment_workflow.py`, which owns the shared identity, size, and
  final-write policy; and
- `scripts/tests/test_ci_comment_workflows.py`, which exercises that policy.

## The short mental model

There are two separate security decisions:

1. **May this pull request spend CI compute?** The parent `pull_request`
   workflow answers this before running expensive jobs.
2. **May data from this completed run be written to this pull request?** A
   separate `workflow_run` reporter answers this while holding the write token.

The data flow is:

```text
untrusted pull-request code
        |
        | structured artifact (untrusted data)
        v
privileged workflow_run reporter
  trusted default-branch checkout
        |
        | resolve identity -> bound inputs -> render -> revalidate destination
        v
one marker-owned pull-request comment
```

The reporter authenticates **where the data came from** and **where the comment
will be written**. It does not authenticate that a fork's measurements are
honest. Identity binding turns an anonymous claim into a claim attributable to
one workflow run; it does not turn the claim into trusted data.

## Actors and sources of authority

| Actor or input | Trust level | What it is allowed to decide |
| --- | --- | --- |
| Pull-request workflow and artifact | Untrusted for fork PRs | Measurement payload only |
| `workflow_run` event | GitHub-authenticated identity | Triggering run ID, head repository, branch, SHA, workflow, and conclusion |
| GitHub pull-request and Actions APIs | Identity authority | Current PR identity, run history, artifact ownership and size, merge base |
| Reporter checkout | Trusted | Policy and Markdown rendering, because it is checked out from the repository default branch |
| Reporter `GITHUB_TOKEN` | Privileged capability | Read Actions metadata and upsert one issue comment |

The reporter never checks out or executes the pull request's version of its
helper or renderer. It treats every downloaded file, file name, JSON value, and
claimed commit as attacker-controlled input.

## Reporter state machine

Both reporters follow the same six states. A transition that cannot establish
its invariant stops or degrades the report as described below.

### 1. Accept a reportable completion

The reporter accepts completed `pull_request` runs whose conclusion is not
`cancelled`. Failed runs are intentional inputs: they can contain useful partial
results or a structured failure summary. Cancelled runs are omitted because
their report artifact is not expected to be complete.

This trigger does not authorize expensive compute. The parent workflow already
made that decision before producing the artifact.

### 2. Resolve exactly one destination PR

The reporter first uses GitHub's native workflow-run-to-PR association. GitHub
can omit that association for cross-fork runs, so the fallback searches open
pull requests by the exact fork owner and branch. It accepts a fallback match
only when all of these values agree:

- head repository;
- head branch;
- full head commit SHA; and
- base repository.

Zero matches, multiple matches, incomplete identities, and malformed full
commit IDs fail closed. After resolution, the reporter fetches the PR and
records its number plus the complete head and base identity:

```text
(PR number,
 head repository, head branch, head SHA,
 base repository, base branch, base SHA)
```

That tuple is the destination capability for the rest of the job. A PR number
alone is never sufficient.

### 3. Resolve the current artifact before download

The reporter asks GitHub for exactly one unexpired artifact with the expected
name on the triggering run. GitHub's artifact metadata must bind it to the
resolved head SHA. The artifact size reported by GitHub must be at most
5,000,000 bytes before the download step receives an artifact ID.

Artifact identity and size checks happen in trusted code. The artifact contents
remain untrusted after these checks.

### 4. Establish optional comparison identities

The current-run summary is the primary report. A main or previous-run
comparison is optional and must not lend its label to unauthenticated data.
The benchmark and timing reporters obtain those identities differently.

#### Benchmark comparisons

The benchmark producer includes current, merge-base, and previous-run summaries
in one artifact. It also includes `baseline-metadata.json`, whose identities are
claims until the reporter checks them.

The merge-base comparison is accepted only when GitHub's comparison API returns
the same full merge-base SHA claimed by the artifact. Its summary directory and
human-readable label are both enabled by that successful check.

The previous-run comparison is accepted only when GitHub confirms all of the
following:

- the claimed run is earlier than the current run;
- both runs have the same workflow ID and expected workflow name;
- both runs are `pull_request` runs;
- the previous run completed with `success` or `failure`;
- its full head SHA equals the artifact's claim;
- its head repository and branch equal the current fork PR;
- its PR association, when GitHub supplies one, includes the current PR; and
- it still has exactly one unexpired, correctly head-bound artifact within the
  artifact metadata limit.

The previous commit is deliberately **not required to be an ancestor** of the
current head. A rebase or force-push is a valid new update of the same fork
branch and PR. Here, continuity means “an earlier run of this workflow for this
fork repository, branch, and PR,” not “a commit in the current Git ancestry.”

The merge-base and previous-run checks are independent. Rejecting one clears
both its summary directory and its label, while leaving the current summary and
the other authenticated comparison available.

#### Test-timing comparisons

The timing reporter selects comparison artifacts itself, so their identities do
not come from the current artifact:

- **Main** is the most recent successful `push` run on the default branch with
  the expected workflow name and a bounded timing artifact.
- **Previous** is the first other eligible completed `pull_request` run in
  GitHub's newest-first listing for the same branch. The current run is excluded;
  the candidate must have the same workflow name, fork repository and branch,
  the current PR association when GitHub supplies one, an allowed `success` or
  `failure` conclusion, and a bounded timing artifact.

These baselines answer different questions. The benchmark “main” column is the
current PR's Git merge base measured alongside the head on the benchmark
runner. The timing “main” column is the latest available successful default-
branch run. Neither should be relabeled as the other.

### 5. Render with trusted code and bounded inputs

Only the default-branch renderer constructs Markdown. The reporter checks each
structured input that it will read as a regular, non-symlink file of at most
2,000,000 bytes. Missing optional baseline files are ignored; a missing primary
benchmark summary produces no benchmark comment, while the timing reporter can
construct a trusted failure summary for a missing current timing summary.

The final UTF-8 comment must:

- start with the workflow's exact ownership marker;
- be at most 60,000 bytes; and
- be rendered to a local file before the write step.

The marker is an ownership key, not content supplied by the artifact. It lets
the reporter update its own prior comment without selecting another bot or
human comment.

### 6. Revalidate immediately before the write

Immediately before creating or updating the comment, the reporter fetches the
PR again. It writes only if the PR is still open and every recorded destination
field still agrees: PR number, head repository, head branch, head SHA, base
repository, base branch, and base SHA.

This final check closes the race between initial resolution and publication. A
new push, rebase, retarget, closed PR, deleted fork, or other identity change
causes the old reporter to skip its write instead of publishing stale results.

## Failure semantics

“Fail closed” does not always mean “fail the workflow job.” It means that
unestablished authority is never converted into a comment or comparison.

| Failure | Result |
| --- | --- |
| Destination cannot be resolved exactly | No artifact download and no comment |
| Current artifact is missing, ambiguous, expired, oversized, or bound to another head | No download and no comment |
| Primary benchmark summary is missing | No benchmark comment |
| Current timing summary is missing | Render a trusted failure summary from the resolved run identity |
| One benchmark baseline identity is invalid | Omit that baseline's data and label; keep the current report and any independently authenticated baseline |
| Optional timing baseline is unavailable | Render without that comparison |
| A selected timing artifact cannot be downloaded or its summary is invalid | Fail the reporter job and retain the existing PR comment |
| Structured input is oversized, a symlink, or not a regular file | Do not read it; the render step fails if it is required |
| Rendered comment has the wrong marker, invalid UTF-8, or exceeds the body limit | Skip the write |
| PR head or base identity changes before publication | Skip the write as stale |
| GitHub rejects an otherwise authorized write | Fail the reporter job so the operational failure is visible |

Policy rejections normally emit a workflow warning and exit successfully after
suppressing the unsafe output. GitHub API and write failures normally fail the
job because retry or maintainer attention can resolve them.

## Resource limits

| Boundary | Current limit | Purpose |
| --- | ---: | --- |
| Actions artifact size reported by GitHub | 5,000,000 bytes | Reject unexpectedly large downloads before passing an artifact ID to the download action |
| Each structured file read by a renderer | 2,000,000 bytes | Bound memory and parser input for current and baseline summaries |
| Final rendered comment | 60,000 bytes | Bound bot output and stay below GitHub's comment limit with margin |
| Reporter job | 10 minutes | Bound privileged job lifetime |

The artifact metadata limit is not an uncompressed-total guarantee. The
per-file checks are the boundary before the trusted Python renderers read
structured inputs. Changing either limit changes a security policy and requires
tests at the exact boundary and one byte beyond it.

## Compute authorization is a separate plane

The parent workflows can execute contributor-controlled code, so expensive jobs
use path filters, timeouts, bounded matrix fan-out, and author/repository policy.
External fork authors need either an eligible repository association or the
`ci-approved` label in addition to GitHub's repository-level Actions approval.
Repository-level approval remains authoritative because YAML cannot override a
run that GitHub has not allowed to start.

The reporter does not repeat the author gate. It executes only trusted
default-branch code, consumes bounded data, and is intentionally responsible
for reporting already-completed cross-fork runs. Its privilege is constrained
instead by read-only repository permissions plus `issues: write`, exact
destination binding, and the final identity revalidation.

## Change checklist

When changing either reporter:

1. Identify whether the change affects compute authorization, report
   publication, or both. Do not let one plane implicitly stand in for the other.
2. State which value is untrusted and which GitHub API value authenticates it.
3. Keep a comparison's identity, directory, and label under one acceptance
   decision.
4. Keep optional comparisons independent from the primary report and from each
   other.
5. Preserve exact head and base revalidation at the final write.
6. Add regression cases to `scripts/tests/test_ci_comment_workflows.py` for
   malformed types, missing associations, cross-fork identity, rebases, size
   boundaries, and failure isolation as applicable.
7. Run the script/workflow tests and documentation guardrails. Expensive
   benchmark, schedule-regeneration, and full Rust validation are not needed
   for reporter-only changes unless another changed path requires them.
