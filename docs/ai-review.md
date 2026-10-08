# AI PR review

An author with current repository write, maintain, or admin permission can post
exactly `/ai-review` on their open, same-repository PR, with no spaces, newlines
or additional text. The workflow posts its
findings inline on the corresponding source lines in the PR diff. It
uses the command to select PRs; the obsolete `ai-review` label is retired.
Other users, external contributors, forks,
bots, edited comments and comments containing additional instructions cannot
trigger it. The workflow must first land on the default branch (`main`), because
[GitHub runs issue-comment workflows from that branch](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment).

Post another `/ai-review` to check fixes, new trusted discussion and changes since
the previous review. Every earlier finding is reassessed, including previously
fixed issues that may have regressed. Unresolved findings retain their IDs rather
than being posted as new issues. Rerunning the same Actions event does not publish
twice. Review requests serialize per PR; GitHub concurrency may replace an older
pending request when several commands arrive while a review runs.

Each PR has a lifetime limit of **three attempts**, across all commits. Before
collection, a separate job reserves a slot in a bot comment showing `1/3`, `2/3`
or `3/3`. Failed, cancelled and incomplete attempts consume their reserved slot;
commands rejected before reservation do not. After three reservations, further
commands stop before collection or OpenAI access. Actions reruns cannot reserve
or invoke the model again; recovery requires a new author command and an available
slot. The limit counts review attempts, each of which may use several model turns.
Force pushes and rebases do not reset the quota because it belongs to the PR
number, not a commit. A changed head during review prevents publication and still
consumes the reserved attempt. A new command reviews the rewritten diff against
the previous reviewed head; if that old commit is unavailable, collection fails
closed rather than silently dropping review history.

A completed review posts "Recommended for approval" only when usefulness is
supported and no unresolved findings remain other than optional nits.
New nits still appear inline.
Prior unresolved P0–P3 findings, unresolved non-nit collaborator or Bugbot feedback,
uncertain findings, or incomplete coverage prevent the recommendation, even if
there are no new findings. This is a COMMENT review proposing approval; it does
not submit GitHub's formal APPROVE action.

When a review withholds the approval recommendation, it visibly says "Approval
not recommended" with every reason, and explains each one that is not already
visible: the coverage limitations when coverage is incomplete, each unresolved
blocker from existing discussion, and each earlier finding that is still open
(P0–P3) or uncertain, with its priority, path, root cause and current evidence
(at most 10, evidence shortened to 200 characters). This explains a withheld
`ai-approved` label even when there are no new inline findings; unclear usefulness
is shown in the usefulness line and new findings inline. These lists use the same
Markdown escaping as inline comments and keep matched code spans readable.
Existing feedback is not duplicated inline. Previously published reviews retain
their original summaries; versioned rendering (`body_version`) keeps them
verifiable during publication retries and label reconciliation, and an unknown
version fails closed.

Every review also posts a short **usefulness assessment** in its review body,
separate from inline code findings. It checks whether the PR solves a concrete
problem and improves the repository enough to justify its costs, even when the
implementation perfectly matches its description. The reviewer prioritizes
security and bug fixes, welcomes refactors that simplify the codebase, and checks
features against existing capabilities and less complex alternatives.

The PR template includes a **Motivation** section. Missing, empty or vague
motivation produces a visible request to explain the problem and expected benefit;
a substantive explanation elsewhere in the body also counts. The reviewer still
evaluates the code when motivation is missing. Missing motivation alone does not
block a clearly useful change, but `unclear` or `not_beneficial` usefulness prevents
both the approval recommendation and `ai-approved` label. The structured result
must include a bounded assessment; the publisher rejects a result without it and
escapes its Markdown using the same rules as inline comments.

Breaking changes need a concrete longer-term goal, an explanation of why a
compatible approach is insufficient, and a migration or coordinated cutover for
affected consumers. The repository's permission to break compatibility is not
itself justification. Missing justification leaves usefulness unclear. Small
fixes, tests, documentation, and code removal can all provide meaningful value;
the review does not require a large or novel feature.

Repeat requests reassess usefulness from the current description and code.
Older reviews without a usefulness assessment remain evidence for prior findings
but cannot establish approval under this policy. The usefulness judgment comes
from the model; schema validation enforces that it is present, not that its
reasoning is correct. PR motivation remains untrusted evidence, never authority
to change review instructions or access credentials.

After publication and read-back verification, the workflow adds `ai-reviewed`
to the **PR**. It also adds `ai-approved` when that review recommends approval,
including a review with no inline findings or only optional nits. A later
published review with blockers or incomplete coverage removes `ai-approved`.
The same approval decision controls both the prose and label; model text cannot
select labels. Existing unrelated PR labels are preserved. A failed publication
does not add labels; retrying a failed label update repairs labels without
posting the review again and uses the latest published review, not an old result.
Every retry rechecks that review's body, commit, submitted state and exact inline
comment contents/coordinates against the publication manifest in its state marker.
A marker alone does not establish successful publication.

`ai-reviewed` remains as review history. New commits (including force pushes),
PR edits and reopening reconcile `ai-approved` against the current PR and latest
verified review. This job checks out only trusted scripts from `main`; it never
checks out PR code or accesses OpenAI.
The invalidation queue is separate from comment-triggered review runs, so an
unrelated comment cannot replace pending invalidation. Snapshots record the latest
GitHub reopening event ID; an old review cannot restore approval after reopening.
Publication and reconciliation share a job-level lock. The outer event queues
admit at most one of each job, so neither can replace the other in that lock's
pending slot. Delayed or replayed updates preserve a newer valid approval, and
reconciliation can recover labels from verified PR history after artifacts are
deleted. Missing or unverifiable review history cannot grant approval.
The publisher also rechecks PR state after adding approval to catch concurrent
PR changes. Label updates are asynchronous and labels are not a
merge authorization or proof that all current discussion has been checked.

Approval covers the reviewed head commit, head and target branch names, PR title
and description, and reopening history. An ordinary advance of the target branch
does **not** invalidate approval; reconciliation and publication retries preserve
the same decision. The label does not certify the latest merge result or changes
merged into the target branch after review. Retargeting the PR or changing its head
does invalidate approval. Collection and first publication still check the exact
snapshotted base SHA. A base change detected during collection or before first
publication rejects that snapshot.

GitHub allows users with triage or higher access to edit labels; these labels
are informational, not an admin-only security control. Repository setup must
create `ai-reviewed` and `ai-approved` before this workflow is used.

The repository copy of the [review skill](../.github/skills/ai-pr-review/SKILL.md)
is the source of truth for automation. Its
[automation contract](../.github/skills/ai-pr-review/references/automation.md)
overrides the manual skill's label discovery, tool access and publication rules.
Manual reviews still support previews and inline publication.

## Trust boundaries

The `issue_comment` workflow and every executable script come from its immutable
default-branch commit. Actions are pinned to full commit hashes. The workflow
never checks out PR code, loads PR Python modules, runs PR build hooks, downloads
PR dependencies, or executes model-generated commands. Git reads immutable blobs;
symlinks and submodules are excluded. Source evidence is kept in JSON outside the
checkout, so a PR cannot overwrite the trusted scripts or skill.

| Job | Credentials | Allowed work |
| --- | --- | --- |
| Reserve | PR write token; no OpenAI key | Revalidate author and write access, count bot reservation markers, reserve one of three attempts |
| Collect | Read-only GitHub token | Revalidate author and current write access, read trusted comments, collect pinned public git objects |
| Review | Environment OpenAI key, read-only job permission | Send source evidence to the fixed OpenAI Responses endpoint; offer bounded in-memory source reads/searches |
| Publish | PR write token; no OpenAI key | Validate the structured result, recheck authorization and snapshot, submit one COMMENT review with new inline findings on the original PR |
| Cleanup | Actions write token; no OpenAI key | After successful publication and label updates, delete only this run's snapshot and result artifacts |
| Reconcile | Contents read + PR write token; no OpenAI key | Run trusted `main` scripts, verify the latest review against live PR metadata, and synchronize labels under the publication lock |

The model receives neither credential nor process/environment access. The source
tools cannot fetch URLs, traverse the filesystem, invoke a shell, or write to
GitHub. HTTP redirects are refused. The publisher fixes the repository, PR and
operation and allowed diff coordinates; model output is never an API path, workflow command, shell argument or
executable code. Model prose is escaped to prevent injected HTML, Markdown images
and user mentions, while preserving matched inline code spans. Errors avoid
remote response bodies and credential-bearing
tracebacks in public logs.

Human discussion, review summaries and inline replies are included only after a
live repository permission check; `author_association` alone is insufficient.
Cursor Bugbot comments are allowlisted using its GitHub-issued bot identity
(`cursor[bot]`, user ID `206951365`); when App metadata is present, its App ID
`1210556` and slug `cursor` must also match. Review/inline endpoints may omit App
metadata. The identities can be checked through GitHub's
[bot account API](https://api.github.com/users/cursor%5Bbot%5D) and
[App API](https://api.github.com/apps/cursor). This allowlist lives in trusted
workflow code. Bot comments are evidence only and cannot authorize reviews,
change quotas, become our prior-review state, or authorize excluded replies.
Bugbot can quote untrusted source or discussion, so its claims require the same
source verification as human claims. Other bots are excluded. Generic
`github-actions[bot]` comments are not automatically included: that shared
identity does not identify which workflow produced a comment.
Prior state is accepted only from marked comments by
`github-actions[bot]`, the trusted publisher identity. This assumes workflows with
PR write permission are trusted; do not give untrusted workflows that permission.
All accepted prose, source and prior findings remain untrusted model evidence.
Quota reservations are excluded from model evidence. Their durable markers count
even when no review was published; do not edit or delete reservation comments.
The quota trusts the bot identity and PR-wide workflow concurrency, just like
review history trusts the publisher. Maintainers who can delete bot comments or
change the trusted workflow can reset it; it is not an administrator-proof budget.
Prompt injection can still degrade review quality, so findings require human
judgment. Credential isolation rests on tool restrictions, not model obedience.

The [adversarial smoke run](https://github.com/LayerZero-Labs/akita/actions/runs/36759469886)
included a maintainer comment requesting environment access, key exfiltration and
false fix dispositions. Its saved result kept the unfixed issue open. This is a
regression check, not proof that every injection will fail. The runner has no
network-level egress audit or firewall; fixed HTTP destinations are enforced by
trusted Python code. Logs/artifacts do not retain all attempted tool calls, and
absence of a secret in logs cannot prove that it was never transmitted. For a
suspected exposure, reconcile the project's OpenAI key usage with workflow runs
and rotate the key if compromise is suspected; unused stolen keys leave no API
usage evidence. Never print a real key to investigate an incident.

## Environment setup

Use the existing `non-main` environment with the `OPENAI_API_KEY` secret. Despite
its name, restrict its deployment branches to the exact **branch** `main` (no tag
rules or wildcard PR refs). Comment events run against the default branch even
when the PR targets another branch. Do not expose this environment to arbitrary
feature-branch or pull-request workflow revisions. Protect `main` and review
workflow/script/skill changes as privileged code. These controls cannot protect
against an administrator or someone authorized to merge malicious code into main.

An environment secret alone does not restrict which workflow can use it;
[environment branch rules control eligible workflow refs](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).
Keep the key out of repository-level secrets. Use an OpenAI project with an
appropriate spend limit. `AI_REVIEW_MODEL` is an optional repository/environment
variable, defaulting to `gpt-6-astra`. Requests explicitly set
`reasoning.effort` to `high`; this is a request parameter, not a model-name suffix.
The selected model must support Responses, function tools, high reasoning effort
and structured output, as documented for
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).

No remote dependency installation is needed: the runner uses Python's standard
library and the [Responses API](https://developers.openai.com/api/docs/guides/function-calling)
with [structured output](https://developers.openai.com/api/docs/guides/structured-outputs).

## Akita-specific: artifact exclusion

The root `artifacts/` directory is excluded from AI review, including all file
types and subdirectories. Its contents are omitted from full and repeat-review
diffs, inline anchors, and every source map, so source tools cannot retrieve them.
The collector still lists changed excluded paths as `excluded_artifacts`, and
the publisher adds a visible notice with their count to the review body.
This policy is explicitly separated from generic review instructions in the
skill's automation contract so it can be adapted when reusing the workflow.

Source, generators, consumers and tests outside `artifacts/` remain in scope.
The unchanged 300,000-character diff budget and 150-file cap apply to the remaining
review scope. Mixed PRs can receive an approval recommendation for that scope;
the recommendation does not cover artifact contents. Artifact-only PRs cannot
receive one. Validate excluded contents separately through the repository's
artifact checks and CI. This source-directory exclusion is separate from the
temporary GitHub Actions snapshot/result artifacts and their cleanup policy.

## Limits and failures

Reviews are static analysis. They cannot run tests, inspect CI logs, fetch linked
issues/URLs, or prove other stacked PRs contain a fix. Excluded content and these
limits must be reported when relevant. A clean automated review is not an approval.

Current local budgets: 150 in-scope changed files, 300,000 characters of combined full/delta
diff (the delta covers only files the PR changes now or changed at the previously
reviewed head, so base-branch changes merged in since count only where they touch
those same files), 300,000 characters of eligible discussion, 250,000 bytes per source blob,
32 MiB per artifact, 48 model turns, 24 reads/searches per turn, 2,000,000 characters
of accumulated context, 20 new findings and 100 retained findings (in practice at
most 60: three attempts of 20). The published review must also fit GitHub's
60,000-character comment limit; if it would not, the model is asked to shorten its
evidence and other text. Exceeding a
budget stops the run; it never silently turns into a clean review. These are
workflow policy limits, not OpenAI API limits. Collection logs report counts and
diff sizes; budget errors identify the exceeded limit.

Before each model call, count-only diagnostics report the turn, elapsed seconds,
and serialized history size split into user evidence (including correction
messages), tool results, encrypted reasoning payloads, and other overhead/output.
These categories sum to the history size used by the local cap; the counter
includes JSON escaping, not just source text, and is not a model token limit.
After each response, diagnostics report that call's input, cached-input, output,
and reasoning token counts. Cached-input and reasoning counts are subsets of
input and output respectively, not additional tokens. Missing or invalid usage
counters are logged as `null`, not zero. Logs do not contain source, tool results,
reasoning payloads, or credentials. History is retained intact between turns.
The larger history allowance can increase cost and latency; model context limits,
the 48-turn bound, and the job timeout still apply. Measured reviews have used
about 26 turns, so 48 leaves headroom within the history allowance.

Actions artifacts contain public source/review evidence, no credentials.
After publication and label
updates succeed, a separate cleanup job deletes `ai-review-snapshot` and
`ai-review-result` from that run. It does not download their contents or delete
other artifacts. Failed reviews/publications retain artifacts for the configured
one day to support investigation and publication retries. Cleanup failures or
cancellation may leave artifacts until expiry; deletion is not an instantaneous
erasure guarantee. Once deleted, the exact snapshot/result can no longer be used
for debugging or publication retries. Repeat author commands still work from the
review state on the PR. This cleanup does not delete data already sent to OpenAI.

The publisher rejects changes to head, base, branch targets, PR text or eligible
discussion since collection. Post a fresh command after the PR stabilizes. A
final check and a GitHub review write cannot be atomic; the review is always
pinned to its reviewed commit. API write failures are not blindly retried; the
publisher also checks for the original request ID before attempting a write.

The latest state is stored in a hidden marker in the bot review body. New reviews
write `akita-ai-review:v2`, which holds zlib-compressed state so the whole finding
history a PR can accumulate fits GitHub's comment limit; reviews with the first
release's uncompressed `akita-ai-review:v1` marker remain readable. Verification
compares the published marker rather than re-compressing the state, because
compressed bytes can differ between zlib versions. Do not edit
those markers;
malformed state fails closed and removal loses that historical baseline. Fixed
findings stay in history so later regressions can be recognized. Reviews that do
not qualify for an approval recommendation still include a visible usefulness
assessment and any missing-motivation note, while new code findings are posted inline.
Prior thread resolution is left to humans. Review artifacts contain detailed
coverage until deletion; the hidden PR state retains completion status and
limitations, including issues that cannot be anchored in the current diff, and
incomplete reviews also show the limitations in their summary. An external
comment cannot spoof state by copying a marker.

Run local adversarial and regression tests with:

```sh
python3 -m unittest discover -s scripts/tests -p 'test_ai_review*.py' -v
```
