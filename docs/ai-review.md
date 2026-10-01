# Author-requested AI reviews

An author with current repository write, maintain, or admin permission can post
exactly `/ai-review` on their open, same-repository PR, with no spaces, newlines
or additional text. The workflow posts its
findings inline on the corresponding source lines in the PR diff. It
does not require the `ai-review` label. Other users, external contributors, forks,
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

A completed review with no unresolved findings, or only optional nits, posts
"Recommended for approval" in its review body. New nits still appear inline.
Prior unresolved P0–P3 findings, unresolved non-nit collaborator or Bugbot feedback,
uncertain findings, or incomplete coverage prevent the recommendation, even if
there are no new findings. This is a COMMENT review proposing approval; it does
not submit GitHub's formal APPROVE action.

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

## Limits and failures

Reviews are static analysis. They cannot run tests, inspect CI logs, fetch linked
issues/URLs, or prove other stacked PRs contain a fix. Excluded content and these
limits must be reported when relevant. A clean automated review is not an approval.

Current budgets: 150 changed files, 300,000 characters of combined full/delta
diff, 300,000 characters of eligible discussion, 250,000 bytes per source blob,
32 MiB per artifact, 32 model turns, 24 reads/searches per turn, 900,000 characters
of accumulated context, 20 new findings and 100 retained findings. Exceeding a
budget stops the run; it never silently turns into a clean review. Artifacts
contain public source/review evidence, no credentials, and expire after one day.

The publisher rejects changes to head, base, branch targets, PR text or eligible
discussion since collection. Post a fresh command after the PR stabilizes. A
final check and a GitHub review write cannot be atomic; the review is always
pinned to its reviewed commit. API write failures are not blindly retried; the
publisher also checks for the original request ID before attempting a write.

The latest state is stored in a hidden marker in the bot review body. Do not edit
those markers;
malformed state fails closed and removal loses that historical baseline. Fixed
findings stay in history so later regressions can be recognized. Reviews that do
not qualify for an approval recommendation carry only this hidden marker in
their review body, while any new findings are posted inline.
Prior thread resolution is left to humans. Review artifacts retain coverage and
limitations, including issues that cannot be anchored in the current diff. An external
comment cannot spoof state by copying a marker.

Run local adversarial and regression tests with:

```sh
python3 -m unittest discover -s scripts/tests -p 'test_ai_review.py' -v
```
