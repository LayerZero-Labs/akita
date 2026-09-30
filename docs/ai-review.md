# Author-requested AI reviews

An author with current repository write, maintain, or admin permission can post
exactly `/ai-review` on their open, same-repository PR. The workflow posts its
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
| Collect | Read-only GitHub token | Revalidate author and current write access, read trusted comments, collect pinned public git objects |
| Review | Environment OpenAI key, read-only job permission | Send source evidence to the fixed OpenAI Responses endpoint; offer bounded in-memory source reads/searches |
| Publish | PR write token; no OpenAI key | Validate the structured result, recheck authorization and snapshot, submit one COMMENT review with new inline findings on the original PR |

The model receives neither credential nor process/environment access. The source
tools cannot fetch URLs, traverse the filesystem, invoke a shell, or write to
GitHub. HTTP redirects are refused. The publisher fixes the repository, PR and
operation and allowed diff coordinates; model output is never an API path, workflow command, shell argument or
executable code. Model prose is escaped to prevent injected HTML, Markdown images
and user mentions. Errors avoid remote response bodies and credential-bearing
tracebacks in public logs.

Human discussion, review summaries and inline replies are included only after a
live repository permission check; `author_association` alone is insufficient.
Other bots are excluded. Prior state is accepted only from marked comments by
`github-actions[bot]`, the trusted publisher identity. This assumes workflows with
PR write permission are trusted; do not give untrusted workflows that permission.
All accepted prose, source and prior findings remain untrusted model evidence.
Prompt injection can still degrade review quality, so findings require human
judgment. Credential isolation rests on tool restrictions, not model obedience.

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
variable, defaulting to `gpt-5.4`; the selected model must support Responses,
function tools, high reasoning effort and structured output.

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
pinned to its reviewed commit. API write failures are not blindly retried: a rerun checks
for the original request ID before attempting a write.

The latest state is stored in a hidden marker in the bot review body. Do not edit
those markers;
malformed state fails closed and removal loses that historical baseline. Fixed
findings stay in history so later regressions can be recognized. A review with no
new findings carries only this hidden marker; no visible summary is posted.
Prior thread resolution is left to humans. Review artifacts retain coverage and
limitations, including issues that cannot be anchored in the current diff. An external
comment cannot spoof state by copying a marker.

Run local adversarial and regression tests with:

```sh
python3 -m unittest discover -s scripts/tests -p 'test_ai_review.py' -v
```
