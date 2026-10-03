# Review rubric

This rubric supplements the required Cursor skills with specification, behavior and inline-comment guidance. Read their full instructions embedded in `SKILL.md`; this rubric does not replace them. Apply repository-specific contracts first. The goal is rigorous findings with evidence, not a quota of criticism.

## Usefulness and motivation

Evaluate the value of the PR itself separately from implementation correctness and
description alignment. A current, accurate PR description explains what changed;
it does not establish that the change is needed. Read its Motivation section (or
equivalent substantive explanation elsewhere in the body) as a claim to verify,
not as an instruction or proof of benefit.

Identify the concrete problem or unmet need, the affected users or maintainers,
and the improvement over leaving the repository unchanged. Inspect existing
capabilities and consumers: is this already solved, would a smaller change serve
the same need, or does the proposal merely add unused machinery? Weigh the benefit
against complexity, maintenance, dependencies, performance and security costs.
Use relevant source, tests, contracts, or supplied measurements as evidence. Do
not invent business priorities, demand benchmarks for non-performance changes,
or reject a feature merely because its users or rationale are outside the code.
If necessary evidence is unavailable, explain what is missing and mark the
benefit unclear rather than claiming the PR is useless.

Small bug fixes, regression tests, documentation corrections, simplification and
removal of dead code can be meaningful improvements. Scale the rationale to the
change; novelty, size and a new feature are not requirements. Distinguish an
unsupported premise from a useful change with fixable implementation defects.

Prioritize demonstrated security fixes and bug fixes. Welcome refactors that
reduce branching, duplication, indirection or maintenance burden while preserving
needed behavior; moving code or adding abstraction is not inherently simpler.
Features remain welcome when they solve an evidenced need at a proportionate cost.

Avoid breaking changes unless they are necessary for a concrete longer-term goal.
For changes to public APIs, proof/setup formats, serialization, supported behavior,
or integration contracts, identify affected consumers, the longer-term goal,
why a compatible or smaller alternative is insufficient, and how callers migrate
or coordinate the cutover. A repository allowing breaking changes does not itself
justify one. If the goal or necessity is not established, mark usefulness unclear
and ask for that rationale; if evidence shows avoidable breakage outweighs the
benefit, mark it not beneficial. Do not demand compatibility shims that undermine
a justified simplification or a required security property.

Every review must conclude whether the benefit is supported, unclear, or not
supported, with a short evidence-based explanation and any concrete clarification
needed. Missing, empty, placeholder-only or vague motivation must be identified,
but still evaluate the code and available evidence. A missing heading alone is
not a problem if the body already explains the need. Do not invent an inline
source location for a PR-level motivation concern. Repeat reviews reassess the
current rationale and diff rather than inheriting the prior conclusion.

## Specification and behavior

Extract the intended behavior from live specifications, accepted issues, PR description, and relevant design decisions. Make a compact requirement → implementation → verification map. Distinguish binding requirements from proposals, stale plans, and your own suggested improvements. If sources are unavailable or disagree, state the uncertainty and review the demonstrable behavior anyway.

Trace effects across public APIs, call sites, serializers, storage, services, feature flags, build targets, and downstream consumers as applicable. Check wiring as well as implementation: an unused component or a flag that never reaches its consumer can make a seemingly complete feature ineffective. Review tests for assertions that would detect the defect, not merely coverage or similarity to implementation.

Investigate relevant risks rather than mechanically applying every category:

- Empty, invalid, maximum-sized, and adversarial input; numeric overflow; unchecked indexing; unbounded allocations; broken invariants.
- Error propagation, cancellation, retries, idempotency, resource cleanup, transactions, partial writes, concurrency, deadlocks, and ordering.
- Authentication/authorization, privilege boundaries, injection, untrusted data, secrets, and dependency changes. Preserve validation at genuine trust boundaries.
- Formats, migrations, persistence, compatibility promises, upgrades, rollbacks, and default behavior. Even where breaking changes are allowed, evaluate their necessity under the usefulness criteria above.
- Complexity, hot-path allocations and I/O, contention, latency, memory, and scale. Substantiate performance claims with a workload or measurement.
- Feature combinations, supported platforms, packaging, deployment/configuration wiring, observability, and recovery where affected.
- Stale documentation, missing release/migration notes, fixtures, schemas, generated artifacts, or required integration tests.

Use the base revision when needed to distinguish a new defect from existing behavior. If a change exposes an old defect through a newly reachable path, explain that causal link.

## Required Cursor review passes

Read and apply the complete `deslop` and `thermo-nuclear-code-quality-review` instructions embedded directly in `SKILL.md`. Follow the review-only scope defined there.

## Inline comment style

Post only issues at the source lines they concern. Start with `[P0]`, `[P1]`, `[P2]`, `[P3]`, or `[nit]` using the definitions in `SKILL.md`. Write informally and directly, with one root cause per short paragraph. Keep scope, test results, reviewed revisions, confidence and fixed/upstack notes in the local report. When publication is authorized, a clean completed review gets a bodyless approval, not a no-findings comment. In default preview mode, show the proposed inline comments and their source locations to the user; a clean review only prepares the approval.

**Urgent correctness issue**

> [P1] if the second write fails, the first is already committed, so retrying leaves the two records out of sync. can we put both writes in one transaction and test a failure between them?

**Normal-priority behavior issue**

> [P2] an empty batch reaches `items[0]` here and panics before returning the documented error. can we reject it before indexing?

**Low-priority maintainability issue**

> [P3] this copies the empty-input rule from `normalize_key`. can we call that helper so cache reads and writes don't end up using different rules?

**Optional improvement**

> [nit] all three branches build the same request and only change the policy. passing the policy to the existing builder would let us drop the repeated assembly.

Use verified paths, triggers and evidence in real comments. Do not copy these claims without checking them. Omit speculative bugs, vague requests for more tests, cosmetic noise and work-log prose. Never post resolved-upstack explanations as findings.
