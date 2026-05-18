# Model-Agnostic Spec Review

This document defines Akita's structured spec review workflow. It is deliberately model-agnostic: a human reviewer, local agent, CI agent, or hosted model can all perform the same review if they follow this rubric.

The goal is not to predict every implementation detail. The goal is to make sure the spec is clear enough that implementation can proceed without discovering basic ambiguity about intent, invariants, scope, evaluation, or affected protocol surfaces.

## Workflow

1. Create a spec from `specs/TEMPLATE.md`.
2. Open a PR containing the spec.
3. The spec-tracking workflow labels the PR as `spec`.
4. Add `spec-review-request` when the author wants structured review.
5. A reviewer runs this rubric and posts a single review comment.
6. If questions remain, update the spec and request another review.
7. Add `spec-approved` only when the ambiguity score is at most 20% and all hard gates pass.
8. Implement the approved spec in the same branch or a follow-up branch, depending on reviewer preference.

The reviewer can be a person or any model. The review output should disclose the reviewer identity when useful, but approval is based on this rubric rather than on which model or tool produced the review.

## Reviewer Rules

- Read the full spec before scoring it.
- Inspect relevant repository context before asking codebase questions.
- Cite concrete evidence by path, symbol, command, prior spec, or existing pattern.
- Ask questions that expose assumptions, not questions the reviewer could answer by searching the repo.
- Separate blocking ambiguity from optional suggestions.
- Do not approve a spec just because the implementation direction seems plausible.
- Do not require implementation details that should reasonably be discovered during implementation.
- For proof, transcript, serialization, verifier, security, or planner changes, be stricter about invariants and evaluation criteria.

## Review Modes

### Single-Pass PR Review

Use this for PR comments. Post one comment containing all scores, blocking questions, and suggested spec edits.

This is preferred for asynchronous review because it avoids drip-feeding questions.

### Interactive Local Review

Use this when working directly with the spec author. Ask one question at a time, starting with the lowest-scoring dimension. Re-score after each answer. Stop when the ambiguity score is at most 20%, the author explicitly stops, or the review has reached diminishing returns.

After the interactive review, update the spec with the resolved answers before marking it approved.

## Scoring

Score each dimension from 0.0 to 1.0.

| Dimension | Weight | Question |
| --- | ---: | --- |
| Goal Clarity | 0.35 | Can the reviewer state exactly what is being built in one sentence? |
| Constraint Clarity | 0.20 | Are invariants, non-goals, safety requirements, and scope boundaries clear? |
| Evaluation Clarity | 0.30 | Could the reviewer write or identify tests, benchmarks, assertions, generated artifacts, or checks that prove success? |
| Context Clarity | 0.15 | Does the spec identify the affected crates, modules, traits, proof objects, configs, schedules, docs, and prior work well enough to modify the repo safely? |

Calculate:

```text
weighted_score =
  goal * 0.35
+ constraints * 0.20
+ evaluation * 0.30
+ context * 0.15

ambiguity = 1.0 - weighted_score
```

Status thresholds:

| Ambiguity | Status |
| ---: | --- |
| 0-20% | Approved if all hard gates pass |
| 21-35% | Questions remain |
| 36-50% | Needs revision before implementation |
| >50% | Not ready for implementation |

## Score Calibration

### Goal Clarity

Score high when:

- the primary objective is a single concrete outcome;
- key abstractions and ownership boundaries are named;
- the spec distinguishes user-facing behavior from internal refactors;
- a reviewer can tell when the feature is done.

Score low when:

- the spec names a topic but not the actual change;
- multiple incompatible interpretations are possible;
- major terms are undefined;
- the spec mixes several features without stating which one is primary.

### Constraint Clarity

Score high when:

- invariants are explicit and connected to affected protocol or API surfaces;
- non-goals rule out tempting scope creep;
- failure modes and compatibility expectations are stated;
- serialization, transcript, setup, verifier, and proof-size semantics are covered when relevant.

Score low when:

- correctness is described only as "should work";
- prover/verifier consistency is implicit;
- security or soundness assumptions are not named;
- non-goals are missing or too vague to constrain implementation.

### Evaluation Clarity

Score high when:

- acceptance criteria are concrete and checkable;
- tests are mapped to behaviors or invariants;
- performance/proof-size claims include commands, datasets, thresholds, or expected direction and magnitude;
- generated artifacts and dry-run outputs are tied to reproducible commands;
- the spec says what existing checks must remain green.

Score low when:

- criteria are subjective or untestable;
- "run tests" is the only evaluation plan for a protocol change;
- performance claims lack measurement commands;
- proof-size or security claims do not specify how they are computed.

### Context Clarity

Score high when:

- affected crates, modules, traits, config presets, generated tables, examples, and docs are listed;
- related specs and prior implementation patterns are referenced;
- the design explains how the change fits existing architecture;
- alternatives considered are specific enough to prevent re-litigating the same decisions.

Score low when:

- the affected subsystem is not identified;
- the spec assumes knowledge that is not in the document or references;
- alternatives are missing for a design with obvious forks;
- implementation would require a broad search just to find the intended surface area.

## Hard Gates

Do not approve the spec until these pass, regardless of the numeric score.

- If the spec changes proof, transcript, verifier, serialization, setup, planner security, or field arithmetic behavior, it must state the relevant consistency and soundness invariants.
- If the spec makes a performance, proof-size, memory, or security claim, it must name how the claim will be measured or checked.
- If the spec changes serialization, it must state canonical byte layout, validation behavior, and compatibility expectations.
- If the spec changes transcript or Fiat-Shamir behavior, it must state prover/verifier ordering and domain-separation expectations.
- If the spec changes generated schedules, setup artifacts, or cached data, it must state regeneration and validation requirements.
- If the spec intentionally excludes a plausible simpler or broader path, that path must appear in Non-Goals or Alternatives Considered.
- If implementation risk is concentrated in an unsafe block, cryptographic assumption, or external estimator/tool, the spec must name that risk and its verification plan.

## Required Review Output

Use this format for PR review comments:

```markdown
## Spec Review: <spec title>

Reviewer: <human/tool/model identifier, optional>
Rubric: `specs/SPEC_REVIEW.md`

| Dimension | Score | Gap |
| --- | ---: | --- |
| Goal Clarity | 0.00 | <clear/gap> |
| Constraint Clarity | 0.00 | <clear/gap> |
| Evaluation Clarity | 0.00 | <clear/gap> |
| Context Clarity | 0.00 | <clear/gap> |
| **Ambiguity** | **00%** | <approved/questions remain/needs revision> |

### One-Sentence Goal

<state what is being built>

### Evidence Reviewed

- `<path>`: <why it matters>
- `<path>`: <why it matters>

### Blocking Questions

1. [<dimension>] <question grounded in spec or codebase evidence>
2. [<dimension>] <question grounded in spec or codebase evidence>

### Non-Blocking Suggestions

- <optional suggestion>

### Status

<Approved / Questions remain / Needs revision>. <Explain next step.>
```

If there are no blocking questions and all hard gates pass, the reviewer may add or request `spec-approved`.

If there are blocking questions, leave `spec-review-request` in place and do not add `spec-approved`.

## Reviewer Prompt

Any model can use this prompt:

```text
Review this Akita spec using specs/SPEC_REVIEW.md.

First read the spec, the PR diff, prior PR comments, specs/TEMPLATE.md, CONTRIBUTING.md, and any referenced specs or modules. Gather repository evidence before asking questions.

Score Goal Clarity, Constraint Clarity, Evaluation Clarity, and Context Clarity from 0.0 to 1.0. Compute ambiguity with the rubric formula. Apply the hard gates. Post a single review with the required output format. Ask only blocking questions needed to make the spec implementable without major clarifying questions.
```

## Approval Semantics

`spec-approved` means:

- the spec has passed this rubric;
- maintainers agree it is ready to implement;
- implementation can proceed without reopening basic intent/scope/evaluation questions.

`spec-approved` does not mean:

- the implementation must follow every optional execution detail exactly;
- all code design questions are resolved;
- performance claims are proven before implementation;
- reviewers waive normal implementation review.
