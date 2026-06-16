# Contributing

Akita uses a lightweight spec-driven workflow for major features and architectural changes.

## When To Write A Spec

Open a direct PR for bug fixes, small improvements, documentation, and tightly scoped cleanups.

Start with a spec when the implementation will be difficult to review directly, especially for:

- public API changes
- proof, transcript, serialization, or verifier behavior changes
- large refactors or crate-boundary changes
- changes expected to exceed roughly 500 non-trivial lines

The goal is to make review cheaper: discuss the shape, invariants, and evaluation plan before the large implementation lands.

## Workflow

1. Create a spec from [`specs/TEMPLATE.md`](specs/TEMPLATE.md).
2. Open a PR containing the spec file.
3. The spec-tracking workflow labels the PR as `spec`.
4. Add `spec-review-request` when the spec is ready for structured review.
5. Review the spec using the model-agnostic rubric in [`specs/SPEC_REVIEW.md`](specs/SPEC_REVIEW.md).
6. Resolve review questions and update the spec.
7. Add `spec-approved` once maintainers agree the spec is ready for implementation.
8. Implement the approved spec in the same branch or a follow-up implementation branch, depending on reviewer preference.

## Labels

| Label                 | Meaning                                      |
|-----------------------|----------------------------------------------|
| `spec`                | PR contains a spec file under `specs/`        |
| `no-spec`             | PR does not contain a spec file              |
| `implementation`      | PR contains implementation changes with spec |
| `spec-review-request` | Maintainer or author requests spec review    |
| `spec-approved`       | Spec is approved for implementation          |

These labels describe review state, not a specific review tool.
Humans, local agents, CI agents, and hosted models can all participate in the same workflow if they follow the same rubric.

## Spec Review Rubric

A spec is ready for implementation when reviewers can answer:

- What is being built, in one sentence?
- Which invariants must never change?
- What is explicitly out of scope?
- Which tests, fixtures, benchmarks, or compile checks prove the change works?
- Which modules, crates, APIs, or protocol surfaces are affected?
- What alternatives were considered, and why were they rejected?

Structured review uses the detailed scoring rubric in [`specs/SPEC_REVIEW.md`](specs/SPEC_REVIEW.md):

- Goal clarity: 35%
- Constraint clarity: 20%
- Evaluation clarity: 30%
- Context clarity: 15%

Reviewers compute an ambiguity score from those dimensions. A spec can be approved when ambiguity is at most 20% and all hard gates in the review rubric pass.

If the answers are unclear, keep the PR in `spec-review-request` and ask questions before implementation begins.

## Documentation

Implementation PRs must keep docs from drifting. See [`docs/documentation.md`](docs/documentation.md)
for the full policy. In short:

- Update the spec `Status` / acceptance criteria when the PR completes spec work.
- Review the **documentation blast-radius** PR comment (`<!-- akita-doc-blast-radius -->`).
- Run `./scripts/check-doc-guardrails.sh` before pushing when you touch book, specs, or `docs/`.
- Fold shipped specs into the book and archive them per [`specs/PRUNING.md`](specs/PRUNING.md).

## Offline SIS table regen

Regenerating `generated_sis_table/` is a manual Sage workflow, not part of Rust CI.
See [`scripts/sis_golden/README.md`](scripts/sis_golden/README.md) for submodule init,
golden refresh (`refresh_golden.py`), replay check (`check.py`), per-family regen
(`gen_sis_table.py --jobs`), and full table stitch
(`stitch_generated_sis_table.py --jobs 6 --max-rank 20`).
