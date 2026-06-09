# Spec: [Feature Name]

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     |                                |
| Created       | YYYY-MM-DD                     |
| Status        | proposed                       |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  |                                |

`Status` must be one of the lifecycle values below. Keep it accurate after the
spec lands; stale `proposed`/`in progress` headers on shipped work are the main
signal `scripts/check-spec-references.sh` and the quarterly audit look for. See
[`specs/PRUNING.md`](PRUNING.md) for the full lifecycle and archive workflow.

| Status        | Meaning |
|---------------|---------|
| `proposed`    | Not approved / not started |
| `approved`    | `spec-approved`, awaiting implementation |
| `active`      | Approved, implementation in flight |
| `implemented` | Shipped; durable reference value |
| `superseded`  | Replaced by another spec (set `Superseded-by`) |
| `historical`  | A retrospective/log of completed work; low forward value |
| `archived`    | Moved to `specs/archive/`; the book owns the durable content |

Fill `Book-chapter` with the Akita Book page that owns this spec's durable
content once it is folded in (e.g. `book/src/how/security/security.md`).

## Summary

One paragraph: what is this feature and why does it matter? State the problem being solved, not just the solution.

## Intent

### Goal

What are we building? State the primary objective in one sentence without qualifiers. Define the key abstractions, types, APIs, crate boundaries, and protocol surfaces this feature introduces or modifies.

### Invariants

What properties must hold? List the correctness, safety, consistency, serialization, transcript, and performance invariants that the implementation must preserve. For PCS/prover/verifier changes, include prover/verifier consistency requirements.

Whenever possible, name the existing test, benchmark, generated schedule, or protocol relation that protects each invariant. If the feature needs a new invariant test, describe where it should live.

### Non-Goals

What is explicitly out of scope? Listing non-goals prevents scope creep and clarifies the feature's boundaries.

## Evaluation

### Acceptance Criteria

Concrete, testable criteria. Each should be verifiable by a test, benchmark, assertion, compile check, CI check, or reviewable artifact.

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

### Testing Strategy

Which existing tests must continue passing? What new tests are needed? Specify feature combinations, release/debug requirements, deterministic fixtures, benchmark modes, generated schedule requirements, or prover/verifier transcript checks where applicable.

### Performance

What are the performance expectations? Specify benchmarks, acceptable regressions, proof-size effects, memory budgets, setup-size effects, or throughput targets. "No regression" is acceptable if there is a benchmark or metric to verify against.

If this moves proof-size/security/planner tradeoffs, state the expected direction and magnitude, and name the planner script, generated table, or profile command that should verify it.

## Design

### Architecture

How does this feature fit into the existing system? Describe which modules, crates, traits, data structures, proof objects, config hooks, setup artifacts, and prover/verifier boundaries are affected. Include a diagram if the interaction is non-trivial.

### Alternatives Considered

What other approaches were evaluated? Why was this design chosen over them? This section prevents re-litigating decisions during implementation review.

## Documentation

What README, spec, paper note, example, profile guide, crate docs, or developer-guide changes are required? List new pages, sections to update, or diagrams to add. If no documentation changes are needed, state why.

## Execution

Optional implementation direction — algorithmic approach, optimizations to consider, modules to touch, migration notes, ownership boundaries, task checklist, and risks to resolve first. The implementer should be able to derive most of this from Intent and Evaluation.

## References

Links to papers, related specs, relevant issues/PRs, prior art, local design notes, or benchmark/profiling commands.
