# Spec: Modular Planner and Distinct Precommit Roles

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-23 |
| Status        | proposed |
| PR            | |
| Supersedes    | Planner-architecture portions of [`archive/2026-Q2/planner-refactor.md`](archive/2026-Q2/planner-refactor.md) and the shared-precommit model in [`distributed-setup-offloading.md`](distributed-setup-offloading.md) |
| Superseded-by | |
| Book-chapter  | book/src/how/configuration.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Akita currently uses one broad planner for several decisions with different
lifecycles and objectives. In particular, two unrelated mechanisms are both
described as "precommitted":

1. A user polynomial group may be committed before its eventual grouped root
   opening is known.
2. A prefix of public setup data may be committed for one known recursive
   transition and opened in the suffix.

The first is a reusable root-commitment policy. The second is a schedule-owned
recursion edge. They must not share a selector, public decision type, or
optimization contract merely because both materialize commitments early.

This specification separates those roles and restructures the offline planner
around explicit modules:

- root-precommit recipe selection;
- root candidate enumeration;
- recursive transition enumeration;
- setup-prefix geometry and placement;
- terminal planning;
- objective/frontier management;
- compact schedule emission.

All planner work remains offline. The runtime boundary is specified by
[`runtime-schedule-boundary.md`](runtime-schedule-boundary.md).

## Status and baseline

This design was written against `origin/main` commit
`af482ab7c2d8d3edcdc901f0c3950378f954787a`, informed by the PR 323 mirror at
`75d7a6ddc246a97c9da86b03c0ac0ac5356d14a1`.

PR 323 establishes an important prerequisite:

- root `log_basis` is fixed to 2 for all field widths;
- recursive levels remain planner-selected;
- the stale small-field and decomposition-basis floors are removed;
- conservative rank widening for root-precommitted groups is removed.

Its remaining root-precommit implementation still runs a hypothetical complete
schedule under singleton basis range `(2, 2)` and extracts the root commitment
parameters. That probe is a behavior-preserving bridge, not the desired
architecture: a hypothetical suffix can influence root geometry even though
the suffix will never be used.

## Terminology

- A **root-precommitted group** is a user polynomial group committed before the
  final grouped root opening is selected.
- A **root-precommit recipe** is the canonical block geometry, matrix
  dimensions, SIS parameters, and commitment bases for such a group.
- The **root-precommit compatibility domain** is the finite set of catalog
  families and root chunk counts in which one recipe is promised to work.
- A **setup prefix** is a prefix of public shared setup data offloaded from a
  direct fold and committed for a known recursive consumer.
- A **setup-prefix slot** identifies one schedule-owned setup-prefix
  commitment and its exact consuming transition.
- A **planner decision** contains independent choices. Derived ranks, widths,
  and costs are materialized by canonical functions.
- A **transition** maps a validated planner state and decision to a successor
  state or a typed rejection.
- A **frontier** retains nondominated partial schedules for explicitly named
  objectives.

## Core distinction

| Property | Root-precommitted group | Setup-prefix commitment |
|---|---|---|
| Contents | User polynomial group | Prefix of public setup data |
| Chosen | Before final batch is known | For a known recursive edge |
| Opened | Root only | Recursive suffix |
| Reuse | Across a declared finite compatibility domain | Only by its owning schedule transition |
| Basis | Fixed root basis 2 | Exact consuming suffix basis |
| Geometry context | Group layout and root compatibility domain | Producer, consumer, prefix length, basis, and chunk count |
| Optimization owner | Root-precommit recipe selector | Recursive transition planner |
| Runtime representation | Commitment-bound descriptor plus catalog recipe | Schedule-owned setup-prefix slot |

The code and documentation **MUST** use distinct names for these concepts.
`PrecommittedLevelParams` or another public type that can represent either role
is forbidden.

## Intent

### Goals

1. Select root-precommit layouts directly, without planning a fake standalone
   proof.
2. Keep setup-prefix offloading inside the recursive transition where its full
   context is known.
3. Make planner components independently testable and replaceable.
4. Make objectives and frontier dimensions explicit.
5. Emit compact runtime decisions that are expanded and validated by
   `akita-schedule`.
6. Preserve the planner-free verifier boundary.

### Invariants

#### Root-precommit recipes

1. The root basis **MUST** be 2.
2. Recipe generation **MUST NOT** invoke full root-plus-suffix planning.
3. Recipe selection **MUST** enumerate only commitment layouts that are secure,
   structurally valid, and usable throughout their declared compatibility
   domain.
4. The initial implementation **MUST** provide one canonical objective, not a
   speculative public profile enum.
5. The selected recipe **MUST** be deterministic and cataloged.
6. A commitment **MUST** bind its complete root-precommit descriptor.
7. The final schedule planner **MUST** treat that descriptor as immutable.
8. Runtime prover and verifier **MUST NOT** recompute the selector.
9. Conservative rank widening across possible future bases is forbidden. The
   fixed root basis and exact candidate geometry determine the ranks.

#### Setup-prefix commitments

1. A setup-prefix slot **MUST** belong to one compact schedule and identify its
   producer and consuming recursive transition.
2. Its basis **MUST** be the consuming transition's basis, not the root basis.
3. Its local geometry selector **MAY** be greedy because producer, consumer,
   prefix length, basis, and chunk count are known.
4. The decision whether to offload, and at which edge, **MUST** remain a global
   suffix-planner decision.
5. Setup-prefix parameters **MUST NOT** be exposed as reusable root-precommit
   recipes.
6. Setup generation **MUST** materialize exactly the slots named by validated
   catalog rows.

#### Planner structure

1. Each independent decision domain **MUST** have one canonical enumerator.
2. Each derived quantity **MUST** have one canonical materialization function.
3. Candidate rejection **MUST** be typed during planning and **MUST NOT** rely
   on matching error strings.
4. Objectives and tie-breakers **MUST** be explicit and deterministic.
5. Frontier state **MUST** retain every dimension needed by later choices.
6. Planner modules **MUST NOT** duplicate security or sizing formulas owned by
   runtime validation primitives.
7. Generated rows **MUST** pass `akita-schedule` expansion and validation
   before emission.
8. No planner module **MAY** be reachable from verifier runtime code.

### Non-goals

This specification does not:

- add multiple root-precommit objective profiles before a real workload
  requires them;
- require precommitted groups to be smaller than the final group;
- impose `live_blocks >= witness_chunks`;
- fix a global maximum machine count;
- permit unbounded or arbitrary future compatibility;
- make setup-prefix commitments reusable;
- guarantee that the chosen greedy root recipe globally minimizes the complete
  proof;
- redesign verifier equations beyond the empty-range question identified
  below.

## Root-precommit recipe selection

### Inputs

The selector receives:

```text
root runtime family
group layout
root basis = 2
security policy
setup capability
compatibility domain
```

The compatibility domain is finite. "Open-world" means the commitment may be
created before choosing among supported future grouped roots; it does not mean
compatibility with arbitrary future protocols, bases, or unbounded machine
counts.

The selector **MUST NOT** receive a hypothetical recursive suffix.

### Candidate enumeration

For each legal root commitment split, the selector derives:

- positions per block;
- number of live blocks;
- inner and outer matrix dimensions;
- secured A and B ranks and norm bounds;
- digit widths for the fixed root basis;
- exact setup requirement;
- exact contribution to the root's outgoing witness for each supported chunk
  count.

Candidates are rejected if they fail:

- group-shape divisibility or padding rules;
- commitment-layout invariants;
- SIS security;
- setup capability;
- integer or allocation bounds;
- compatibility with any advertised runtime family.

Candidate derivation **MUST** call the same canonical arithmetic and security
functions used by runtime validation.

### Canonical objective

The initial objective is the root-precommitted group's exact incremental
contribution to the outgoing root witness.

For a group \(g\) and root chunk count \(W\), write:

- \(Z_g\) for one replicated folded-response segment;
- \(E_g\) for all block-opening segments across the group;
- \(T_g\) for all matrix-image segments across the group.

The selector scores:

\[
  C_g(W) = W Z_g + E_g + T_g.
\]

`E_g` and `T_g` are partitioned across chunks; `Z_g` is present once per
group/chunk slot. Implementations **MUST** compute this score from exact
materialized segment widths, not from a separately maintained approximation.

When one recipe is shared across several supported chunk counts, the initial
selector **SHOULD** minimize:

\[
  \max_{W \in \mathcal W} C_g(W).
\]

Because \(C_g(W)\) is monotone in \(W\) for a fixed layout, this is normally the
score at the largest advertised \(W\). The compatibility set
\(\mathcal W\), not a hard-coded global machine cap, determines that value.

Deterministic tie-breakers **MUST** be documented next to the selector. They
should prefer lower setup requirements before raw geometry fields unless
measurement justifies another order.

This objective is a canonical local policy, not a theorem that it globally
minimizes the final proof. If large precommitted groups later require a
setup-first policy, that is a separate design change with a real compatibility
identifier; it is not added speculatively here.

### Catalog and commitment binding

Generation emits one canonical recipe for each supported recipe key. The
runtime catalog binds complete schedule rows to those recipes as specified in
[`runtime-schedule-boundary.md`](runtime-schedule-boundary.md).

The commitment descriptor **MUST** include all frozen values needed to detect a
recipe mismatch. A later grouped planner may score the descriptor's consequence
but **MUST NOT** change its geometry, ranks, bounds, or bases.

## Multi-chunk groups with empty block ranges

### Required layout

Let a group have \(B\) live blocks and let the root use \(W\) witness chunks.
The distributed layout retains exactly \(W\) group/chunk slots even when
\(B < W\).

Blocks are partitioned deterministically:

```text
q = B / W
r = B % W

chunk i owns q + 1 blocks when i < r
chunk i owns q blocks otherwise
```

For \(B=4\) and \(W=8\):

```text
0: 0..1
1: 1..2
2: 2..3
3: 3..4
4: 4..4
5: 4..4
6: 4..4
7: 4..4
```

Each group/chunk slot remains:

```text
[ z[g,i] | e[g,i] | t[g,i] ]
```

The layout rules are:

- `z[g,i]` retains its full per-chunk width for every \(i < W\);
- `e[g,i]` and `t[g,i]` are proportional to the assigned block range;
- an empty range has empty `e` and `t` segments;
- the honest prover **MUST** write the canonical zero value into `z` for an
  empty range;
- the planner **MUST** still charge all \(W\) `z` segments.

Therefore the group width remains \(W Z_g + E_g + T_g\). Allowing empty ranges
removes an accidental feasibility restriction; it does not make replicated
`z` free.

This rule also applies mechanically to a setup-prefix group if its selected
geometry has fewer live blocks than the consuming transition has chunks.

### Security gate: aggregate versus local relation

The current horizontally concatenated relation constrains the aggregate
\(\sum_i z[g,i]\). It does not appear to prove that each `z[g,i]` is the fold of
exactly that chunk's block range. In particular, the verified relation may be
invariant under a bounded redistribution:

\[
  (z_i, z_j) \mapsto (z_i + \delta, z_j - \delta).
\]

Requiring every chunk to own a block does not remove this freedom.

Before empty ranges are enabled, the implementation **MUST** complete and
record a focused soundness audit choosing one of two contracts:

1. **Aggregate-share contract.** Verification proves bounded additive shares
   with the required aggregate. The SIS and fold-norm analysis must cover
   arbitrary bounded redistributions, while the honest prover still emits
   machine-local folds and canonical zeroes for empty ranges.
2. **Strict-local contract.** Verification proves each machine-local fold. The
   relation must add the constraints required to enforce zero for an empty
   range and locality for nonempty ranges.

The specification recommends the aggregate-share contract if the existing
security argument already matches the implemented relation. This is not an
accepted security claim until the audit identifies the exact theorem and bound
used by verifier validation.

No code may remove `live_blocks >= witness_chunks` merely on the basis of the
layout arithmetic. The security gate, prover construction, verifier layout,
structured evaluators, serialization, and tests must move together.

## Setup-prefix planning

### Local geometry

For one proposed offload edge, the planner knows:

- producer fold and direct setup footprint;
- natural and padded prefix lengths;
- consuming recursive fold;
- consuming basis and ring dimension;
- consuming challenge and chunk configuration;
- successor witness shape;
- setup capability.

The local setup-prefix geometry selector enumerates secure compatible splits
and minimizes the exact physical contribution to that consumer's witness.
This selector **MUST NOT** choose whether the offload exists.

### Global placement

The suffix planner decides whether and where to offload. Its frontier must
retain the dimensions affected by that choice, including:

- first-direct setup requirement;
- proof payload;
- successor witness width;
- setup-prefix slot count and sizes;
- contraction state;
- any resource envelope that affects later feasibility.

A setup-prefix candidate is an edge:

```text
producer state
  + offload decision
  + exact setup-prefix geometry
  + consumer fold decision
  -> successor state
```

The frontier compares the offloaded and direct alternatives under explicit
objectives. A locally smallest prefix commitment may lose globally because it
changes contraction, setup coverage, or later suffix cost.

### Runtime representation

The emitted compact schedule names every selected `SetupPrefixSlotId` and
records the independent geometry decisions needed for deterministic expansion.
The runtime validator checks:

- the slot names the correct producer and consumer;
- the committed prefix length matches the producer's setup footprint;
- the basis and chunk layout match the consumer;
- the commitment parameters are secure;
- the slot is included in the setup capability.

No reusable setup-prefix recipe catalog is introduced.

## Modular planner architecture

### Modules

The planner should converge on these responsibilities:

```text
request normalization
    ↓
root-precommit recipe selector
    ↓
root candidate enumerator
    ↓
recursive transition enumerator
    ├── direct transition
    └── setup-prefix offload transition
    ↓
objective/frontier engine
    ↓
terminal planner
    ↓
compact decision row
    ↓
akita-schedule expansion + validation
```

The exact Rust module names are not normative. The ownership boundaries are.

### Decision IR

Planner decisions **SHOULD** be small values that contain only independent
choices. A conceptual form is:

```rust
struct RootDecision {
    split: BlockSplit,
    log_basis: u32,
    witness_chunks: u32,
}

struct RecursiveDecision {
    split: BlockSplit,
    log_basis: u32,
    offload: Option<SetupPrefixDecision>,
}
```

Root basis 2 may be represented implicitly if it is a protocol constant. The
planner must not reintroduce it as a per-preset optional policy merely to make
the IR self-describing.

Derived values such as secured ranks, norm bounds, segment widths, and setup
footprints are materialized canonically. They should not be independently
stored in the decision unless needed for stable generated representation and
validated against recomputation.

### Typed rejection

Enumerators return either a materialized candidate or a typed planner rejection
such as:

```text
InvalidGeometry
ArithmeticOverflow
InsufficientSetup
InsecureSis
NoContraction
UnsupportedChunkLayout
InvalidSetupPrefixEdge
TerminalUnavailable
```

These types support diagnostics and tests. They are planner concepts and need
not become public verifier errors.

### Objectives and frontiers

Every frontier must state:

- which cost coordinates it retains;
- the dominance relation;
- deterministic tie-breakers;
- which future transition reads each coordinate.

Do not collapse setup footprint and proof payload into one scalar if later
choices can prefer different trade-offs. Conversely, do not keep speculative
coordinates that no transition or final objective reads.

The root-precommit recipe selector is not a recursive frontier. It uses the
canonical local objective defined above.

### Extension points

A new planner feature should normally add or modify one of:

- a candidate enumerator;
- a transition variant;
- an objective coordinate;
- a terminal strategy;
- a compact decision field plus runtime validation.

It should not require edits across unrelated root, suffix, grouped, generation,
and verifier paths. Shared changes belong in canonical decision
materialization or runtime validation.

## Evaluation

### Acceptance criteria

#### Root precommit

- [ ] The full-schedule precommit probe is removed.
- [ ] Root-precommit recipes are selected directly at root basis 2.
- [ ] The selector uses exact materialized \(WZ+E+T\) widths.
- [ ] One recipe is deterministic across its declared compatibility domain.
- [ ] Conservative basis-envelope rank widening is absent.
- [ ] Commitments bind the complete selected descriptor.
- [ ] Runtime code only looks up and validates recipes.

#### Setup prefix

- [ ] Setup-prefix types cannot be confused with root-precommit descriptors.
- [ ] Local geometry selection receives the exact consuming transition.
- [ ] The suffix frontier decides whether and where offloading occurs.
- [ ] Emitted slots expand and validate without planner code.
- [ ] Setup generation materializes exactly the slots named by catalog rows.

#### Empty ranges

- [ ] Block partitioning permits empty trailing ranges for \(B < W\).
- [ ] Every group retains exactly \(W\) `z` segments.
- [ ] Empty ranges have zero-width `e` and `t`.
- [ ] The honest prover writes canonical zero `z` for empty ranges.
- [ ] Planner costing retains \(WZ+E+T\).
- [ ] The aggregate-versus-local security audit is completed and referenced.
- [ ] SIS and norm validation match the chosen security contract.
- [ ] Prover, verifier, transcript, serialization, and structured evaluators
  agree on empty ranges.

#### Planner modularity

- [ ] Root, recursive, setup-prefix, and terminal decisions have independent
  enumerators.
- [ ] Candidate derivation uses canonical runtime arithmetic and security
  primitives.
- [ ] Rejections are typed.
- [ ] Objectives and frontier coordinates are documented and tested.
- [ ] Compact emitted rows expand and validate through `akita-schedule`.
- [ ] Adding a transition strategy does not require verifier changes.

### Test strategy

Root-precommit tests **MUST** cover:

- direct selection parity with known secure fixtures;
- deterministic tie-breaking;
- exact score recomputation;
- descriptor binding and mismatch rejection;
- shared recipe behavior across its advertised chunk counts;
- a case where a fake suffix would have selected different root geometry.

Empty-range tests **MUST** cover:

- \(B=4, W=8\) yielding four nonempty and four empty ranges;
- \(B=0\) only at the per-chunk range level, never as an invalid empty group;
- zero-width `e` and `t` layouts;
- canonical honest zero `z`;
- total witness width retaining eight `z` segments;
- a grouped proof where another group uses all chunks;
- malformed range and serialization rejection without panic;
- the security property chosen by the audit.

Setup-prefix tests **MUST** cover:

- local geometry at multiple consumer bases and chunk counts;
- direct versus offloaded frontier alternatives;
- exact producer-prefix length binding;
- slot identity and setup-capability mismatch;
- empty per-chunk ranges where applicable;
- deterministic compact emission.

Planner-module tests **SHOULD** use small exhaustive domains to compare frontier
results with brute force. Generated-family tests then cover full production
domains.

## Execution

The work should proceed in reviewable slices:

1. Record the empty-range soundness audit and choose the verified contract.
2. Make per-group chunk ranges empty-safe across shared types, prover, verifier,
   and structured evaluators.
3. Add a direct root-precommit recipe selector and parity fixtures.
4. Remove the hypothetical full-schedule probe.
5. Split root-precommit and setup-prefix public types.
6. Isolate setup-prefix local geometry from global placement.
7. Introduce compact root and recursive decision types.
8. Extract typed enumerators and transition functions.
9. Make objectives and frontier dimensions explicit.
10. Regenerate catalogs through the planner-owned binary and validate every
    row through `akita-schedule`.

The runtime dependency cut in
[`runtime-schedule-boundary.md`](runtime-schedule-boundary.md) may be
implemented before these planner slices. None of these steps may restore a
runtime planner dependency.

## Risks

### Local objective is mistaken for global optimality

The root-precommit selector intentionally uses a stable local contract because
the eventual batch is unknown. Benchmarks must not describe it as globally
optimal. If the intended workload changes, introduce a separately reviewed
compatibility policy instead of silently changing existing recipes.

### Recipe compatibility is underspecified

A commitment cannot promise compatibility with an undefined future. Catalog
generation must make the finite compatibility domain explicit and test every
advertised family.

### Empty ranges expose an existing proof-language ambiguity

The verifier may already prove aggregate bounded shares rather than literal
machine-local folds. Empty ranges make that distinction visible but do not
create it. Enabling them without recording the security contract would hide a
protocol assumption in layout code.

### Shared helpers recreate coupling

Extracting generic helpers too early can produce thin wrappers or large
parameter bags. Modules should share canonical arithmetic and security
primitives, not an all-purpose candidate builder.

### Generated churn obscures objective changes

Planner refactors can preserve validity while selecting different schedules.
Each slice should distinguish expanded-schedule parity, intentional selection
changes, and textual regeneration.

## Open questions

The following choices remain open and must not be encoded accidentally:

1. What finite chunk-count compatibility domain should shipped root-precommit
   recipes advertise?
2. Should the first recipe tie-break prefer setup footprint or a simpler
   geometry after equal \(WZ+E+T\)?
3. Does the existing security proof establish the aggregate-share contract, or
   are strict machine-local constraints required?
4. Should a future large-precommit workload introduce a setup-first recipe
   policy? No such policy is part of this specification.
5. Is a repository-wide maximum machine count of eight desirable as a runtime
   resource bound? Empty-range compatibility does not depend on that answer.

## References

- [`runtime-schedule-boundary.md`](runtime-schedule-boundary.md)
- [`distributed-planner.md`](distributed-planner.md)
- [`distributed-setup-offloading.md`](distributed-setup-offloading.md)
- [`digit-innermost-layout.md`](digit-innermost-layout.md)
- [`schedule-catalog-ownership.md`](schedule-catalog-ownership.md)
- [`../docs/verifier-contract.md`](../docs/verifier-contract.md)
