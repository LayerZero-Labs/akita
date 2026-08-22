# Spec: Certified Planner Architecture

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-08-21 |
| Status        | proposed |
| PR            | [#434](https://github.com/LayerZero-Labs/akita/pull/434) |
| Supersedes    | Planner architecture portions of [`archive/2026-Q3/modular-planner-and-precommit-roles.md`](archive/2026-Q3/modular-planner-and-precommit-roles.md) |
| Superseded-by | |
| Book-chapter  | book/src/how/configuration.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Akita needs one offline planner whose result is exact under an explicit policy
and whose normal execution is fast. The planner must not choose between a
small heuristic search and a second exhaustive fallback. It must search one
audited decision domain, use guidance to visit strong candidates early, and
remove unexplored regions only through checked proofs of irrelevance.

This specification defines that architecture. It separates the candidate
language from traversal order, feasibility from progress, and local scoring
from the complete schedule objective. It defines a diagnostic oracle and a
guided execution as two settings of the same search engine. An expensive
oracle run may produce versioned guidance, but the guided run must still prove
that its selected schedule is optimal.

The pruning proof addendum in this document states the first concrete
theorems for that result. It derives an exact mandatory recursive witness body,
an incumbent interval for split search, an admissible relaxed suffix search,
and transition dominance rules. It also gives separate proof contracts for
selective L2 routes and setup first slice choices. Those two current shortcuts
must be removed or certified in later implementation pull requests.

The specification also replaces the narrow semantic model of one main group
plus precommitted groups. A commitment workload may contain several semantic
groups, several commitment epochs separated by transcript challenges, and one
or more shared opening batches. Each group owns its source and extraction
contract. The opening batch separately identifies the group that closes the
current grouped root. That closing group is algebraically new, but it is not
the semantic main witness.

Aerie Falcon version 1 is the main integration case. It commits five groups
before the JL seed and three JL digit groups after the seed, then opens all
eight together. The last JL shard closes the current Akita batch only because
the transcript commits it last. The planner must jointly price the complete
workload without calling that shard the main route or allowing its source
family to own the shared opening policy.

All planning remains offline. The planner emits a compact complete plan which
`akita-schedules` expands and validates. Runtime setup, commitment, proving,
and verification consume an approved catalog artifact. Runtime code never
invokes planner search or trusts a planner cost estimate.

## Current baseline and unresolved problems

The target architecture starts from several settled changes.

1. [PR #408](https://github.com/LayerZero-Labs/akita/pull/408) replaced the
   contractive root attempt and noncontractive fallback with one root search.
   Root contraction is not an admission rule or objective coordinate.
2. [PR #416](https://github.com/LayerZero-Labs/akita/pull/416) consolidated
   schedule parameters and removed several duplicate representations.
3. [`setup-offloading-planner.md`](setup-offloading-planner.md) defines the
   current recursive setup offload semantics and setup first objective.
4. [`heterogeneous-group-source-contracts.md`](heterogeneous-group-source-contracts.md)
   defines group owned honest fold sizing while keeping runtime profiles free
   of source identity.
5. Generated catalogs, rather than planner execution, remain the runtime path.

The remaining problems concern the meaning and proof of the search.

### Search policy still changes the candidate language

`RecursiveSplitSearchPolicy::BoundedBalancedExtremesV1` searches the two split
extremes and a fixed radius around a balance estimate when the variable count
is large. The omitted splits are not currently excluded by a proof that they
cannot win the complete objective. Root totality therefore holds only within
this configured split domain.

Selective L2 planning currently derives a candidate from the split selected
by the best modeled Linf candidate. Setup first slice pruning keeps one local
slice choice before successor witness sizing and suffix search. Some levels
retain a split frontier while others keep a local best candidate according to
level and policy conditions. Each choice may be useful, but the code does not
give every omission a common proof contract.

There is no theorem which says that the best L2 route uses the best Linf split.
There is also no theorem which says that the smallest local padded setup gives
the best complete setup first schedule. The audited domain must include the
other L2 splits and slice choices until a checked certificate covers their
complete objective effect.

### Local and global choices remain mixed

Candidate generation applies local scoring, successor progress, slice
selection, security route selection, and frontier retention in the same call
chain. A local winner can be globally inferior when it changes the next
witness, the first direct setup capacity, a later security route, or the
canonical descriptor.

The planner must state which choices are exact domain definitions and which
are traversal hints. It must also state which local dominance rules are valid
for every possible suffix consumer.

### Group planning begins after earlier profiles are frozen

`AkitaScheduleLookupKey` receives one final group layout and exact profiles for
earlier groups. This is the correct lookup shape for an already compiled batch,
but it is too late for joint planning. The planner cannot compare alternative
commitment recipes for the earlier groups once those profiles are part of the
input key.

[PR #409](https://github.com/LayerZero-Labs/akita/pull/409) explores exact
precommit profile planning. [PR #412](https://github.com/LayerZero-Labs/akita/pull/412)
explores symmetry reduction and streamed grouped root search. This
specification keeps those design goals but places them inside a wider workload
contract.

### Planner performance is not yet a contract

The repository checks catalog drift, but it does not yet define a supported
fast search setting, a guide artifact, or per fixture time and memory budgets.
An implementation can remain exact and still be unusable for iterative
schedule work. It can also become fast by silently narrowing the domain.
Neither outcome is acceptable.

## Intent

### Goal

Build one exact offline planner that searches a complete audited decision
domain, supports phased heterogeneous commitment workloads, uses certified
guidance for a fast normal execution, and emits one deterministic validated
plan under an explicit complete schedule objective.

### Invariants

#### Search meaning

1. The planner **MUST** have one semantic search path.
2. Oracle and guided executions **MUST** use the same request normalization,
   decision enumerators, candidate materializer, objective, frontier rules,
   descriptor ordering, and output validator.
3. Guidance **MAY** change traversal order and provide an incumbent.
4. Guidance **MUST NOT** remove a candidate unless a checked certificate proves
   that the candidate cannot change the selected complete schedule.
5. A missing bound, arithmetic overflow in a bound, unsupported certificate,
   or stale guide **MUST** retain the affected candidate.
6. The selected result **MUST** be independent of traversal order, parallel
   execution, batching, memo capacity, and cache eviction.
7. `UnsupportedSchedule` **MUST** mean that the audited domain contains no
   complete feasible schedule. It **MUST NOT** mean that a fast subset found no
   schedule.

#### Objective and frontier meaning

1. Every selection policy **MUST** define one total order on complete
   schedules, including the canonical descriptor tie break.
2. Every partial frontier **MUST** name the future observations that justify
   each retained coordinate.
3. A dominance rule **MUST** prove irrelevance for every parent or suffix that
   can consume the dominated candidate.
4. A local layout score **MUST NOT** be used as a complete schedule lower bound
   unless the missing suffix and parent costs are proven nonnegative and
   independent in the required order.
5. Memo eviction **MAY** cause recomputation. It **MUST NOT** change frontier
   contents or the final result.

#### Security and arithmetic

1. Candidate materialization **MUST** use the same role typed security,
   sizing, and layout primitives that `akita-schedules` uses to validate an
   expanded row.
2. Generic checked `usize` arithmetic **MUST** use `akita_error::checked`.
   Planner modules **MUST NOT** introduce local checked product, sum, range,
   alignment, division, or power of two helpers.
3. A recursive state **MUST** retain every value needed to price and certify a
   future transition. This includes the predecessor response basis and any
   source moment or energy class used by a later security route.
4. A planner **MUST NOT** replace a certified rank, norm route, or digit basis
   with a locally smaller value before all later consumers have been priced.
5. Security route selection **MUST** be part of the audited decision domain or
   have a proof that an omitted route cannot win.

#### Commitment workloads

1. A group semantic role, source contract, commitment epoch, and grouped root
   role **MUST** be separate fields or separate derived concepts.
2. A group which closes a batch **MUST NOT** be described as the semantic main
   group unless the application independently assigns that role.
3. The shared opening policy **MUST** be selected explicitly for the batch. It
   **MUST NOT** be inferred from the source family of the closing group.
4. The planner **MUST** jointly compare every group commitment plan and the
   shared opening whenever those choices can affect the complete objective.
5. The protocol supplied epoch order and challenge boundaries **MUST** be
   preserved. The planner **MUST NOT** reorder groups across a transcript
   challenge.
6. Any schedule or source bound choice that affects extraction or verifier
   behavior **MUST** be publicly bound before the first commitment whose
   meaning depends on that choice.

#### Performance

1. The guided execution **MUST** be the supported normal planner execution.
2. At least one supported exact path **MUST** meet the time and memory budgets
   in this specification without reducing the audited decision domain.
3. The oracle **MAY** be expensive, but it **MUST** remain available for small
   domains, scheduled audits, new guide generation, and focused high pressure
   fixtures.
4. Every performance result **MUST** report compile exclusion, machine details,
   thread count, peak resident memory, guide identity, domain identity, and
   selected schedule identity.
5. The guided and oracle results **MUST** agree exactly on every fixture where
   the oracle is run.

#### Runtime boundary

1. No planner crate or search function **MAY** be verifier reachable.
2. Planner estimates **MUST NOT** enter proof bytes, transcript state, or
   verifier acceptance.
3. Every emitted plan **MUST** expand and validate through `akita-schedules`
   before publication.
4. Runtime catalog trust, versioned artifact loading, and row authentication
   are consumer contracts. They are not alternate planner algorithms.

### Non-goals

This specification does not change the Akita proof equations.

This specification does not change the current setup offload feasibility or
contraction rules. It gives those rules a clearer transition and pruning
boundary.

This specification does not move source identity into commitments, proofs, or
runtime schedule keys. Source laws remain offline group policy inputs.

This specification does not require the production planner to evaluate every
materialized candidate. It requires the planner to account for the complete
audited domain through evaluation, exact equivalence, or certified pruning.

This specification does not define an arbitrary dependency graph for protocol
execution. Ordered commitment epochs cover the current Fiat-Shamir use cases.
A later protocol may extend the workload model if ordered epochs are
insufficient.

This specification does not permit empirical observations to prove candidate
irrelevance. Empirical observations may choose traversal order. A mathematical
or exact structural argument must justify pruning.

This specification does not require one universal objective for every Akita
application. It requires every policy to define one complete deterministic
order and every catalog to bind its policy identity.

This specification does not make the planner a runtime service. Applications
continue to consume generated schedules.

## Terminology

An **audited decision domain** is the complete finite set of independent
choices admitted by a versioned planner policy.

A **decision** contains independent choices such as a split, basis, ring
dimension, opening method, security route, slice count, or setup offload edge.

A **materialized candidate** is the exact typed schedule fragment derived from
a decision by canonical arithmetic and security functions.

A **feasible candidate** satisfies protocol geometry, security, setup, and
resource constraints.

**Progress** is a transition property. A recursive fold progresses when its
successor satisfies the required witness contraction. Root contraction is not
a feasibility rule.

A **complete schedule** contains the root, every recursive fold, and the
terminal response.

A **guide** is a versioned offline artifact that provides candidate ordering,
an incumbent, and optional certified exclusions.

The **oracle execution** disables optional certified pruning and explores the
complete audited domain through the shared engine.

The **guided execution** uses a guide and every available certified bound to
prove the same optimum with less work.

A **pruning certificate** is a checked argument that a candidate or region
cannot improve the complete schedule total order.

A **frontier** is a set of partial schedules that remain distinguishable to a
future consumer.

A **commitment epoch** is an ordered set of groups whose values are available
and whose commitments are absorbed before the next named transcript challenge.

A **closing group** is the new group whose commitment parameters complete one
grouped root after the prior group profiles are fixed. It is an algebraic and
temporal role, not a semantic role.

A **frozen group** is an earlier committed group whose exact commitment profile
cannot be changed by the grouped root planner.

A **batch opening policy** owns the shared root and recursive opening choices
for a group batch. It is separate from every individual group source contract.

## Planner architecture

### Data flow

```text
workload and planner policy
    |
    v
request normalization
    |
    v
canonical decision enumerators
    |
    v
candidate materialization and typed rejection
    |
    v
certified bounds and guided traversal
    |
    v
exact suffix DP and parent observable frontiers
    |
    v
complete schedule selector
    |
    v
compact commitment and opening plan
    |
    v
akita-schedules expansion and validation
    |
    v
versioned catalog artifact
```

The exact Rust module names are not normative. The ownership boundaries are.

### Component responsibilities

Request normalization owns default removal, policy validation, stable ordering,
and construction of the finite audited domain.

Decision enumerators own independent choices. They do not compute a suffix
objective or retain a local winner.

Candidate materialization owns derived widths, ranks, bounds, matrices,
witness lengths, setup requirements, proof byte estimates, and typed
rejections. It calls canonical shared primitives directly.

The bound library owns admissible lower bounds and exact equivalence proofs.
It does not own candidate enumeration.

The search orchestrator owns traversal order, work queues, batching, incumbent
management, and memo use. It does not redefine feasibility or objectives.

The frontier engine owns consumer projections and dominance. It does not
reconstruct protocol parameters.

The selector owns the total order on complete schedules.

The emitter owns compact plan serialization for generated source and diagnostic
reports. It does not authorize runtime use.

`akita-schedules` owns expansion, canonical validation, and runtime schedule
identity.

### One canonical function per concept

The implementation **MUST** keep one canonical function for each decision
domain, derived quantity, objective, descriptor, and validation rule. A type
method may assemble its fields into that function. Thin aliases which only
recompose an existing API are forbidden.

Shared code means shared arithmetic, materialization, and validation. It does
not mean one large candidate builder with optional fields for every planner
mode.

## Formal planning model

Let (R) be a normalized request and (S) a planner state. Let
(mathcal D(R,S)) be the finite decision domain at that state. Materialization
maps one decision to a transition or a typed rejection:

\[
M(R,S,d) \rightarrow T \;\text{or}\; E,
\qquad d\in\mathcal D(R,S).
\]

A transition contains its exact current cost, successor state, parent visible
geometry, and compact decision. A complete path through transitions forms a
schedule (C).

Let (mathcal C(R)) be every complete feasible schedule in the audited domain.
For the configured total objective (O_R), the planner contract is:

\[
\operatorname{plan}(R)
=
\arg\min_{C\in\mathcal C(R)} O_R(C).
\]

The descriptor tie break is part of (O_R). Enumeration order is not.

If (mathcal C(R)) is empty, planning returns `UnsupportedSchedule` with
diagnostics that report the typed rejection counts. If the domain itself is
invalid, normalization returns the corresponding input or policy error before
search begins.

### Feasibility, progress, objective, and traversal

These four concepts must remain separate.

Feasibility asks whether a candidate satisfies protocol and resource rules.

Progress asks whether a recursive successor contracts enough for the selected
edge. The root may be noncontractive.

The objective compares complete feasible schedules.

Traversal decides which still possible schedule is examined first.

No Boolean such as `contractive_path`, `fast_mode`, or `fallback_mode` may
select a second planner implementation.

## Decision language and canonical materialization

### Independent decisions

The audited domain must explicitly cover every independent choice which may
change the selected complete plan.

For a root or recursive fold, this includes as applicable:

- block split;
- A source basis;
- response and opening basis;
- A, B, and D ring dimensions;
- opening method and its challenge configuration;
- commitment slice count;
- payload mode;
- witness chunk layout;
- honest fold policy result;
- Linf or L2 security route;
- setup prefix edge and exact prefix geometry;
- terminal strategy;
- commitment group profile choice;
- commitment epoch and closing group where the protocol permits a choice.

The protocol or catalog policy may fix any coordinate to a singleton domain.
A fixed coordinate is still documented as part of the domain identity.

### Derived values

The following values are derived and must not become independent knobs:

- digit counts implied by a source contract and basis;
- accepted balanced digit ranges;
- source and response norm bounds;
- matrix input widths and minimum secure ranks;
- physical segment widths;
- commitment and opening payload bytes;
- next witness length;
- setup field requirements;
- contraction status;
- proof shape and proof byte estimate;
- parent observable geometry;
- canonical descriptor bytes.

A compact generated plan may store some derived values for stable source
generation. Expansion must recompute or validate them through the canonical
owner. No unchecked duplicate becomes authoritative.

### State sufficiency

A memo key must contain every fact that can change future feasibility, pricing,
or descriptor order. It must not contain values which cannot affect a future
consumer.

In particular, recursive state must preserve:

- current level and witness length;
- source basis inherited from the preceding response;
- any source moment or certified energy class used by response modeling;
- incoming setup prefix state;
- active ring dimensions or their admissible ceiling;
- payload phase;
- workload and policy identities needed by later materialization.

The state may use a stable compact identity for large immutable workload data.

### Typed rejection

Candidate materialization returns typed planner rejections. The initial set
should distinguish at least:

```text
InvalidGeometry
ArithmeticOverflow
InsufficientSetup
InsecureSis
UnsupportedOpeningMethod
UnsupportedChunkLayout
NoRecursiveProgress
InvalidSetupPrefixEdge
UnavailableSecurityRoute
TerminalUnavailable
WorkloadOrderViolation
InvalidGuideCertificate
```

Diagnostics may add structured context such as level, group, split, basis, or
dimension. Planner logic must not branch on error message strings.

## Phased commitment workloads

### Why the existing key is not enough

`AkitaScheduleLookupKey` is a compiled lookup key. It contains one new group
layout and exact frozen profiles for every prior group. It remains useful after
planning, but it cannot express the joint search that chooses those profiles.

The planner therefore needs a request which exists before any individual group
recipe is frozen.

### Conceptual request

The intended information content is:

```text
CommitmentWorkload
    groups: ordered GroupRequirement list
    epochs: ordered CommitmentEpoch list
    opening_batches: OpeningBatchRequirement list
    batch_opening_policy
    selection_policy

GroupRequirement
    id
    semantic_label
    logical_layout
    source_contract
    allowed_commitment_policy
    epoch

CommitmentEpoch
    ordered_group_ids
    challenge_after_epoch

OpeningBatchRequirement
    ordered_group_ids
    closing_group_id
    reorder_policy
```

The final Rust types may differ. The separation of information is normative.

`semantic_label` is for protocol review and diagnostics. Core planner decisions
must use the layout and source contract, not a match on an application name.

`source_contract` selects the offline honest fold policy and the declared
extraction requirements. Runtime profiles remain source free as required by
[`heterogeneous-group-source-contracts.md`](heterogeneous-group-source-contracts.md).

`epoch` states when the witness values exist and where their commitment is
absorbed. It does not select a commitment profile.

`closing_group_id` names the group which is new when the grouped root is
formed. Every other group in that opening batch is frozen at that point.

`batch_opening_policy` owns the shared opening and recursion policy. It is not
copied from the closing group configuration.

### Ordered epochs

Each epoch contains groups whose commitments may be computed in parallel after
their values become available. Their payloads are absorbed in the exact order
given by the epoch.

If an epoch produces a challenge, every earlier group profile, schedule choice,
and commitment payload required by that challenge must already be bound. A
later epoch may depend on that challenge.

The planner may compare alternative group orders only when `reorder_policy`
explicitly permits them. It may never move a group across a challenge boundary
without a protocol revision.

### Aerie Falcon version 1

Aerie is the required heterogeneous phased workload fixture.

| Group | Semantic role | Epoch | Extraction family | Grouped root role |
|---|---|---|---|---|
| `S1` | Falcon intermediate | before JL seed | bound 18 | frozen |
| `Epsilon` | four square slack | before JL seed | bound 18 | frozen |
| `S2SourcesFour` | decoder and high part | before JL seed | bound 6 | frozen |
| `BudgetTwo` | body budget | before JL seed | bound 6 | frozen |
| `BudgetOne` | body budget | before JL seed | bound 6 | frozen |
| `JlFour` | four low projection digits | after JL seed | bound 6 | frozen |
| `JlTwo` | two high projection digits | after JL seed | bound 6 | frozen |
| `JlOne` | top projection digit | after JL seed | bound 6 | closing |

The JL digit values depend on the seed, but their layouts and extraction
contracts do not. The offline planner can therefore select the complete eight
group plan before proving begins.

The selected plan identity must be bound before the first commitment if any
choice affects extraction, group profiles, transcript shape, or verifier
behavior. Prover only conversion shortcuts which preserve the same committed
profile do not require a new schedule identity.

The closing role of `JlOne` follows from the Aerie transcript order. It does not
make JL digits the semantic main route. It also does not make the shared opening
a bound 6 policy by implication.

### Joint group profile planning

For each group requirement, the planner enumerates a local set of secure exact
commitment profiles. It then compares combinations through the complete batch
opening objective.

A local profile may be removed only when it is dominated for every opening
batch and policy which can consume that group. A locally larger profile may
win globally if it reduces the grouped root witness, a later fold, total setup,
or the canonical schedule descriptor.

The implementation must stream group profile combinations into the root
frontier. It must not materialize the full Cartesian product.

Groups with identical requirements and exchangeable transcript positions may
use a multiset domain. The symmetry proof must show that omitted permutations
have identical feasibility, cost, parent observations, and descriptor handling.
If transcript order distinguishes two groups, they are not exchangeable.

### Compiled lookup and execution plan

After joint planning, the selected profiles compile to the exact ordered batch
profile needed by runtime lookup and commitment execution.

The output must distinguish:

```text
GroupCommitPlan
    group id
    exact committed profile
    epoch and absorb position

BatchOpeningPlan
    ordered group ids
    closing group id
    exact fold schedule
    catalog and row identity
```

The current `final_group` and `precommitteds` representation may remain inside
the compiled runtime schedule while the implementation migrates. Planner APIs
and diagnostics should use `closing_group`, `new_group`, `frozen_groups`, or
`prior_groups` according to the exact role. `main_group` is forbidden when the
code means only the new or last group.

## Candidate domains

### Domain identity

Every catalog binds a stable identity for the complete candidate language.
The identity includes:

- basis ranges;
- split domain version;
- ring dimension domain;
- opening method domain;
- slice and payload domains;
- source policy identities;
- security route domain;
- setup offload policy;
- terminal strategies;
- workload reorder policy;
- objective policy;
- security table and modulus profile identities.

A guide is valid only for the exact domain identity or for a declared region
whose validity predicate accepts that domain.

### Root decisions

Root candidate enumeration must cover every split in the audited root split
domain. Root contraction may order candidates but may not admit or reject them.
The root enumerator must materialize both contractive and noncontractive
candidates through the same suffix search.

A bounded root split domain is acceptable only if the policy declares that
domain as an approximation, or if every excluded split has a checked optimality
certificate. Production exact catalogs must use the second choice.

### Recursive decisions

Recursive folds require strict progress under the active successor policy.
Progress is checked after exact witness materialization or through a
conservative lower bound that proves progress impossible.

The production planner must account for every recursive split in the audited
domain. A balance estimate and fixed radius may seed traversal, but they cannot
define an exact domain without a proof.

### Basis and dimension decisions

The A source basis and response basis remain independent coordinates where the
policy permits both. A balanced recursive source uses the basis which produced
it and is not decomposed again under a new A basis.

Dimension domains come only from `RingDimensionScheduleMode`. A guide may order
dimension tuples. It may exclude a tuple only through a complete feasibility or
objective certificate.

### Security route decisions

Linf and selective L2 routes are independent candidate routes when both are
eligible. An L2 route may reuse a split selected under Linf only if a proof
shows that no other L2 split can win the complete objective.

The production search must first apply geometry and witness bounds which do not
depend on the security route. It must then evaluate every eligible route for
every surviving split cell. A complete route dominance certificate may remove
a route from a region. The split chosen by another route is never a route
domain definition.

The route decision carries its source moment or energy class through every DP
state which may price a later witness. The selected route fixes exact verifier
visible bounds and ranks in the emitted schedule. Runtime expansion never
reruns the response model.

### Slice decisions

Slice candidates must remain until exact future irrelevance is established.
For setup first selection, choosing the smallest local padded setup and then
the smallest slice count is not sufficient by itself. The proof must cover the
next witness, suffix proof, total setup envelope, and descriptor order.

The current slice domain has four values, 1, 2, 4, and 8. The planner should
materialize their cheap transition signatures before it starts suffix search.
It may retain every signature which is not worse on all relevant coordinates.
It may remove a slice only when the transition dominance theorem in the proof
addendum applies.

### Setup offload decisions

The direct and offloaded forms are transitions in one recursive domain. Local
prefix geometry may use a canonical local minimizer only when the setup
offloading contract proves that every other local geometry is irrelevant to
the global suffix. The decision whether to offload remains global.

### Terminal decisions

Every state which may legally terminate must enumerate its terminal routes.
The planner compares termination with another fold through the complete
objective. A terminal candidate does not require an unused successor witness
to contract.

## Complete objectives and frontiers

### Current complete orders

This architecture preserves the current complete schedule orders until a
separate policy revision changes them.

`MinEstimatedProofPayload` uses:

```text
(proof bytes, total setup field elements, canonical descriptor)
```

`MinFirstDirectSetupThenPayload` uses:

```text
(first direct padded setup capacity,
 proof bytes,
 total setup field elements,
 canonical descriptor)
```

The descriptor is compared only after all numeric coordinates tie.

If joint commitment profile planning needs commitment payload bytes in the
objective, the policy must add that coordinate explicitly and change its
identity. The planner must not silently reinterpret `proof bytes`.

### Parent observations

A partial suffix may be merged with another only when every possible parent
observes them as equivalent or prefers one under the complete order.

The frontier contract must list:

- the exact parent visible key;
- admission classes such as required fold depth and setup capacity;
- numeric projections used by each selection policy;
- descriptor context needed when numeric coordinates tie;
- the proof that a discarded candidate cannot win for any parent.

No frontier coordinate may remain merely because it once affected an older
planner. No parent visible value may be omitted merely because it does not
affect the current local score.

### Complete lower bounds

A lower bound used to prune a partial schedule must be a prefix of the same
lexicographic objective used for complete selection. It may omit the descriptor
only when pruning requires strict numeric inferiority.

Equal numeric lower bounds cannot prune because an unvisited schedule may win
the canonical descriptor tie break.

## One search engine

### Shared semantics

The planner has one work queue and one candidate materialization path. The
engine accepts an execution configuration which controls optional certificates,
diagnostics, and guide use. It does not select a different planner.

Conceptually:

```text
normalize request
load and validate optional guide
seed incumbent from guide if feasible
enqueue the complete audited region
while work remains:
    take the region with the best priority
    if a checked bound is strictly worse than the incumbent:
        record the certificate and discard the region
    else if the region is splittable:
        refine it and enqueue its children
    else:
        materialize candidates and update exact frontiers
select the best complete schedule
expand and validate the plan
```

Priority may use empirical models. Discarding work may not.

### Oracle execution

The oracle uses the same engine with optional pruning disabled. Exact
equivalence reductions which define the audited domain, such as a proved
symmetry quotient, remain enabled.

The oracle must support:

- exhaustive small domains in ordinary tests;
- focused full split scans for named recursive states;
- focused group profile products;
- scheduled high pressure fixture runs;
- guide generation and guide audits.

The oracle output includes the selected plan, complete objective, runner up
gap, domain counts, and a stable trace digest.

### Guided execution

The guided execution is the default for catalog generation. It uses:

- a known feasible incumbent;
- preferred decision order;
- certified feasible intervals;
- certified impossible intervals;
- symmetry classes;
- complete objective lower bounds;
- cached exact local profile frontiers.

The guide may identify the schedule found by a previous oracle run. That makes
the incumbent available immediately. Fast completion still requires the engine
to certify every remaining region as worse or equivalent.

There is no restart after a guide miss. An invalid incumbent is rejected and
the same queue continues. An invalid optional exclusion is ignored and the
region remains queued. The diagnostic report records each ignored guide item.

### Guide artifact

A guide artifact must contain:

```text
format version
planner domain identity
objective identity
security table identity
workload region predicate
incumbent compact plan
ordering hints
certified exclusions
generator revision and evidence reference
guide digest
```

Ordering hints may be broad and empirical. Certified exclusions must name a
checker implemented by the bound library and contain only the data needed by
that checker.

The planner must validate the incumbent as if it had just materialized it. The
planner must validate every exclusion before using it. The guide is not a
trusted proof system input.

A guide for one exact catalog row may use an equality region predicate. A
guide intended for nearby rows must state checked ranges for every workload
coordinate on which its certificates depend.

## Certified pruning

### Rule contract

Every pruning rule must have a stable name and a specification with these
fields.

| Field | Required meaning |
|---|---|
| Domain | States, workloads, and policies where the rule applies |
| Predicate | Exact checked condition used by code |
| Bound | Quantity bounded and its objective position |
| Proof | Why every removed candidate is unable to win |
| Unknown behavior | Retain the candidate or region |
| Oracle check | Test which compares the rule with pruning disabled |
| Diagnostics | Counts, time, and region size removed by the rule |

A code comment which says that a candidate is unlikely to win is not a pruning
proof.

### Recursive witness body bound

The mandatory Z, E, and T body provides the first certified recursive split
bound. The full proof appears in the addendum below.

Let \(N\) be the current ring element count. Let \(p\) be the number of
position bits inside a block. Define

\[
M_p=2^p,
\qquad
q_p=\left\lceil\frac{N}{2^p}\right\rceil
\]

where \(M_p\) is the number of positions in one block and \(q_p\) is the number
of live blocks. In any split cell where all ranks, digit depths, compression
choices, and relation geometry are fixed, the exact current group body has the
form

\[
F(p)=a q_p+b2^p,
\]

where

\[
a=m\left(\delta_o w_E+n_A\delta_B d_A\right),
\qquad
b=c\delta_i\delta_f d_A.
\]

Here \(m\) is the claim count, \(c\) is the chunk count, \(w_E\) is the
physical E row width, \(n_A\) is the A row count, and \(d_A\) is the A ring
dimension. The four digit depths are the exact inner, fold, opening, and outer
depths for that cell. This identity follows from the canonical witness layout,
which is shared by the planner, prover, and verifier.

Frozen group bodies and every setup prefix, quotient, compression, and
alignment term are nonnegative. They can be added exactly when known or
omitted from a conservative lower bound. Therefore, if the body lower bound is
at least the current witness length, that split cannot produce a strictly
contracting recursive fold.

The sequence is discrete convex for the relevant positive domain. If
\(d_p=q_p-q_{p+1}\), then \(d_p\) is nonincreasing and

\[
F(p+1)-F(p)=-a d_p+b2^p
\]

is nondecreasing. The strict contraction sublevel is therefore contiguous.
The same result holds for every fixed split cell. The planner can inspect the
small number of integer split values with checked arithmetic, without building
their matrices or recursive suffixes.

This theorem proves recursive progress impossibility. It does not prove that a
remaining split is globally optimal.

### Local layout lower bound

Adding the mandatory challenge and chunk work to \(F(p)\) gives a lower bound
on the local layout score. Once a local best candidate exists, this bound may
remove a split from a `Best` search when it is strictly worse in the first
local score coordinate.

This rule is valid only for a consumer whose contract is exactly that local
best choice. It is not sufficient for a global frontier because a locally
larger next witness may expose different parent geometry, setup capacity,
security routes, or descriptor order.

### Complete schedule bounds

The main guided speedup must come from complete schedule lower bounds. A bound
for a partial transition must include every already fixed cost and a
conservative lower bound for every possible suffix cost which appears before
the descriptor tie break.

The implementation should derive bounds in increasing cost order:

1. impossible geometry and recursive progress;
2. mandatory current level proof bytes;
3. minimum possible successor and terminal bytes;
4. minimum first direct setup capacity;
5. minimum total setup envelope;
6. parent visible payload and admission class.

The engine may stop evaluating later bound terms once an earlier objective
coordinate is already strictly worse than the incumbent.

### Symmetry certificates

Interchangeable groups may use nondecreasing profile assignments rather than
all permutations. The certificate must prove that permutation does not change:

- transcript meaning;
- group source policy;
- feasibility;
- exact root widths and proof cost;
- setup envelope;
- parent observations;
- canonical descriptor comparison.

If the descriptor includes group order, the canonical representative must be
defined before quotienting. If semantic or transcript roles distinguish two
groups, the rule does not apply.

### Slice and security route certificates

A slice dominance proof must cover the complete future objective. Until that
proof exists, all feasible slice choices remain in the exact frontier.

A security route dominance proof must compare the exact norm proof, A payload,
next witness, later suffix, terminal response, and setup consequences. A lower
A rank alone is not enough.

### Memo and cache rules

Memoization stores exact completed frontier results keyed by sufficient state.
Cache quotas are performance settings, not semantic settings.

Eviction removes only the cached result. A later lookup recomputes the same
state. Tests must run with several small capacities, including zero effective
reuse, and obtain the same selected descriptor.

Guide caches and local profile caches follow the same rule. A cache hit may
save work. It may not add authority.

## Addendum: Certified pruning proofs

This addendum is normative. It states the proof obligations behind the
certified pruning architecture. A later implementation may use a stronger
bound, but it must prove at least the same safety claim and expose the same
unknown behavior.

### Proof boundary

The planner minimizes a total order on complete schedules. The numeric prefix
is one of these orders.

\[
(P,S)
\]

or

\[
(C_1,P,S).
\]

Here \(P\) is proof bytes, \(S\) is the total setup envelope, and \(C_1\) is
the first direct padded setup capacity. The canonical descriptor follows the
numeric prefix in both orders.

A lower bound contains only numeric coordinates. It may prune a region only
when it is strictly worse than a completed incumbent on the numeric order.
Equality is not enough because the canonical descriptor can still choose a
candidate from that region.

All formulas use mathematical integers. Their checkers use the canonical
checked arithmetic functions. An overflow or an unsupported table lookup
returns unknown and retains the region.

### Proof status

| Result | Status in this specification | Current implementation status |
|---|---|---|
| Exact mandatory Z, E, and T body identity | Proved below from the canonical witness layout | The canonical formula exists, but the full cell bound is not yet used |
| Discrete convexity inside one split cell | Proved below for every positive coefficient choice | A weaker lower bound is used for local split checks |
| Incumbent interval around the analytic balance point | Proved below | Not implemented |
| Relaxed suffix lower bound | Proved below by induction over remaining depth | Current direct edge bound uses a zero suffix cost |
| Same state transition dominance | Proved below | Exact completed suffix frontiers implement part of this idea |
| Interchangeable group symmetry | Proved below under an explicit equivalence relation | PR #412 applies a narrower form |
| L2 route omission at the Linf winning split | Not proved and not accepted as an exact domain | Current code uses this shortcut |
| Local setup first slice pruning | Not proved and not accepted as an exact domain | Current code keeps one local slice |
| Fixed radius recursive split search | Not proved and not accepted as an exact domain | Current bounded policy uses this shortcut |

The last three rows are migration requirements. This PR specifies their
replacement. It does not claim that the present planner already satisfies the
target contract.

### Theorem 1: exact mandatory body in one split cell

Fix a planner state, one current group, and one region of split values where
the following data are constant.

- The claim count \(m\).
- The witness chunk count \(c\).
- The A ring dimension \(d_A\).
- The physical E row width \(w_E\).
- The A row count \(n_A\).
- The inner, fold, opening, and outer digit depths
  \(\delta_i,\delta_f,\delta_o,\delta_B\).
- The source encoding, security route, relation geometry, and compression plan.

Call such a region a split cell. Let \(M_p=2^p\), and let
\(q_p=\lceil N/M_p\rceil\). The canonical witness layout creates one Z range
per chunk. It creates E and T ranges whose block counts sum to \(q_p\) across
all chunks. Their exact physical coefficient counts are

\[
Z(p)=c M_p \delta_i \delta_f d_A,
\]

\[
E(p)=m q_p \delta_o w_E,
\]

and

\[
T(p)=m q_p n_A \delta_B d_A.
\]

Therefore the current group body is exactly

\[
F(p)=Z(p)+E(p)+T(p)=a q_p+b2^p,
\]

with

\[
a=m(\delta_o w_E+n_A\delta_B d_A)
\]

and

\[
b=c\delta_i\delta_f d_A.
\]

Proof. The canonical `witness_unit_lengths` function gives the three lengths
for one group and one chunk. Summing Z over \(c\) chunks gives the first
formula because every chunk has one Z range of the same length. The dyadic
chunk ranges partition the \(q_p\) live blocks. Summing their lengths gives
\(q_p\), which gives the E and T formulas. Adding the three terms gives the
result.

Every coefficient is positive for a valid candidate. Frozen group bodies,
setup prefixes, quotient rows, compression layers, and alignment can only add
physical coefficients. The planner may add any of those terms when it can
compute them cheaply. Omitting them preserves a lower bound.

### Corollary 1.1: discrete convexity

For positive \(a\) and \(b\), the sequence

\[
F(p)=a\left\lceil\frac{N}{2^p}\right\rceil+b2^p
\]

is discrete convex on the integer split values in one cell.

Proof. Let \(q_p=\lceil N/2^p\rceil\). Then

\[
q_{p+1}=\left\lceil\frac{q_p}{2}\right\rceil
\]

and

\[
q_p-q_{p+1}=\left\lfloor\frac{q_p}{2}\right\rfloor.
\]

The last quantity does not increase with \(p\). The first difference is

\[
F(p+1)-F(p)
=-a\left\lfloor\frac{q_p}{2}\right\rfloor+b2^p.
\]

Its negative term becomes less negative while its positive term increases.
The first difference therefore does not decrease. This is discrete convexity.

Every sublevel set of a discrete convex sequence is an integer interval. In
particular, all splits which can satisfy a contraction threshold form one
interval inside a cell.

### Split cell construction

The theorem does not assume that security ranks or digit depths stay fixed over
the whole split domain. The planner must create a new cell at every checked
change to any value which affects \(a\), \(b\), fixed body terms, edge cost, or
the successor state. These changes include:

- a security table key or selected rank;
- an inner, fold, opening, or outer digit depth;
- an opening relation width or A row count;
- a source encoding or response basis;
- a compression plan or setup offload form;
- a selective L2 eligibility or norm proof shape change.

The cell builder does not need to guess these boundaries. It can evaluate the
cheap signature at each supported integer split and group adjacent equal
signatures. The split count is tiny compared with matrix construction and
recursive suffix search. This exact scan removes the need for a fixed semantic
radius.

### Theorem 2: incumbent interval

Suppose a checked lower bound in one cell has the following form in the
coordinate being pruned.

\[
L(p)\geq a\frac{N}{2^p}+b2^p+C.
\]

Here \(a>0\), \(b>0\), and \(C\geq0\). Let \(U\) be the largest value in the
same coordinate which could still tie or beat the incumbent after the other
fixed lower bound terms are included. Define

\[
x=2^p,
\qquad
x_0=\sqrt{\frac{aN}{b}},
\qquad
\rho=\frac{U-C}{2\sqrt{abN}}.
\]

If \(\rho<1\), no split in the cell can win. If \(\rho\geq1\), every split
which can win satisfies

\[
\rho-\sqrt{\rho^2-1}
\leq
\frac{x}{x_0}
\leq
\rho+\sqrt{\rho^2-1}.
\]

Proof. Since \(q_p\geq N/2^p\), a winning split must satisfy

\[
a\frac{N}{x}+bx+C\leq U.
\]

Divide by \(\sqrt{abN}\) and set \(y=x/x_0\). The condition becomes

\[
y+\frac{1}{y}\leq2\rho.
\]

For positive \(y\), this is equivalent to

\[
y^2-2\rho y+1\leq0.
\]

The roots give the stated interval. The quadratic has no real root when
\(\rho<1\), so no split can meet the bound in that case.

The interval width in split bits is

\[
2\log_2\left(\rho+\sqrt{\rho^2-1}\right).
\]

The following table gives a conservative integer count. It uses one plus the
ceiling of that width, then clips the result to the cell.

| \(\rho\) | Width in split bits | Splits to inspect at most |
|---:|---:|---:|
| 1.05 | 0.91 | 2 |
| 1.10 | 1.28 | 3 |
| 1.25 | 2.00 | 3 |
| 1.50 | 2.78 | 4 |
| 2.00 | 3.80 | 5 |

This theorem replaces a fixed radius with a checked interval. A strong guided
incumbent makes \(\rho\) close to one, so the exact surviving interval is
usually small. A weak incumbent leaves a wider interval but never changes the
answer.

### Corollary 2.1: exact split traversal

An exact split search may use this order.

1. Compute the cheap signature for every supported split.
2. Form maximal adjacent split cells with equal signatures.
3. Materialize and validate the guided incumbent.
4. Apply contraction and feasibility bounds to each cell.
5. Apply the incumbent interval to each remaining cell.
6. Evaluate the exact integer splits in the surviving intervals.
7. Use the complete suffix lower bound below before expanding a child.

The scan in step 1 is part of the exact path. It creates small checked values,
not matrices or suffix schedules. The oracle can disable steps 4, 5, and 7
while using the same enumerator and materializer.

### Theorem 3: relaxed suffix lower bound

Let \(V(s)\) be the best complete remaining objective from a real planner state
\(s\). Define a relaxed suffix problem with these properties.

1. Every real transition from \(s\) has a corresponding relaxed transition.
2. The relaxed edge cost is no greater than the real edge cost in every
   objective coordinate which a parent can observe.
3. Every real child state maps to a relaxed child state.
4. The relaxed terminal cost is no greater than the real terminal cost.
5. The operation which combines a prefix edge with a suffix objective is
   monotone.

Let \(h(s)\) be the optimal value of the relaxed problem. Then

\[
h(s)\leq V(s)
\]

in the complete numeric order.

Proof. Use induction on the maximum remaining fold depth. At a terminal state,
property 4 gives the result. Assume the result for every child. For any real
transition \(t\), properties 1 to 3 provide a relaxed transition \(t'\).
The induction hypothesis gives a relaxed child value no greater than the real
child value. Property 5 preserves this order when the edge and child values are
combined. The relaxed optimum is no greater than this mapped value because it
minimizes over a superset of transitions. It is therefore no greater than the
best real value.

For proof bytes, prefix combination is addition. For total setup, it is the
maximum of the edge setup and suffix setup. For the first direct setup
coordinate, it keeps the first direct capacity already chosen by the prefix or
uses the suffix value when the prefix is offloaded. Each operation is
monotone.

The planner can combine the exact fixed prefix with \(h(s)\). It may prune the
region only when the result is strictly worse than a completed incumbent on
the numeric order. It must retain an equal bound for descriptor comparison.

### Relaxed state and useful bound terms

A relaxed state can merge details only when the merge preserves the theorem.
The first implementation should retain at least:

- the payload phase and the minimum and maximum remaining fold depth allowed
  by admission;
- a checked witness length interval;
- the incoming setup prefix capacity class;
- the available ring dimension ceiling;
- the source moment or energy class;
- the response basis and security route eligibility;
- the parent admission class;
- the descriptor context needed to detect numeric equality.

The first useful relaxed edge should include:

- exact bytes already fixed at the current level;
- minimum terminal bytes for the witness interval;
- minimum mandatory bytes for every fold which admission still requires;
- a lower bound on the first direct setup capacity;
- a lower bound on the total setup envelope.

The current direct edge bound is the valid but weak special case where the
suffix contributes zero. The implementation can strengthen one term at a time.
Every added term needs a focused proof test and an oracle comparison.

### Theorem 4: transition dominance

Consider two transitions \(t_a\) and \(t_b\) from the same exact planner state.
Transition \(t_a\) dominates \(t_b\) only if all of these conditions hold.

1. Every parent and suffix form which admits \(t_b\) also admits \(t_a\).
2. The transitions have the same sufficient child state, or a separate proof
   maps every child completion of \(t_b\) to a no worse completion of \(t_a\).
3. Every exact edge and parent visible objective projection of \(t_a\) is no
   worse than the matching projection of \(t_b\).
4. If the numeric projections can tie, descriptor composition proves that
   \(t_a\) is no worse. Otherwise one numeric coordinate before the descriptor
   must be strictly better.

Under these conditions, removing \(t_b\) cannot change the selected complete
schedule.

Proof. Take any complete schedule which begins with \(t_b\). Condition 2 gives
a completion after \(t_a\). Conditions 1 and 3 show that the replacement is
admitted and is no worse on every numeric coordinate seen by any consumer.
Condition 4 handles the only remaining tie. Thus every schedule removed with
\(t_b\) has a retained schedule which the complete selector prefers or treats
as equal.

If the child states differ and no mapping proof exists, a lower bound on one
child is not enough to establish dominance over all completions of the other.
The planner must retain both transitions. It may still prune one later when a
completed incumbent is strictly better than that transition plus \(h(child)\).

### Selective L2 route completeness

Selective L2 and Linf are separate routes. Their security ranks, proof shapes,
and successor witnesses can cross at different split values. There is no
general implication from the best Linf split to the best L2 split.

The current shortcut creates selective L2 only at the split chosen by the best
modeled Linf candidate. It also rejects L2 when its inner A rank is not smaller
than the Linf rank. Neither fact alone compares the norm proof, the B rank, the
next witness, the suffix, the setup envelope, or the descriptor. The target
exact planner must not use either fact as a complete route proof.

The complete route search works as follows.

1. Apply geometry, body, and incumbent interval bounds shared by both routes.
2. For every surviving split, derive each eligible Linf and L2 route from the
   same source state.
3. Partition each route at its own security table, digit, and proof shape
   boundaries.
4. Build a route transition signature and retain its exact frontier.
5. Apply `l2_route_dominance_v1` only when the checker proves Theorem 4.

The L2 transition signature contains at least:

- the certified L2 table key and exact A rank;
- the source moment and challenge L2 bound;
- the norm proof shape and bytes;
- the resulting A payload and B rank;
- the next witness body, relation tail, and exact length when available;
- the current edge proof bytes;
- the level setup envelope;
- the sufficient child state and parent admission class;
- the canonical descriptor prefix.

A Linf region can dominate an L2 region only if the checker proves all of the
following over the whole region.

- Linf admits every parent and suffix admitted by L2.
- Its first direct setup coordinate is no worse.
- Its proof bound, including the absence or presence of a norm proof, is no
  worse.
- Its setup envelope is no worse.
- Its child state is equal or is covered by a certified suffix mapping.
- Any numeric tie is safe under the descriptor order.

The reverse comparison uses the same conditions. If any input is unknown, both
routes remain. The route frontier must include tests where the best L2 and Linf
splits differ.

### Setup first slice completeness

The current setup first shortcut chooses the smallest local padded setup and
then the smallest slice count before it computes every successor witness and
suffix. Slice count changes the outer commitment input width and can change the
B rank. It can also change relation rows, compression, the next witness, proof
bytes, total setup, parent admission, and descriptor order.

The exact planner treats slice count as an ordinary transition decision. Since
the current domain has only four values, it first builds all feasible cheap
slice signatures. It then applies `setup_slice_dominance_v1` or retains a
frontier.

The slice transition signature contains at least:

- the exact physical B input width;
- the B security table key and exact secure rank;
- the logical B row count and complete B source coefficients;
- the compression plan identity and admission result;
- the next witness body and exact length when available;
- the active and padded setup capacities;
- the current edge proof bytes and setup field elements;
- the sufficient child state, including source moment and response basis;
- the parent admission class and descriptor prefix.

One slice may remove another before suffix expansion only when Theorem 4 holds.
The usual cheap case is equal sufficient child state with no worse exact edge
cost and a safe descriptor prefix. When child states differ, the planner keeps
both until it has either a mapping proof or a completed incumbent which is
strictly better than the other edge plus its relaxed suffix bound.

The setup first implementation must test all four slice values against a run
with slice pruning disabled. Boundary cases include a B rank change, a
compression plan change, equal padded setup with different next witnesses, a
smaller first setup with a larger total proof, a parent which masks the child
setup envelope, and an equal numeric score decided by the descriptor.

### Theorem 5: interchangeable group symmetry

Let \(g\) groups have the same allowed set of \(k\) profile choices. Suppose a
permutation of those groups preserves all of the following.

- Semantic role and source contract.
- Commitment epoch and transcript position class.
- Candidate feasibility and security sizing.
- Exact current proof and setup costs.
- Sufficient successor state and parent observations.
- Batch opening and closing group behavior.
- Canonical descriptor comparison after a declared representative is chosen.

Then the planner may search profile multiplicities instead of labeled profile
assignments. The number of multiplicity choices is

\[
\binom{k+g-1}{g}
\]

instead of \(k^g\).

Proof. Every labeled assignment maps to a vector of \(k\) nonnegative profile
counts which sums to \(g\). The assumptions make all assignments with the same
count vector equal under feasibility, cost, successor state, parent
observations, and descriptor representative. Keeping one representative per
vector therefore preserves the selected schedule. The number of nonnegative
count vectors which sum to \(g\) is the stated binomial coefficient.

If group identifiers, transcript order, semantic roles, source policies, or
the closing group role change any observed value, the groups are not
interchangeable. The planner then retains the labeled assignments or proves a
smaller equivalence class.

### Certificate registry

The first implementation should expose these stable rule names.

| Rule | Removes | Checked basis | Unknown behavior |
|---|---|---|---|
| `recursive_body_cell_v1` | Splits which cannot meet progress or a local body threshold | Theorem 1 and Corollary 1.1 | Retain the split |
| `recursive_incumbent_interval_v1` | Splits outside the incumbent interval | Theorem 2 with a same coordinate budget | Retain the cell |
| `relaxed_suffix_dp_v1` | Regions whose prefix plus relaxed suffix is strictly worse | Theorem 3 | Retain the region |
| `transition_dominance_v1` | One transition from an exact state | Theorem 4 | Retain both transitions |
| `l2_route_dominance_v1` | Linf or L2 route regions | Theorem 4 plus the route signature | Retain both routes |
| `setup_slice_dominance_v1` | Slice transitions | Theorem 4 plus the slice signature | Retain every slice |
| `interchangeable_group_symmetry_v1` | Permutations inside an equivalence class | Theorem 5 | Retain labeled assignments |

Each diagnostic record includes the rule name, checker version, normalized
input region, bound value, incumbent value, strict comparison result, and the
number of candidates removed. A guide may refer to these names and inputs. It
may not provide a trusted boolean result.

### Proof test matrix

Every rule must pass three levels of testing.

1. Formula tests compare the checked bound with exact canonical materialization
   at every small domain point and at every table or digit boundary.
2. Rule tests compare the candidates removed by one enabled certificate with
   the same search where only that certificate is disabled.
3. Search tests compare guided and oracle complete objectives and descriptors
   across randomized enumeration order, memo capacity, and batch size.

The split tests cover every integer split in small domains, every cell boundary,
and incumbent equality. The L2 tests pair every split with every route. The
slice tests pair every split and route with all four slice values. The symmetry
tests compare labeled enumeration with multiplicity enumeration.

For larger named fixtures where the full oracle is expensive, the repository
keeps a diagnostic oracle run outside routine CI. A guide generated from that
run does not prove later pruning. The checked certificates do. Routine CI
replays the certificates, verifies the selected descriptor, and samples the
interior and boundary of every declared region.

## Traversal and memory architecture

### Best first region traversal

The guided engine should order regions by an admissible complete objective
bound when one is available. An empirical estimate may break equal priority or
order regions which have no strong bound.

The engine should materialize the guided incumbent first. It should then prove
large regions irrelevant before enumerating individual candidates.

### Streaming group products

Group profile products, opening method products, and dimension products must be
generated incrementally. A batch is sent directly into the shared exact root
frontier. The full union is never stored.

The batch size is a performance setting. Changing it must not change the final
schedule. Tests must compare several batch sizes and reversed batch order.

### Parallelism

Independent regions may be evaluated in parallel. Workers return exact
candidates or certified exclusions. A deterministic reducer applies the shared
frontier and complete selector.

Wall clock race order must not decide an equal score. The canonical descriptor
is the final tie break.

### Resource limits

The planner may enforce explicit memory and work budgets in diagnostic tools.
A budget exhaustion result is distinct from `UnsupportedSchedule` and does not
authorize a partial optimum claim.

Production catalog generation must choose budgets which meet the supported
fixture contract. If those budgets are exhausted, generation fails with a
resource error and preserves diagnostics.

## Diagnostics and evidence

Every planner run should be able to report:

- normalized request and domain identities;
- guide identity and validation result;
- raw region and candidate counts;
- counts removed by each exact equivalence;
- counts removed by each pruning certificate;
- candidates fully materialized;
- typed rejection counts;
- suffix calls, memo hits, recomputations, and evictions;
- frontier sizes and peak retained candidates;
- time and peak memory by phase;
- selected complete objective and descriptor;
- runner up objective and gap;
- selected decisions at every level;
- ignored or stale guide entries;
- expansion and validation result.

An oracle run additionally emits a stable decision census. A guide generation
tool records which observations were used only for ordering and which facts
became checked certificates.

Diagnostics are not schedule identity unless a separate artifact format names
them. Timing, memory, and empirical estimates never enter proof or transcript
state.

## Output and runtime boundary

### Compact plan

The planner emits independent decisions and stable identities. It does not
emit unchecked mirrors of every derived field.

The compact plan contains:

- workload and domain identity;
- selected group commitment profiles;
- commitment epoch and closing group assignments;
- root decision;
- recursive decisions;
- setup prefix decisions;
- terminal decision;
- objective policy identity;
- guide evidence identity for audit only.

`akita-schedules` expands the compact plan with canonical arithmetic and
security functions, validates the complete schedule, and computes the row
identity.

### Runtime catalog

The runtime consumer obtains an approved versioned catalog and resolves a row
from public workload geometry and profiles. Prover and verifier reconstruct the
same selection without running the planner.

If trusted catalog artifacts replace generated Rust tables, as proposed by
[PR #428](https://github.com/LayerZero-Labs/akita/pull/428), the planner output
contract remains the same. Artifact trust, loading, and digest binding are
owned by `akita-schedules` and application configuration.

### Compatibility

Akita provides no backward compatibility guarantee. The workload request,
compact plan, guide artifact, and catalog identity may use new versioned
formats. The cutover must not retain old and new planner request types through
pass through wrappers.

Existing runtime schedule formats may migrate separately when a direct cut is
safer. During that migration, one canonical adapter may compile the new plan
to the existing exact runtime key. The adapter must contain no search or policy
logic.

## Performance contract

### Supported normal path

The guided execution is the only supported path for routine full catalog
generation. The oracle is a diagnostic and evidence path.

The first implementation must include at least these named fixtures:

1. A standard single group direct row.
2. A high pressure recursive row with at least 36 source variables.
3. An adaptive or setup offload row which requires a split frontier.
4. The Aerie Falcon version 1 eight group target workload.
5. The Aerie minimum batch workload where several groups use the commitment
   floor.

### Initial budgets

After compilation, on the checked in planner reference runner:

- each named guided row **MUST** complete in at most 60 seconds;
- the full stock catalog **MUST** complete in at most 10 minutes;
- peak resident memory **MUST** remain at or below 4 GiB;
- each high pressure guided fixture **MUST** use at most 20 percent of the
  median oracle wall time when the oracle completes within the audit window.

The benchmark record must name the exact CPU, memory, operating system, Rust
version, feature set, thread count, and commit. Until the repository selects a
shared reference runner, the relative 20 percent requirement is the blocking
cross machine criterion and absolute budgets are reported rather than enforced
in ordinary pull request CI.

Three fresh process runs are required. The median wall time and maximum peak
memory are reported. Compilation and dependency download are excluded.

An implementation revision may change these budgets only through a reviewed
spec update with current measurements. It may not relax a budget by narrowing
the decision domain.

### Measurement commands

The implementation must add a planner performance harness which accepts:

```text
--fixture <name>
--execution oracle|guided
--guide <path>
--json <path>
```

The JSON report contains the metrics listed under Diagnostics and Evidence.
The harness runs the same public planning entry point used by catalog
generation.

Full catalog validation continues to use:

```bash
scripts/generate-schedule-tables.sh --row-progress
```

The implementation may add a wrapper for repeat runs and peak memory capture.
It must not maintain a second fixture planner.

## Evaluation

### Acceptance criteria

#### Search semantics

- [ ] One normalized request has one audited domain and one search engine.
- [ ] Oracle and guided executions use the same materializer, objective,
      frontier, descriptor, and validator.
- [ ] Root contraction is not an admission rule or search mode.
- [ ] Every production domain restriction is documented as an exact
      equivalence, a certified pruning rule, or an explicitly approximate
      policy which cannot publish an exact catalog.
- [ ] `UnsupportedSchedule` is returned only after the complete audited domain
      has no feasible complete schedule.
- [ ] Traversal order, batch size, parallel order, and memo capacity do not
      change the selected descriptor.

#### Certified pruning

- [ ] Every pruning rule follows the rule contract in this specification.
- [ ] Unknown or overflowing bounds retain work.
- [ ] The exact Z, E, and T body identity matches
      `grouped_witness_body_coefficients` at every small split and chunk shape.
- [ ] Split cells break at every rank, digit, relation, compression, source,
      setup, and security route signature change.
- [ ] Recursive witness body contraction intervals and incumbent intervals are
      checked against full split enumeration.
- [ ] A fixed balance radius is used only for ordering and never defines an
      exact production split domain.
- [ ] Local layout bounds are not used to prune global frontiers without a
      complete consumer proof.
- [ ] The relaxed suffix value is no greater than the exact suffix optimum on
      every tractable state.
- [ ] Transition dominance checks admission, sufficient child state, every
      parent visible projection, and descriptor ties.
- [ ] Selective L2 is evaluated at every surviving route cell unless a complete
      route dominance certificate removes that cell.
- [ ] All feasible setup first slice choices remain until a complete transition
      dominance certificate removes one.
- [ ] Symmetry quotients have permutation invariance tests.
- [ ] Every optional pruning rule can be disabled independently for oracle
      comparison.

#### Guided execution

- [ ] A versioned guide artifact seeds an exact incumbent and checked
      exclusions.
- [ ] Invalid or stale guide entries are ignored without changing the semantic
      domain.
- [ ] Guided output equals oracle output on every tractable grid and named
      audit fixture.
- [ ] The named performance fixtures meet the time, memory, and relative
      speed requirements.
- [ ] The performance report includes exact domain, guide, and selected row
      identities.

#### Commitment workloads

- [ ] The public planner request represents ordered commitment epochs.
- [ ] Group semantic role, source contract, epoch, and grouped root role are
      separate.
- [ ] The batch opening policy is not inferred from the closing group source
      family.
- [ ] Earlier group profiles and the shared opening can be planned jointly.
- [ ] Group profile products are streamed rather than fully materialized.
- [ ] The Aerie eight group workload preserves its before seed and after seed
      transcript boundary.
- [ ] `JlOne` is represented as the closing group without being called the
      semantic main group.
- [ ] The selected Aerie plan is bound before the first commitment when its
      choices affect extraction or verifier behavior.

#### Ownership and output

- [ ] Decision enumerators, materialization, bounds, traversal, frontiers,
      selection, and emission have separate owners.
- [ ] Derived arithmetic uses canonical shared functions and
      `akita_error::checked`.
- [ ] Typed rejections replace error string matching.
- [ ] Compact plans contain independent decisions and validated stable
      identities.
- [ ] Every emitted plan expands and validates through `akita-schedules`.
- [ ] Runtime prover and verifier do not depend on `akita-planner`.

### Testing strategy

Small exhaustive grid tests must compare every guided result with the oracle.
The grid must vary split, basis, dimensions, slice count, opening method,
security route, setup offload choice, terminal choice, and group profile choice
where each coordinate is supported.

The L2 grid must include cases where the best Linf and L2 routes use different
splits. It must include every L2 table boundary and a missing or stale modeled
cap. The setup first grid must include every slice count and cases where local
setup order disagrees with complete schedule order.

Property tests must randomize enumeration order, reverse candidate order, vary
streaming batch size, and vary memo capacity. Every run must return the same
complete objective and descriptor.

Each pruning certificate must have a focused proof test and an integration
test which disables only that rule. Boundary tests must cover equality with the
incumbent because equal numeric bounds cannot prune the descriptor tie break.

Overflow tests must force every checked bound to return unknown. The candidate
must remain in the search and later materialization must return the canonical
typed arithmetic error if the exact candidate itself overflows.

Group workload tests must cover:

- one epoch and one group;
- one epoch with several groups;
- two epochs separated by a challenge;
- heterogeneous source contracts;
- a closing group narrower than a frozen group;
- exchangeable groups with a valid symmetry quotient;
- ordered groups which must not be quotiented;
- the two Aerie fixtures named in the performance contract.

Catalog tests must distinguish:

- same revision drift;
- intentional selection changes;
- workload key changes;
- guide changes which preserve the selected schedule;
- guide changes which expose a stale certificate;
- compact plan expansion changes;
- row identity changes.

The standard repository checks remain required. Planner implementation pull
requests must also run all Clippy feature graphs listed in `AGENTS.md` and the
full schedule drift job command from `.github/workflows/ci.yml`.

## Execution plan

### Slice 1: Domain and pruning census

Inventory every current candidate omission and pruning rule. Record its domain,
current rationale, selected schedule effect, and oracle switch. Classify it as:

```text
exact domain definition
proved equivalence
certified pruning
ordering heuristic
unproved semantic restriction
```

No unproved semantic restriction may remain in an exact production catalog.

### Slice 2: Canonical decisions and typed rejection

Introduce small decision types for root, recursive, terminal, security route,
and group commitment choices. Extract canonical materialization and typed
rejections without changing selected schedules.

### Slice 3: Shared oracle

Make every enumerator expose its complete audited domain. Add the oracle
configuration to the shared engine and establish exact small grid results.

### Slice 4: Exact split cells and local bounds

Implement the exact mandatory body identity, split cell builder, discrete
convex interval, and incumbent interval. Keep the cheap scan over every integer
split. Remove the fixed radius as a production domain while preserving the
balance estimate as traversal order.

### Slice 5: Relaxed suffix bounds and transition dominance

Add `relaxed_suffix_dp_v1` with exact current edge bytes and a proved minimum
terminal cost. Extend it one monotone term at a time. Add the shared transition
dominance checker and replace level based frontier retention with consumer
proved bounds.

### Slice 6: Selective L2 route frontier

Remove the dependency on the best modeled Linf split. Enumerate L2 at every
surviving split cell, build the full route transition signature, and apply
`l2_route_dominance_v1` only after all complete objective effects are covered.
Add exhaustive route by split tests.

### Slice 7: Setup first slice frontier

Replace `prune_locally_unprofitable_slices` with one canonical slice transition
frontier. Materialize the four cheap slice signatures, retain different child
states, and apply `setup_slice_dominance_v1` only when the shared dominance
theorem holds. Add exhaustive slice tests under both complete objectives.

### Slice 8: Guide artifacts

Add guide generation, validation, incumbent seeding, ordering hints, and
certificate replay. Produce guides for the named recursive fixtures and stock
catalog rows.

### Slice 9: Workload and group profile planning

Introduce `CommitmentWorkload` information, ordered epochs, and an explicit
batch opening policy. Integrate exact group profile candidates, symmetry, and
streamed root frontiers.

### Slice 10: Aerie integration

Express the Aerie eight group workloads through epochs and a closing group.
Bind the selected plan before the first commitment. Remove application code
which constructs seven precommits plus a final JL group as the planner facing
model.

### Slice 11: Runtime and catalog cutover

Emit one compact plan, validate it through `akita-schedules`, regenerate stock
catalogs, and remove superseded planner request and wrapper paths. Keep any
trusted catalog artifact migration separate from search semantics.

### Slice 12: Performance gate and documentation

Land the performance harness, named fixture records, guide evidence, and CI or
scheduled audit jobs. Update the Book configuration chapter and archive this
spec after the durable architecture is implemented and taught there.

## Intended code ownership

The implementation should converge on ownership close to this map.

| Responsibility | Intended owner |
|---|---|
| Workload and domain normalization | `akita-planner` request module |
| Root decisions | root candidate module |
| Recursive decisions | recursive candidate module |
| Terminal decisions | terminal candidate module |
| Group commitment profile decisions | group profile candidate module |
| Exact materialization | focused schedule parameter modules |
| Split cells and certified bounds | planner bound module |
| Security route frontier | recursive security route candidate module |
| Slice transition frontier | recursive slice candidate module |
| Relaxed suffix value | suffix bound module |
| Complete objectives | objective module |
| Parent observable frontier | suffix frontier module |
| Search queue and guide replay | search module |
| Diagnostics and census | diagnostics module |
| Compact plan emission | emitter module |
| Expansion and validation | `akita-schedules` |
| Runtime catalog resolution | `akita-schedules` and `akita-config` |

The current files may move incrementally. The final architecture should not
leave `planner.rs`, `recursive.rs`, `suffix_dp.rs`, or the emitter as a general
owner for unrelated decisions merely to avoid adding a focused module.

## Alternatives considered

### Fixed split window

A fixed radius around the analytic balance point is fast, but it gives no
complete schedule optimality result. It remains useful as candidate ordering
inside the guided engine. It is not an exact production domain.

### Exhaustive production search

Full enumeration is a clear oracle but is too slow for routine catalog work.
The production path instead visits a strong incumbent first and certifies the
remaining regions.

### Guided attempt followed by exhaustive fallback

Two passes duplicate traversal and obscure the failure contract. One engine
with a persistent queue handles valid, stale, or absent guidance without a
restart.

### Trust the stored winning schedule

A stored schedule proves feasibility but not optimality under a changed
domain, objective, or security table. The guide must be versioned, and every
exclusion must be rechecked.

### Local best candidate at every level

Local minimization loses schedules whose next witness or parent visible
geometry improves the suffix. It is allowed only when a dominance theorem
covers every consumer.

### Use the Linf split for selective L2

The two routes can cross at different splits because their security ranks and
proof costs are different functions. Sharing bounds which do not depend on the
route is safe. Using the winner from one route as the domain of the other is not
safe without a complete route dominance certificate.

### Keep one setup first slice by local setup

The four value slice domain is cheap to inspect. Removing three values before
successor sizing gives little proven benefit and can hide a better complete
schedule. The target planner keeps their transition frontier and uses the
shared dominance theorem.

### Full Cartesian group product

Materializing every group profile combination consumes too much time and
memory. Streaming, symmetry, and complete lower bounds preserve exactness with
bounded working storage.

### Main group and precommitted groups

This vocabulary confuses semantic importance with commitment order. The
compiled grouped root still has one new group and frozen prior groups, but the
planner request describes semantic groups, epochs, source contracts, and a
closing role separately.

### Arbitrary commitment dependency graph

No current integration needs the extra complexity. Ordered epochs express the
Fiat-Shamir transcript and Aerie JL dependency directly.

### Add more public heuristic modes

Extra mode enums make heuristic choices part of catalog identity without
proving better semantics. Empirical models belong in versioned guide ordering.
Only audited domain and objective changes become public policy.

## Risks

### Complete bounds may be too weak

A mathematically valid lower bound can still leave most of the domain. The
guide must combine strong incumbents, region bounds, symmetry, and streaming.
Performance acceptance is separate from correctness acceptance.

### A frontier may omit a hidden parent observation

This is the main exactness risk. Every frontier proof must begin from the code
which prices a child in its parent. Tests must compare against the oracle under
all parent forms, including setup offload and grouped roots.

### A split cell signature may miss a boundary

A missing rank, digit, relation, compression, source, or route field can apply
one formula outside its proved domain. The signature must be built from the
canonical materialization inputs. Boundary tests must compare every adjacent
split, not only the selected candidates.

### A relaxed suffix may be safe but ineffective

Merging too many states or charging zero for most future work preserves the
lower bound but leaves a large search. Diagnostics must report the incumbent
gap and candidates removed by each term. Performance work should strengthen
the weakest term without changing the real candidate domain.

### Guidance can become a second policy language

A guide which carries unchecked exclusions can silently replace the audited
domain. The guide format therefore names only checked certificate kinds. Any
other content is ordering data.

### Joint group planning can recreate a Cartesian explosion

Exact local profile frontiers, symmetry, and streamed combination are required
before broad workload support. The Aerie fixture is the acceptance test for
this risk.

### Closing group terminology may hide a real algebraic asymmetry

The new group does have distinct root commitment work. The design does not
pretend that all groups are internally identical. It names the asymmetry by its
actual role and prevents that role from owning unrelated semantic policy.

### Performance budgets can become machine noise

Every measurement records the exact environment and uses repeated fresh
process runs. Relative oracle comparisons are the cross machine gate until a
shared reference runner is selected.

### A guide may overfit checked catalog rows

Exact row guides are still useful for reproducible regeneration. Broader guides
must declare a region predicate and pass oracle audits on interior and boundary
samples before they may prune that region.

## Documentation

While this specification is proposed or active, it remains the normative
planner architecture record. Existing implemented specs continue to own setup
offloading and group source security details.

When implementation is complete:

1. Fold the stable planner model into `book/src/how/configuration.md`.
2. Update `book/src/how/architecture.md` with the workload, planner, schedules,
   and runtime ownership boundary.
3. Update the Aerie integration documentation to use commitment epochs and
   closing group terminology.
4. Mark this spec implemented, set its implementation PRs, and check every
   completed acceptance item.
5. Archive it only after the Book owns the durable content.

## References

- [`crates/akita-planner/README.md`](../crates/akita-planner/README.md)
- [`crates/akita-types/src/witness.rs`](../crates/akita-types/src/witness.rs)
- [`crates/akita-planner/src/schedule_params/objective.rs`](../crates/akita-planner/src/schedule_params/objective.rs)
- [`crates/akita-planner/src/schedule_params.rs`](../crates/akita-planner/src/schedule_params.rs)
- [`crates/akita-planner/src/schedule_params/candidate/recursive.rs`](../crates/akita-planner/src/schedule_params/candidate/recursive.rs)
- [`crates/akita-planner/src/schedule_params/suffix_dp/frontier.rs`](../crates/akita-planner/src/schedule_params/suffix_dp/frontier.rs)
- [`crates/akita-planner/src/schedule_params/candidate/recursive/split.rs`](../crates/akita-planner/src/schedule_params/candidate/recursive/split.rs)
- [`book/src/how/configuration.md`](../book/src/how/configuration.md)
- [`book/src/how/architecture.md`](../book/src/how/architecture.md)
- [`setup-offloading-planner.md`](setup-offloading-planner.md)
- [`heterogeneous-group-source-contracts.md`](heterogeneous-group-source-contracts.md)
- [`selective-l2-fold-security-sizing.md`](selective-l2-fold-security-sizing.md)
- [`archive/2026-Q3/modular-planner-and-precommit-roles.md`](archive/2026-Q3/modular-planner-and-precommit-roles.md)
- [PR #408, total root selection](https://github.com/LayerZero-Labs/akita/pull/408)
- [PR #409, exact precommit profile planning](https://github.com/LayerZero-Labs/akita/pull/409)
- [PR #412, grouped root scaling](https://github.com/LayerZero-Labs/akita/pull/412)
- [PR #416, parameter consolidation](https://github.com/LayerZero-Labs/akita/pull/416)
- [PR #428, trusted runtime catalogs](https://github.com/LayerZero-Labs/akita/pull/428)
- [Aerie Falcon version 1 specification at the reviewed commit](https://github.com/a16z/aerie/blob/33eeae94d49c4f9148a2b7f2cb7d9ad9087d76a1/specs/falcon-v1.md)
