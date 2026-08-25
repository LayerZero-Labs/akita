# Spec: Quotient-free tail ring relations by reduced evaluation

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-08-25 |
| Status        | proposed |
| PR            | |
| Supersedes    | |
| Superseded-by | |
| Book-chapter  | book/src/how/proving/akita-fold-realizations.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Akita currently turns every nonterminal physical ring relation into an
ordinary polynomial identity by adding a private quotient for division by
`X^d + 1`. Those quotient coefficients are digit-decomposed, range-checked,
committed in the successor witness, and folded again. This specification adds
a second ring-relation mode for a deliberately narrow tail suffix. The
new mode transposes public negacyclic multiplication through the existing
random `alpha` evaluation, checks the reduced ring relation directly over the
extension field, and omits every polynomial-modulus quotient row from the
successor witness.

The feature is **quotient-free tail ring relations**. Its protocol mechanism is
**reduced evaluation**: reduce each ring product modulo `X^d + 1`, then apply
the existing evaluation functional. “Functional” and “direct linear
functional” remain useful descriptions of the mathematics, but are too generic
for a protocol enum. The schedule field is therefore:

```rust
pub enum RingRelationMode {
    QuotientLift,
    ReducedEvaluation,
}
```

`QuotientLift` is the current protocol. `ReducedEvaluation` adds no proof element,
opening, or Fiat–Shamir challenge. It changes the public relation weights and
deletes relation-quotient digits from the committed witness.

The production feature is a one-way tail cutover, not a freely selectable bit
at every level:

```text
root L0       recursive L1       recursive L2 ... last committed fold    terminal
QuotientLift  QuotientLift       QuotientLift ... ReducedEvaluation suffix    clear/direct
                                      ^ planner-selected cutover
```

The cutover is eligible only at absolute fold level 2 or later. The selected
fold and every later committed fold MUST consume no incoming setup prefix and
MUST create no setup prefix for a successor. The terminal already verifies a
clear response without carrying a recursive relation quotient, so it does not
store this mode.

This scope has four consequences.

1. The root and level 1 never use reduced evaluation, even if their setup
   contribution is direct.
2. Reduced evaluation composes with evaluation trace, raw or compressed
   commitment payloads, Linf or selective L2 security, mixed role dimensions,
   witness chunking, and extension-opening reduction (EOR).
3. It does not compose with an incoming setup prefix or Stage 3 in this
   feature. The rank-two reduced-evaluation Stage-3 construction remains
   valid future work, but is outside this tail scope.
4. Subring coefficient packing remains confined to levels 0 and 1 by its
   existing policy. Reduced evaluation therefore needs no packing-specific
   implementation in this scope.

## Decision at a glance

| Question | Decision |
|---|---|
| Feature name | Quotient-free tail ring relations |
| Protocol mechanism | Reduced evaluation |
| Schedule enum | `RingRelationMode::{QuotientLift, ReducedEvaluation}` |
| Selection granularity | One mode per nonterminal fold |
| Search shape | At most one monotone cutover per complete schedule |
| Earliest cutover | Absolute fold level 2 |
| Setup-prefix eligibility | No incoming prefix at the cutover or later; no later offload edge |
| Opening method in the supported suffix | `OpeningMethod::EvaluationTrace` |
| Ordinary quotient rows | Omitted in `ReducedEvaluation` |
| Compression F/H quotient rows | Omitted in `ReducedEvaluation` |
| Packing consistency quotient | Out of scope because packing is ineligible at level 2+ |
| Proof fields and challenges | Unchanged |
| Direct setup contribution | One public setup scan with reduced-evaluation coefficient weights |
| Stage 3 | Forbidden after the cutover in this scope |
| Prover realization | Baseline dense extension-field relation-weight oracle |
| Verifier realization | Succinct recurrence plus the existing fused public setup scan |
| Planner objective | Existing complete-schedule objective; no hidden heuristic penalty |
| Compatibility | Breaking schedule, descriptor, catalog, and proof-shape cutover |

## Intent

### Goal

Add a descriptor-bound reduced-evaluation relation mode that lets the offline
planner remove all polynomial-modulus quotient digits from a setup-direct,
evaluation-trace suffix beginning at level 2 or later, while keeping one clean
verifier equation, one canonical witness layout, and a bounded exact planner
search.

### Invariants

#### Protocol and algebra

- `ReducedEvaluation` MUST enforce the same reduced relation in
  `F[X] / (X^d + 1)` as `QuotientLift`.
- Prover and verifier MUST derive reduced-evaluation weights from the same public
  multipliers, row order, role-native dimensions, witness addresses, `tau1`,
  and `alpha`.
- The implementation MUST NOT use `A(alpha) * alpha^j` as the reduced weight of
  witness coefficient `j`. That expression evaluates the unreduced ordinary
  product and is incorrect when multiplication wraps modulo `X^d + 1`.
- Every public multiplier and witness segment that affects the relation MUST be
  fixed before `alpha` is sampled.
- Reduced evaluation MUST reuse the existing ring-switch `alpha` and row
  batching point `tau1`. It MUST NOT add a challenge or a proof opening.
- The public right-hand side MUST be evaluated in exactly the same native row
  dimension and canonical row order as the corresponding reduced relation.
- Mixed A, B, D, and compression row dimensions MUST use their native
  cyclotomic modulus. No row may be silently widened to the A dimension.
- Compression mode MUST delete both the ordinary quotient tail and the
  compression F/H quotient rows. Keeping hidden compression quotients would
  make “reduced evaluation” an incomplete and misleading proof-size mode.

#### Tail eligibility

- `FoldSchedule` MUST reject `ReducedEvaluation` at absolute fold levels 0 and 1.
- A `ReducedEvaluation` fold MUST have
  `FoldParams::incoming_setup_prefix().is_none()`.
- Once a schedule enters `ReducedEvaluation`, every later nonterminal fold MUST
  remain in `ReducedEvaluation`.
- Once a schedule enters `ReducedEvaluation`, no later successor MAY carry an
  incoming setup prefix. Equivalently, the reduced-evaluation suffix contains no
  Stage-3 edge.
- The terminal fold MUST remain outside `RingRelationMode`. Its existing
  clear-response relation is not encoded as a fake reduced-evaluation fold.
- The feature MUST accept only `EvaluationTrace` in a
  reduced-evaluation fold. The current planner already makes coefficient packing
  unavailable after level 1; runtime validation MUST repeat the restriction.
- These checks MUST be schedule-validation rules, not planner conventions.
  Hand-built or malformed rows MUST be rejected before proving or verification.

#### Layout and proof shape

- `RelationRhsLayout` and `RelationRowFamily` MUST remain the semantic sources
  for physical relation row identities, native moduli, and row order.
- `WitnessLayout` MUST remain the sole source for the actual committed witness
  ranges. In `QuotientLift` it allocates the current R rows. In
  `ReducedEvaluation` it allocates no R row.
- `ReducedEvaluation` MUST create no zero-width placeholder quotient ranges, empty
  quotient objects, or dummy digits. An absent quotient is an absent witness
  segment.
- Stage 1 MUST range-check exactly the live digit witness. It MUST not retain a
  virtual range-image interval for omitted quotient digits.
- Proof sizing, successor witness length, source moments, response bounds,
  commitment geometry, and terminal input length MUST all derive from the same
  mode-aware `WitnessLayout`.
- A reduced-evaluation proof MUST contain no new serialized proof field. The mode
  comes from the trusted, transcript-bound effective schedule.

#### Verifier

- The verifier MUST evaluate reduced-evaluation public weights without
  materializing a witness-sized functional table.
- For an active public setup of `S` base-field coefficients and native role
  dimensions bounded by `d`, direct setup evaluation MUST use `O(S + d)`
  extension-field work and `O(d)` auxiliary extension-field storage, excluding
  existing checked setup-plan metadata.
- The verifier MUST scan each active public setup coefficient at most once per
  existing fused direct setup pass. A separate A, B, or D matrix rescan is not
  acceptable implementation.
- Verifier-reachable construction MUST validate dimensions, point lengths,
  row counts, and setup bounds before allocation or indexing.
- Malformed schedules, proofs, and setup views MUST return `AkitaError` or
  `SerializationError`. They MUST NOT panic.

#### Planner and generated artifacts

- The reduced-evaluation choice MUST be part of the planner’s audited decision
  domain.
  It MUST NOT be an environment-variable sizing override in production.
- A complete schedule MAY contain zero or one transition from `QuotientLift`
  to `ReducedEvaluation`. It MUST NOT switch back.
- The planner MUST NOT enumerate one independent quotient bit per level. For a
  fixed `m`-fold eligible suffix, the mode language has `m + 1` sequences, not
  `2^m` sequences.
- The suffix memo key MUST distinguish quotient-prefix and reduced-evaluation-suffix
  states. The state MUST be sufficient to reject later setup offloading without
  inspecting an already constructed complete schedule.
- The existing complete-schedule objective remains authoritative. The planner
  MUST NOT introduce an unreported empirical prover-cost penalty to delay the
  cutover.
- If a future objective prices prover or verifier work, that coordinate MUST be
  explicit in `PlannerPolicy`, catalog identity, diagnostics, and comparison
  evidence.
- Generated rows, catalog identity, canonical descriptors, effective schedule
  digests, reports, and drift checks MUST include the selected mode at every
  nonterminal level.
- Search-cache quotas MUST NOT be raised merely to hide a Cartesian mode
  explosion.

#### Transcript and security

- The effective schedule digest bound in `AkitaInstanceDescriptor::plan` MUST
  change when any fold’s ring-relation mode changes.
- The mode MUST be bound before the outgoing witness commitment and before
  `alpha`.
- Prover and verifier MUST preserve the existing ordering of outgoing-witness
  absorption, `alpha`, `tau0`, and `tau1`.
- If transcript grinding is present, its query immediately before `alpha` and
  its packed nonce pricing MUST see the mode-aware witness geometry. Reduced
  evaluation MUST NOT insert a challenge on either side of the grinding query.
- The soundness analysis MUST apply random evaluation to the reduced residual,
  whose degree is less than its native modulus dimension. It MUST NOT argue
  soundness from an unreduced product identity after removing the quotient.

### Non-goals

- Enabling reduced evaluation at the root or level 1.
- Selecting independent quotient modes for separate row families in one fold.
- Supporting reduced evaluation for `SubringCoefficientPacking` in this
  feature.
- Supporting a reduced-evaluation fold that consumes a setup prefix.
- Supporting setup offloading or rank-two Stage 3 after the cutover.
- Replacing the terminal clear-response protocol.
- Removing the algebraic concept of quotient lifting from Akita.
- Adding a new proof field, commitment, sumcheck, or Fiat–Shamir challenge.
- Matching the current factored prover’s time or memory in this feature.
- Introducing streamed, checkpointed, GPU, packed, or rank-two reduced-evaluation
  prover optimizations in this feature.
- Changing commitment-compression granularity. `payload_mode` remains one
  fold-level raw-or-compressed choice with its current monotone cutover policy.
- Changing the Linf/L2 security argument, challenge distribution, or norm-proof
  semantics.
- Changing coefficient-packing eligibility, EOR policy, role-native layouts,
  or setup-offload feasibility outside the restrictions above.
- Preserving old schedule descriptors, generated catalog rows, setup artifacts,
  or proof bytes.

## Terminology and ownership

### Preferred terms

| Term | Meaning |
|---|---|
| Quotient lifting | The current identity in the ordinary polynomial ring with a private `(X^d + 1)R(X)` term |
| Reduced evaluation | Reduce the product in `F[X]/(X^d+1)`, then apply the public evaluation functional by transposing the public multiplication map |
| Residue kernel | The coefficient weights `kappa_(A,alpha)(j)` for one public multiplier `A` |
| Terminal residue kernel | The `H_k(r,alpha)` weights used by the verifier to evaluate the MLE of a residue kernel |
| Quotient prefix | The initial nonterminal schedule segment in `QuotientLift` |
| Reduced-evaluation suffix | The final committed segment after the one-way cutover |
| Incoming setup prefix | `FoldParams::incoming_setup_prefix()`, the successor-owned group produced by the preceding Stage-3 edge |

The implementation SHOULD use these terms consistently. It SHOULD avoid a
bare enum variant named `Functional`, because that name does not say which
functional is applied or what protocol object disappears.

### Existing authorities that remain authoritative

| Concept | Existing authority |
|---|---|
| Semantic relation rows and native geometry | `RelationRhsLayout`, `RelationRowFamily`, `RelationRowGeometry` |
| Physical recursive witness ranges | `WitnessLayout` |
| Flat Stage-2 address split | `RelationAddressGeometry` |
| Public A/B/D setup contraction | `SetupContributionPlan` |
| Per-fold effective parameters | `CommittedGroupParams` |
| Absolute schedule positions and adjacency | `FoldSchedule` |
| Transcript preamble binding | `AkitaInstanceDescriptor::plan` and effective schedule digest |
| Planner complete-schedule objective | `PlannerPolicy`, suffix DP, and parent-observable frontiers |

The new mode extends these authorities. It MUST NOT create a second relation
layout, a verifier-only row order, or a planner-only witness-length formula.

## Mathematical design

### Current quotient-lifted relation

Let

\[
R_d = F[X]/(X^d+1).
\]

One public-linear physical row has the form

\[
\sum_c A_c(X)\circledast W_c(X)=Y(X)
\quad\text{in }R_d,
\]

where `A_c` and `Y` are public after transcript challenges are fixed, `W_c`
comes from the private recursive witness, and `circledast` is negacyclic
multiplication.

The current relation introduces a private polynomial `Q` and proves

\[
\sum_c A_c(X)W_c(X)-Y(X)=(X^d+1)Q(X)
\]

in `F[X]`. After sampling `alpha`, this becomes

\[
\sum_c A_c(\alpha)W_c(\alpha)-Y(\alpha)
-(\alpha^d+1)Q(\alpha)=0.
\]

This identity explains the current rank-one coefficient factor:

\[
A_c(\alpha)W_c(\alpha)
=\sum_j A_c(\alpha)\alpha^j w_{c,j}.
\]

The cost is that `Q` is private. Akita computes it, digit-decomposes it, adds
its digits to `WitnessLayout`, range-checks them in Stage 1, uses them in Stage
2, commits them in the next witness, and folds them at later levels.

### Reduced evaluation

For a public multiplier `A`, define

\[
\kappa_{A,\alpha}(j)
=\left(A(X)X^j\bmod(X^d+1)\right)(\alpha).
\]

Because reduction and evaluation are linear,

\[
(A\circledast W)(\alpha)
=\sum_{j=0}^{d-1}w_j\kappa_{A,\alpha}(j).
\]

The row can therefore be checked as

\[
\boxed{
\sum_c\sum_{j=0}^{d-1}
  \kappa_{A_c,\alpha}(j)w_{c,j}=Y(\alpha)
}
\]

without exposing `Q` as a witness.

Write

\[
A(X)=\sum_{k=0}^{d-1}a_kX^k.
\]

The exact signed wrap kernel is

\[
\kappa_{A,\alpha}(j)
=\sum_{k=0}^{d-1} a_k
\begin{cases}
\alpha^{k+j}, & k+j<d,\\
-\alpha^{k+j-d}, & k+j\ge d.
\end{cases}
\]

This formula is the reduced-evaluation reference oracle. It is quadratic if evaluated
literally for every `j`, but it is not the production algorithm.

### Linear-time residue-kernel recurrence

Let

\[
D_\alpha=\alpha^d+1.
\]

The residue kernel satisfies

\[
\kappa_{A,\alpha}(0)=A(\alpha)
\]

and, for `0 <= j < d-1`,

\[
\boxed{
\kappa_{A,\alpha}(j+1)
=\alpha\kappa_{A,\alpha}(j)
-D_\alpha a_{d-1-j}.
}
\]

The subtraction is exactly the coefficient that crosses the `X^d=-1`
boundary when the reduced polynomial is shifted by `X`. One kernel therefore
costs `O(d)` field operations and `O(d)` output storage, or `O(1)` state when
streamed once.

The production algebra module MUST implement this recurrence. The quadratic
quadratic formula MUST remain available under tests as an independent oracle.

### Where the private quotient went

For every public basis product,

\[
A(X)X^j
=\operatorname{red}_d(A(X)X^j)+(X^d+1)Q_{A,j}(X).
\]

The current private quotient is the linear combination

\[
Q(X)=\sum_jw_jQ_{A,j}(X).
\]

Reduced evaluation substitutes that linear function into the public coefficient
weights:

\[
\kappa_{A,\alpha}(j)
=A(\alpha)\alpha^j-D_\alpha Q_{A,j}(\alpha).
\]

The quotient contribution has not been assumed away. It has moved from
private witness coordinates to public verifier-computable weights.

### Batched rows

Let `lambda_rho = eq(tau1, rho)` be the existing row batching weight. For row
`rho`, native modulus `d_rho`, public multipliers `A_(rho,c)`, and public
right-hand side `Y_rho`, reduced evaluation checks

\[
\sum_\rho\lambda_\rho
\left(
  \sum_c\sum_{j=0}^{d_\rho-1}
  \kappa_{A_{\rho,c},\alpha}(j)w_{c,j}
  -Y_\rho(\alpha)
\right)=0.
\]

Canonical witness addresses may split one logical source ring into native B or
D subcolumns. The public multiplier for each stored coefficient MUST include
the existing subcolumn power, gadget digit, row weight, challenge, and setup
column semantics before the reduced evaluation transform is applied. The
reduced-evaluation mode changes the native coefficient functional; it does not change witness
address semantics.

### Verifier MLE of a residue kernel

Stage 2 ends at a multilinear point. Let `r` be the coefficient-coordinate
part of that point, and let

\[
e_r(j)=\operatorname{eq}(r,j).
\]

For one public multiplier `A`, the verifier needs

\[
\widetilde\kappa_{A,\alpha}(r)
=\sum_j e_r(j)\kappa_{A,\alpha}(j).
\]

Swap the public multiplier and witness-coordinate sums:

\[
\widetilde\kappa_{A,\alpha}(r)
=\sum_k a_k H_k(r,\alpha),
\]

where

\[
H_k(r,\alpha)
=\sum_j e_r(j)
(-1)^{\lfloor(k+j)/d\rfloor}
\alpha^{(k+j)\bmod d}.
\]

The complete `H` vector has the recurrence

\[
H_0(r,\alpha)
=\sum_j e_r(j)\alpha^j
=\prod_b\left((1-r_b)+r_b\alpha^{2^b}\right)
\]

and

\[
\boxed{
H_{k+1}(r,\alpha)
=\alpha H_k(r,\alpha)
-D_\alpha e_r(d-1-k).
}
\]

The verifier builds `e_r` and `H` once per distinct native role dimension,
then evaluates each public multiplier by one base-field-by-extension-field dot
product with `H`.

### Direct public setup scan

Let `S_s` be one active public setup coefficient. Existing setup tensors
derive a high-address scalar

\[
\theta_s(r_{\mathsf{lane}},\tau_1,\text{gadget},\text{group geometry}).
\]

The current lifted direct scan accumulates terms of the form

\[
S_s\,\theta_s\,\alpha^{k(s)}.
\]

The reduced-evaluation scan accumulates

\[
\boxed{S_s\,\theta_s\,H_{k(s)}(r,\alpha).}
\]

The outer setup traversal, group fusion, row weights, setup bounds, and role
projection geometry are unchanged. Only the native coefficient functional
changes from powers of `alpha` to `H` weights.

If `S` is the active setup coefficient count, both scans cost `O(S+d)` and one
base-by-extension multiply-accumulate per setup coefficient. Reduced evaluation
adds `O(d)` extension work to build the equality table and `H` recurrence.
Mixed A/B/D dimensions require one `H^(d_role)` per distinct native dimension,
which is the same dimension set for which the verifier currently prepares
native alpha powers.

The verifier MUST extend the existing fused `SetupContributionPlan` scan. It
MUST NOT add independent A, B, and D scans or materialize one residue kernel
per setup lane.

### Structured non-setup terms

The reduced-evaluation verifier does not need a dense functional table for the
remaining public multipliers.

- **Evaluation trace.** Its trace target and structured equality term remain
  field-linear and unchanged. Ring-reduced multipliers that act on its witness
  coordinates use the same `H` kernel as their native role. No new trace or EOR
  claim is introduced.
- **Sparse fold challenges.** Once `H` exists, a challenge of Hamming weight
  `h` evaluates as a signed dot product over its `h` public coefficients. The
  cost is `O(h)` per distinct challenge, after the shared `O(d)` preparation.
- **Gadget and native subcolumn axes.** These are public scalar factors and
  preserve their current tensor/address ownership. They do not create an
  additional residue kernel.
- **Equality-tensor multipliers.** Negacyclic addition has one carry bit across
  the common coefficient block. A later prover optimization may preserve these
  weights as two factored terms. The verifier does not need that optimization;
  it can use the `H` recurrence and the existing equality tensors.
- **Compression F/H relations.** Their maps and row coefficients are public.
  Reduced evaluation applies the same reduced transpose to those maps and omits their
  quotient rows. The existing negative-binary range term remains a separate
  pointwise relation and is not deleted.

### Soundness

For row `rho`, define the reduced residual

\[
Z_\rho(X)=
\sum_c A_{\rho,c}(X)\circledast W_c(X)-Y_\rho(X)
\in F[X]_{<d_\rho}.
\]

The reduced-evaluation scalar for that row is exactly `Z_rho(alpha)`. If the ring
relation is false, at least one reduced residual is nonzero. Random evaluation
of a nonzero polynomial of degree below `d_rho` has the usual
Schwartz–Zippel bound over the extension challenge field. Existing `tau1` row
batching then combines the native row claims exactly as in the current Stage-2
analysis.

The implementation does not divide by `D_alpha`. It therefore need not reject
an `alpha` satisfying `alpha^d + 1 = 0`; the residue recurrence and reduced
polynomial evaluation remain well-defined. Security accounting SHOULD state
the bound through the reduced residual rather than through invertibility of
the recurrence.

Removing the quotient does not relax witness binding. The outgoing witness,
public relation instance, effective schedule, and any transcript-grinding
nonce are bound before `alpha`. The prover cannot choose the witness after
learning the evaluation point.

## Supported feature matrix

The reduced-evaluation mode is an additional ring-relation axis. It is not a
replacement for the planner’s other choices.

| Existing axis | Reduced-evaluation suffix | Rule |
|---|---|---|
| `EvaluationTrace` | Supported | Required in this feature |
| `SubringCoefficientPacking` | Not reachable | Packing is restricted to L0/L1; reduced evaluation starts at L2 |
| Linf A security | Supported | Relation mode does not change the SIS security route |
| Selective L2 A security | Supported | The physical Z norm proof is unchanged; source moments omit R |
| Compressed payload | Supported | F/H digits remain; all ordinary and compression quotient digits disappear |
| Raw payload | Supported | Ordinary quotient digits disappear; no compression suffix exists |
| Compressed-to-raw cutover | Supported | Independent monotone phase; both phase orders are admissible when otherwise valid |
| Direct setup scan | Required | Uses the fused `H`-weighted setup scan |
| Incoming setup prefix | Forbidden | Cutover waits until the prefix is absent |
| Outgoing setup offload / Stage 3 | Forbidden after cutover | The reduced-evaluation suffix cannot create a later prefix |
| Extension-opening reduction | Supported | Evaluation-trace final claim and EOR transcript remain unchanged |
| Mixed role dimensions | Supported | Prepare one terminal residue kernel per distinct native dimension |
| Multi-chunk witness | Supported | `WitnessLayout` remains chunk-major; mode only removes the shared R ranges |
| Frozen root precommitments | Not present in the suffix | Current recursive folds reject precommitted groups independently |
| Clear terminal response | Existing behavior | Not represented by the new enum |

The implementation MUST test every supported row of this table. It
MUST test every forbidden row as an explicit typed rejection.

### Compression and relation cutovers are independent

`CommitmentPayloadPhase` currently permits a compressed prefix followed by a
raw suffix. `RingRelationMode` permits a quotient prefix followed by a
reduced-evaluation suffix. The planner state is the small product of two monotone
phases:

```text
                           ring-relation mode
                      QuotientPrefix   ReducedEvaluationSuffix
payload Compressed       supported          supported
phase   RawSuffix        supported          supported
```

The table is not four independent flags. Each axis changes at most once. A
schedule may therefore use reduced evaluation while its payload is still
compressed, or may first stop compression and switch ring-relation mode later.

Compression granularity remains fold-wide because `CommittedGroupParams` owns
one `payload_mode`. This feature MUST NOT introduce per-group B/D compression
choices.

### Selective L2 remains independent

Selective L2 starts at level 3 under its existing eligibility rules. A
reduced-evaluation suffix may therefore contain Linf folds, L2 folds, or both. The L2 norm
proof continues to cover the complete physical folded Z response. It does not
need an R coordinate because R is no longer a response source in
reduced-evaluation mode.

The response model’s R component is zero in reduced-evaluation mode. This change may alter
modeled caps, selected A ranks, and later schedule geometry. The planner MUST
derive those effects from the mode-aware witness layout and typed source model;
it MUST NOT subtract quotient bytes only at the final proof-size report.

## Tail eligibility and state machine

### Absolute level convention

This specification uses the repository’s existing absolute levels:

```text
L0 = FoldSchedule::root
L1 = FoldSchedule::recursive_folds[0]
L2 = FoldSchedule::recursive_folds[1]
...
T  = FoldSchedule::terminal
```

`ReducedEvaluation` is valid only for `L >= 2`. A schedule with no second
recursive fold simply has no eligible reduced-evaluation level.

### Setup-prefix direction

An incoming setup prefix belongs to the consuming successor. For a recursive
fold `Li`,

```text
Li.params.setup_prefix().is_some()
```

means `L(i-1)` ran Stage 3 and `Li` consumes the resulting precommitted setup
group. The reduced-evaluation eligibility check uses this exact successor-owned
field. It MUST NOT introduce a second “setup mode” bit.

The one-way suffix rule is stronger than checking only the selected fold. Once
the cutover occurs, candidate generation MUST suppress offloaded child edges,
so a setup prefix cannot reappear later.

### Planner phase

The suffix search adds one semantic phase:

```rust
enum RingRelationPhase {
    QuotientPrefix,
    ReducedEvaluationSuffix,
}
```

This phase is a planner state, not a protocol type. The protocol type remains
the per-fold `RingRelationMode` stored in `CommittedGroupParams`.

Transitions are:

```text
QuotientPrefix --QuotientLift--> QuotientPrefix

QuotientPrefix --ReducedEvaluation--> ReducedEvaluationSuffix
    only if level >= 2
    and incoming_setup_prefix is None
    and opening method is EvaluationTrace
    and a setup-direct child edge is selected

ReducedEvaluationSuffix --ReducedEvaluation--> ReducedEvaluationSuffix
    only with incoming_setup_prefix None
    only through setup-direct child edges
```

There is no `ReducedEvaluationSuffix -> QuotientPrefix` transition.

### Search-size argument

Suppose a fixed schedule skeleton has `m` consecutive eligible committed
folds. An independent bit per fold would create `2^m` relation-mode sequences.
The monotone language creates exactly:

```text
no cutover
cut over at eligible fold 0
cut over at eligible fold 1
...
cut over at eligible fold m - 1
```

or `m + 1` sequences.

The suffix DP SHOULD share completed reduced-evaluation suffixes through its memo,
rather than rebuilding a suffix once for every earlier quotient prefix. The
memo key MUST include `RingRelationPhase`. It MUST continue to include
the exact witness length, basis, source moment, incoming prefix, dimensions,
and payload phase that affect future pricing.

No dominance rule is required for this feature.
If a later optimization claims that the earliest eligible cutover always wins,
it MUST prove that claim against the complete objective, downstream ranks,
L2 route changes, setup capacity, and canonical descriptor. Empirical proof
size results are guidance, not a pruning proof.

## Architecture

### End-to-end data flow

```text
Planner decision
    |
    v
CommittedGroupParams.ring_relation_mode
    |----> canonical level/schedule descriptor
    |----> generated catalog row and identity
    |----> FoldSchedule eligibility validation
    |
    v
RelationWitnessGeometry + WitnessLayout
    |---- QuotientLift: Z | E | T | R | compression digits/quotients
    `---- ReducedEvaluation: Z | E | T | compression digits
    |
    +----> exact successor witness length / source moments / proof sizing
    +----> prover relation-weight compiler
    `----> verifier relation evaluator
              |
              +---- structured challenge and trace terms
              +---- fused direct setup scan with H weights
              `---- no quotient-tail evaluation
```

### Protocol type and descriptor ownership

`RingRelationMode` belongs in `akita-types`. `CommittedGroupParams` SHOULD
store it beside `payload_mode`, because both values describe how one complete
fold realizes its relation and outgoing witness. It MUST NOT live on an
individual `GroupOpenPhaseParams`: one Stage-2 relation batches every group
and owns one shared quotient policy.

Required type-layer changes include:

1. Add `RingRelationMode` with stable descriptor tags:
   `QuotientLift = 1` and `ReducedEvaluation = 2`. The implementation MUST use an
   explicit `tag()` match, not a Rust discriminant cast.
2. Add `ring_relation_mode` to `CommittedGroupParams::try_new` and every
   canonical builder/materializer.
3. Append exactly one relation-mode tag immediately after the existing
   payload-mode tag in
   `CommittedGroupParams::append_descriptor_bytes_with_payload_mode`, before
   `source_encoding`. Root and recursive schedule descriptors already invoke
   this canonical parameter encoding; they MUST NOT append a second copy.
4. Bump the effective schedule descriptor epoch because the previous byte
   language did not contain this field.
5. Include the mode in `GeneratedFoldCore` or another one-per-fold generated
   owner. Do not repeat it in `GeneratedGroup` and `GeneratedRecursiveFold`.
6. Include the tag in generated catalog identity and policy reports.
7. Keep the mode out of proof serialization. The verifier obtains it from the
   already authenticated schedule.

### Schedule validation

`FoldSchedule::validate_structure` is the canonical adjacency checker. It
SHOULD validate the complete nonterminal mode sequence in one pass while it
already checks payload phase and setup-prefix topology.

The pass keeps two booleans or typed phases:

```text
relation phase = QuotientPrefix
for each nonterminal level in absolute order:
    validate current mode against absolute level and incoming prefix
    if current mode is ReducedEvaluation:
        relation phase = ReducedEvaluationSuffix
        reject any future incoming prefix
    if relation phase is ReducedEvaluationSuffix:
        reject QuotientLift
```

The generated walker MUST exercise the same validator after expansion. The
planner MUST not have a private copy of this schedule rule.

### Mode-aware witness layout

`WitnessLayout` currently places the Z/E/T units, ordinary R rows, compression
digits, compression quotient rows, alignment, and zero suffix. The new mode
must alter that one construction:

```text
QuotientLift, raw:
    Z | E | T | ordinary R

ReducedEvaluation, raw:
    Z | E | T

QuotientLift, compressed:
    Z | E | T | ordinary R | F/H digits | F/H quotient rows

ReducedEvaluation, compressed:
    Z | E | T | F/H digits
```

The semantic `RelationRhsLayout::row_families()` remains complete in both
modes because the reduced-evaluation compiler and verifier still need every physical row.
Only `WitnessLayout::r_rows()` becomes empty in reduced-evaluation mode.

APIs that address an R coefficient SHOULD return a typed error when called on
a reduced-evaluation layout. They MUST NOT return zero, alias another range, or use an
unchecked optional value. The existing live-length, successor padding, and
Boolean-domain calculations then update automatically.

The normative native quotient-tail section of
[`role-native-projected-digit-layout.md`](role-native-projected-digit-layout.md)
continues to define `QuotientLift`. This specification is the additional
authority for the no-R `ReducedEvaluation` case until both designs are folded into
the Book.

### Shared algebra primitive

The residue recurrence belongs in `akita-algebra::ring`, where both prover and
verifier can test it without depending on schedule or proof types. The module
SHOULD expose one checked primitive for each actual concept, for example:

```rust
pub fn residue_kernel<F, E>(
    coefficients: &[F],
    alpha: E,
) -> Result<Vec<E>, AkitaError>;

pub fn terminal_residue_kernel<E>(
    coefficient_point: &[E],
    alpha: E,
) -> Result<Vec<E>, AkitaError>;
```

Exact names may change during implementation. The ownership requirements do
not:

- one checked recurrence implementation;
- one independent quadratic reference under tests;
- no prover and verifier copies of the recurrence;
- no `_for_level` wrappers;
- power-of-two dimension and point-arity validation before allocation;
- no division by `alpha^d + 1`.

`terminal_residue_kernel` SHOULD generate the equality weights and `H` in one
allocation when that is clearer and faster than retaining two vectors. Tests
must still expose both mathematical quantities to the oracle.

### Prover relation-weight representation

The current Stage-2 prover stores a rank-one
`RelationWeightFactorization<E>`:

```text
common alpha factor: d0 elements
relation lane weights: W / d0 elements
```

Generic reduced-evaluation weights are not rank one across lane and coefficient.
The implementation MUST not hide that fact behind an invalid
factorization.

The clean minimum is one canonical Stage-2 relation-weight state with two
realizations:

```rust
enum RelationWeightOracle<E> {
    QuotientFactored(RelationWeightFactorization<E>),
    ReducedDense(DenseRelationWeights<E>),
}
```

The exact type name is not normative. Its behavior is:

- `QuotientFactored` preserves the current optimized coefficient/lane path.
- `ReducedDense` contains the complete padded reduced-evaluation weight MLE over the
  existing Stage-2 witness domain and folds in place with sumcheck challenges.
- Dispatch occurs once per sumcheck round or construction phase, not once per
  witness coordinate.
- Both variants use the same witness state, range-image term, structured
  opening term, transcript, and terminal claim API.
- The dense reduced-evaluation table is ephemeral extension-field prover state. It is not
  committed, serialized, or counted as proof bytes.

This baseline path uses `O(W)` extension-field storage and `O(W)` folding work
for a Stage-2 domain of size `W`. The acceptance criteria prioritize verifier
cleanliness and proof size, while still requiring the prover cost to be
measured and reported. A future optimization may add streamed, checkpointed,
sparse-kernel, or rank-two prover variants behind the same semantic oracle
without changing this protocol mode.

The reduced-evaluation compiler MUST derive weights from semantic row families and
canonical witness ranges. It MUST NOT construct a second dense relation matrix
or replay setup rows in a new order.

### Prover construction path

In `QuotientLift`, `ring_switch_build_w` keeps the current order:

1. Prepare group Z/E/T values.
2. Compute ordinary and compression relation quotients.
3. Allocate the mode-aware witness.
4. Emit Z/E/T, quotient digits, and compression digits.

In `ReducedEvaluation`, it becomes:

1. Prepare the same group Z/E/T values and compression digits.
2. Construct the mode-aware `WitnessLayout` with no R rows.
3. Allocate and emit only the live non-quotient witness segments.
4. Skip `compute_multi_group_relation_quotient` and every R decomposition.
5. After `alpha` and `tau1`, compile the dense reduced-evaluation relation-weight oracle
   from public setup, public challenges, row weights, and canonical addresses.
6. Run the existing fused range-image/relation sumcheck through the reduced-evaluation
   oracle variant.

No empty `RelationQuotientOutput` SHOULD be constructed. The mode match should
occur before quotient computation.

### Compression path

Compression retains the F/H digit witness and its negative-binary restriction.
It removes only the polynomial-modulus quotient rows associated with the F/H
ring relations.

The current verifier keeps F/H relation weights in
`CompressionRelationWeights` and evaluates their quotient contribution through
the compression-specific path. Reduced-evaluation mode SHOULD reuse the same compression
map authority to compile reduced functional weights over the F/H digit ranges.
It MUST not reconstruct map coefficients from witness offsets.

The reduced-evaluation prover may merge these linear weights into `ReducedDense`. The
negative-binary pointwise term remains in `AdditionalRelationTerms`. This
separation keeps “ring reduction” distinct from “digit alphabet”.

### Verifier relation evaluator

`RelationMatrixEvaluator` remains the one verifier-side prepared object. It
SHOULD store the trusted `RingRelationMode` and dispatch once in
`evaluate_relation_at_point`:

```text
QuotientLift:
    prepare alpha-power coefficient functionals
    evaluate structured groups
    evaluate direct or deferred setup
    evaluate quotient tail
    multiply by common low alpha MLE factor

ReducedEvaluation:
    require deferred_setup_claim == None
    prepare H coefficient functionals
    evaluate structured groups with reduced evaluation weights
    evaluate fused direct setup with H
    do not evaluate a quotient tail
    return the already complete flat MLE
```

The reduced-evaluation branch MUST NOT call
`PreparedRelationPoint::common_alpha_evaluation`
as an outer factor. `H` already includes the coefficient equality
contraction. Multiplying by the old common factor would double-count it.

`PreparedRelationPoint` SHOULD be generalized around a checked native
coefficient functional rather than accreting optional alpha-power and H fields.
One possible internal shape is:

```rust
enum PreparedCoefficientFunctional<E> {
    LiftedPower { powers: Arc<[E]>, lane_powers: Arc<[E]> },
    ReducedEvaluation { terminal_kernel: Arc<[E]> },
}
```

The exact representation may differ. The important boundary is that
`SetupContributionPlan` and structured-term evaluators request the coefficient
weights they need without knowing planner policy or proof serialization.

### Fused setup scan

`SetupContributionPlan::evaluate_direct` currently receives native alpha-power
slices. It SHOULD evolve to consume a checked coefficient-functional view for
each role. The outer fused scan, setup bounds, segment scheduling, parallel job
partition, group fusion, and base-ring projection remain common.

The per-ring inner operation becomes:

```text
QuotientLift:  dot(setup_ring, [1, alpha, ..., alpha^(d-1)])
ReducedEvaluation: dot(setup_ring, H^(d)(r_coeff, alpha))
```

The common scanner MUST specialize the lifted power path where
`eval_ring_at_pows_fast` is faster. Sharing the scanner does not require
discarding its optimized inner product.

Because reduced evaluation is forbidden when setup is deferred, the
reduced-evaluation branch
does not cache a Stage-3 `SetupContributionPlan` and does not consume
`setup_prefix_eval`.

### Planner integration

Candidate materialization receives the selected per-level relation mode before
it computes:

- `WitnessLayout`;
- outgoing live and padded witness length;
- source moments and typed component counts;
- candidate A/B/D ranks that depend on the successor geometry;
- commitment payload bytes;
- EOR and Stage-2 domain sizes;
- terminal input length;
- complete suffix proof bytes.

The earlier diagnostic prototype that merely toggled quotient inclusion in a
scalar length formula is useful for proof-size intuition, but is not a valid
production architecture. Production planning MUST put the mode in the typed
candidate and replay the exact generated row through the same layout and proof
accounting as runtime.

The suffix DP SHOULD branch as follows:

1. In `QuotientPrefix`, materialize the existing quotient candidate.
2. If the current state is eligible, also materialize a reduced-evaluation
   candidate with the same independent geometry choices.
3. Recurse from the reduced-evaluation candidate with
   `RingRelationPhase::ReducedEvaluationSuffix` and no offloaded child.
4. In `ReducedEvaluationSuffix`, materialize only reduced-evaluation candidates and
   suppress setup-prefix search.
5. Retain candidates through the existing complete objective and
   parent-observable frontier.

The opening method, payload mode, security route, ring dimensions, split,
slice count, chunking, and relation mode remain independent decisions where
the feature matrix permits them. Candidate generation SHOULD enumerate the
relation mode outside low-level split loops when both modes share the same
geometry domain, so it does not duplicate expensive matrix derivation before
the witness length differs.

### Generated schedules and reports

Generated row types and emitted Rust MUST store the exact relation mode for
every root and recursive fold. For readability, the emitter MAY omit the
`QuotientLift` token only if the generated type supplies that default in an
unambiguous versioned schema. It MUST emit `ReducedEvaluation` explicitly.

`catalog_policy_report` SHOULD add `rel=quotient` or `rel=reduced-evaluation` to
each nonterminal level. It SHOULD report:

- selected cutover level or `none`;
- ordinary quotient coefficient count removed per level;
- compression quotient coefficient count removed per level;
- input and output witness lengths;
- payload mode;
- opening method;
- Linf or L2 route;
- incoming setup-prefix presence;
- setup-direct and Stage-3 proof bytes.

Dense fp32, fp64, and fp128 evidence MUST compare the generated baseline and
head schedules row by row. An aggregate proof-size delta without per-level
witness geometry is insufficient.

## Transcript and serialization contract

### Effective schedule binding

The proof does not serialize `RingRelationMode`. The mode is public
configuration selected by the trusted schedule. `PlanSection` binds the digest
of `FoldSchedule::append_descriptor_bytes`, which in turn includes each
`CommittedGroupParams` descriptor. Adding the mode there binds it into the
transcript preamble before any commitment or challenge whose meaning depends
on it.

Changing only the mode MUST change:

- the level canonical descriptor;
- the complete schedule descriptor;
- the effective schedule digest;
- generated catalog identity and row digest;
- the derived witness length and proof shape when quotient rows are nonempty.

It MUST NOT change a commitment to an already frozen source group whose
commitment profile is independent of the consuming fold’s relation mode.

### Challenge order

The order remains:

```text
bind public instance and effective schedule
bind current commitments and outgoing witness commitment
[grind at the scheduled ring-switch-alpha site, if enabled]
sample alpha
sample tau0
sample tau1
run Stage 1 and Stage 2 transcript events in their existing order
```

Reduced evaluation uses `alpha` only after the outgoing witness is fixed. The
dense prover table is derived after `alpha`; it is not another witness that
needs commitment.

No new transcript query label or domain separator is introduced. The existing
ring-switch `alpha`, `tau0`, and `tau1` query labels remain unchanged; mode
separation comes from the effective schedule digest already absorbed in the
instance preamble. Prover and verifier MUST use the same bumped descriptor
epoch and canonical parameter bytes before reaching those existing labels.

### Compatibility

This is a breaking protocol and artifact change. The implementation MUST
regenerate every affected schedule table and any trusted schedule artifact.
It MUST reject old rows under the new catalog identity. It MUST NOT add a
legacy decoder, descriptor fallback, or implicit default based on a missing
wire field.

The proof serialization schema need not gain a field, but exact proof bytes
change because recursive witness and commitment shapes change.

## Performance model

### Verifier

Let:

- `S` be the active direct setup coefficient count;
- `D` be the sum of distinct native role dimensions prepared at the terminal
  Stage-2 point;
- `C` be the number of non-setup public sparse coefficients actually queried.

Then the reduced-evaluation verifier target is:

\[
O(S+D+C)
\]

extension-field work and `O(D)` auxiliary extension-field storage. Current
quotient lifting is `O(S+D+C)` as well: it prepares alpha powers, scans the
same setup, evaluates structured terms, and reads quotient-tail MLE weights.
Reduced evaluation replaces quotient-tail work with `H` preparation and changes
the coefficient multiplier used during the scan.

Concrete benchmarking MUST separate:

- coefficient-functional preparation;
- structured group evaluation;
- direct setup scan;
- quotient-tail evaluation, which is zero in reduced-evaluation mode;
- complete Stage-2 verifier time;
- total verifier time.

The primary verifier acceptance condition is asymptotic and architectural: no
witness-sized functional table and no extra setup scan. Concrete regressions
must be reported before expanding eligibility earlier than this tail scope.

### Prover

For Stage-2 witness domain `W`, the reduced-evaluation implementation may use:

| Mode | Relation-weight work | Extra extension state |
|---|---:|---:|
| Quotient lift | `O(W)` factored Stage 2 plus quotient construction/decomposition | `O(d0 + W/d0)` plus quotient witness |
| Reduced evaluation, dense | `O(W)` dense-table generation and folding | `O(W)` ephemeral |

This specification accepts the dense prover cost. It does not accept
accidentally computing both the quotient and reduced-evaluation table. Whole-fold
benchmarks MUST show quotient construction and quotient digit emission at zero
in reduced-evaluation mode.

Future optimizations may explore:

- streaming and recomputation;
- checkpointing after one or more coefficient rounds;
- one kernel per distinct sparse challenge;
- two carry-state factors for equality tensors;
- a rank-two Stage-3 setup product.

Those are alternatives behind the same reduced-evaluation semantics. They MUST
not add planner-visible proof modes unless they change proof bytes or verifier
behavior.

### Proof size

Reduced evaluation adds zero proof fields. Its structural saving at one fold is
the effect of removing the quotient digits from that fold’s outgoing witness:

```text
ordinary removed coefficients
    = quotient_depth * sum(native ordinary row dimensions)

compressed-only removed coefficients
    = quotient_depth * sum(native F/H quotient row dimensions)
```

The downstream byte saving is not just those coefficients times one byte. A
smaller witness can change:

- successor Boolean capacity and zero suffix;
- A/B/D ranks;
- digit depths;
- compression payloads;
- Linf/L2 route choice and norm-proof shape;
- fold count and terminal response;
- grinding nonce pricing, if enabled.

Therefore only the complete generated schedule estimate and serialized proof
benchmark count as proof-size evidence.

### Planner complexity

The implementation MUST record, for baseline and head:

- raw relation-mode transitions considered;
- reduced-evaluation transitions rejected by each eligibility rule;
- suffix calls and memo hits by relation phase;
- peak memo entries under the existing direct/prefixed quotas;
- frontier candidate counts;
- wall time and peak resident memory for dense fp32, fp64, and fp128 generation;
- the selected schedule descriptor and proof bytes.

The implementation MUST NOT increase `MAX_SUFFIX_SEARCH_CACHE_ENTRIES` as part
of enabling the feature. If generation no longer fits the existing bound, the
implementation must improve state sharing or candidate traversal before the
feature is accepted.

## Evaluation

### Acceptance criteria

#### Algebra and soundness

- [ ] The linear residue recurrence matches literal negacyclic reduction for
      random powers-of-two dimensions, public multipliers, witnesses, and
      `alpha` values.
- [ ] The terminal `H` recurrence matches an independently materialized
      residue-kernel MLE at random coefficient points.
- [ ] The direct setup scan matches a dense public-matrix oracle for A, B, D,
      mixed dimensions, chunks, and compressed F/H rows.
- [ ] Quotient-lift and reduced-evaluation modes produce the same scalar relation
      claim on identical valid witnesses.
- [ ] A nonzero reduced residual is rejected in reduced-evaluation mode without a quotient
      witness.
- [ ] Tests cover `alpha^d + 1 == 0` in a field where such a test point is
      constructible, or explain why the test field lacks one. No division is
      used.

#### Eligibility and schedule binding

- [ ] Reduced evaluation is rejected at root L0.
- [ ] Reduced evaluation is rejected at recursive L1.
- [ ] Reduced evaluation is accepted at L2 or later when the complete suffix has
      no setup prefix and uses evaluation trace.
- [ ] A reduced-evaluation fold with an incoming setup prefix is rejected.
- [ ] A schedule that returns from reduced evaluation to quotient lifting is
      rejected.
- [ ] A schedule that adds an offloaded setup edge after the cutover is
      rejected.
- [ ] A reduced-evaluation coefficient-packing fold is rejected.
- [ ] Changing only the relation mode changes the effective schedule digest
      and transcript preamble.
- [ ] Prover and verifier reject a proof replayed under the other mode’s
      schedule.

#### Witness and proof shape

- [ ] A raw reduced-evaluation layout contains exactly Z/E/T and no R range.
- [ ] A compressed reduced-evaluation layout contains Z/E/T and F/H digits, but no
      ordinary or compression R range.
- [ ] Stage 1 domain, Stage 2 domain, outgoing commitment length, response
      model, and proof estimate all equal values derived from the same
      mode-aware `WitnessLayout`.
- [ ] Reduced-evaluation mode never calls quotient construction or quotient digit
      decomposition; an operation counter or focused mock test proves this.
- [ ] Reduced-evaluation mode adds no serialized proof field or sumcheck round.

#### Feature combinations

- [ ] Raw/Linf/evaluation-trace reduced-evaluation suffix.
- [ ] Compressed/Linf/evaluation-trace reduced-evaluation suffix with F/H relations.
- [ ] Raw/L2/evaluation-trace reduced-evaluation suffix.
- [ ] Compressed/L2/evaluation-trace reduced-evaluation suffix.
- [ ] Small-field evaluation trace with EOR.
- [ ] Mixed A/B/D dimensions.
- [ ] A multi-chunk eligible fold, if any production or focused fixture reaches
      level 2 with more than one chunk; otherwise a constructed type-level
      fixture covers it.
- [ ] Each forbidden matrix cell has a negative schedule-validation test.

#### Planner

- [ ] The planner may choose no cutover, an L2 cutover, or a later cutover from
      the same exact search engine.
- [ ] Every complete candidate contains at most one quotient-to-reduced-evaluation
      transition.
- [ ] A small exhaustive oracle enumerates all `m + 1` monotone cutovers and
      matches the suffix DP’s selected complete descriptor.
- [ ] Reversing relation-mode traversal order does not change the selected
      descriptor.
- [ ] Reduced-evaluation suffix states cannot invoke setup-prefix candidate search.
- [ ] Existing suffix-cache quotas remain unchanged.
- [ ] Generated row replay recomputes the exact reduced-evaluation witness lengths and
      proof estimate.
- [ ] Catalog identity and policy reports include relation mode and cutover.

#### Verifier performance and safety

- [ ] The reduced-evaluation verifier allocates `O(d)` coefficient-functional state, not
      `O(W)` witness-sized state.
- [ ] The reduced-evaluation verifier uses the existing fused setup traversal and scans
      each active setup coefficient once.
- [ ] Benchmarks separately report preparation, setup scan, Stage 2, and total
      verifier time for quotient-lift and reduced-evaluation modes on selected tail folds.
- [ ] Fuzz or property tests reject malformed mode, dimension, row, point, and
      setup combinations without panic or unbounded allocation.

#### Generated proof-size evidence

- [ ] Regenerate every affected schedule table with
      `scripts/generate-schedule-tables.sh`.
- [ ] Produce checked baseline/head evidence for dense fp32, fp64, and fp128
      generated families.
- [ ] For every changed row, report cutover, per-level witness length,
      quotient coefficients removed, payload mode, security route, setup-prefix
      presence, setup-direct bytes, Stage-3 bytes, and total proof bytes.
- [ ] Serialize representative proofs and confirm the measured byte delta
      agrees with the generated proof estimate.
- [ ] Report planner wall time, peak resident memory, and search counters for
      the same dense families.

### Testing strategy

#### Algebra tests

Add independent reference tests under `akita-algebra`:

1. Construct random `A` and `W` in `R_d`.
2. Compute `A circledast W` by the existing cyclotomic ring implementation.
3. Evaluate the result at `alpha`.
4. Generate `kappa` by recurrence and compute `sum_j kappa_j w_j`.
5. Compare both values.
6. Compare `H` recurrence against literal `sum_j eq(r,j) Phi(k,j)`.

Use every dispatch dimension exercised by fp32, fp64, and fp128 schedules. Add
small exhaustive dimensions when they make wrap and carry failures easier to
localize.

#### Shared-layout tests

Extend `akita-types` relation and witness tests to build the same geometry in
both modes. Assert exact segment ranges and live length. Cover raw and
compressed payloads, mixed row dimensions, alignment boundaries, and no-R
access errors.

#### Prover/verifier equivalence tests

Focused fixtures SHOULD expose both modes under the same public relation even
when production eligibility would reject the early level. Algebra equivalence
belongs in the relation engine; schedule eligibility belongs in separate
tests. End-to-end PCS tests MUST use only eligible L2+ tail schedules.

Tamper tests should modify:

- one Z/E/T digit;
- one retained compression digit;
- one public setup coefficient view;
- one sparse challenge coefficient;
- the row order or `tau1` point;
- the trusted mode or schedule digest.

Each tamper must reject under the verifier without relying on an absent
quotient range.

#### Planner tests

Add a small unpruned relation-cutover oracle beside the existing suffix-search
oracles. It should enumerate the monotone cutover index explicitly, construct
all feasible schedules through canonical materializers, and compare the exact
complete objective and descriptor with production suffix DP.

Property tests should vary:

- number of eligible tail folds;
- setup-prefix disappearance point;
- compressed-to-raw cutover;
- Linf/L2 route availability;
- role dimensions and bases;
- EOR availability;
- traversal order and memo capacity.

#### Repository gates

The implementation must run the CI preflight and feature-graph commands from
`AGENTS.md`. Protocol, Book, or spec edits additionally run:

```bash
./scripts/check-doc-guardrails.sh
scripts/check-spec-references.sh --all
```

Generated-table drift and profile-specific workflows are required when their
source files change.

### Performance evidence format

The implementation PR SHOULD add a checked TSV or Markdown report under
`specs/evidence/quotient-free-tail/`. Every record must include the exact base
and head SHA. At minimum, one table should contain:

```text
family
num_vars
base_proof_bytes
head_proof_bytes
delta_bytes
cutover_level
fold_level
relation_mode
input_witness_len
output_witness_len
ordinary_quotient_coefficients_removed
compression_quotient_coefficients_removed
payload_mode
opening_method
security_route
incoming_setup_prefix
setup_direct_payload_bytes
stage3_payload_bytes
```

A separate planner table should record wall time, peak RSS, suffix calls, memo
hits, peak memo entries, and frontier candidates. Do not combine compilation
time with planner execution.

## Alternatives considered

### Keep quotient lifting everywhere

This preserves the current factored prover and mature implementation. It also
keeps paying quotient construction, decomposition, range checks, commitments,
and later folds when quotient rows dominate the remaining tail witness. It is
the retained baseline, not the selected new feature.

### Enable reduced evaluation at every fold

The verifier can perform reduced evaluation efficiently even for dense setup, but
the generic prover loses the current compact rank-one relation table. Early
folds also contain more lanes and challenges while quotient rows are a smaller
fraction of the recursive witness. This feature therefore starts at L2
and only in a setup-direct tail. Earlier activation requires new benchmarks and
a scope revision.

### Independent per-level mode bits

This permits `QuotientLift`/`ReducedEvaluation`/`QuotientLift` oscillation and creates `2^m` mode
sequences over an `m`-fold tail. No protocol advantage requires switching back
after quotient rows have been removed. The monotone cutover is easier to
validate, plan, report, and optimize.

### A fixed cutover after two or three folds

A fixed threshold is useful for diagnostics and produced the initial
proof-size preview. It is not a durable planner policy: witness geometry,
payload compression, L2 availability, and fold count vary by family and row.
The production planner searches the one cutover under its complete objective,
while eligibility fixes only the lower bound `level >= 2`.

### Treat reduced evaluation as a Boolean sizing toggle only

Subtracting R widths from a planner estimate does not update source moments,
downstream ranks, commitment geometry, terminal response, generated replay, or
verifier semantics. The mode must be a typed schedule and layout input.

### Retain compression quotient rows

This would reduce implementation work but makes reduced evaluation depend on
payload mode and leaves a material quotient tail precisely when commitment
compression is selected. Compression relations are public-linear ring
relations too. The implementation therefore covers them.

### Add packing-specific reduced evaluation now

The mathematics applies to the packing consistency relation over its smaller
modulus and coordinate planes. Current schedules use coefficient packing only
at L0 and L1, while reduced evaluation is forbidden there. Implementing the
combination would add untestable production surface and weaken the desired
scope boundary. It is deferred.

### Force setup offloading and use rank-two Stage 3

Keeping the setup coefficient as an independent Stage-3 axis admits an exact
two-carry-state factorization and avoids dense prover setup kernels. It is a
valuable future optimization when Stage 3 is already selected. Forcing a new
setup prefix in the tail adds a subproof and successor group solely to avoid
prover state, outside this feature’s proof-size and verifier-first scope.

### Stream or checkpoint the dense reduced-evaluation prover table

Streaming can reduce extension-field memory but may rescan public sources in
each coefficient round. Checkpointing gives intermediate time-memory points.
Neither changes proof bytes or verifier behavior. This implementation
uses one dense oracle behind an abstraction that can admit these optimizations
later.

### Add a prover-cost coordinate to the planner now

The current complete objective does not price prover time. Adding an
uncalibrated penalty would make catalog selection harder to audit and could
hide proof-size improvements. The implementation measures prover cost and
keeps the initial eligibility conservative. A future multi-objective policy
may add an explicit versioned coordinate with measured evidence.

## Open implementation risks

### Compression map transpose

The ordinary A/B/D setup path already exposes public ring coefficients in
canonical geometry. Compression F/H uses a separate compact map authority.
The implementation must prove that its reduced transpose uses the exact same
map, row order, native modulus, and witness digit addresses. This is the most
likely place for a correct ordinary path and an incorrect compressed path to
diverge.

Mitigation: land scalar reference oracles and compressed equivalence tests
before optimizing the dense reduced-evaluation table compiler.

### Common-factor assumptions in Stage 2

`RelationRangeImageProver`, `PreparedRelationPoint`, and verifier terminal
evaluation currently assume a common low alpha factor. Reduced evaluation breaks
that rank-one assumption. Leaving one outer multiplication or one lane-power
projection in the reduced-evaluation branch can produce a plausible but incorrect
relation.

Mitigation: make the relation-weight representation and prepared coefficient
functional typed enums, and compare full materialized tables in debug tests.

### Setup scanner duplication

A naïve reduced-evaluation implementation can accidentally build `H`, materialize one
kernel per setup lane, or rescan A/B/D separately. Any of these defeats the
verifier design even though the proof remains correct.

Mitigation: extend `SetupContributionPlan` first and benchmark its existing
fused traversal with power and H functionals before integrating the full
protocol.

### Planner state widening

Removing quotient rows changes witness length, response moments, and later
candidate geometry. The relation phase itself is only monotone, but those new
lengths can expose suffix states that did not exist in the baseline.

Mitigation: preserve cache quotas, add phase-specific diagnostics, compare an
unpruned small oracle, and publish dense-family wall/RSS evidence.

### Concurrent protocol PRs

Several open PRs touch the exact files this feature will need. The execution
plan below separates the stable algebra/verifier work from volatile transcript,
planner, and prover integration. All required slices remain commits on this
single feature branch; the separation is an implementation and review order,
not a proposal for stacked feature PRs.

## Execution plan

### Slice 0: approve the protocol and scope

- Review this spec with `specs/SPEC_REVIEW.md`.
- Resolve whether implementation begins after transcript grinding or on a
  stack whose base is its exact reviewed head.
- Confirm that the reduced-evaluation mode removes compression quotients and
  forbids all setup-prefix edges after cutover.
- Record baseline dense fp32/fp64/fp128 catalog and planner diagnostics.

Exit condition: no ambiguity remains about mode semantics, eligibility,
transcript order, or proof-size evidence.

### Slice 1: shared type, descriptor, and layout authority

- Add `RingRelationMode` and the one-per-fold field.
- Bind it into descriptors, instance schedule digests, generated types, and
  catalog identity.
- Add complete `FoldSchedule` eligibility validation.
- Make `WitnessLayout` mode-aware and remove R ranges in reduced-evaluation mode.
- Route proof sizing and source moments through that layout.

This slice may use a test-only reduced-evaluation schedule but need not prove it yet.

Exit condition: typed schedule and layout tests can distinguish both modes,
all malformed sequences reject, and no planner-only quotient toggle remains.

### Slice 2: algebra oracle and fused verifier setup scan

- Add the residue-kernel and terminal-kernel recurrences plus independent
  references.
- Generalize prepared native coefficient functionals.
- Extend `SetupContributionPlan` to evaluate power or H coefficient weights
  through the same fused scan.
- Add reduced-evaluation structured challenge and compression-map terminal evaluation.

Exit condition: verifier-focused dense oracles pass for raw/compressed and
mixed-dimension fixtures, with one setup scan and `O(d)` auxiliary state.

### Slice 3: verifier protocol integration

- Add the exhaustive mode dispatch to `RelationMatrixEvaluator`.
- Reject deferred setup claims in reduced-evaluation mode.
- Remove quotient-tail evaluation and the common-alpha outer factor in the
  reduced-evaluation branch.
- Add transcript-order, schedule-digest, tamper, and no-panic tests.

Exit condition: the verifier accepts reduced-evaluation scalar fixtures and rejects every
cross-mode or malformed replay without any proof-format addition.

### Slice 4: baseline dense prover

- Skip quotient construction and emission in reduced-evaluation mode.
- Introduce the canonical factored-or-dense relation-weight oracle.
- Compile all ordinary and compression reduced weights into the dense variant.
- Integrate it with the existing fused range-image/relation sumcheck.
- Preserve evaluation-trace/EOR structured terms and negative-binary terms.

Exit condition: quotient-lift and reduced-evaluation end-to-end proofs agree on valid relations;
reduced-evaluation mode executes zero quotient work and supports the declared feature
matrix.

### Slice 5: exact planner cutover

- Add `RingRelationPhase` to suffix state and memo keys.
- Enumerate the one-way cutover at eligible states.
- Suppress setup-prefix search in the reduced-evaluation suffix.
- Price exact mode-aware witness shapes, source moments, proof bytes, and any
  grinding nonce stream.
- Add the small exhaustive cutover oracle and phase diagnostics.

Exit condition: traversal order does not change selection, cache quotas remain
unchanged, and generated replay matches planner estimates.

### Slice 6: generated catalogs and evidence

- Regenerate all affected catalogs.
- Produce dense fp32/fp64/fp128 baseline/head proof-size evidence.
- Serialize representative proofs and benchmark verifier phases.
- Record planner wall time, peak RSS, and search counters.
- Update Book chapters only after behavior and evidence are stable.

Exit condition: checked evidence supports the proof-size change, verifier
architecture, and bounded search claims.

### Slice 7: optional prover optimization

This slice is not required for acceptance of this feature branch. Profile the dense oracle
before choosing among sparse kernel banks, one-round checkpointing, full
streaming, or rank-two Stage 3. Any optimization must preserve the shared
algebra oracle and verifier equation.

## Pull-request landscape and stacking plan

This section records the open Akita branches inspected on 2026-08-25. SHAs are
included so the recommendation does not silently apply to later rewrites.

### Transcript grinding PR 417

PR [#417](https://github.com/LayerZero-Labs/akita/pull/417), head
`aa4efc3074d652abfab166e7bada3fc5a3fed397`, is open and review-required. It
changes transcript query sites around `alpha`, `tau0`, and `tau1`; proof-level
nonce serialization; planner cost composition; suffix-DP state and frontiers;
schedule estimates; generated catalogs; and both ring-switch implementations.

This feature has no mathematical dependency on grinding, but its integration
has a strong code and transcript dependency. The implementation SHOULD wait
for PR 417 to land, or explicitly stack on that exact reviewed head. Building
the full feature independently on current main would create avoidable conflict
in the verifier alpha site and would price the wrong planner cost type if PR
417 lands later.

This single feature branch remains based on current `main`. It will carry the
specification and the complete implementation. Algebra Slice 2 may begin on
the current base because it avoids transcript-facing types. Before Slices 3–6,
the same branch SHOULD be rebased onto a `main` containing PR 417, or explicitly
stacked on that exact reviewed head if implementation cannot wait. Do not open
a replacement implementation branch.

### Suffix EOR and packed prover stack

PR [#398](https://github.com/LayerZero-Labs/akita/pull/398), head
`dd0a9fbdb6dfecd2b363a7ad82ffbc8a65366a2b`, rewrites suffix EOR and touches
evaluation-trace prover tables. PR
[#437](https://github.com/LayerZero-Labs/akita/pull/437), head
`f72ae5683e2af3051d5b805b21cf61c27a832ae6`, stacks packed recursive witness
storage on #398 and changes `ring_switch`, `ring_switch/coeffs`,
`ring_relation_witness`, Stage 1, Stage 2, and witness emission. PR
[#439](https://github.com/LayerZero-Labs/akita/pull/439), head
`fb4fa643b22953f90085d919e31c006105b5cf51`, adds packed dense prover storage.

These PRs do not change the reduced-evaluation verifier equation. They strongly
intersect the baseline dense prover slice. The prover implementation SHOULD
be rebased after the accepted packed-witness cutover rather than teaching a new
reduced-evaluation path to storage that is about to be replaced. The reduced-evaluation oracle
may remain unpacked extension-field scratch; only the compact recursive witness
must follow the final packed ownership model.

### Trusted schedule artifacts PR 428

PR [#428](https://github.com/LayerZero-Labs/akita/pull/428), head
`d6499748e121851b1fcc5967256dff3403f59d0e`, is open, behind main, and
review-required. It changes runtime schedule authority, trusted artifacts,
schedule resolution, descriptors, generated catalogs, witness types, and
verifier setup boundaries.

The reduced-evaluation mode must be authenticated by whichever schedule authority
lands. It has no need to stack on the current behind head. If #428 lands first,
Slice 1 must add the mode to its trusted artifact schema and validation. If it
does not, current effective-schedule digest and generated-row authority remain
the integration target.

### Certified planner spec PR 434

PR [#434](https://github.com/LayerZero-Labs/akita/pull/434), head
`d7261d9167667ce71a38152a2f5d7d3867cdb621`, is a draft documentation PR. It
does not supply implementation code, but its audited-decision-domain rule is
directly relevant. This spec follows that rule: the cutover is an exact
decision, traversal order is guidance, and no candidate is removed without a
complete dominance proof.

There is no code-stack dependency. If #434 is approved, the implementation PR
should cite its planner architecture and add relation phase to the documented
state sufficiency and oracle tests.

### Grouped planner PRs 409 and 412

PR [#409](https://github.com/LayerZero-Labs/akita/pull/409), head
`5bef0c1a54d7ac7c8718e4a7aca803f64f83f24b`, plans exact precommit profiles.
PR [#412](https://github.com/LayerZero-Labs/akita/pull/412), head
`733efbea094e02756b88aa8662f37323198e9f9f`, changes grouped-root planner
scaling and several suffix candidate files.

Reduced evaluation is forbidden at the root, and current recursive suffixes contain no
frozen precommitted group, so these PRs are not semantic prerequisites. Their
planner file overlap argues for rebasing the planner slice after any accepted
planner stack, not for stacking the verifier or algebra work on them.

### Commitment-stage PR 441

PR [#441](https://github.com/LayerZero-Labs/akita/pull/441), head
`4fb5264b3be6e076a925ce88ba837932a2940ed9`, stacks a prover commitment-stage
refactor on the packed recursive witness branch. It may change where the
smaller reduced-evaluation witness is committed, but not how its relation is verified.
Treat it as a prover integration surface, not a protocol dependency.

### Recommended stack shape

```text
codex/quotient-free-tail-relations
  |-- specification and stable shared algebra on the current main base
  |-- rebase the same branch after accepted transcript/planner dependencies land
  |-- implement verifier, prover, and exact planner cutover on that base
  `-- generate catalogs, evidence, Book updates, and the review-ready PR
```

Do not create a separate specification PR or implementation branch. Do not
stack this branch on all open feature branches. Re-evaluate exact heads before
each implementation slice, and rebase this branch only onto dependencies that
are accepted or intentionally chosen as its PR base.

## Documentation plan

This proposed spec owns the in-flight design. It intentionally does not cite
an unpublished Akita paper or require a private research note. The Book must
explain the feature from code and approved specifications once implementation
lands.

Expected durable destinations are:

- `book/src/how/proving/akita-fold-realizations.md`: quotient-lift and
  reduced-evaluation realizations, witness shapes, and cutover;
- `book/src/how/verifying/matrix_evaluation.md`: terminal residue kernel and
  fused setup scan;
- `book/src/how/proving/sumcheck-stages.md`: Stage-2 equation in both modes;
- `book/src/how/configuration.md`: planner cutover and supported feature
  matrix;
- `book/src/how/security.md`: reduced-residual soundness statement and
  unchanged Linf/L2 boundary.

When the implementation and Book updates land, mark this spec `implemented`.
Archive it after the durable content is fully folded, following
[`specs/PRUNING.md`](PRUNING.md).

## Reviewer map

| Review concern | Primary current files |
|---|---|
| Protocol mode and schedule binding | `crates/akita-types/src/layout/params.rs`, `layout/params/descriptor.rs`, `schedule.rs`, `instance_descriptor/mod.rs` |
| Semantic rows and physical layout | `crates/akita-types/src/proof/relation_layout.rs`, `proof/relation.rs`, `witness.rs`, `witness/scalar_len.rs` |
| Shared residue algebra | `crates/akita-algebra/src/ring/` |
| Prover quotient removal | `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`, `ring_switch/coeffs.rs` |
| Prover Stage-2 weights | `crates/akita-prover/src/protocol/ring_switch/relation_weights/`, `sumcheck/relation_range_image/` |
| Verifier terminal MLE | `crates/akita-verifier/src/protocol/ring_switch/prepared_relation_point.rs`, `relation_evaluation.rs` |
| Fused direct setup scan | `crates/akita-types/src/setup_contribution/plan/` |
| Compression reduced transpose | `crates/akita-types/src/proof/compression_relation_weights.rs`, prover/verifier ring-switch compression paths |
| Planner state and cutover | `crates/akita-planner/src/schedule_params/suffix_dp/`, recursive candidate materialization, response model |
| Generated rows and identity | `crates/akita-schedules/src/generated/`, `catalog_identity.rs`, planner emitter and reports |
| Transcript grinding interaction | PR #417 ring-switch query sites, packed proof cost, and grinding plan |
| End-to-end protocol tests | `crates/akita-pcs/src/scheme/tests/`, `crates/akita-pcs/tests/protocol_soundness.rs` |

## References

- [`role-native-projected-digit-layout.md`](role-native-projected-digit-layout.md),
  current native quotient-row order and witness layout.
- [`structured-e-term.md`](structured-e-term.md), current verifier structured
  evaluation-trace term.
- [`setup-offloading-planner.md`](setup-offloading-planner.md), successor-owned
  incoming-prefix topology and Stage-3 selection.
- [`selective-l2-fold-security-sizing.md`](selective-l2-fold-security-sizing.md),
  independent Linf/L2 route selection and typed source moments.
- [`subring-coefficient-packing.md`](subring-coefficient-packing.md), current
  L0/L1 packing policy and later evaluation-trace cutover.
- [`specs/SPEC_REVIEW.md`](SPEC_REVIEW.md), required review rubric before
  implementation approval.
- [PR #417](https://github.com/LayerZero-Labs/akita/pull/417), transcript
  grinding integration inspected at head `aa4efc307`.
- [PR #428](https://github.com/LayerZero-Labs/akita/pull/428), trusted schedule
  artifact proposal inspected at head `d6499748e`.
- [PR #434](https://github.com/LayerZero-Labs/akita/pull/434), certified planner
  architecture inspected at head `d7261d916`.
- [PR #398](https://github.com/LayerZero-Labs/akita/pull/398) and
  [PR #437](https://github.com/LayerZero-Labs/akita/pull/437), suffix EOR and
  packed recursive witness stacks inspected at heads `dd0a9fbdb` and
  `f72ae5683`.
