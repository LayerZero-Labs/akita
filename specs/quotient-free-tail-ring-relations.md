# Spec: Quotient-free tail ring relations by reduced evaluation

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-08-25 |
| Status        | active |
| PR            | [#466](https://github.com/LayerZero-Labs/akita/pull/466) |
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

Implementation re-audit (2026-09-04): the protocol, layout, prover, verifier,
and planner paths described here are implemented in PR #466, with code and
evidence pinned at `04111dedf` and the main documentation sync at `5fd356d0c`.
Full cross-mode replay, relation-mode traversal invariance, small-field
reduced/EOR coverage, production-profile phase timing, bounded malformed-input
coverage, serialized-proof agreement, and planner telemetry are present. The
specification remains `active` until the PR merges; aggregate measurements
belong in the PR, while exact generated schedules are neither compatibility
fixtures nor additional protocol modes.

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
formula MUST remain available under tests as an independent oracle.

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

Stage 2 ends at a multilinear point. Let `r` be that complete point and let
`o` be the physical start address of one native coefficient window. Define the
exact terminal equality weights

\[
e_{r,o}(j)=\operatorname{eq}(r,o+j).
\]

The caller MUST derive this window from the canonical relation address through
the checked offset-equality authority. It MUST NOT assume that every native A,
B, D, F, or H window begins at zero merely because the shared coefficient block
is aligned. Two windows with the same native dimension but different physical
starts are not interchangeable.

For one public multiplier `A`, the verifier needs

\[
\widetilde\kappa_{A,\alpha}(r,o)
=\sum_j e_{r,o}(j)\kappa_{A,\alpha}(j).
\]

Swap the public multiplier and witness-coordinate sums:

\[
\widetilde\kappa_{A,\alpha}(r)
=\sum_k a_k H_k(r,\alpha),
\]

where

\[
H_k(r,\alpha)
=\sum_j e_{r,o}(j)
(-1)^{\lfloor(k+j)/d\rfloor}
\alpha^{(k+j)\bmod d}.
\]

The complete `H` vector has the recurrence

\[
H_0(r,\alpha)
=\sum_j e_{r,o}(j)\alpha^j
\]

The familiar product formula is available only when the checked window
authority proves the corresponding zero-aligned complete Boolean block. The
general implementation consumes the exact equality-weight slice.

and

\[
\boxed{
H_{k+1}(r,\alpha)
=\alpha H_k(r,\alpha)
-D_\alpha e_{r,o}(d-1-k).
}
\]

The verifier builds `H` from each exact equality window and reuses it only for
an identical physical start and native dimension. It then evaluates each public
multiplier by one base-field-by-extension-field dot product with `H`.

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
Mixed or unaligned A/B/D/F/H windows require their exact checked equality
weights. The implementation MAY stream or cache these kernels, but auxiliary
storage MUST remain bounded by the largest active native window and it MUST NOT
materialize a witness-sized functional table.

The verifier MUST extend the existing fused `SetupContributionPlan` scan for
the ordinary A, B, and D setup tensor. It MUST NOT add independent A, B, and D
scans or materialize one residue kernel per setup lane. Compression F/H maps
remain owned by the canonical compression-relation program: they use the same
reduced coefficient-functional semantics, but are not falsely described as
columns of the ordinary A/B/D setup tensor.

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
| Mixed role dimensions | Supported | Prepare terminal residue kernels from exact checked physical equality windows |
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

## Normative implementation contract

The cross-crate architecture, verifier and prover realization, performance
model, acceptance criteria, alternatives, and open implementation risks live in
the focused normative companion
[`quotient-free-tail-ring-relations-implementation.md`](quotient-free-tail-ring-relations-implementation.md).
Both files form the active protocol contract.

## Non-normative rollout record

Slice order, exact dependency heads, restack notes, documentation destinations,
and reviewer routing live in
[`docs/quotient-free-tail-ring-relations-rollout.md`](../docs/quotient-free-tail-ring-relations-rollout.md).
That record is operational context and is not normative.

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
