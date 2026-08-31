# Supporting Design: Certified Planner Pruning Proofs

Parent specification:
[Certified Planner Architecture](../certified-planner-architecture.md)

This file is normative support for the parent specification. It inherits the
parent's status, pull request, ownership, acceptance process, and retirement.
It is not an independent live specification.

The parent defines the planner architecture and the authority of certified
pruning. This file owns the detailed rule contracts, mathematical arguments,
checker obligations, and unresolved proof gaps. An implementation may use a
stronger theorem, but it must preserve the same safety claim and conservative
behavior when a checker cannot establish its predicate.

## Rule contracts and current bounds

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
bound. The full proof appears below.

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
6. root output-witness length when it is not already fixed by the prefix;
7. parent visible payload and admission class.

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
- root output-witness length;
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
A rank alone is not enough. Equality on all earlier V2 coordinates also
requires the root output-witness and descriptor order.

### Memo and cache rules

Memoization stores exact completed frontier results keyed by sufficient state.
Cache quotas are performance settings, not semantic settings.

Eviction removes only the cached result. A later lookup recomputes the same
state. Tests must run with several small capacities, including zero effective
reuse, and obtain the same selected descriptor.

Guide caches and local profile caches follow the same rule. A cache hit may
save work. It may not add authority.

The post-#445 complete-source compression-plan cache, response-model caches,
setup-prefix search cache, and bounded suffix memo follow this contract. The
suffix key must distinguish `RingRelationPhase`; a quotient prefix and a
reduced-evaluation suffix do not have the same legal transition domain even if
their other scalar fields happen to match.

## Formal pruning proofs

This section states the formal obligations behind the certified pruning
architecture. A later implementation may use a stronger bound, but it must
prove at least the same safety claim and expose the same unknown behavior.

### Proof boundary

The planner minimizes a total order on complete schedules. The numeric prefix
is one of these orders.

\[
(P,S,W)
\]

or

\[
(C_1,P,S,W).
\]

Here \(P\) is proof bytes, \(S\) is the total setup envelope, and \(C_1\) is
the first direct padded setup capacity. \(W\) is the root output-witness
length introduced into the V2 objectives by #445. The canonical descriptor
follows the numeric prefix in both orders.

A lower bound contains only numeric coordinates. It may prune a region when an
earlier coordinate is strictly worse than a completed incumbent. If all
earlier coordinates tie, it must bound \(W\) before using it to prune; equality
through \(W\) is not enough because the canonical descriptor can still choose
a candidate from that region.

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
| Monotone ring-relation transition | Implemented prerequisite from #445; not a pruning theorem | Typed transition authority and independent `m + 1` cutover oracle are present |
| L2 route omission at the Linf winning split | Not proved and not accepted as an exact domain | Production uses this shortcut; the test oracle derives L2 independently at every split |
| Local setup first slice pruning | Not proved and not accepted as an exact domain | Current code retains every minimum-padded-setup tie but removes larger local setup choices |
| Fixed radius recursive split search | Not proved and not accepted as an exact domain | Current bounded policy uses this shortcut |

The final three rows are migration requirements. This PR specifies their
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
- The source encoding, security route, typed relation transition, relation
  geometry, and compression plan.

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
setup prefixes, mode-dependent quotient rows, compression layers, and
alignment can only add physical coefficients. Reduced evaluation makes the
corresponding quotient term zero; quotient lift charges it canonically. The
planner may add any of those terms when it can compute them cheaply. Omitting
them preserves a lower bound, but the typed relation transition remains part
of the cell and successor-state signature.

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
- a typed relation transition or relation phase;
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
uses the suffix value when the prefix is offloaded. The root output-witness
length is fixed by the root edge and carried unchanged while its suffix is
compared. Each operation is monotone.

The planner can combine the exact fixed prefix with \(h(s)\). It may prune the
region only when the result is strictly worse than a completed incumbent on
the numeric order. It must retain an equal bound for descriptor comparison.

### Relaxed state and useful bound terms

A relaxed state can merge details only when the merge preserves the theorem.
The first implementation should retain at least:

- the payload phase and the minimum and maximum remaining fold depth allowed
  by admission;
- the ring-relation phase and its legal setup-offload consequence;
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
- the root output-witness length when it is not already fixed by the exact
  prefix.

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
4. If the numeric projections, including the root output-witness coordinate,
   can tie, descriptor composition proves that \(t_a\) is no worse. Otherwise
   one numeric coordinate before the descriptor must be strictly better.

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

The production shortcut creates selective L2 only at the split chosen by the
best modeled Linf candidate for each admitted relation mode. The independent
test oracle added by #445 derives eligible L2 candidates at every split. The
production path also rejects L2 when its inner A rank is not smaller than the
Linf rank. Neither fact alone compares the norm proof, the B rank, the next
witness, the relation phase, the suffix, the setup envelope, the root output
witness, or the descriptor. The target exact planner must not use either fact
as a complete route proof.

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
- the root output-witness projection when the route can affect it;
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

The current setup-first shortcut retains every slice tied at the smallest local
padded setup, but removes every strictly larger local setup before it computes
all successor witnesses and suffixes. Slice count changes the outer commitment
input width and can change the B rank. It can also change relation rows,
compression, the next witness, proof bytes, total setup, root output witness,
parent admission, and descriptor order.

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
- the root output-witness projection when the slice can affect it;
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
- Root output-witness projection.
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
