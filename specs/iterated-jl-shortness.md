# Spec: Iterated JL shortness certification

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-09-03 |
| Status | proposed |
| PR | [#471](https://github.com/LayerZero-Labs/akita/pull/471) |
| Supersedes | |
| Superseded-by | |
| Book-chapter | |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they
appear in all capitals.

This specification describes a proposed protocol. It does not describe the
currently implemented digit-range and selective-L2 paths.

## Summary

Akita will replace the generic digit-range tree with two kinds of shortness
evidence:

1. iterated Johnson--Lindenstrauss (JL) projections that give coarse,
   role-specific no-wrap bounds; and
2. separate exact squared-L2 claims for the three vectors that determine the
   `A`, `D`, and `B` Module-SIS radii.

The stage split is intentionally asymmetric:

- Stage 1 has one job: prove the exact norm of the aggregate semantic `Z`
  consumed by the `A` relations.
- Stage 2 proves the exact norms of literal `Ehat` and `That` directly inside
  the ordinary witness-domain sumcheck.

`Z` needs a separate norm sumcheck because it is a virtual sum of
recompositions of committed `Zhat` chunk planes. `Ehat` and `That` need no
virtualization because they are already literal intervals of the Stage-2
witness.

The fp32 protocol preserves this split. When one role norm cannot lift through
the base field, fp32 partitions that role into fixed shards, certifies each
shard below the modulus, proves each shard norm in the role's existing stage,
and reconstructs the total role norm with checked integer addition.

## Intent

### Goal

Replace the generic digit-range protocol at every enabled fold level without
losing exact, independently priced Euclidean bounds for `Z`, `Ehat`, and
`That`.

The first implementation target is fp64 and larger. The fp32 design is a
shape specialization of the same protocol, not a fallback to digit range.

### Non-goals

This proposal does not:

- prove that `Zhat`, `Ehat`, or `That` is a canonical gadget decomposition;
- project `Zhat` merely to recover gadget injectivity;
- assign an independent shortness meaning to quotient rows or padding;
- replace the binary predicates required by commitment compression;
- make a prover-selected projection matrix an unbounded grinding surface;
- reuse one JL matrix at different projection layers or fold levels;
- commit a projected image in the first cutover; or
- make the Book describe this design before it is implemented.

### Global protocol selector

The schedule MUST choose one shortness protocol for all nonterminal levels:

```text
DigitRangeV1
IteratedJlExactL2V1
```

A schedule MUST NOT mix digit-range levels and iterated-JL levels. The legacy
selector MAY remain during benchmarking and migration. Enabling
`IteratedJlExactL2V1` disables the generic digit-range tree at every level.
The negative-binary compression predicate remains independent of this
selector.

## Security roles

For commitment group `g` and response chunk `c`, define the chunk-local field
response and the group response consumed by that group's `A_g` relation as

```text
Z_g^(c) = G_b Zhat_g^(c) in F_q,
Z_g     = ctr_q(sum_c Z_g^(c)).
```

The sum is taken in the field and `ctr_q` is applied coefficientwise after the
sum. Define the level's semantic `Z` table to be the relation-ordered
concatenation

```text
Z = ||_g Z_g.
```

Every later use of semantic `Z`, `ProjZ`, `S_Z`, `C_Z`, or `Q_Z` refers to
this table. It does **not** mean the concatenation
`||_(g,c) ctr_q(Z_g^(c))`, and it does not mean one selected response chunk.
This distinction is load-bearing: `A_g` binds the sum over chunks, so a norm
of concatenated chunk responses would omit cross terms in `||Z_g||_2^2` and
would underprice the existing shared `A_g` rows unless an extra chunk-count
factor were introduced.

The protocol has three independent exact norm authorities:

| Exact vector | Integer claim | Scheduled cap | Security consumer | Proof location |
|---|---:|---:|---|---|
| aggregate semantic `Z = ||_g Z_g` | `S_Z` | `C_Z` | every group-local `A_g` | Stage 1 norm sumcheck; Stage 2 virtualization |
| literal `Ehat` | `S_E` | `C_E` | `D` | direct Stage-2 square term |
| literal `That` | `S_T` | `C_T` | `B` | direct Stage-2 square term |

The caps are independent. In particular, a large `C_Z` MUST NOT be reused as
the `Ehat` or `That` cap.

The JL protocol supplies separate coarse no-wrap authorities:

```text
Q_Z  for aggregate semantic Z,
Q_ET for Ehat || That.
```

The baseline wide-field plan requires `Q_Z < q` and `Q_ET < q`. The JL caps
establish unique integer lifting. They MUST NOT replace `C_Z`, `C_E`, or `C_T`
when sizing `A`, `D`, or `B`.

Each `Zhat_g^(c)` is an existential algebraic preimage. The accepted semantic
response for group `g` is

```text
Z_g = ctr_q(sum_c G_b Zhat_g^(c)).
```

The protocol constrains the centered aggregate field vector `Z_g`, not an
unreduced integer gadget sum or a list of independently short chunk responses.
A noncanonical `Zhat_g^(c)` is acceptable when the chunk planes together
satisfy the field relation and all other witness constraints.

### Certified source coverage

The cutover MUST constrain exactly the objects that carry an independent
shortness obligation:

| Object | JL source | Exact norm | Security use | Other constraint |
|---|---|---|---|---|
| `Zhat` | none | none | none directly | committed gadget preimage |
| aggregate semantic `Z` | `ProjZ` | Stage 1 | every `A_g` | `Z_g = ctr_q(sum_c G_b Zhat_g^(c))` in Stage 2 |
| literal `Ehat` | `ProjET` | direct Stage 2 | `D` | ordinary witness relation |
| literal `That` | `ProjET` | direct Stage 2 | `B` | ordinary witness relation |
| compression `F`/`H` digits | none | none | compression matrices | negative binary |
| quotient rows and padding | none | none | none independently | relation and layout only |

Projecting the complete recursive witness is therefore not the baseline. It
would spend projection energy and verifier work on coordinates that do not
enter an independent norm reduction.

### Why `Zhat` need not be canonical

Removing the generic digit predicate makes gadget recomposition noninjective.
For adjacent positions, `(b, -1, 0, ...)` is in the integer kernel of
`[1, b, b^2, ...]`. A malicious prover can therefore use carry-cancelling
digits without changing the recomposed field element.

This does not break the intended relation. Akita proves an existential
statement: there are committed group/chunk `Zhat` planes whose summed field
recompositions are the accepted aggregate semantic `Z`. Each `A_g` reduction
consumes the exact norm bound of `Z_g`, not the norm or uniqueness of any
`Zhat_g^(c)`. The honest prover MAY keep using the canonical balanced
decomposition, but the verifier MUST NOT require canonicality after this
cutover.

The distinction is load-bearing. For every group, the verifier and security
inventory MUST use

```text
Z_g(u) = ctr_q(sum_c sum_h b^h Zhat_g^(c)(u,h) mod q),
```

not the unreduced integer gadget sum and not an independently centered
chunk-energy sum.

## Public level plan

Each level plan MUST fix:

- the `Z` projection tree;
- the `Ehat || That` projection tree;
- every local matrix law, row count, column count, block count, and layer;
- the CertifiedJL lower and upper constants for each local matrix;
- the final clear-image shapes and accepted integer energies;
- the role caps `C_Z`, `C_E`, and `C_T`;
- the no-wrap caps `Q_Z` and `Q_ET`;
- the maximum projection retry count;
- the retry-selection rule and its transcript domain;
- every sumcheck shape and transcript label; and
- the complete schedule-wide JL failure ledger.

The proof stream MUST remain headerless. The schedule, not the proof, selects
all vector lengths and variants.

One canonical level-plan constructor MUST feed the prover, verifier,
serialization shape, proof-size estimator, schedule audit, and SIS security
inventory.

### Logical proof shape

The schedule-derived proof stream contains, in order:

```text
JL prelude
  retry index for ProjZ, if the plan permits retries
  clear final ProjZ image
  retry index for ProjET, if the plan permits retries
  clear final ProjET image
  reverse projection-reduction proofs and terminal source claims

Stage 1
  exact integer S_Z
  aggregate-semantic-Z norm sumcheck
  terminal aggregate-semantic-Z evaluations

Stage 2
  exact integers S_E and S_T
  ordinary fused Stage-2 proof with direct E/T squares and deferred bindings
  final authenticated witness evaluation
```

fp32 replaces each role item with its schedule-fixed shard array. The proof
MUST NOT carry row counts, shard counts, layer counts, or a mode tag. Deserialization
uses the public level plan and rejects a missing, extra, or malformed element.

## Exact protocol order

For one nonterminal level, the transcript order is:

```text
1. Bind the outgoing recursive witness.
2. Derive every JL matrix from the bound transcript and the scheduled retry
   index.
3. Send the final clear Z and ET projection images.
4. Check both clear-image energies against their public thresholds.
5. Run the projection reduction sumchecks.
6. Send S_Z and run Stage 1, whose only relation is the exact aggregate Z
   norm.
7. Fix every projection and Z-norm terminal claim.
8. Send S_E and S_T and check their independent public caps.
9. Sample the Stage-2 batching challenges.
10. Run Stage 2: ordinary relation, direct Ehat/That norms, compression
    relation and binary predicate, and all deferred linear bindings.
11. Authenticate the final witness MLE evaluation against the outgoing
    commitment.
```

Every value on the right-hand side of a batched relation MUST be fixed before
the batching challenge for that relation is sampled.

The object that binds `Zhat`, `Ehat`, and `That` MUST be absorbed before the
JL matrices are derived. Housing that object in a later in-memory proof struct
does not permit late transcript binding.

The Stage-1 norm point and the compression-binary equality point belong to
different domains. They MUST be sampled independently even if they have the
same Boolean width.

## JL prelude

### Logical certificates

The wide-field baseline uses two logical certificates:

```text
ProjZ  : aggregate semantic Z -> clear v_Z,
ProjET : Ehat || That     -> clear v_ET.
```

Here the source of `ProjZ` is `||_g Z_g`, after summing the recomposed response
chunks within each group. Chunk index is therefore not a source-concatenation
axis. The prover MAY compute the first projection layer from chunk-local
contributions and add projected contributions in `F_q`, by linearity, before
the clear output is centered. The resulting claim MUST be the projection of
`sum_c G_b Zhat_g^(c)`. Projecting chunks as independent source blocks and
checking only the sum of their squared norms is not equivalent.

Their final image predicates and no-wrap caps remain separate. Their
projection sumchecks MAY be transcript-batched, but the implementation MUST
NOT collapse the two public energy predicates into one bound that forces
`Ehat` or `That` to inherit `Z` headroom.

The `Ehat || That` certificate MUST avoid a ragged first-layer boundary. It
uses private stems until both branches have the same power-of-two output
length:

```text
Ehat  -> private ET stem --+
                            +-> aligned selector join -> shared ET tail
That  -> private ET stem --+
```

The `Z` certificate has its own stem and tail. All intermediate vectors remain
prover-local.

The first layer is role-aligned rather than one flat, ragged matrix:

```text
aggregate semantic Z ----------> Z stem ----> Z tail ----> clear v_Z

literal Ehat ----> E stem --+
                              +--> ET tail ---> clear v_ET
literal That ----> T stem --+
```

The join uses a public selector axis after both private stems reach the same
scheduled power-of-two output length. This keeps prover tables regular and
lets the verifier evaluate only the local matrices used by each stem or tail.

### Public energy predicate

For a clear final image `v`, the verifier computes

```text
E(v) = sum_i ctr_q(v_i)^2
```

with checked integer arithmetic and compares it with the schedule's final
threshold. The verifier does not compare `E(v)` with a prover-supplied norm or
an informal expected value.

For local layer lower floors `L_i`, upper thresholds `U_i`, an honest source
energy cap `H`, and accepted final energy `Omega`, the planner establishes the
completeness inequality

```text
H * product_i U_i <= Omega.
```

For one unstructured chain, the strict threshold theorem gives the conservative
integer source cap

```text
Q = floor(Omega / product_i L_i) + 1.
```

The strict `+1` MUST NOT be replaced by an unchecked ceiling identity. For a
structured repeated-block layer, the planner applies the pinned structured
lifting theorem and includes its public block-allocation factor in `Q`. It MUST
NOT apply the iid full-matrix theorem to `I_r tensor J`.

The plan checks `Q < q` and every intermediate CertifiedJL modular-threshold
premise. For the 256-row L2 theorem below, each invoked threshold `b` must
satisfy `3 b <= q`; the actual adversarial input norm is otherwise
unrestricted.

The foundation registry MUST pin CertifiedJL revision
`8ac6eda09c6f8b6fe38770f78489af610eb05023`. Its tight 256-row base L2 point is:

```text
Rows256Bits128.ternaryL2ThresholdLower29: L = 29, margin = 3,
Rows256Bits128.ternaryL2Upper338:          U = 338.
```

These are kernel results, not automatically a schedule-wide 128-bit claim. A
production plan with several tail events MUST pin a strengthened per-event
frontier or otherwise prove that its complete checked ledger meets the target.
Fewer rows or different constants MAY be used only when their matching lower
and upper theorems are generated, named, and pinned.

### L2 versus Linf tails

The baseline MUST use the L2 lower and upper tails unless the planner proves a
complete Linf plan is smaller for a specific field profile.

| 256-row result | Failure bound | Lower condition/result | Upper result |
|---|---:|---|---|
| L2 | `2^-128` each tail | `b^2 <= ||w||_2^2`, `3b <= q` implies output energy is at least `29b^2` | output energy at most `338 ||w||_2^2` |
| Linf | lower `2^-130`, upper `2^-128` | `b^2 <= ||w||_2^2`, `2b <= q` implies output Linf exceeds `(21/50)b` | output Linf at most `(39/4)||w||_2` |

L2 matches the later exact square sums and composes additively across the E/T
branches. Its one-layer energy distortion is `338/29`. Linf has a better
modulus margin, but converting its available lower and upper constants back to
an energy cap loses roughly `(975/42)^2` per layer. Linf is therefore only a
small-field planner alternative when the `2b` margin changes feasibility; it
MUST NOT be selected merely because both theorem families exist.

### Matrix derivation and reuse

One transcript-derived master seed MAY generate all matrices through distinct
domain separators. The actual matrix at every `(level, certificate, stem,
layer)` MUST be fresh.

The domain separator MUST bind at least:

```text
protocol version, schedule identity, fold level, certificate,
role or stem, layer, rows, columns, matrix law, retry index.
```

The block index is intentionally absent only when the public plan selects
`I_r tensor J` reuse. Every other semantically distinct matrix gets a distinct
domain. Expansion MUST implement the balanced-ternary law
`Pr[0] = 1/2`, `Pr[-1] = Pr[+1] = 1/4`; binary sign matrices are not a
compatible substitution for the pinned theorems.

One local matrix MAY be reused across the equal-width blocks of one layer:

```text
P_i = I_(r_i) tensor J_i.
```

The structured lower-tail proof allocates a proportional threshold to every
nonzero block and union-bounds the failure probabilities over blocks. Matrix
reuse within the layer is therefore compatible with the CertifiedJL vector
theorem.

The same `J_i` MUST NOT be reused at another layer. The next layer input
depends on the preceding matrix, so reusing the matrix would violate the
fixed-input premise of the JL theorem. Matrices also MUST be fresh across fold
levels.

The default wide-field plan uses one projection attempt. If a schedule permits
bounded retries for honest upper-tail failures, it MUST fix a maximum count.
The prover selects only among transcript-derived candidates for the already
bound witness, and the selected retry index is absorbed before the chosen
matrix is used. The verifier rejects an out-of-range or noncanonical index.
Retries multiply the lower-tail opportunity and therefore enter the ledger.

The global JL failure satisfies a checked bound of the form

```text
epsilon_JL <= sum_(levels,certificates,layers) blocks_i * retries_i * delta_i.
```

The selected per-kernel frontier MUST make the complete protocol ledger meet
the configured 128-bit target. The implementation MUST NOT silently treat one
`2^-128` theorem as a `2^-128` schedule-wide result after adding the lower and
upper tails or multiplying by block and retry counts. The target is 128 bits;
the planner MUST NOT default to a 192-bit frontier when a tighter 128-bit
ledger can be certified.

### Succinct projection verification

For one block layer

```text
Y[b,u] = sum_v J[u,v] X[b,v],
```

an output evaluation claim satisfies

```text
Y~(r_b,r_u)
  = sum_(b,v) X[b,v] eq(r_b,b) J~(r_u,v).
```

The prover runs a degree-two sumcheck over `(b,v)`. At the terminal point
`(s_b,s_v)`, the verifier evaluates

```text
eq(r_b,s_b) J~(r_u,s_v).
```

The verifier scans the one local `J`, not the repeated block matrix and not
the dense product of all layers. A reverse chain of these reductions leaves
source MLE claims for aggregate semantic `Z`, literal `Ehat`, and literal
`That`.

For each certificate, the verifier samples an evaluation point only after the
clear final image is absorbed. It evaluates that clear image directly, reduces
the last projection layer, and continues backward until it reaches the source
stem. Every intermediate evaluation claim is transcript-fixed before the next
reduction challenge. The proof does not serialize intermediate projection
vectors or any JL matrix.

The projection sumchecks are a prelude. They do not create a third Akita
witness stage. Their source claims are deferred and closed by Stage 2.

## Stage 1: exact aggregate `Z` norm only

Stage 1 MUST have exactly one semantic responsibility:

```text
sum_u Z(u)^2 = S_Z mod q.
```

Equivalently, across all relation-ordered commitment groups,

```text
S_Z = sum_g sum_u ctr_q(sum_c sum_h b^h Zhat_g^(c)(u,h))^2.
```

There is one Stage-1 responsibility and one total claim even when the witness
has multiple groups or response chunks. Since `S_Z` is the norm of a
concatenation over groups, every group response satisfies
`||Z_g||_2^2 <= S_Z`. It is **not** the sum of the separate chunk norms.

Before the sumcheck, the verifier checks `S_Z <= C_Z` and that `S_Z` has the
scheduled checked integer representation. `ProjZ` establishes
`||Z||_2^2 < Q_Z < q`. Therefore the field equality and the public integer
claim lift uniquely to an exact integer equality.

The norm sumcheck terminates in one or more evaluations of semantic `Z`.
Those evaluations are not authenticated in Stage 1. Stage 2 MUST bind them to
the committed `Zhat` planes through the gadget-recomposition functional.

Stage 1 MUST NOT contain `Ehat` or `That` norm terms.

Stage 1 replaces the current digit-range tree; it is not fused with a surviving
range leaf. Its scheduled proof shape contains only the semantic-`Z` norm
sumcheck and the resulting virtual evaluations.

## Stage 2: direct literal norms and relation

Let `W` be the literal Stage-2 witness table. Let `chi_E` and `chi_T` be the
multilinear extensions of the public indicators for all `Ehat` and `That`
addresses in `WitnessLayout`.

Before sampling their batching coefficients, the verifier receives and checks
the independent integer claims:

```text
S_E <= C_E,
S_T <= C_T.
```

`ProjET` establishes

```text
||Ehat||_2^2 + ||That||_2^2 < Q_ET < q.
```

The exact norm terms are included directly in the ordinary Stage-2 sumcheck:

```text
rho_E chi_E(X) W(X)^2 + rho_T chi_T(X) W(X)^2.
```

The corresponding input claim is

```text
rho_E S_E + rho_T S_T.
```

No E/T virtual table, preliminary norm sumcheck, or E/T norm terminal
evaluation is permitted. At the Stage-2 terminal point, the verifier uses the
already authenticated `W(r)` and evaluates:

```text
rho_E chi_E(r) W(r)^2 + rho_T chi_T(r) W(r)^2.
```

These direct quadratic terms fit the existing degree-three Stage-2 bound.

The complete Stage-2 summand has the schematic form

```text
W(X) relation_weight(X)
+ rho_E chi_E(X) W(X)^2
+ rho_T chi_T(X) W(X)^2
+ rho_bin eq(tau_bin,X) chi_bin(X) W(X)(W(X)+1)
+ W(X) deferred_linear_weight(X)
+ opening and compression-relation terms.
```

`WitnessLayout` MUST own the exact E/T supports, including their chunk and
group unions. Padding, quotient rows, and compression rows MUST evaluate to
zero under both `chi_E` and `chi_T`.

### Deferred linear bindings

Stage 2 also batches linear source claims left by:

- `ProjZ`;
- `ProjET`;
- the Stage-1 `Z` norm sumcheck; and
- existing opening and compression relations.

Every semantic-`Z` functional is rewritten as a public linear functional of
all committed group/chunk `Zhat` planes:

```text
L(Z) = sum_g sum_u L_(g,u) sum_c sum_h
         b^h W(pi_Z(g,c,u,h)).
```

This equality is a field equality. Centered lifting is needed for the JL and
exact-norm predicates, while the linear binding uses the same field sum that
the existing relation assigns to `A_g`. `WitnessLayout` MUST supply the
relation-order, chunk count, gadget-plane address, and coefficient position in
`pi_Z`; Stage 2 MUST NOT bind only one chunk or a chunk concatenation.

The `Ehat` and `That` projection terminals map directly to their literal
witness intervals.

All right-hand-side source claims are absorbed before the fresh Stage-2
batching coefficients. Stage 2 then reduces them to the same final `W(r)` that
is authenticated against the outgoing commitment.

### Compression binary predicate

The `F` and `H` compression digits retain their negative-binary predicate:

```text
eq(tau_bin,X) chi_bin(X) W(X)(W(X)+1).
```

The generic digit-range predicate is absent.

The compression equality point MUST be sampled independently of the Stage-1
`Z` norm point. The old implementation's reuse of a range-tree Stage-1 point
cannot survive this cutover because the new Stage-1 point belongs to the
semantic `Z` domain.

## Exact SIS radii

The JL distortion affects only the no-wrap proof. The three exact scheduled
caps determine the collision radii.

For group `g` with accepted challenge multiplication-operator norm `Gamma_g`:

```text
eta_A_g,2^2 = 64 Gamma_g^2 C_Z.
```

This is sound because `||Z_g||_2^2 <= S_Z <= C_Z`. No response-chunk factor
appears: `C_Z` already caps the exact norm of the sum consumed by `A_g`. A
design that instead certified `sum_c ||Z_g^(c)||_2^2` would need the weaker
bound `||sum_c Z_g^(c)||_2^2 <= C sum_c ||Z_g^(c)||_2^2`, and would price `A_g`
with that extra factor. This specification deliberately avoids that loss.

For two accepted openings, the unsliced `Ehat` and `That` difference bounds
are:

```text
eta_D,2^2 <= (sqrt(C_E) + sqrt(C_E'))^2,
eta_B,2^2 <= (sqrt(C_T) + sqrt(C_T'))^2.
```

One uniform cap on both openings gives `4 C_E` and `4 C_T`. The security
inventory MUST consume these role caps rather than `Q_Z`, `Q_ET`, or a global
witness cap.

Compression `F` and `H` remain on their coefficientwise binary-difference
route.

## Cost model

For a block layer with `r_i` blocks and local matrix
`J_i in {-1,0,1}^{m_i x n_i}`:

- prover projection work is proportional to `r_i m_i n_i` sparse integer
  operations;
- verifier matrix work is proportional to `m_i n_i` field operations for the
  one local matrix MLE, not `r_i m_i n_i`;
- the reduction has `log2(r_i) + log2(n_i)` Boolean variables; and
- the proof transmits sumcheck polynomials and terminal claims, but not `J_i`
  or the repeated block matrix.

The clear-image payload is exactly the sum of the scheduled final row counts
for `ProjZ` and `ProjET`, encoded with the schedule's canonical centered-integer
format. The proof-size estimator MUST derive its byte count from those row
counts and the encoding; this spec does not hard-code a `256`-row or one-
kilobyte wire shape.

Adding E/T norm terms to Stage 2 does not add rounds. It adds selector and
square work to the existing degree-three round computation. Stage 1 retains
only the semantic-`Z` norm rounds and removes the generic range-product tree.

## fp32 adaptation

The production fp32 modulus profile is `Q32Offset99`, with
`q = 2^32 - 99 = 4,294,967,197`. fp32 uses the same logical certificates and
the same stage ownership. It
changes only the representation of a role norm when one field square-sum
would wrap.

### Fixed role shards

For a role vector `R`, the schedule MAY choose a public disjoint partition:

```text
R = R_0 || ... || R_(k_R-1).
```

Each shard has an independent coarse no-wrap cap:

```text
||R_j||_2^2 < Q_(R,j) < q.
```

The prover supplies an exact integer shard norm `S_(R,j)`. The verifier proves
the field square-sum for that shard, uses the JL cap to lift it uniquely, and
reconstructs the total role norm with checked integer addition:

```text
S_R = sum_j S_(R,j).
```

The role's scheduled cap remains independent:

```text
S_R <= C_R.
```

Sharding therefore removes the field-lifting obstruction without replacing
the exact norm by a distorted JL bound.

### fp32 Stage 1

If `Z` needs `k_Z` shards, Stage 1 proves:

```text
sum_u Z_j(u)^2 = S_(Z,j) mod q      for every j,
S_Z = sum_j S_(Z,j),
S_Z <= C_Z.
```

These sumchecks MAY be batched over a public shard selector. They all remain
inside Stage 1, whose only job is exact aggregate-semantic-`Z` norm
certification. Their terminal semantic claims are virtualized to the
corresponding group/chunk `Zhat` intervals in Stage 2.

### fp32 Stage 2

If one global `Ehat` or `That` square-sum would wrap, partition its literal
witness intervals into fixed shards. Stage 2 adds one direct square term per
shard:

```text
sum_j rho_(E,j) chi_(E,j)(X) W(X)^2,
sum_j rho_(T,j) chi_(T,j)(X) W(X)^2.
```

It reconstructs `S_E` and `S_T` from the exact shard claims and checks the
independent caps `C_E` and `C_T`. These terms remain direct Stage-2 terms; fp32
MUST NOT introduce E/T virtualization or an E/T Stage-1 norm proof.

### fp32 projection layout

The clear projection payload uses a typed outer axis:

```text
Y_Z[shard,row] = J_Z[shard] Z_shard,
Y_E[shard,row] = J_E[shard] Ehat_shard,
Y_T[shard,row] = J_T[shard] That_shard.
```

Equal-width shards MAY reuse one local matrix within a layer. The verifier
checks a separate accepted energy for every no-wrap shard. Projection
sumchecks MAY batch the arrays over the public role/shard selector, while
preserving the individual public energy predicates.

For E/T, the outer axis can retain the role selector used by the wide-field
aligned join and add a shard selector beneath it. A batch is only algebraic:
each shard keeps its own public accepted energy and extracted cap.

The planner SHOULD first try:

1. one `Z` certificate and one combined `ET` certificate;
2. sharded `Z` with one combined `ET` certificate;
3. sharded `Z` plus separate or sharded `Ehat` and `That` certificates.

It selects the first plan satisfying all no-wrap, modulus-margin, failure,
proof-size, and verifier-work constraints.

Every fp32 `Z` shard is a coordinate shard of the already chunk-aggregated
semantic table `||_g Z_g`. For a shard support `I_s`, its exact claim is

```text
S_Z,s = sum_(g,u in I_s) ctr_q(sum_c sum_h
          b^h Zhat_g^(c)(u,h))^2.
```

Stage 1 checks these shard norms and checked-integer addition reconstructs
`S_Z = sum_s S_Z,s`. Sharding the committed chunk planes first and then adding
their separate norm claims is forbidden, because that again omits the cross
terms of the `A_g`-consumed response.

### Coefficient packing

Subring coefficient packing is compatible with the Euclidean `A` route. The
carrier embedding preserves coefficient L2 norm, and multiplication by the
embedded challenge is block diagonal with the same L2 operator norm as the
carrier challenge.

Concretely, for `d_A = k h d_car`, the carrier embedding

```text
iota(c(Y)) = c(X^(k h))
```

is a coefficient-L2 isometry. Multiplication by `iota(c)` is block diagonal
with `k h` copies of multiplication by `c`, so its induced L2 operator norm is
the carrier operator norm. Packing therefore does not require a new
coefficientwise range argument.

The coefficient-packing challenge sampler MUST therefore be able to apply the
same certified operator-norm predicate used by the L2 evaluation-trace route.
The current Linf-only admission rule is an implementation restriction, not an
fp32 security requirement.

### Norm sharding versus matrix sharding

Norm sharding and Module-SIS matrix sharding are separate decisions.

- If one `A`, `B`, or `D` matrix is secure for the reconstructed total role
  cap, the implementation uses that matrix and the exact total cap.
- If the total cap does not fit a secure q32 estimator row, the protocol MAY
  use independently domain-separated or block-diagonal matrix instances
  aligned with the role shards.

Merely splitting the norm proof while retaining one dense matrix does not
reduce that matrix's collision radius. The planner MUST model this distinction
explicitly.

### fp32 decision procedure

For every fold level, the planner:

1. computes honest role energies for chunk-aggregated semantic `Z`, literal
   `Ehat`, and literal `That`;
2. tries the unsharded wide-field plan;
3. increases only the shard count of a role whose no-wrap or matrix-security
   constraint fails;
4. derives CertifiedJL row counts and lower/upper constants for the complete
   block/layer/retry ledger;
5. verifies every shard cap is below `q` and every modular threshold premise;
6. derives exact total role caps from shard caps, without JL distortion;
7. prices `A`, `D`, and `B` from `C_Z`, `C_E`, and `C_T` respectively;
8. introduces matrix shards only when the corresponding total cap has no
   admissible q32 estimator row; and
9. rejects the level if no complete assignment exists.

There is no per-level fallback to generic digit range checking.

### Wide-field and fp32 correspondence

| Contract | fp64 and larger | fp32 specialization |
|---|---|---|
| JL sources | one aggregate-source `ProjZ`, one aligned `ProjET` | same roles, split only the aggregate/literal roles that fail no-wrap |
| Stage 1 | one exact `S_Z` | exact `Z` shard norms, checked sum to `S_Z` |
| Stage 2 | direct exact `S_E`, `S_T` terms | direct exact E/T shard terms, checked role totals |
| SIS pricing | total `C_Z`, `C_E`, `C_T` | same totals unless matrix sharding is independently required |
| compression | negative binary | unchanged |
| generic digit range | absent | absent |

## Architecture and implementation slices

### Ownership

| Owner | Responsibility in the cutover |
|---|---|
| `akita-challenges` | balanced-ternary law and deterministic seed expansion |
| `akita-algebra` | checked packed matrix, integer/field projection, block application, and local-matrix MLE |
| `akita-types` | inert theorem records, level plans, witness supports, proof shapes, and checked cap formulas |
| `akita-transcript` | versioned labels and ordering only; no JL algebra |
| `akita-prover` | projection construction, reverse reductions, Stage-1 Z norm, and Stage-2 E/T terms |
| `akita-verifier` | clear-image checks, reduction replay, exact-claim lifting, and fused Stage 2 |
| planner, config, and schedules | choose one feasible plan and feed the same caps to the security inventory |
| serialization and proof-size code | derive the headerless wire shape and exact byte count from that plan |

The implementation MUST keep one canonical function for every geometry, cap,
or proof-shape calculation. Prover, verifier, planner, schedule audit, and
proof-size code MUST NOT reimplement the formulas independently.

### Slice 1: certified projection primitives

- Add the pinned CertifiedJL bound registry and digest.
- Implement deterministic balanced-ternary matrix expansion with
  `Pr[0] = 1/2` and `Pr[-1] = Pr[+1] = 1/4`.
- Implement scalar and packed block projection kernels.
- Implement local matrix MLE evaluation.
- Test dense-reference equality, transcript domain separation, centered
  lifting, checked integer overflow, and malformed geometry rejection.

### Slice 2: projection reduction

- Add schedule-derived `Z` and `ET` projection plans.
- Implement the reverse block-layer reduction sumchecks.
- Implement aligned E/T stem joining.
- Batch source claims without combining the two energy predicates.
- Add mutation tests for seeds, images, layers, terminal claims, and retry
  nonces.

### Slice 3: exact role norms

- Refactor Stage 1 to contain only chunk-aggregated semantic-`Z` norm
  certification.
- Add direct `Ehat` and `That` square selectors to Stage 2.
- Add a dedicated compression-binary equality point.
- Extend `WitnessLayout` authority for role and shard supports.
- Add exact integer reconstruction and role-cap tests.

### Slice 4: wide-field protocol cutover

- Add the schedule-wide protocol selector.
- Add headerless proof shapes and transcript sites.
- Add `A`, `D`, and `B` Euclidean security inventory entries.
- Enable larger decomposition bases.
- Cut over fp64 and larger schedules only when every level validates.

### Slice 5: fp32

- Enable the L2 route for coefficient packing.
- Add role-shard planning and proof shapes.
- Add exact coordinate shards of chunk-aggregated `Z` to Stage 1.
- Add direct E/T shard square terms to Stage 2.
- Add optional role-aligned matrix sharding.
- Regenerate and audit fp32 schedules.

### Slice 6: remove the legacy path

After every supported field family uses `IteratedJlExactL2V1`, remove:

- the generic range-image tree;
- its Stage-1 proof objects and transcript sites;
- the generic range-image term in Stage 2;
- digit-range-derived A/B/D security caps; and
- the legacy schedule selector if no benchmark or compatibility use remains.

The negative-binary compression predicate remains.

## Compatibility and migration

This is an intentional protocol and wire-format break. Enabling the cutover
changes transcript labels, proof shapes, schedule identity, proof-size
accounting, and the security inventory. Existing proofs and serialized
descriptors MUST NOT be accepted under the new schedule identity.

The first implementation MAY retain the legacy selector for side-by-side
benchmarks, but one schedule cannot mix the two shortness protocols across
levels. After fp64, fp128, and fp32 all have admitted JL plans, the repository
removes the generic digit-range path. No setup matrix is added for JL because
all JL matrices are transcript-derived.

The following behavior is preserved:

- ordinary Akita relation semantics and the final outgoing-witness opening;
- commitment-compression relations and negative-binary `F`/`H` digits;
- headerless, schedule-owned proof geometry; and
- the verifier no-panic contract.

## Parameters still selected by the planner

This specification fixes the protocol contract but does not freeze one global
matrix geometry. Before a schedule is admitted, generated configuration MUST
resolve and audit:

- the number and shape of Z, E, and T stem layers;
- the shared ET tail and every final row count;
- a strengthened per-event CertifiedJL frontier for the full 128-bit ledger;
- whether bounded retries are worth their soundness and transcript cost;
- the canonical clear-image encoding;
- fp32 role shard counts; and
- whether any fp32 security cell needs matrix sharding in addition to norm
  sharding.

An unresolved item rejects the schedule. It does not select a hidden default
or re-enable generic digit range.

## Acceptance criteria

- [ ] The outgoing witness is transcript-bound before every JL seed.
- [ ] Every projection layer and fold level uses a fresh domain-separated
      matrix.
- [ ] The verifier scans only local JL matrices, not repeated block matrices
      or dense products.
- [ ] `ProjZ` and `ProjET` have independent public no-wrap caps.
- [ ] `ProjZ` sources `||_g ctr_q(sum_c G_b Zhat_g^(c))`, not a selected
      response chunk or a concatenation of chunk responses.
- [ ] `Zhat`, quotient rows, padding, and compression digits are absent from
      the JL sources unless a later security proof adds a named shortness use.
- [ ] Stage 1 contains only exact aggregate-semantic-`Z` norm certification.
- [ ] Stage 2 proves exact literal `Ehat` and `That` norms directly.
- [ ] Stage 2 virtualizes every `Z` claim and no E/T norm claim.
- [ ] Compression F/H remains negative binary under an independent equality
      point.
- [ ] `C_Z`, `C_E`, and `C_T` independently price every `A_g`, `D`, and `B`.
- [ ] fp32 reconstructs exact total role norms from independently no-wrap
      shard norms.
- [ ] fp32 norm sharding does not silently change a dense matrix's SIS radius.
- [ ] The complete JL block/layer/retry ledger meets the configured 128-bit
      target.
- [ ] Headerless serialization rejects every shape mismatch without panicking.
- [ ] All supported field families verify without a generic digit-range tree.

## Testing strategy

The implementation requires:

- scalar-versus-packed projection tests;
- matrix-MLE-versus-dense-reference tests;
- prover/verifier transcript parity tests;
- noncanonical-`Zhat` acceptance tests where semantic `Z` is unchanged;
- carry-cancelling `Zhat` tests that preserve every Stage-2 semantic claim;
- independent `S_Z`, `S_E`, and `S_T` corruption tests;
- multi-chunk tests where equal chunk energies but different cross terms change
  `||sum_c Z_g^(c)||_2^2`;
- Stage-2 `Z` virtualization mutations that omit, duplicate, or reorder one
  group/chunk plane;
- Stage-2 E/T support-selector mutation tests;
- compression binary point and support mutation tests;
- fp32 shard-boundary, overflow, and checked-total tests;
- fp32 tests proving `Z` shards are taken after chunk aggregation;
- fp32 tests distinguishing norm-only sharding from matrix sharding;
- schedule identity and proof-size agreement tests; and
- malformed-proof and allocation-bound tests for the verifier no-panic
  contract.

## Documentation

This proposed spec is the design authority until implementation. The Akita
Book MUST continue to describe current behavior and MUST NOT present this
proposal as implemented. After the cutover ships, fold the durable protocol
and security explanation into:

- `book/src/how/proving/sumcheck-stages.md`;
- `book/src/how/security.md`;
- `book/src/how/configuration.md`; and
- `book/src/how/verification.md`.

Then archive this spec according to `specs/PRUNING.md`.

## References

- `specs/selective-l2-fold-security-sizing.md` for the implemented selective
  physical-`Z` norm route that this proposal generalizes.
- `specs/subring-coefficient-packing.md` for the implemented packing geometry
  and current Linf-only security-route restriction.
- `specs/role-native-projected-digit-layout.md` for literal witness intervals.
- `specs/transcript-grinding.md` for bounded retry and transcript accounting.
- CertifiedJL revision `8ac6eda09c6f8b6fe38770f78489af610eb05023`,
  especially `Rows256Bits128.ternaryL2ThresholdLower29`,
  `Rows256Bits128.ternaryL2Upper338`,
  `Rows256Bits130.ternaryLInfThresholdLower21Over50`, and
  `Rows256Bits128.ternaryLInfUpper39Over4`.
- LaBRADOR, RoKoko, and PikkuFold for clear projected images, structured
  block projections, and layered projection reductions.
