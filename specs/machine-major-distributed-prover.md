# Spec: Machine-major distributed prover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao; Codex |
| Created       | 2026-07-11 |
| Status        | proposed |
| PR            | |
| Supersedes    | `distributed-prover.md`; layout portions of `distributed-verifier-row-eval.md` and `distributed-planner.md` |
| Superseded-by | |
| Book-chapter  | book/src/how/proving/distributed-prover.md |

This spec follows the lifecycle in [`PRUNING.md`](PRUNING.md). It defines the
protocol and implementation cutover for a genuinely distributed recursive
prover. The implementation must land with this spec in one PR.

## Summary

Akita's current multi-chunk prover emits one contiguous `[z | e | t]` unit per
machine, but the next recursive level flattens their concatenation into one
`RecursiveWitnessFlat` and interprets it with a single global block-fast index.
That interpretation crosses machine boundaries: a machine that owns one
contiguous unit no longer owns complete recursive blocks.

The correct layout is hierarchical:

```text
global witness
  machine 0: local block-fast witness [z_0 | e_0 | t_0 | r_0]
  machine 1: local block-fast witness [z_1 | e_1 | t_1 | r_1]
  ...
  machine W-1: local block-fast witness [z_{W-1} | e_{W-1} | t_{W-1} | r_{W-1}]
```

The machine axis is outermost. Each machine owns one literally contiguous local
witness. Within that witness, the local block index remains the fastest axis.
The prover, relation, commitment, trace, setup, and sum-check paths consume this
layout natively. There is no runtime permutation wrapper and no universal
digit-fast cutover.

The relation is the horizontal concatenation of native local extended relations.
At distributed levels, machine `j` retains its own full-shaped ring-switch
quotient `r_j`. The quotient values are not replicas, but their layouts are
identical. This removes quotient communication and gives every machine the same
witness shape. At the distributed-to-single cutover, the machines explicitly
aggregate the smaller fold output and emit one ordinary single-machine witness.

## Decision

### Physical order

Let `W = 2^k` be the machine count. Every distributed level stores a vector of
`W` local witnesses, not one flat recursive witness:

```text
DistributedRecursiveWitness {
    machines: [RecursiveWitnessFlat; W]
}
```

All machine witnesses have the same live length `C`. Machine `j` owns exactly
the contiguous global interval `[jC, (j+1)C)` when the proof is serialized or a
single-host oracle is materialized.

Let `Q` be the number of recursive blocks owned by one machine and `L` their
local block length. Inside a machine:

```text
local_index(position, local_block) = position * Q + local_block
```

The complete hierarchical address is therefore:

```text
index(machine, position, local_block)
    = machine * C + position * Q + local_block
```

This is machine-major globally and block-fast locally.

### Native relation, not a permutation layer

For explanation only, if `P` denotes the permutation from the single-machine
column order to machine-major order, then `w_dist = P w` and
`M_dist = M P^{-1}` preserve `M_dist w_dist = M w`.

The implementation must not construct `P`, permute a materialized witness, or
wrap the old evaluator. It must derive the distributed relation columns directly
in machine-major, locally block-fast order. The same rule applies to setup
columns, trace weights, commitment columns, and quotient columns.

### Compact storage and local opening stride

The live local length `C` need not be a power of two. Define

```text
P = C.next_power_of_two()
```

for the local multilinear opening stride. Positions `C..P` are structural zero.
They are not serialized, committed, range-checked, or counted as setup columns.
Because `W` is a power of two,

```text
next_power_of_two(W * C) = W * P.
```

Moving the global suffix padding into equal virtual machine strides does not
increase the opening domain. The opening point splits as:

```text
[local witness bits | machine bits].
```

Every verifier path must evaluate the machine selector and the local compact
witness weight without materializing the `W * P` table.

## Protocol

### Input ownership

At a distributed fold level, machine `j` owns a contiguous power-of-two block
window `I_j`. The windows partition the current relation blocks. Each machine
stores all positions and digits for its blocks in its own local block-fast
witness.

For a singleton commitment group, every machine uses the same public `A`, `B`,
and `D` matrices, regenerated from the shared setup seed. `B_j` and `D_j` mean
the native setup-column views selected by machine `j`'s block window; they are
not copied matrices.

### Local fold and local relation

Machine `j` computes:

```text
z_j = sum_{i in I_j} c_i s_i
```

in the full ambient fold space. It also holds the corresponding `e_j` and `t_j`
digits. Its local ring relation has the ordinary global row set but only its
native local columns:

```text
M_j [z_j | e_j | t_j] = h_j             in R_q.
```

The partial right-hand sides satisfy `sum_j h_j = h`; only `h` is public.
The global relation is defined natively as

```text
[M_0 | M_1 | ... | M_{W-1}]
[w_0 | w_1 | ... | w_{W-1}]^T = h.
```

There is one transcript challenge schedule. Machine `j` restricts every
block-indexed challenge to `I_j` and computes a contribution to the same public
sum-check claim.

### Partial-fold norm contract

A partial fold is not automatically bounded by the global fold norm: cancellation
between machines may make the global sum smaller than an individual `z_j`.
The planner must price the infinity norm of each partial fold directly. The
prover must grind or reject until every live `z_j` satisfies the scheduled cap.
Stage 1 range-checks every decomposed `z_j` independently.

### Local ring-switch quotients

Machine `j` lifts its local relation before reduction modulo `X^d + 1`:

```text
M_j w_j = h_j + (X^d + 1) r_j.
```

Each `r_j` has the complete global relation-row shape. Its entries are partial
row contributions, not copies of another machine's quotient. Machine `j`
digit-decomposes `r_j` locally and appends the digits to its local witness:

```text
machine j: [z_j | e_j | t_j | rhat_j].
```

The global lifted relation is

```text
sum_j (M_j w_j - (X^d + 1) G rhat_j) = h.
```

This is correct because `G rhat_j = r_j` for every machine and recomposition is
linear. The protocol does not claim
`decompose(sum_j r_j) = sum_j decompose(r_j)`, which is false.

The quotient row count and digit depth are identical across machines, so local
witnesses remain uniform. The planner charges `W` quotient segments during
distributed levels. This is deliberate: shipped W=8 schedules have quotient
segments hundreds of rings long while `e/t/z` machine bodies contain tens or
hundreds of thousands of rings at the distributed levels.

### Alternative: row-sharded summed quotient

An implementation may later replace local quotients with a summed quotient if
benchmarks justify the communication:

1. Sum `r = sum_j r_j` coefficient-wise.
2. Reduce-scatter complete relation rows of `r` across machines.
3. Digit-decompose only after reduction.
4. Give each machine an equal padded row-slot count.
5. Route each real quotient row to exactly one owner in the native relation.

This preserves the single-machine quotient size but adds one quotient
reduce-scatter. It is not the first implementation target and must not be added
as a second live layout mode in the initial PR.

### Next-level commitment

Each machine treats its output as an ordinary local block-fast recursive
witness. It computes the inner commitment for its complete local blocks and the
native local contributions to the outer commitment. The machines sum only the
short public commitment vectors.

The implementation must expose a batch-of-local-witness source to commitment
kernels. It must not concatenate the buffers and call today's monolithic
`SuffixWitnessView::block_elem`.

### Commitment-compression interaction

Commitment compression introduces global images and payloads but does not
require a global witness owner. The protocol distinguishes three objects:

1. the raw B/D image, which is the sum of short machine contributions;
2. one canonical schedule-generated F/H compression chain for that global
   image; and
3. the chain's digit witnesses, whose matrix columns are partitioned across
   machines after the canonical decomposition is fixed.

For the B/F chain, workers first compute and reduce the short raw image

```text
u = sum_j u_j.
```

The workers then execute one distributed canonical chain:

```text
u = G_{A_F,1} xi_F,1
u_k = F_k xi_F,k = G_{A_F,k+1} xi_F,k+1    for 1 <= k < L_F
u_pub = F_LF xi_F,LF.
```

1. Each worker derives only its scheduled coefficient/ring-column shard of
   `xi_F,1` from the canonical `u`.
2. At layer `k`, the domain-selective native-ring kernel computes each partial
   negacyclic image `u_neg,k,j`. It may also compute the cyclic contribution and
   quotient eagerly, or defer F-cyclic completion to an equal-shape opening
   bucket as specified by the compression execution policy.
3. The short reduction `u_k = sum_j u_neg,k,j` fixes the canonical image. When
   quotient completion is eager, quotient contributions are available at this
   point; when it is deferred, each worker later combines its F-cyclic
   contribution with the partial negacyclic image it computed and retained.
   The globally decomposed successor image is not a machine-local negacyclic
   RHS. Workers do not communicate a separate cyclic image.
4. If `k < L_F`, each worker derives only its scheduled shard of
   `xi_F,k+1` from that global image and continues. Otherwise the reduction is
   the public payload `u_pub`.

Thus matrix multiplication and digit storage are both distributed, while the
chain remains exactly the single global scheduled F-chain instance. Decomposition, map
shapes, setup prefix, shard boundaries, and encoding are descriptor-bound. No
compression-digit vector crosses the worker boundary. The H chain follows the
same rule after reducing `v = sum_j v_j`.

The implementation must **not** compress each partial `u_j` or `v_j`
independently and sum the final payloads. In general,

```text
decompose(sum_j u_j) != sum_j decompose(u_j).
```

More importantly, independently compressed partials would change the binding
instance for the terminal map `F_L xi_L` to repeated columns
`[F_L | ... | F_L] [xi_{L,0}; ...; xi_{L,W-1}]`. Standalone certification of
the original terminal map would no longer price that wider witness. Sending one
standalone payload per machine would preserve independent security but multiply
wire size by `W`; that is not this protocol.

#### Compression-witness sharding

Each F/H witness segment is partitioned independently in its own native ring
column axis. Equal F/H ring dimensions are unnecessary, and a role's column
count need not divide `W`. For a role with `L` native ring columns, machine `j`
owns the deterministic near-even contiguous interval

```text
[floor(j L / W), floor((j + 1) L / W)).
```

The maximum local slot count is `ceil(L/W)`. Missing final slots are structural
zero, not serialized digits, setup columns, range-check entries, or quotient
inputs. This padding is at most `W-1` absent columns per role; it is not
digit-depth padding and does not round a digit count to a power of two.

The machine-local witness order becomes schedule-derived rather than fixed to
only four hard-coded ranges. Conceptually:

```text
machine j:
  z_j | e_j | t_j
  | (xi_Fk,j)_{k in [L_F]} | (xi_Hk,j)_{k in [L_H]}
  | local quotient families_j
```

F segments repeat per independent commitment identity in canonical relation
order. H segments belong to the current opening. The public compressed payload
is not sharded: it appears once in proof serialization and once in the public
right-hand side.

`MachineWitnessLayout` must therefore consume the canonical semantic relation-
witness layout rather than owning a second hard-coded list of compression
roles. Every semantic segment carries one distribution policy:

```text
PerMachineFull       z and local quotient contributions; same shape, different values
BlockPartitioned     e/t segments selected by the machine's block window
ColumnPartitioned    F/H compression digit segments
```

The distributed layout applies these policies and returns checked local spans.
Compression code does not independently assign worker offsets.

#### Local compression relations and quotients

Let `F_{k,j}` denote the column restriction of layer `k` to machine `j`'s digit
shard. The global chain rows are sums of local contributions:

```text
sum_j (B_j t_j - G_b,j xi_F1,j)                 = 0
sum_j (F_k,j xi_Fk,j - G_{A_F,k+1},j xi_F(k+1),j) = 0  for k < L_F
sum_j  F_LF,j xi_F,LF,j                         = u_pub.
```

For each compression map `K`, worker `j` computes

```text
u_neg,j  = neg(K_j xi_j)
r_prod,j = (cyc(K_j xi_j) - u_neg,j) / 2
K_j xi_j = u_neg,j + (X^d + 1) r_prod,j.
```

The successor digits come from the reduced global image, so generally
`u_neg,j != G_j xi_next,j`. The local extended relation therefore equals the
residual `delta_j = u_neg,j - G_j xi_next,j`, not zero; the coordinated
sum-check proves `sum_j delta_j = 0`. The H rows are analogous. Exactly one
designated RHS owner subtracts `u_pub` or `v_pub`; every other machine uses zero
RHS. This is only an additive convention and does not make the owner's local
row valid independently. The public
payload and transcript contain no owner field because the owner is derived
canonically from the schedule (initially machine zero).

Every machine digit-decomposes `r_prod,j`; its quotient contribution retains the
complete row-family shape, including F/H families, and uses each family's
native ring dimension. Every F/H row has a product-quotient contribution; there
is no scalar or quotient-free compression exception. The coordinator sums
local round polynomials before each challenge. Summing the local residuals and
lifted contributions recovers the one mixed-dimension global relation without
quotient-reduction communication.

The negative-binary support is the union of every real local F/H input span
whose authenticated map alphabet is negative binary, at any layer. Structural
zero slots are excluded. A map is priced at coefficient bound one only when its
complete global span is the disjoint union of these verifier-enforced local
supports. Each worker contributes the restricted-equality round polynomial for
its real sparse support to the fused Stage-2 sum-check; the coordinator sums it
with the ordinary carried-range and relation polynomials before deriving the
single challenge. After that challenge each worker sibling-folds its local
sparse weights, so work and memory remain proportional to projected real
support. Structural-zero shards contribute neither entries nor work. No
separate global binary table is materialized.

#### B/D source geometry

This implementation uses the unsliced B/D source images. Machine ownership
partitions the input columns, and the short reduction reconstructs the single
canonical image consumed by the first F/H map. B/D block-axis slicing is a
separate extension: when implemented, it must use the exact slice-major source
image, descriptor, and structured-matrix security contract specified by the
commitment-compression design. It does not change F/H column sharding, and it
must not identify slice count with machine count.

#### Hints, groups, and cutover

Each compressed commitment identity remains independent. A frozen root group
has one payload and one canonical F-chain hint. When distributed proving starts,
the hint is exposed as canonical semantic segments and each worker loads only
its scheduled column shards. If the commitment was created with a different
machine count, repartitioning is input provisioning and is measured separately
from proof communication; it does not alter the commitment or its descriptor.

Recursive commitments created while `W > 1` retain their F-chain shards on the
same workers for the next level. At W-to-1 cutover, the coordinator receives or
recomputes the now-small canonical F/H segments needed by the single-machine
suffix. Multi-group payloads remain separate public objects and their F digits
remain separate semantic segments; machine sharding never concatenates
commitment identities into one compression chain.

### Distributed sum-check

All machines prove restrictions of one padded global MLE. In every round:

1. Each machine computes its local round-polynomial contribution.
2. The coordinator sums the coefficient vectors.
3. The transcript absorbs the summed polynomial.
4. The verifier samples the next challenge.

The machine-prefix bits are fixed for a local prover. The remaining local point
uses the same block-fast factorization as the single-machine prover. Challenges
must never be sampled from a local transcript.

### Distributed range tree

Each machine builds the range-product subtree for its local witness. The `W`
local roots become the leaves of the short machine-prefix tree. The coordinator
combines only those roots and broadcasts the resulting carried claims. No
machine materializes another machine's digit table.

### Trace and setup contribution

Trace and setup weights use the same native machine-major address contract.
For partitioned `e/t` data, machine supports tile the single-machine columns.
For replicated `z_j`, contributions sum over machines. Setup entries are
seed-shared, so the verifier combines machine weights before scanning each setup
entry; it must not repeat the expensive setup scan `W` times.

The relation, trace, and setup evaluators must match an independent dense
machine-major oracle. No evaluator may flatten the witness and fall back to a
digit-fast or global-block-fast interpretation.

## Cutover to one machine

Let `R` be the number of distributed output levels. A fold has separate public
input and output ownership:

```text
level L input machines  = machines emitted by level L-1
level L output machines = W if L < R, otherwise 1
```

The current single `LevelParams.witness_chunk` field is insufficient because it
is used ambiguously for both input interpretation and output pricing. Replace it
with schedule-owned input/output machine counts, or derive the input count from
the predecessor while storing the output count explicitly.

The cutover fold consumes `W` native local witnesses. The machines continue to
run its fold and sum-check distributively. They then aggregate the smaller
folded response and quotient needed for the single-machine output, emit one
ordinary block-fast `RecursiveWitnessFlat`, and elect the suffix prover. The
planner must price this communication. It must not reinterpret the machine-major
concatenation through the monolithic recursive backend.

After cutover, all later levels use the existing single-machine protocol.

## True distributed execution

The current `CommitCluster`, `OpeningCluster`, `TensorCluster`, and
`RingSwitchCluster` types delegate directly to `CpuBackend`. They are compute
capability markers, not a distributed runtime. Likewise, the current
multi-chunk prover loops over every chunk inside one process, retains the global
fold, commits the full flattened witness, and runs each sum-check over the full
table. That path is a correctness oracle only.

The implementation PR must add a real coordinator/worker execution path, but it
must not attempt to build a production distributed-systems framework. One local
coordinator must spawn persistent worker processes and communicate through a
small framed byte-channel interface. The reference implementation uses child
stdin/stdout, available through Rust's standard process API on supported
platforms. Separate address spaces demonstrate ownership and expose
serialization, synchronization, communication volume, and coordinator-memory
costs. Remote hosts, reconnects, discovery, encryption, and transport plugins
are out of scope.

### Runtime roles

The coordinator owns:

- the canonical Fiat--Shamir transcript;
- the public schedule and descriptor digest;
- ordered collective results;
- the serialized proof;
- the single-machine suffix after cutover.

Worker `j` owns:

- its root blocks or current local recursive witness;
- its commitment hints and local setup views;
- `z_j`, `e_j`, `t_j`, and `r_j`;
- its local sum-check tables and range subtree.

For the reference harness, each worker receives an assigned shard path or
independently generated fixture seed at process launch. The coordinator does
not serialize the union witness into worker pipes. Input provisioning outside
the proof is not a network-storage design problem for this PR, but the measured
proof path must begin with already-sharded ownership.

The coordinator must not retain the union of worker witnesses during a
distributed level. A debug oracle may materialize the union only under tests.

### Minimal worker protocol

Keep the coordinator/worker interface protocol-specific and small, separate
from arithmetic compute backends. The first implementation needs only:

```text
send_public_challenge(value)
receive_local_contribution(short_vector_or_round_poly)
receive_acceptance(bool)
receive_cutover_payload(payload)
```

Protocol messages are encoded independently of the channel. The runtime has a
minimal `send_frame`/`recv_frame` byte-channel boundary; proving logic must not
depend on child-process handles, file descriptors, Unix-only APIs, or the
reference process launcher. This is a portability seam, not a general
collective framework. A later remote transport can carry the same frames and
worker state machine without changing prover arithmetic or transcript logic.

The coordinator performs the reductions itself; workers do not communicate
with one another. Workers persist across all distributed levels so process
startup is outside proving measurements. Each frame carries a compact
phase/round tag and checked payload length. Unexpected phase, premature EOF, or
wrong shape returns `AkitaError`. The first implementation assumes trusted
local workers and fail-stop execution. General collective APIs, TCP, MPI/NCCL,
Byzantine-worker robustness, retries, and elastic membership are deliberately
deferred.

CI must spawn independent worker processes and prove the same bytes as the
single-host oracle. The process harness must report bytes and time per message
class so that replacing local IPC with a real deployment transport later does
not hide the protocol's communication pattern.

### Distributed phase schedule

For each distributed level:

1. **Opening and partial commitments.** Each worker computes local `e/t`,
   partial raw B/D images, and its partial claimed value. Short images and
   scalars are summed. Workers run the scheduled distributed H-chain on the
   global D image, retaining only their H digit shards.
2. **Transcript synchronization.** The coordinator absorbs the sums and
   broadcasts the transcript-derived challenge material.
3. **Fold grinding.** For each candidate nonce, every worker computes only its
   `z_j` and reports acceptance. A nonce is accepted iff all workers accept.
   The coordinator commits the minimum accepted nonce and broadcasts it.
4. **Local relation and quotient.** Each worker builds `M_j`, `h_j`, and `r_j`
   without communication.
5. **Next commitment.** Each worker commits its local recursive blocks using
   native setup columns. After reducing the raw B image, workers run the
   scheduled distributed F-chain, retain only their F digit shards, and absorb the
   single compressed payload.
6. **Range tree.** Workers build local subtrees; only local roots cross the
   network.
7. **Sum-check stages.** Workers submit one small polynomial per round. The
   coordinator sums, absorbs, and broadcasts the next challenge before workers
   advance.
8. **State handoff.** Each worker retains its local next witness. No witness
   gather occurs while the next output level remains distributed.

When verifier offloading is enabled, its stage-3 public setup term must not be
computed redundantly by every worker. Partition its setup-index range across
workers or assign that public term to one worker, then sum it with the
witness-local round contributions. Without verifier offloading, no stage-3
setup-product phase is present.

### Communication contract

Before cutover, communication is limited to:

- raw B/D images, intermediate F/H images, and final compressed payload
  contributions;
- claimed values and carried claims;
- one acceptance bit per grind probe;
- local range-tree roots;
- constant-degree round-polynomial coefficients;
- transcript challenges and phase metadata.

There is no communication proportional to the root witness, `e_j`, `t_j`,
`z_j`, or `r_j`. The planner and profiler must report bytes by collective and
level. A hidden witness-sized collective is a correctness failure, not merely a
performance regression.

At cutover, the coordinator explicitly gathers the smaller single-machine
output. This includes the partitioned `e/t` output, the summed folded response,
and the ordinary global quotient formed by summing quotient coefficients before
decomposition. The schedule prices these bytes and selects the cutover with
communication included.

## Multi-group interaction

Semantic groups and machines are independent axes. The future combined layout is:

```text
machine 0: [group-order local segments | local quotient]
machine 1: [group-order local segments | local quotient]
...
```

Within each machine, groups appear in canonical relation order. Transcript order
remains separately defined by `OpeningClaimsLayout`. Every distributed group
must partition evenly across the same active machine count, or the schedule must
provide explicit inactive-machine zero segments. Group-specific `A_g/B_g`
parameters and frozen commitment metadata remain unchanged.

The first implementation keeps multi-group combined with `W > 1` rejected. The
layout types must not conflate a semantic group with a machine. Once singleton
distributed recursion passes, add a dense reference test for the product
`groups x machines` before lifting the guard.

The existence of digit-fast root hints does not require digit-fast recursive
witnesses. Ring-switch assembly reads each machine's block window from the hint
and emits the machine's local recursive witness directly in local block-fast
order.

## Implementation design

### Ownership across the compression PR

Commitment compression and distributed proving must not introduce competing
layout or relation abstractions. Their ownership boundary is:

- the compression work owns semantic commitment identities,
  `CompressionChainPlan`, the schedule-derived semantic relation-row and
  relation-witness layouts, mixed-ring relation providers, and F/H security;
- this work owns machine input/output geometry, distribution policies applied
  to semantic witness segments, local relation contributions, process
  orchestration, and cutover;
- setup contribution, trace, range, and sum-check consume the composed semantic
  plus distributed layout rather than choosing one feature's layout first and
  patching the other afterward.

The preferred implementation stack lands compression's descriptor, semantic
layout, mixed-ring quotient, and relation-provider foundations first. This
branch then rebases and implements machine sharding against those authorities.
The distributed spec may be reviewed against `main`, but implementation must
not freeze a `z/e/t/r`-only public type that compression immediately has to
replace.

The compression spec's current instruction to “extend `WitnessLayout`” must be
read as extending the canonical semantic relation-witness layout. It must not
append global compression spans to the old flat multi-chunk layout. The final
composition is semantic layout first, machine distribution second.

### Types

Replace the flat/chunk-list ambiguity with explicit ownership types:

```text
DistributedRecursiveWitness
MachineWitnessLayout
DistributedWitnessLayout
```

`MachineWitnessLayout` owns local `z/e/t/r` ranges, local live length, virtual
opening stride, block count, block length, and the scheduled shards of any
compression suffixes. It derives those spans by applying distribution policies
to the canonical semantic relation-witness layout; it does not duplicate the
F/H role list. `DistributedWitnessLayout` owns the ordered machine layouts,
common-shape validation, relation order, total live length, and total opening
domain.

Do not add thin aliases over `WitnessLayout`. Remove or rewrite the old type once
all production callers move.

### Prover

- Emit one `RecursiveWitnessFlat` per machine.
- Compute one folded response and one local quotient per machine.
- Add batch-local opening, fold, inner-commit, and outer-commit kernels.
- Aggregate only short commitments and sum-check messages at distributed levels.
- Implement an explicit W-to-1 cutover fold.

### Verifier

- Derive the hierarchical layout from the schedule.
- Evaluate relation rows by summing native per-machine structured evaluations.
- Evaluate every local quotient segment.
- Reuse setup scans across machine contributions.
- Enforce work and allocation caps before building tables.
- Reject malformed machine counts, unequal live shapes, invalid point lengths,
  and unsupported multi-group products with `AkitaError`.

### Verifier performance design

The machine-major layout must not introduce a per-machine copy of the dominant
verifier work. Equal local strides give an exact machine-prefix factor. Exploit
it before evaluating component weights.

- **`z` and local quotient columns.** Their local matrix coefficients are
  identical across machines. Sum the machine equality weights first (equal to
  one over the complete machine domain), then evaluate the local structured
  formula once.
- **`e/t` consistency columns.** Machine block windows tile the original block
  axis. Build one global challenge summary by concatenating or folding the local
  summaries; total heavy work remains `O(B)`, not `O(WB)`.
- **`A/B/D` setup columns.** Combine all machine column weights before the
  shared setup scan. Each physical setup entry receives one alpha evaluation.
- **Trace.** Fold machine selectors into one structured `e`-support evaluator.
  Do not build one dense virtual table per machine.
- **Stage 3.** Reuse the same combined setup-index weights and active prefix.
  Machine count may affect cheap weight construction, never setup scan length.

The verifier must not materialize the padded `W * P` opening table or a dense
relation-weight table. It may allocate `O(P_local + W + rows)` structured
tables, subject to existing caps.

Normative operation-count invariants:

```text
setup alpha evaluations(distributed) = setup alpha evaluations(single)
B/D setup entries scanned(distributed) = B/D entries scanned(single)
A setup entries scanned(distributed) = A entries scanned(single)
```

The intrinsic verifier overhead is limited to machine-prefix contractions,
extra proof rounds caused by the genuinely larger witness, and the short local
quotient/range bookkeeping.

`origin/main` is the normative performance and operation-count baseline. PR
#294 is compared only as an alternative-layout data point; beating it does not
establish success.

CI performance gates for the implementation PR:

- `W = 1`: verifier wall time no more than 5% above `origin/main`.
- `W in {2,4,8}`: verifier wall time no more than 20% above the single-chunk
  sibling after separately reporting extra transcript rounds.
- no shipped profile may regress by more than 10% without an approved spec
  amendment identifying an intrinsic cost.
- the setup alpha-evaluation counters above must match exactly; wall-time noise
  cannot waive this requirement.

PR #294's reported multi-chunk verifier regressions are not an acceptable
baseline. The target is the structured block-fast verifier on `main` plus the
small intrinsic overhead above.

### Planner and generated schedules

- Separate input and output machine counts per level.
- Price `W` copies of `z` and local quotient segments on distributed outputs.
- Price local virtual padding without charging physical witness bytes.
- Price per-machine partial-fold norm bounds and grinding.
- Price W-to-1 cutover communication.
- Enlarge generated rows and descriptor digests with all ownership geometry.
- Regenerate the D64 multi-chunk families.

### Native kernels

Every affected kernel must accept the hierarchical layout directly:

- recursive opening and folding;
- inner and outer commitment;
- ring relation and quotient;
- relation-column MLE;
- trace weight;
- setup contribution and Stage 3;
- terminal witness handling;
- proof-size accounting.

No production path may materialize a permutation, transpose the entire witness,
or select PR #294's universal digit-fast layout as a fallback.

## Security and transcript invariants

- `W` is a descriptor-bound power of two.
- All machines use the same transcript and public schedule.
- Every local witness is committed before fold challenges are sampled.
- Every local `z_j` satisfies the scheduled partial-fold norm cap.
- Every local quotient digit satisfies the generic digit range proof.
- The global public right-hand side equals the sum of partial right-hand sides.
- The verifier checks one global relation claim and one global commitment.
- Machine order, group order, local ranges, and cutover level are transcript-bound.
- `W = 1` is byte-identical to the ordinary single-machine layout.
- ZK with `W > 1` remains explicitly unsupported in the initial PR.

## Implementation sequence and expected diffs

These are ordered review checkpoints in one spec-and-implementation PR. Each
checkpoint must compile and pass its focused tests before the next begins. The
order separates four failure classes: layout, algebra, verifier cost, and
process orchestration.

### Migration rules

The following rules apply throughout the implementation:

1. There is one production layout authority at every commit. A temporary dense
   oracle may coexist under `#[cfg(test)]`, but two selectable production
   layouts may not.
2. No compatibility adapter may flatten `DistributedRecursiveWitness` and call
   `SuffixWitnessView`. Consumers move to the hierarchical source directly.
3. `W = 1` remains on the ordinary single-machine types and byte path. Do not
   emulate it with a one-element distributed batch.
4. The process runtime is not introduced until the complete single-host
   hierarchical path proves and verifies. It orchestrates already-correct local
   providers; it does not define protocol arithmetic.
5. Generated schedules change only after the planner prices the new runtime
   shape. Hand-edited generated rows are forbidden.
6. The old shared-quotient and flat-chunk path may remain temporarily only until
   the native path reaches end-to-end parity. At that checkpoint it is deleted,
   not retained behind a flag.
7. Every performance-sensitive checkpoint records operation counts before wall
   time. An implementation does not advance while a dominant scan unexpectedly
   gains a factor of `W`.

### S0: Freeze the `origin/main` baseline

Before protocol changes, add instrumentation that explains verifier work rather
than relying only on elapsed time.

Expected diffs:

- `crates/akita-types/src/setup_contribution/`: add test/profile counters for A,
  B, and D entries scanned, setup alpha evaluations, equality-table elements,
  and packed-segment fallbacks. Counters must be compiled out or no-op outside
  tests/profiling.
- `crates/akita-verifier/src/protocol/ring_switch.rs` and
  `crates/akita-verifier/src/stages/stage3.rs`: attribute relation, trace, and
  setup work without changing evaluator selection.
- `crates/akita-pcs/examples/profile/`: report counters, verifier allocations,
  proof rounds, and `W` beside wall time for the existing W1/W2/W4/W8 profiles.

Deliverable: a checked-in benchmark table for `origin/main` that separates
intrinsic extra rounds/columns from repeated evaluator work. No layout or proof
bytes change in S0.

#### S0 baseline captured on 2026-07-11

The instrumentation-only branch runs the same verifier algorithms and shipped
schedules as `origin/main`. One release sample at `nv=25`, q128/D64 one-hot,
compares the ordinary schedule with W8R2. Times are diagnostic rather than a CI
threshold; operation counts are deterministic.

Direct setup contribution:

| Profile | Verify | Relation groups | Chunk bodies | Direct setup evaluations | Setup ring visits | Setup segments |
|---------|-------:|----------------:|-------------:|-------------------------:|------------------:|---------------:|
| W1 | 7.961 ms | 6 | 6 | 6 | 49,461 | 37 |
| W8R2 | 20.188 ms | 7 | 21 | 7 | 136,296 | 45 |
| ratio | 2.54x | 1.17x | 3.50x | 1.17x | 2.76x | 1.22x |

Recursive setup contribution:

| Profile | Verify | Stage 3 instances | Setup ring scans | Setup-index eq elements | Ring eq elements | Succinct weights | Generic-plan weights | Packed segments |
|---------|-------:|------------------:|-----------------:|------------------------:|-----------------:|-----------------:|---------------------:|----------------:|
| W1 | 9.081 ms | 5 | 47,933 | 52,224 | 320 | 0 | 5 | 32 |
| W8R2 | 20.807 ms | 6 | 134,704 | 202,752 | 384 | 0 | 6 | 40 |
| ratio | 2.29x | 1.20x | 2.81x | 3.88x | 1.20x | — | 1.20x | 1.25x |

The extra level/round work is visible in group, Stage 3, and ring-coordinate
counts (1.17--1.20x). It does not explain setup work growing 2.76--3.88x or
relation chunk bodies growing 3.50x. Both recursive profiles also bypass the
succinct setup-weight evaluator entirely in this shipped mixed-role schedule.
These counts establish the first optimization targets for S5.

### S1: Replace ambiguous schedule geometry

Make input and output ownership explicit before changing witness bytes.

Expected diffs:

- `crates/akita-types/src/witness.rs`: add descriptor-bound
  `DistributedOwnershipGeometry` containing input/output machine counts, local
  block windows, and cutover validation, but no witness-segment offsets. The
  existing shared-tail `WitnessLayout` remains the production byte-layout
  authority only until the atomic S4 cutover.
- consume compression's canonical semantic segment IDs and distribution
  policies when that prerequisite is present; do not add distributed copies of
  F/H plan or row-layout types.
- `crates/akita-types/src/layout/params.rs`: replace the per-level ambiguous
  `witness_chunk` interpretation with descriptor-bound `input_machines` and
  `output_machines`. Preserve `ChunkedWitnessCfg` only as a planner policy input
  until S7; it is not a resolved protocol geometry.
- `crates/akita-types/src/schedule.rs` and proof-shape/descriptor code: bind both
  counts and all local geometry. Validation checks `input_machines` against the
  predecessor and `output_machines` against the current fold output.
- `crates/akita-planner/src/catalog_identity.rs` and generated schedule identity:
  hash the new geometry in canonical field order.

The canonical geometry constructor takes block counts and machine counts and
returns ownership windows. Callers may not reconstruct those windows manually.
Its independent test oracle uses the equations in this spec, not production
indexing methods.

Deliverable: geometry and serialization tests for W1/W2/W4/W8 and invalid
predecessor/output transitions. No witness ranges, prover behavior, or proof
bytes change in S1.

### S2: Introduce the hierarchical witness source

Change ownership in the recursive backend without yet changing the relation.

Expected diffs:

- `crates/akita-prover/src/backend/recursive/witness.rs`: keep
  `RecursiveWitnessFlat` as the owner of one machine-local buffer; add
  `DistributedRecursiveWitness { machines: Vec<RecursiveWitnessFlat> }` with
  layout-checked construction. Add a borrowed machine-local block-fast source
  used by kernels.
- Commitment/opening compute plans in `crates/akita-prover/src/compute/`: accept
  a batch of local sources and explicit local geometry. Do not add a method that
  concatenates digits or forwards to the flat plan.
- CPU kernels: loop over local sources at the outer boundary and reuse the
  existing block-fast inner kernels. Return one local contribution per machine;
  reduction occurs at the protocol layer.

Deliverable: local opening, fold, inner commitment, and outer commitment match a
dense machine-major oracle. A test records that no local kernel calls
`block_elem` with another machine's block range.

### S3: Emit native local relations and local quotients

Move ring-switch construction to the final distributed witness shape.

Expected diffs:

- `crates/akita-types/src/witness.rs`: define the target
  `DistributedWitnessLayout` and `MachineWitnessLayout`. A machine layout owns
  local `z/e/t/r` ranges, live length, virtual stride, local blocks, and global
  block base; it never has `Option<r_len>`. The canonical constructor returns
  all ranges and validates equal local shapes and the exact `WP` domain. During
  S3 these types are exercised by the native test/oracle path; the existing
  layout remains the sole production authority until the S4 atomic cutover.
- apply `ColumnPartitioned` to every scheduled F/H digit segment and derive the
  local negative-binary supports from every real shard whose map is tagged
  negative binary.
- `crates/akita-prover/src/protocol/fold_grind.rs`: return accepted local
  `z_j` witnesses as the primary output. Stop aggregating them into a global
  fold during distributed output levels. Acceptance requires every scheduled
  local norm check.
- `crates/akita-prover/src/protocol/ring_relation.rs`: build `h_j` and the native
  local `M_j [z_j|e_j|t_j]` contribution for each machine.
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`: compute
  and decompose one complete-row `r_j` per machine, then emit
  `[z_j|e_j|t_j|rhat_j]` directly into its `RecursiveWitnessFlat`.
- `crates/akita-types/src/proof/relation_matrix_cols.rs`: materialize native
  machine-major columns for the prover only; keep the dense implementation as a
  test oracle, not as the verifier path.

Deliverable: every ordinary additive local lifted identity passes
independently; compression product quotients recompose their partial products,
compression residuals and round polynomials vanish only after summation; every
local quotient digit is range-checked; and a fixture guards against assuming
additive digit decomposition.

### S4: Cut over recursive prover consumers atomically

Make `prove_fold` consume and produce ownership-aware state end to end. This is
the point at which the old flat distributed production path is removed.

Expected diffs:

- `crates/akita-prover/src/protocol/core/fold.rs`: replace the single
  `logical_w` flow with an enum or state type that distinguishes ordinary
  single-machine and distributed witnesses. Dispatch once at the fold boundary,
  not inside arithmetic loops.
- `crates/akita-prover/src/protocol/core/{root_fold,suffix}.rs`: root assembly
  emits machine-local buffers; suffix levels preserve them while
  `output_machines > 1`.
- `crates/akita-prover/src/protocol/ring_switch/{commit,finalize,evals}.rs`:
  accept hierarchical sources and return local evaluation providers.
- reduce raw B/D images, deterministically recompute the one canonical
  F/H chain on workers, and retain only scheduled compression-digit shards;
  never compress partial machine images independently.
- `crates/akita-prover/src/protocol/sumcheck/`: expose one local
  round-polynomial provider per machine and a deterministic in-process reducer.
  Challenges still come from the single canonical transcript.
- `crates/akita-prover/src/protocol/core/fold.rs`: implement a fixed-schedule
  W-to-1 cutover for test and shipped schedule geometry. Sum folded responses,
  assemble partitioned `e/t`, sum quotient coefficients, and decompose the
  single global quotient only after summation.
- `crates/akita-types/src/proof/ring_relation.rs`: delete the old shared-tail
  semantics (`last_chunk`, optional `r_len`, and global `r_offset`) in the same
  commit that all production callers switch to machine layouts.
- terminal handling remains single-machine and requires the explicit cutover
  before the terminal fold.

Deliverable: the single-host hierarchical prover produces a complete proof for
W2/W4/W8 whose local round polynomials match the dense oracle. W1 proof bytes
remain identical. The production verifier cutover is S5; no process or channel
code exists yet.

### S5: Native verifier and sum-check contraction

Land the optimized verifier before process orchestration so performance failures
cannot be blamed on IPC.

Expected diffs:

- `crates/akita-verifier/src/protocol/ring_switch.rs`: replace chunk-offset
  iteration with a machine-prefix contraction followed by local block-fast
  structured evaluation.
- `crates/akita-types/src/setup_contribution/`: remove multi-chunk exclusion from
  `prefers_succinct_path`; expose one evaluator that combines machine column
  weights before scanning A/B/D. Do not materialize per-machine setup vectors.
- `crates/akita-types/src/trace_weight/`: express machine support as a prefix
  factor and retain compact trace terms; no dense distributed fallback.
- compression relation providers combine machine column shards before the one
  shared F/H setup-prefix scan; public payload rows occur once.
- `crates/akita-verifier/src/stages/{stage2,stage3}.rs`: use the native relation,
  trace, and setup evaluators. Stage 3 evaluates the public setup term once.
- `crates/akita-verifier/src/protocol/core/suffix.rs`: derive hierarchical input
  and single-machine cutover output from the separate schedule fields.

Deliverable: the production verifier accepts the S4 proofs; every final
structured evaluation matches the dense oracle; setup scan counters equal W1
exactly; W1 stays within 5% of `origin/main`; and W2/W4/W8 meet the performance
gates before S6 begins.

### S6: Price ownership and select the cutover

Only after both sides implement hierarchical input should the planner optimize
where the ownership transition occurs.

Expected diffs:

- `crates/akita-planner/src/{schedule_params,resolve}.rs`: choose the cutover
  using local witness work plus communication bytes. Price `W` local quotients
  only on distributed outputs.
- `crates/akita-planner/src/generated/{walk,expand}.rs`: propagate the two
  ownership counts without reconstructing them from policy activation depth.

Deliverable: the planner-selected W2/W4/W8 cutovers preserve the S4/S5 proof
behavior, match an ordinary single-machine suffix from the cutover commitment
onward, and have proof/communication bytes equal to runtime counters.

### S7: Add the minimal portable process runtime

The process layer wraps the proven local providers from S5; it contains no
cryptographic formulas.

Expected diffs:

- new `crates/akita-prover/src/distributed/` modules for typed protocol frames,
  coordinator state, worker state, and the minimal `send_frame`/`recv_frame`
  byte-channel traits;
- one portable child-stdio channel and persistent worker entry point in the
  integration harness;
- `CommitCluster`, `OpeningCluster`, `TensorCluster`, and `RingSwitchCluster` in
  `compute/delegating_cpu.rs` are deleted or renamed so they no longer pretend
  that CPU delegation is a distributed runtime;
- profiling records worker compute time, coordinator reduction time, blocked
  time, bytes by message class (including raw B/D image reductions and any
  compression-shard transfer), and peak memory per process.

The coordinator owns transcript transitions and reduces already-produced short
messages. Workers receive assigned shard paths/seeds at startup and retain state
across levels. The channel boundary is portable; child stdio is only its first
implementation.

Deliverable: independent processes produce byte-identical proofs to the S5
in-process reducer, the coordinator never owns the union witness, and premature
exit/unexpected-phase/wrong-shape tests fail closed.

### S8: Regenerate schedules and remove migration residue

Expected diffs:

- regenerate D64 W2/W4/W8 schedule families and catalog hashes through the
  planner;
- remove `ChunkedWitnessCfg` from resolved `LevelParams` and retain it only if it
  remains useful as a public planner policy;
- delete flat distributed offsets, shared quotient helpers, obsolete guards,
  delegation-only cluster markers, and test fixtures that encode the old order;
- update the Book from implemented code and mark this spec implemented.

Final search gates must find no production references to old shared-tail
semantics, no distributed concatenation into `SuffixWitnessView`, and no
multi-chunk verifier fallback. Run the complete acceptance suite only after
these deletions, so dead compatibility code cannot make it pass accidentally.

### Expected blast radius by checkpoint

| Checkpoint | Proof bytes | Generated schedules | Principal code effect |
|------------|-------------|---------------------|-----------------------|
| S0 | unchanged | unchanged | instrumentation only |
| S1 | unchanged | geometry-only regeneration | descriptor and ownership geometry |
| S2 | unchanged | unchanged | hierarchical sources and test oracles, not selected in production |
| S3 | changed for `W > 1` | stale until S6 | local quotients and native machine witnesses |
| S4 | same as S3 | stale until S6 | all prover consumers and local sum-check providers |
| S5 | same as S3 | stale until S6 | native structured verifier; no wire change |
| S6 | final | planner regenerates | explicit cutover and final pricing |
| S7 | byte-identical to S6 | unchanged | process orchestration only |
| S8 | final | checked in | cleanup and catalog refresh |

Files expected to be created are limited to the distributed runtime modules,
their integration worker, and focused test/profile support. The implementation
must not create parallel `*_distributed` copies of relation, trace, setup, or
sum-check arithmetic. Existing canonical modules gain ownership-aware inputs.

Public proof structs should change only where the number or ownership of
committed witness segments is actually serialized. Process frames, worker IDs,
local partial right-hand sides, and communication counters are runtime state and
must not enter proof serialization. The verifier remains a single verifier; no
distributed-verifier API is introduced.

## Acceptance criteria

- [ ] New machine-major layout types are the only distributed layout authority.
- [ ] Each machine owns one contiguous local `RecursiveWitnessFlat`.
- [ ] Local recursive indexing remains block-fast.
- [ ] No distributed level concatenates machine buffers into one monolithic recursive view.
- [ ] Relation, quotient, setup, and trace columns are emitted natively in machine-major order.
- [ ] Ordinary additive local quotient identities hold independently;
  compression product quotients recompose partial products and their residuals
  vanish only after summation into the global lifted relation.
- [ ] No live doc claims digit decompositions add across machines.
- [ ] Partial-fold norms are independently priced and checked.
- [ ] Distributed commitment equals an independent dense machine-major commitment.
- [ ] Independent worker processes complete a distributed proof over the reference transport.
- [ ] No worker holds another worker's witness during distributed levels.
- [ ] Coordinator peak witness memory is sublinear in the union of worker witnesses before cutover.
- [ ] Communication counters contain no witness-sized collective before cutover.
- [ ] Distributed opening/fold equals an independent dense machine-major oracle.
- [ ] Summed local round polynomials equal the monolithic reference in every round.
- [ ] Setup and trace evaluations match dense native-layout references.
- [ ] W-to-1 cutover consumes hierarchical input without cross-machine witness reads.
- [ ] `W = 1` remains byte-identical.
- [ ] `W in {2,4,8}` passes layout, quotient, prove, serialize, and verify tests.
- [ ] Non-power-of-two local live lengths pass with virtual zero suffixes.
- [ ] Multi-group with `W > 1` remains fail-closed until its product tests land.
- [ ] Planner proof bytes equal runtime proof bytes for regenerated multi-chunk schedules.
- [ ] Distributed proof bytes equal the single-host machine-major oracle.
- [ ] Verifier setup-scan operation counts equal the single-chunk counts.
- [ ] Verifier benchmark gates in this spec pass for `W in {1,2,4,8}`.
- [ ] PR #294's universal digit-fast layout is not required or retained as a fallback.
- [ ] `cargo fmt -q` passes.
- [ ] `cargo clippy --all --message-format=short -q -- -D warnings` passes.
- [ ] `cargo test` passes.
- [ ] `rtk cargo nextest run --profile ci --no-default-features --features parallel,disk-persistence` passes.
- [ ] `./scripts/check-doc-guardrails.sh` passes.

## Required tests

1. Native dense relation times native witness equals the semantic relation.
2. Each ordinary local lifted relation recomposes from its quotient digits;
   each compression product quotient recomposes its local cyclic/negacyclic
   product difference.
3. Summing local residuals and lifted contributions yields the public global
   relation, including compression rows whose individual residuals are nonzero.
4. A fixture demonstrates that decomposing after summing differs from summing
   decompositions, preventing the old documentation error from returning.
5. Distributed commitment, opening, folding, trace, and setup paths match dense
   independent oracles.
6. Per-round local sum-check polynomials sum to the monolithic polynomial.
7. The cutover fold accepts W local inputs and emits one canonical output.
8. Malformed ownership, lengths, points, padding, and schedules reject without panic.
9. `W = 1` proof bytes match the ordinary preset.
10. Benchmark primarily against `origin/main`, and secondarily against PR #294:
    setup, commit, prove, verify, memory,
    communication bytes, and proof bytes.
11. Spawn-process test with premature exit, unexpected phase, and wrong-shape
    messages; malformed sessions fail closed without transcript divergence.
12. Coordinator-memory test proves that distributed levels do not materialize
    the union witness outside the explicit dense oracle.

## Documentation cutover

The implementation PR must update:

- `book/src/how/proving/distributed-prover.md`;
- `book/src/how/verifying/distributed-relation-verifier.md`;
- `specs/distributed-prover.md`;
- `specs/distributed-verifier-row-eval.md`;
- `specs/distributed-planner.md`;
- `specs/multi-group-batching.md`;
- `book/src/how/proving/sumcheck-stages.md` where ownership affects carried claims;
- `book/src/how/recursion.md` for the W-to-1 cutover.

The older specs must be marked superseded or narrowed to implemented historical
slices. They must not remain live with contradictory witness or quotient rules.
