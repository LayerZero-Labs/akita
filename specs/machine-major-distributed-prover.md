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
   partial `u_j/v_j`, and its partial claimed value. Short images and scalars are
   summed.
2. **Transcript synchronization.** The coordinator absorbs the sums and
   broadcasts the transcript-derived challenge material.
3. **Fold grinding.** For each candidate nonce, every worker computes only its
   `z_j` and reports acceptance. A nonce is accepted iff all workers accept.
   The coordinator commits the minimum accepted nonce and broadcasts it.
4. **Local relation and quotient.** Each worker builds `M_j`, `h_j`, and `r_j`
   without communication.
5. **Next commitment.** Each worker commits its local recursive blocks using
   native setup columns. Short commitment contributions are summed and absorbed.
6. **Range tree.** Workers build local subtrees; only local roots cross the
   network.
7. **Sum-check stages.** Workers submit one small polynomial per round. The
   coordinator sums, absorbs, and broadcasts the next challenge before workers
   advance.
8. **State handoff.** Each worker retains its local next witness. No witness
   gather occurs while the next output level remains distributed.

Stage 3's public setup term must not be computed redundantly by every worker.
Partition its setup-index range across workers or assign that public term to one
worker, then sum it with the witness-local round contributions.

### Communication contract

Before cutover, communication is limited to:

- `u/v` and next-commitment vectors;
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

### Types

Replace the flat/chunk-list ambiguity with explicit ownership types:

```text
DistributedRecursiveWitness
MachineWitnessLayout
DistributedWitnessLayout
```

`MachineWitnessLayout` owns local `z/e/t/r` ranges, local live length, virtual
opening stride, block count, and block length. `DistributedWitnessLayout` owns
the ordered machine layouts, common-shape validation, relation order, total live
length, and total opening domain.

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

## Implementation sequence inside the PR

These are review checkpoints in one spec-and-implementation PR, not separate
protocol variants.

### S0: Layout authority

Add the hierarchical layout and dense independent index oracle. Freeze the
machine-major order, local block-fast order, virtual zero suffix, input/output
machine counts, and descriptor bytes before changing prover behavior.

### S1: Native local relations and quotients

Teach the single-host harness to build `W` local witnesses, local partial
right-hand sides, and local quotient segments. Check every local lifted identity
and their global sum. Remove the shared quotient tail from distributed layouts.

### S2: Recursive consumers

Move opening, folding, commitment, relation-column, trace, setup, and terminal
kernels from a monolithic recursive source to native local witnesses. No caller
may flatten and reinterpret the batch.

### S3: Distributed sum-check oracle

Run the sum-check prover as a sum of local round-polynomial providers. Establish
round-by-round equality with an independent dense global table before optimizing
structured evaluators.

Add the process-level collective runtime in the same slice. The existing
single-host implementation remains only as an oracle and fallback for `W = 1`;
it is not the shipped distributed path.

### S4: Planner and cutover

Price local quotients and partial-fold bounds, separate input/output ownership,
regenerate schedules, and implement the hierarchical-input to single-output
cutover fold.

### S5: Cleanup and documentation

Delete the superseded flattened chunk layout, stale shared-tail helpers, and any
universal digit-fast fallback. Fold the final behavior into the Book and update
the superseded spec headers in the same PR.

## Acceptance criteria

- [ ] New machine-major layout types are the only distributed layout authority.
- [ ] Each machine owns one contiguous local `RecursiveWitnessFlat`.
- [ ] Local recursive indexing remains block-fast.
- [ ] No distributed level concatenates machine buffers into one monolithic recursive view.
- [ ] Relation, quotient, setup, and trace columns are emitted natively in machine-major order.
- [ ] Local quotient identities hold independently and sum to the global lifted relation.
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
2. Each local lifted relation recomposes from its local quotient digits.
3. Summing local lifted relations yields the public global relation.
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
