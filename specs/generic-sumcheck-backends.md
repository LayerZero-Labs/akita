# Spec: Generic Sumcheck Compute Backends

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-20 |
| Status | active |
| PR | [#423](https://github.com/LayerZero-Labs/akita/pull/423) |
| Supersedes | |
| Superseded-by | |
| Book-chapter | book/src/roadmap/compute-backends.md |

## Summary

Akita's sumcheck drivers currently call relation-owned scalar prover objects.
This works on CPU, but the object owns both protocol state and witness storage.
It cannot express a borrowed compact witness, device-resident tables, a fused
bind-and-compute kernel, or one launch for a whole front-loaded batch without
changing the relation itself.

This specification separates the protocol from the computation. The protocol
driver continues to own proof shape, transcript order, batching coefficients,
master-point suffix alignment, and verifier recurrence. Unequal-round
eq-factored groups need a future proof format because they do not share one
accumulated equality factor. This specification checks that geometry but does
not assign it a transcript schedule. A source exposes a
borrowed group view in its native representation. A backend prepares that view
and a checked relation program into a transcript-free round executor. CPU,
packed CPU, Metal, CUDA, and mixed-device implementations may use different
storage and kernels behind that same boundary.

The first implementation is a reference CPU path plus one existing Akita
sumcheck. Optimized CPU work from the closed, unmerged sumcheck-kernel branch is
an implementation source, not the public abstraction. Aerie may continue its
direct Falcon implementation in parallel. Its later Akita migration must
replace source and executor plumbing without changing Falcon proof bytes or
transcript labels.

## Intent

### Goal

Build one protocol-neutral sumcheck compute hierarchy that accepts native
witness views, supports standard front-loaded batches and eq-factored batches
that share one equality factor, and lets CPU and accelerator backends fuse
round work without owning protocol or transcript semantics. It also preserves
checked unequal-round eq geometry for a later protocol design.

### Invariants

1. The protocol layer alone owns transcript absorbs and squeezes, proof-object
   construction, challenge order, batch coefficients, suffix alignment, and
   the standard or eq-factored claim recurrence.
2. The verifier remains independent of prover backends, prepared compute
   state, source layouts, and device crates.
3. A checked batch plan derives every shorter instance point as a suffix of one
   master challenge point. Callers do not supply independently aligned points.
4. Eq-factored linear state is derived and advanced by the protocol engine.
   A kernel receives the checked round values it needs but cannot choose or
   mutate the recurrence.
5. A backend operation consumes a borrowed, source-typed group view. The
   public boundary does not require `Vec<E>`, a dense multilinear table, host
   residency, or a universal evaluation-table wrapper.
6. The group operation is canonical. A singleton is a group of length one;
   there is no required default loop over singleton operations.
7. Backends return typed errors for unsupported shapes, invalid prepared state,
   device failures, and arithmetic overflow. Verifier-controlled input cannot
   reach a backend panic.
8. Backend choice, source representation, packing, and scheduling do not alter
   proof bytes, transcript labels, accepted statements, or final claims.
9. A generic relation program has a reference interpreter. An optimized kernel
   must match it over deterministic fixtures before it can replace it.
10. Dynamic dispatch, if used, is limited to a constant number of calls per
    active group per round. It does not occur per row, factor, table cell, or
    witness element.
11. The protocol has no fixed maximum batch length. Callers and backends may
    enforce explicit resource limits, but those limits are not transcript or
    proof-format constants.

### Non-Goals

- This specification does not implement a production Metal or CUDA backend.
- It does not wait for the generic hierarchy before Aerie produces Falcon
  correctness and performance numbers.
- It does not make proof verification asynchronous or backend-dependent.
- It does not expose CPU packing layouts, GPU buffers, command queues, or
  relation-specific caches as protocol types.
- It does not promise that every relation runs through an interpreter. Hot
  relations may use fused kernels selected during preparation.
- It does not redesign PCS commitment, opening, tensor, or ring-switch kernels.
- It does not change the packed arithmetic decisions in
  [`packed-sumcheck.md`](packed-sumcheck.md). That spec becomes one optimized
  CPU implementation track under this boundary.

## Evaluation

### Acceptance Criteria

- [ ] `akita-sumcheck` has checked standard and eq-factored batch plans that
  reject empty batches, inconsistent degree declarations, invalid round
  counts, invalid suffix offsets, and malformed proof shapes before execution.
- [ ] The standard protocol driver can prove a front-loaded batch through
  grouped round executors while producing the same proof bytes and challenges
  as the reference scalar driver on deterministic fixtures. The eq-factored
  driver can combine same-round groups into the existing proof type and rejects
  unequal-round execution before transcript work.
- [ ] A source trait exposes a group view through a generic associated type.
  Dense, compact integer, sparse or indexed, and externally owned views can be
  implemented without conversion to `Vec<E>`.
- [ ] One canonical prepare-group operation creates a transcript-free executor.
  The engine can start every active group for a round before collecting their
  messages.
- [ ] An executor can fuse ingestion of the previous challenge with computation
  of the next message and can finish the final binding without an extra table
  pass.
- [ ] The first generic product-sum program has a scalar reference interpreter,
  checked degree metadata, and at least one fused CPU specialization.
- [ ] Standard differential tests cover one, many, unequal round counts,
  non-power-of-two logical batch sizes, terminal-only instances, and tampered
  round or terminal claims. Eq-factored tests cover same-factor grouped
  execution, terminal-only batches, unequal-round suffix derivation, and typed
  rejection where the existing proof type cannot represent the plan.
- [ ] A proof-session test proves that scratch and prepared resources are
  released on success and error and cannot be reused with mismatched field,
  relation, or setup identity.
- [ ] The Akita prover stack can route sumcheck as its own operation cluster
  while retaining the uniform-CPU convenience constructor.
- [ ] One current Akita sumcheck is migrated without a compatibility wrapper or
  parallel old/new runtime path. The old relation-owned compute methods are
  removed for that relation.
- [ ] Benchmarks report reference scalar, optimized CPU, and dispatch overhead
  separately, with command, commit, target, hardware, input shape, proof-byte
  equality, peak resident memory, and throughput.
- [ ] `cargo fmt -q`, workspace clippy, affected crate tests, documentation
  guardrails, and live-spec reference checks pass.

### Testing Strategy

Protocol tests use deterministic transcripts and compare standard proof bytes,
sampled challenges, per-round claims, and terminal claims against the current
scalar driver. Standard groups begin at different master rounds so the tests
exercise suffix alignment directly. Eq-factored tests independently recompute
the shared linear factor and scaled recurrence instead of trusting prover
state. Separate geometry tests cover unequal-round eq suffixes and confirm that
execution rejects them before any transcript absorb or squeeze.

Source tests instantiate the same relation over dense extension values,
compact signed digits, and a deliberately non-contiguous borrowed view. The
reference interpreter is the oracle for every source. A fake asynchronous
backend records `start_round` and `finish_round` calls to prove that all active
groups are submitted before any result is awaited.

Backend differential tests compare scalar, packed CPU, and each accelerator on
identical checked plans. Device tests must cover unsupported shapes and injected
submission failures. Property tests vary logical batch length independently of
hypercube padding and backend launch geometry.

The first migrated Akita relation keeps an independent end-to-end verifier
test. Removing the old compute path is part of the migration test, not deferred
cleanup.

### Performance

The abstraction must preserve these properties:

- no required witness copy or dense extension-field materialization at the
  public source boundary;
- one preparation per same-shape group, not one preparation per logical
  instance;
- one group-level start and finish per active group per round, with no
  per-element virtual dispatch;
- all active groups may be submitted before the transcript waits for the
  combined round message;
- challenge binding and next-round accumulation may share one pass;
- round messages are the only required device-to-host traffic during the loop;
- scratch allocation is proof-scoped and reused across rounds;
- batch coefficients are available to the backend before its first scan so a
  fused backend can accumulate the combined message directly;
- compact, sparse, virtual, packed, and device-native inputs remain in that
  representation until a chosen kernel requires otherwise.

The reference CPU implementation establishes correctness, not the performance
ceiling. The first fused CPU migration must measure dispatch overhead and show
that it is below one percent for the production benchmark shape or below the
measurement noise floor. A backend that falls back to the interpreter must
report that choice in benchmark diagnostics. Production APIs do not silently
fall back across devices after proof execution begins.

## Design

### Ownership

`akita-sumcheck` owns:

- proof and verifier types;
- standard and eq-factored format rules;
- checked batch, group, and round plans;
- structural relation programs and their scalar interpreter;
- source-view, prepare-group, and round-executor traits;
- the transcript-driving engine and terminal-claim checks.

`akita-prover` owns:

- the adapter from `OperationCtx` and prepared Akita setup into a sumcheck
  operation context;
- the sumcheck cluster in `ProverComputeStack` and `LevelProveStacks`;
- Akita relation descriptors and source implementations;
- the CPU kernels and their protocol-specific call sites.

Device crates own buffers, pipelines, command submission, synchronization, and
backend-private compiled programs. They implement Akita-owned operation traits;
Akita protocol crates do not depend on device crates.

This split avoids a dependency from `akita-sumcheck` back to `akita-prover`.
Downstream protocols may use the generic engine with their own backend context.
Akita still validates its prepared setup through `OperationCtx` before creating
any executor.

### Control Flow

```text
protocol relation + witness
        |
        v
checked batch plan ---- owns master rounds, suffixes, coefficients, eq state
        |
        +---- source.group_view(checked group)
        |             |
        |             v
        +---- context.prepare_group(view, relation program, plan)
                              |
                              v
                    transcript-free round executor
                              |
            start all groups  |  finish all groups
                              v
                    combined round message
                              |
                              v
                  transcript absorb + challenge
```

For round zero, `start_round` computes from the initial state. For later rounds,
it receives the previous challenge and may bind that challenge while computing
the next message. After the final challenge, `finish_binding` produces terminal
claims without another round message.

The driver starts every active executor before finishing any executor. A CPU
executor may complete during `start_round`; a device executor may only submit
work. This two-phase object-safe boundary permits overlap without putting async
types, device events, or transcript access into the protocol API.

### Protocol Formats

`StandardFormat` checks `g(0) + g(1) = previous_claim` and advances the claim by
evaluating the transmitted round polynomial at the sampled challenge.

`EqFactoredFormat` proves `s(X) = l(X) q(X)`. The engine owns the linear eq
factor, supplies its checked `(l(0), l(1))` values to the round executor, and
advances the scaled claim. The executor only computes the compact message for
`q`. It cannot provide a competing factor state.

The existing `EqFactoredSumcheckProof` can combine several executor groups only
when every group starts at master round zero and therefore shares the same
accumulated `l`. The engine adds those group messages and absorbs one existing
proof message per round. A shorter suffix group starts with a different
accumulated factor, so its compact `q` message cannot be added to the longer
group's message. The checked plan may represent that geometry, but execution
returns a typed unsupported-schedule error before transcript work. This
specification does not define an in-memory frame, serializer, or verifier replay
for unequal-round eq batches.

Formats define message encoding and verifier recurrence. They are not backend
implementations. Adding a format requires prover and verifier logic together;
adding a backend does not.

### Batch Geometry

A `CheckedSumcheckBatch` is created from logical instances before transcript
work. It records:

- the master number of rounds;
- each instance's local rounds and `master_rounds - local_rounds` suffix offset;
- format and degree metadata;
- stable logical order;
- front-loaded batching coefficients;
- groups of instances that share source layout, relation program, local round
  count, and backend placement.

An instance becomes active at its suffix offset. The final local challenge
point is always a borrowed suffix of the master point. Terminal-only instances
have zero local rounds and participate only in the checked terminal reduction.
For eq-factored execution through the current proof type, all groups must have
offset zero, unless every group is terminal-only. Logical batch length is
independent of padding used inside a source or kernel.

The engine combines group round messages in stable logical order. Parallel or
device completion order cannot change claim order, error attribution, or
diagnostics.

### Source Views

The source contract follows Akita's existing root source hierarchy:

```rust,ignore
pub trait SumcheckSource<F, E, R> {
    type GroupView<'a>
    where
        Self: 'a;

    fn group_view<'a>(
        &'a self,
        group: &CheckedSumcheckGroup<R>,
    ) -> Result<Self::GroupView<'a>, AkitaError>;
}
```

`R` is a typed structural relation descriptor. A group view may borrow several
logical instances and may describe dense extension tables, base-field tables,
signed digits, sparse indices, virtual factors, device buffers, or a composite
of these. The trait does not require iteration because forcing a host iterator
would exclude device-native sources and would make gather strategy part of the
public contract.

Sources validate logical shape when creating the view. Backends validate
representation and capability when preparing it. Neither step absorbs a
transcript challenge.

### Relation Programs

The first general relation vocabulary is a checked weighted product-sum:

```text
q(X) = sum_t weight[t] * product_j oracle[factor[t][j]](X)
```

It records oracle roles, term weights, factor lists, claimed degree, output
claim roles, and whether the surrounding protocol uses standard or
eq-factored messages. It does not contain witness buffers or backend handles.
This form covers a broad class of range, product, and consistency sumchecks,
including the planned Falcon digit-product checks.

The scalar interpreter evaluates the structural program over a source that can
provide scalar oracle values. Optimized backends inspect and lower the program
once during `prepare_group`. They may select a handwritten fused kernel, compile
a device pipeline, or reject an unsupported shape. They do not branch on term
shape for every witness element.

This is deliberately smaller than a registry with one backend slot per
protocol relation. Jolt's backend/session/round-scheduler separation is useful,
but its relation-wide backend registry is not the Akita public boundary. New
relations should usually be data in the product-sum program. A dedicated
operation trait is justified only when a relation cannot be represented
without losing material efficiency.

### Prepare Group and Round Executor

The public operation has one group method. The exact Rust lifetimes may be
adjusted during the compile-first slice, but the ownership boundary is fixed:

```rust,ignore
pub trait PrepareSumcheckGroup<S, F, E, R> {
    type Executor<'a>: SumcheckRoundExecutor<E>
    where
        Self: 'a,
        S: 'a;

    fn prepare_group<'a>(
        &'a self,
        source: S,
        relation: &'a R,
        plan: &'a CheckedSumcheckGroup<R>,
        session: &mut SumcheckProofSession,
    ) -> Result<Self::Executor<'a>, AkitaError>;
}

pub trait SumcheckRoundExecutor<E> {
    fn start_round(
        &mut self,
        round: CheckedLocalRound,
        previous_challenge: Option<E>,
        context: CheckedRoundContext<'_, E>,
    ) -> Result<(), AkitaError>;

    fn finish_round(&mut self) -> Result<GroupRoundMessage<E>, AkitaError>;

    fn finish_binding(
        &mut self,
        final_challenge: Option<E>,
    ) -> Result<GroupTerminalClaims<E>, AkitaError>;
}
```

`prepare_group` may borrow sources and immutable prepared state for the
executor lifetime. It cannot retain the mutable session borrow. It takes owned
or shared session leases so the engine can prepare several live executors.

The engine may erase executors after preparation so different backends can
share one batch. Erasure is at group granularity. A homogeneous caller may keep
the associated executor type statically dispatched.

`start_round` cannot be called twice without `finish_round`. `finish_round`
cannot be called before a successful start. `finish_binding` consumes the final
logical state and cannot be followed by another round. The concrete API must
encode these states where practical and return typed errors otherwise.

### Proof Session

A proof session owns scratch arenas, command buffers, temporary device leases,
compiled-program references, and outstanding round submissions. It is not
serialized and does not cross into verification.

The session is opened after plans and prepared setup are validated and before
the first executor is created. Executors receive scoped leases rather than
unrestricted mutable access to a shared allocator. Cleanup is deterministic on
success and error. A session is keyed by field, backend identity, and any
prepared-setup identity the owning context requires.

The first CPU implementation may use a simple reusable arena. The API must not
assume that a submission completes before `start_round` returns.

### Akita Compute Stack

Sumcheck becomes a fifth operation cluster beside commit, opening, tensor, and
ring switch:

```text
ProverComputeStack<Commit, Opening, Tensor, RingSwitch, Sumcheck>
```

`LevelProveStacks` gains a `Sumcheck` associated backend and the stack gains a
`sumcheck()` operation context. `UniformProverStack<B>` still binds every
cluster to `B`. The delegating CPU facade implements the new operation in the
same way it implements existing clusters.

This cluster addition occurs only after the standalone `akita-sumcheck`
boundary compiles and has differential tests. It must be a direct cutover, not
an optional side channel that protocol code may bypass.

### CPU, Metal, and GPU Implementations

The reference CPU backend interprets the product-sum program and accepts simple
borrowed scalar views. It is the correctness oracle.

The optimized CPU backend may reuse the table layouts, packed folds, delayed
reduction, and measurements from `packed-sumcheck.md` and closed PR #368. Those
details stay private to CPU executors. The old branch's public
`EvaluationTable` and extension-field-only operation trait are not carried
forward because they force storage and arithmetic choices on every backend.

Metal and CUDA backends may keep witnesses resident, compile one pipeline per
checked program shape, submit all active groups, and return only round messages
and terminal claims. They use the same source and executor contracts.
Device-specific capabilities are checked during preparation. Hybrid scheduling
assigns whole groups to backends before a round begins; it does not move a
group's mutable state between devices in the first implementation.

### Aerie Migration Boundary

The direct Aerie Falcon path is intentionally allowed to copy and specialize
Akita range logic now. It should still keep four internal layers:

1. checked protocol and batch plan;
2. borrowed witness views;
3. transcript-free round computation;
4. typed terminal claims.

When the Akita hierarchy is ready, Aerie maps layers 2 and 3 to Akita source
views and executors. Layer 1 remains Falcon-owned because it fixes transcript
and proof semantics. Layer 4 remains the typed Falcon handoff to later protocol
blocks. This limits the migration to compute plumbing and prevents the generic
work from blocking current benchmarks.

### Alternatives Considered

**Make `EvaluationTable<E>` the public backend boundary.** Rejected because it
forces dense extension-field host storage and makes compact, sparse, virtual,
and device-native witnesses second-class.

**Put every relation in one `JoltBackend`-style registry.** Rejected as the
public Akita design because each new relation changes a central trait and
backend type. Akita takes Jolt's session and fused round-control ideas while
using structural programs and source-typed group operations.

**Let relation prover objects own compute and challenge ingestion.** Rejected
because protocol and storage remain coupled, group fusion is accidental, and a
device backend cannot overlap submissions under one driver.

**Use only statically dispatched executors.** Rejected for mixed backends in one
front-loaded batch. Static dispatch remains available inside homogeneous
groups; optional erasure occurs once per group.

**Move the transcript into the backend.** Rejected because it would make proof
semantics backend-dependent and prevent a simple CPU/device differential test.

**Wait for the generic hierarchy before implementing Falcon.** Rejected because
it delays correctness and performance measurements. The direct path and the
generic path proceed independently and converge at the source/executor seam.

## Documentation

The durable architecture belongs in
`book/src/roadmap/compute-backends.md` after the first runtime cutover. Until
then this active spec owns the design. `docs/compute-backends.md` points to this
work for the current deferred sumcheck boundary. The live spec index,
`specs/PRUNING.md`, and `scripts/check-spec-references.sh` include this file.

When the boundary and one production migration ship, mark this spec
`implemented`, fold the stable API into the Book, and archive the spec under the
current quarter. `packed-sumcheck.md` remains live until its separate optimized
CPU acceptance criteria are complete.

## Execution

Work proceeds in reviewable slices:

1. **Architecture contract.** Land this spec and synchronized documentation.
2. **Protocol plans and executor shell.** Add checked standard and eq-factored
   batch geometry, object-safe two-phase round control, a fake asynchronous
   executor, and transcript-order tests. Standard execution supports unequal
   rounds. Eq-factored execution combines groups that share the full equality
   factor into the existing proof type and rejects other checked plans before
   transcript work. Do not define a new eq proof format or migrate a production
   relation.
3. **Source and reference program.** Add the source GAT, product-sum program,
   scalar interpreter, proof session, and dense plus compact fixtures.
4. **Fused CPU pilot.** Select one current Akita sumcheck, implement a fused CPU
   executor, prove byte equality, benchmark it, and remove that relation's old
   compute path.
5. **Compute-stack cutover.** Add the fifth stack cluster, uniform and delegated
   CPU wiring, runtime capability tests, and direct protocol routing.
6. **Optimized CPU migration.** Port useful packed and compact kernels from the
   closed branch behind the new executor. Keep storage plans private.
7. **Accelerator pilot.** Implement one device executor and deterministic CPU
   differential test before broad kernel coverage or hybrid scheduling.
8. **Aerie adapter.** Replace the direct Falcon round executor and source
   plumbing while preserving its protocol, proof, transcript, and terminal
   claim types.

Each implementation slice ends with tests, a benchmark or explicit reason one
is not yet meaningful, and removal of any superseded path in that slice.

## References

- [`packed-sumcheck.md`](packed-sumcheck.md), the approved optimized CPU track.
- [`heterogeneous-group-source-contracts.md`](heterogeneous-group-source-contracts.md),
  the existing source-typed group boundary.
- [`akita-compute-backend-metal.md`](akita-compute-backend-metal.md), the active
  Metal implementation track.
- `docs/compute-backends.md`, current Akita backend ownership rules.
- `crates/akita-sumcheck/src/traits.rs`, current relation-owned prover traits.
- `crates/akita-sumcheck/src/batched_sumcheck.rs`, current front-loaded batch
  driver.
- `crates/akita-prover/src/compute/{backend,kernels,poly,stack}.rs`, the existing
  setup, source, operation, and heterogeneous stack hierarchy.
- a16z/Jolt commit `f823a9f85`, especially
  `crates/jolt-kernels/src/backend.rs` and
  `crates/jolt-sumcheck/src/prover.rs`, for backend, proof-session, fused-round,
  and scheduler precedent.
- Closed, unmerged Akita PR #368, as a source of CPU kernel code and benchmark
  evidence only.
