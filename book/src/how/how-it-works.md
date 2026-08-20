# How it works

Akita opens a committed polynomial by repeatedly replacing a large hidden
table with a smaller committed witness. Every replacement comes with a proof
that the new witness carries the same opening claim and satisfies the required
commitment relations. The last witness is small enough for the verifier to
check directly.

That repeated reduction is the central idea behind the implementation. The
configuration decides the exact sequence of folds. The prover executes it. The
verifier derives the same sequence from public inputs and rejects any proof
whose shape or messages do not match.

This part of the Book explains that path from public configuration to final
acceptance. It is for contributors, integrators who need to understand the
proof boundary, and reviewers who want to connect each protocol claim to the
code that enforces it.

The same `batched_prove` and `batched_verify` APIs always use a folded schedule.
Every schedule contains a root fold and a terminal cleartext witness, with zero
or more recursive folds between them. The root may connect directly to the
terminal witness when no recursive fold is needed. Schedule selection rejects
a request when the audited fold domain contains no valid complete schedule.

## The complete proof in one view

An Akita opening moves through the following stages.

1. **Resolve the schedule.** The selected `CommitmentConfig` maps the workload
   shape to one generated schedule row. That row fixes every fold, matrix
   dimension, decomposition basis, opening method, and terminal profile.
2. **Prepare the setup and commitments.** The prover materializes the required
   prefix of the public setup. Each polynomial group is committed with the
   parameters assigned to that group.
3. **Bind the public statement.** The transcript absorbs the configuration,
   setup identity, schedule, commitment groups, opening points, and claimed
   evaluations before deriving challenges that depend on them.
4. **Run the root fold.** The prover prepares each requested opening, builds the
   first fold relation, and proves that relation with the scheduled sum-checks.
   The fold emits a smaller witness and a new opening claim.
5. **Run any recursive folds.** Each nonterminal successor authenticates the
   witness produced by its predecessor and repeats the same reduction. A fold
   may also authenticate a prepared setup prefix when setup offloading was
   selected.
6. **Check the terminal witness.** The final witness is sent in clear encoded
   form. The verifier checks its consistency, commitment relation, evaluation
   trace, and any scheduled norm bound directly. There is no further recursive
   commitment and no terminal sum-check cascade.

The verifier performs the same steps in the same transcript order. It does not
trust the proof to choose a schedule or describe its own lengths. Public
configuration and claims determine the expected shape before proof data is
decoded.

```text
polynomial groups + opening claims
                 |
                 v
       generated root schedule
                 |
                 v
        root relation and fold
                 |
                 v
      smaller committed witness
                 |
          zero or more recursive folds
                 |
                 v
      clear terminal witness check
                 |
                 v
              accept
```

## What one fold preserves

A fold is useful only if it shrinks the witness without weakening the claim.
Akita maintains three invariants at every nonterminal level.

### The opening claim remains attached to the witness

The current level starts with one or more claims that committed polynomials
evaluate to stated values at stated points. Its relation turns those claims
into one evaluation of the next witness. That value becomes the opening claim
consumed by the successor.

Multiple commitment groups may enter the root together. Each group keeps its
own commitment, point, values, and parameters. The root batches their checks,
then returns to the ordinary recursive path with one folded witness.

### The commitment relations remain binding

The prover cannot choose an arbitrary smaller witness. Each fold proves the
relations that connect source digits, inner commitment values, outer
commitments, opening digits, and the folded response. The proof may carry raw
commitment values or smaller compressed payloads, but both forms enforce the
same semantic commitment statements.

### Every size and method comes from the schedule

The schedule owns the ring dimensions for the A, B, and D matrix roles, the
number of blocks, the digit bases, the opening method, the proof shape, and the
terminal policy. Prover and verifier consume these values through the same
typed schedule structures. Unsupported geometry fails during schedule
resolution or validation rather than selecting an improvised protocol.

## The main branches in the protocol

The fold engine is shared, but a generated schedule can select among several
well-defined paths.

| Choice | What changes | Where to read |
| --- | --- | --- |
| Coefficient packing or evaluation trace | How an extension-valued opening is represented in the fold relation | [Fold path and field geometry](./proving/fold-path.md) |
| Raw or compressed commitment payload | Which commitment values appear on the wire and which extra physical rows are proved | [Raw and compressed realizations](./proving/akita-fold-realizations.md) |
| Direct or offloaded setup contribution | Whether the verifier scans setup now or a later fold authenticates a prepared setup prefix | [Setup offloading](./setup-offloading.md) |
| Coefficient or Euclidean response bound | Which range and norm checks certify the folded response | [Security model](./security.md) and [Sum-check stages](./proving/sumcheck-stages.md) |
| Dense or one-hot source | How the prover computes the commitment and source operations | [Setup and commitment](./commitment.md) |

These choices are not independent switches supplied by a proof. The generated
schedule contains one complete compatible assignment.

## Read this section by goal

### Understand the system first

Read these chapters in order:

1. [Architecture overview](./architecture.md) for crate ownership and the
   public `commit`, `prove`, and `verify` lifecycle.
2. [Configuration and planning](./configuration.md) for the generated schedule
   and the difference between offline search and runtime resolution.
3. [Setup and commitment](./commitment.md) for the public matrices and Ajtai
   commitment.
4. [The proving protocol](./proving/proving.md) for one fold from input claim to
   successor witness.
5. [Verification](./verification.md) for transcript replay, final checks, and
   malformed-input rejection.

### Implement or review a proof stage

Start with [The proving protocol](./proving/proving.md), then follow its links
to the relation, layout, and sum-check chapter for the stage being changed.
Pair the prover path with the matching page under Verification. The protocol
is complete only when both sides compute the same claim from the same ordered
messages.

### Review security boundaries

Read [Transcript and instance binding](./transcript.md), [Verification](./verification.md),
and [Security model](./security.md). Then trace each accepted norm bound and
matrix role back to the generated schedule and Module-SIS tables. The useful
review question is not merely whether an equation appears in the prover. It is
whether the verifier derives and enforces every public condition needed by
that equation.

### Review performance

Read [Optimizations](./optimizations.md) after the protocol path is clear. The
fast implementation changes table representations, arithmetic kernels, and
execution order. It does not define a second proof system. Scalar references,
schedule checks, and verifier equations remain the correctness boundary.

## Chapter map

| Chapter | Main question |
| --- | --- |
| [Architecture overview](./architecture.md) | Which crate owns each concept, and how do the public calls flow through them? |
| [Configuration and planning](./configuration.md) | How does one public workload select an exact generated schedule? |
| [Setup and commitment](./commitment.md) | How do public matrices bind a polynomial group? |
| [Transcript and instance binding](./transcript.md) | Which public facts are absorbed before each Fiat-Shamir challenge? |
| [The proving protocol](./proving/proving.md) | How does one nonterminal fold create and authenticate its successor witness? |
| [The distributed prover](./proving/distributed-prover.md) | Which fold computations can remain local to independent machines? |
| [Recursion and proof shape](./recursion.md) | How are root, recursive, and terminal records connected? |
| [Setup offloading](./setup-offloading.md) | How can a prepared setup commitment replace a large online verifier scan? |
| [Verification](./verification.md) | How does the verifier replay every level and reject malformed input without panicking? |
| [Security model](./security.md) | Which assumptions, norm regimes, and generated bounds support acceptance? |
| [Optimizations](./optimizations.md) | How does the implementation make the same protocol fast? |

## Where the lifecycle enters the code

The public orchestration lives in `crates/akita-pcs/src/scheme/`. The prover
walk begins in `crates/akita-prover/src/protocol/core/prove.rs`. The verifier
mirror begins in `crates/akita-verifier/src/protocol/core/verify.rs`.

Those files should remain orchestration layers. Mathematical rules belong to
their canonical layout, algebra, schedule, sum-check, or verifier helpers. A
reviewer should be able to follow the lifecycle here, enter one of those three
paths, and find one implementation owner for every rule.
