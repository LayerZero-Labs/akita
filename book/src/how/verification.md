# Verification

How the verifier replays the proof level by level, and the no-panic contract
that governs every verifier-reachable line.

## Reading this section

The verifier has several checks because one recursive fold must bind a public
opening claim, a digit witness, a ring relation, and the public setup at the
same time. Read the chapters in this order:

1. [Matrix evaluation at a point](./verifying/matrix_evaluation.md)
   defines the physical rows and columns.
2. [The Stage 2 fused check](./verifying/stage2.md) shows how the relation,
   range image, and schedule-selected opening method share one final witness
   evaluation.
3. [Evaluation trace](./verifying/evaluation_trace.md) explains the trace-based
   opening method. [Root fold and ring switch](./proving/root-fold-ring-switch.md)
   explains subring coefficient packing.
4. [Setup contribution and Stage 3](./verifying/setup_contribution.md) explains
   direct setup evaluation and recursive setup offloading. Read the
   [setup offloading overview](./setup-offloading.md) first if you want the
   complete planner, preprocessing, and recursive handoff.
5. [The distributed relation verifier](./verifying/distributed-relation-verifier.md)
   explains exact dyadic ownership and unequal chunks.
6. [Terminal verification](./verifying/terminal.md) explains the final direct
   checks after recursion stops.

The proving section owns the derivation of the ring relations and the logical
multi-group layout. These verifier chapters start from those relations and
explain how one verifier replays them at sampled points. This separation keeps
the mathematical definition and the optimized verifier implementation from
becoming competing sources of truth.

The book describes shipped behavior. The files under `specs/` record design
history and pending changes, so they may describe alternatives that are not
active in the verifier.

## Verifier data flow

For each nonterminal level, the verifier follows this order:

```text
validated schedule and setup
        |
        v
fold replay and outgoing witness binding
        |
        v
ring switch: sample alpha, tau0, and tau1
        |
        +--> prepare relation matrix evaluator
        +--> prepare compression weights
        +--> prepare scheduled opening terms
        |
        v
Stage 1: digit range product
        |
        v
Stage 2: range image + relation + opening claim
        |
        v
Stage 3: setup product when the setup claim is deferred
        |
        v
next recursive level or terminal direct checks
```

Preparation validates public geometry once. The final point kernels then use
typed prepared state and return errors for any remaining mismatch.

## Per-level replay

`batched_verify` (in `crates/akita-verifier/src/protocol/core/verify.rs`) receives
a validated `TrustedScheduleCatalog`. The catalog is a trusted verifier
parameter. It is loaded before proof parsing and is not read from the proof.

At a high level:

1. **Bind the instance** and absorb the opening batch shape into the transcript.
2. **Resolve the exact trusted row** named by
   `OpeningScheduleSelection`. Artifact identity and row digests are checked
   before the
   ordered `GroupCommitPhaseParams` values are compared with the resolved row.
   The verifier never runs planner search.
3. **Replay the structural folds** in `protocol/core`: the root fold followed by
   every recursive fold, using the schedule-selected `CommittedGroupParams`.
4. **Check the terminal witness directly** against its predecessor-bound `t`
   state. The terminal relation is `consistency | A`; it has no outer `u`, B
   block, D block, or quotient sumcheck. If the terminal A matrix uses an L2
   route, the verifier also computes the decoded response's exact integer
   squared norm and compares it with the scheduled cap.

At each nonterminal fold, the verifier checks fixed 128-byte `p_H` and `p_F`
payload shapes, reconstructs the B, D, F, and H relation right hand sides, and
folds the compression relations at their native ring dimensions. It derives
the negative-binary support from `WitnessLayout` and evaluates the stage-1
equality table restricted to those intervals; compression roles never enlarge
or shrink the ordinary A/B/D common address block.

The same validated fold parameters select the Stage-2 ring-relation
realization. `QuotientLift` evaluates the factored ordinary relation and its
explicit quotient spans. `ReducedEvaluation` instead evaluates terminal
signed-wrap residue kernels for the A/B/D rows and the separately owned F/H
compression program; its witness layout contains no ordinary or compression
quotient spans. Both paths close the same schedule-bound native-ring statement
and feed the same next-witness opening into the successor. There is no
proof-controlled mode bit or fallback path.

Schedule validation admits reduced evaluation only as a monotone suffix from
absolute level 2, with evaluation-trace openings and direct setup
contribution. It rejects coefficient packing, an incoming setup prefix,
deferred Stage 3, or a later return to quotient lifting before transcript
replay. Raw versus compressed payload and `L∞` versus `L2` response security
remain independent choices within that admitted suffix.

An L2 fold at D64 or D128 also replays operator norm rejection from the
transcript. The schedule fixes the sparse challenge family and both the true
subset threshold and strict integer threshold. A challenge is accepted only
when the integer interval calculation proves that every spectral magnitude is
within the strict threshold.

Root replay reads each commitment group's point directly from
`PolynomialGroupClaims`.
The verifier prepares the per-group relation and extension-opening factors from
that complete point, without reconstructing a common point.
When EOR is required, the verifier samples an early coefficient vector and
replays one ordinary sumcheck for the combined opening claims. The proof keeps
the individual terminal claims. The verifier checks their early combination
against the sumcheck terminal value and absorbs them. It then absorbs the
complete opening payload before sampling the independent application
coefficients that bind those terminal claims to the committed witness.

At a recursive boundary, Stage 2 supplies the next-witness claim
`(stage2_point, stage2_next_w_eval)`.
Stage 3 independently proves the setup product and supplies
`(stage3_setup_point, stage3_setup_prefix_eval)`.
The successor consumes these as separate witness and setup groups.
Stage 3 does not re-randomize, project, or serialize the witness claim.

The terminal `A * z` check accepts exactly the signed-i16 coefficient class.
Decoded coefficients outside `[-32768, 32767]` are rejected before arithmetic;
there is no alternate i8 or balanced-radix verifier path. The exact
CRT-capability selector keeps the base profile when
`2 * width * D * floor(q/2) * 32768 < product(base primes)` and otherwise adds
the 12289 i16 tail. A schedule whose accumulation exceeds both profiles is
rejected as an invalid setup.

The verifier warms the strongest representation selected by the validated
terminal schedule before transcript replay. Prepared forms are derived from
the coefficient setup, keyed by ring dimension, and never serialized. Groups
share one base prefix; its optional tail is only as long as the largest
tail-requiring group. Thus a base-only schedule never constructs the tail, and
a larger base-only group cannot unnecessarily extend one required by a smaller
group. Shape and setup-prefix checks happen before either kernel indexes
prepared state.

The verifier never constructs prover-only polynomial backends or setup expansion
kernels.

### Schedule and profile admission

Verifier admission uses one canonical row audit. Before setup access or proof
replay, it:

1. validates the trusted artifact family, protocol epoch, active policy, and
   runtime challenge hooks when the catalog is installed;
2. resolves the fixed-width row digest by bounded lookup;
3. checks every ordered public committed profile against the resolved row;
4. re-audits every A, B, D, recursive, and terminal SIS matrix against the
   canonical security table, role, modulus, rank, width, bound, and ring
   dimension;
5. checks root, recursive, setup-prefix, challenge, witness-partition, terminal
   response, and full terminal norm cap geometry;
6. checks that the schedule fits the setup field capacity; and only then
7. binds the instance descriptor and replays the proof.

Private polynomial representations and honest-prover witness models are absent
from this path. The public selection is one digest of the exact ordered
profiles and expanded row. Catalog policy metadata is validated when the
trusted parameter is loaded. Unknown proof row digests reject. There is no
proof field that can replace a catalog row or ask the verifier to decode
schedule content.

## The verifier no-panic contract

Verifier-reachable execution is a **no-panic boundary**.
Malformed verifier-facing proof, setup, schedule, public claim, opening point,
commitment, direct witness, or transcript input must be rejected with
`AkitaError` or `SerializationError`, never by panicking.
`AkitaError` has one canonical definition in `akita-error`.

### Crates in scope

- `akita-verifier`
- Verifier-reachable paths in `akita-types`, `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, and verifier-used `jolt-field` code
- `akita-config` (every `CommitmentConfig` method reachable from `batched_verify`)
- `akita-schedules` artifact identity, row resolution, and canonical
  resolved-row audit paths

### Rules for contributors

1. Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier boundary has validated the invariant.
2. Use `akita_error::checked` for reusable exact `usize` formulas. The functions return `Option`, so the caller maps failure to the `AkitaError` variant that matches the protocol boundary. A direct standard library `checked_*` call remains appropriate for one local operation.
3. Do not use wrapping or saturating arithmetic for exact sizes and indices. Reject arithmetic that cannot represent the required geometry.
4. Strengthen validation at deserialization, setup construction, schedule selection, `CommittedGroupParams` construction, and verifier API entry points rather than sprinkling checks through hot loops.
5. Prover-only panics are acceptable when not reachable from verifier paths.

Maintainer mirror: [`docs/verifier-contract.md`](../../../docs/verifier-contract.md).
Historical audit evidence: [`docs/verifier-panic-audit.md`](../../../docs/verifier-panic-audit.md).
