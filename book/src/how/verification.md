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

`batched_verify` (in `crates/akita-verifier/src/fold/verify.rs`) receives
a `TrustedScheduleCatalog<Cfg>` already validated and bound to the verifier's
exact config. The catalog is a trusted verifier parameter. It is loaded before
proof parsing and is not read from the proof.

`batched_verify` builds a standalone transcript state and calls
`batched_verify_embedded`, which a caller with its own transcript can call
directly. On any error, the embedded entry poisons the borrowed state before
it returns, so every later proof read and the end-of-proof check fail. On
success it returns a `PendingEof`. Acceptance requires
`PendingEof::check_eof`, which checks that the whole proof was consumed;
`batched_verify` calls it right away, and an embedding caller calls it after
its own last message. See
[Running inside a caller's transcript](./transcript.md#running-inside-a-callers-transcript).

At a high level:

1. **Bind the instance** and absorb the opening batch shape into the transcript.
2. **Resolve the exact trusted row** named by
   `OpeningScheduleSelection`. Artifact identity and row digests are checked
   before the
   ordered `GroupCommitPhaseParams` values are compared with the resolved row.
   The verifier never runs planner search.
3. **Replay the structural folds** in `fold/`: the root fold followed by
   every recursive fold, using the schedule-selected `CommittedGroupParams`.
4. **Check the terminal witness directly** against its predecessor-bound `t`
   state. The terminal relation is `consistency | A`; it has no outer `u`, B
   block, D block, or quotient sumcheck. If the terminal A matrix uses an L2
   route, the verifier also computes the decoded response's exact integer
   squared norm and compares it with the scheduled cap.

The verifier source follows this split. `fold/` holds orchestration: the root
and suffix walks, the fold-challenge draw, the fold's public relation instance,
and terminal dispatch. `verify_fold` in `fold/mod.rs` runs one
nonterminal fold as a sequence of stage calls in transcript order:

1. check the opening and commitment payload shapes, then read the fold
   response and draw the fold challenges;
2. assemble the ring-relation instance and receive the successor witness;
3. replay the ring switch (`stages/ring_switch.rs`) and derive the relation
   claim and range-image plan (`stages/relation_claim.rs`);
4. replay Stage 1 (`stages/stage1.rs`), then prepare the Stage 2 opening
   semantics (`stages/opening_semantics.rs`);
5. replay the Stage 2 rounds (`stages/stage2.rs`), then Stage 3 when the
   successor defers the setup contribution (`stages/stage3.rs`);
6. check the Stage 2 output claim, which needs the Stage 3 setup claim.

Each stage returns a typed replay output that the next call consumes. The
Stage 2 round replay and output check are separate calls because Stage 3 sits
between them in the transcript. `relation/` holds the prepared relation
evaluator, the relation point, and the evaluation trace. `terminal/` holds the
direct terminal checks and the exact A-product evaluation.

At each nonterminal fold, the verifier checks fixed 256-byte `p_H` and `p_F`
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

Schedule validation admits reduced evaluation only as a monotone single-chunk
suffix after the first two folds and all multi-chunk folds, with
evaluation-trace openings and direct setup contribution. Folds 0 and 1 require
coefficient packing and quotient lift even when single-chunk. The reduced suffix rejects packing, incoming setup
prefixes, deferred Stage 3, and any later return to quotient lifting before
transcript replay. Raw versus compressed payload and `L∞` versus `L2` response security
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

`AkitaVerifier` construction prepares the terminal matrix before any proof
arrives. It keeps one prepared entry per ring dimension, covering the longest
terminal prefix and the widest terminal row among its admitted schedule rows.
A shorter product reads a prefix of that entry, and exact CRT capacity is
monotone in the row width, so the widest row's profile is exact for every
narrower one. The entry carries the i16 tail only when the widest row needs it.
Prepared forms are derived from the coefficient setup and never serialized.
Verification only reads them, and shape and setup-prefix checks happen before
either kernel indexes prepared state.

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

## What the verifier trusts

The verifier has two kinds of input, and it treats them differently.

Proof bytes, commitments, claims, and opening points are untrusted. Every
check in this chapter exists so that soundness does not depend on who produced
them.

The remaining inputs are preprocessing: the expanded public matrix, the
setup-prefix commitment registry, the optional terminal NTT cache, and the
schedule catalog. The matrix is a function of the setup seed. The registry and
the cache are functions of the seed and of schedule rows. The catalog is
planner output checked against the compiled SIS tables. Together they play
the role of a verifying key.

Three separate claims stand behind a preprocessed object.

- **Identity.** The setup seed names the public stream and the row digest
  names the schedule row. Akita binds both into the transcript.
- **Derivation.** The object is the stated function of the seed and the
  schedule. Only computation establishes this, either by recomputing the
  object or by checking a proof of the derivation. Akita has no such proof
  today.
- **Byte authentication.** The bytes the verifier loaded are the bytes whose
  derivation was established. A release, a verifier program image, or a
  pinned digest can establish this.

Authentication does not replace derivation. A pinned digest authenticates
bytes and inherits the trust placed in whoever derived them. A verifier
therefore either recomputes an object and trusts no one for it, or loads
authenticated bytes and trusts the party that derived them. Recomputation is
always allowed and never required, and it is costly for the setup-prefix
commitments in particular.

Soundness assumes that the derivation claim holds. A verifier that loads
preprocessing that neither it nor a party it trusts has derived is outside
the soundness claim.

### Pinned releases

Akita authenticates preprocessing as a pinned **release**. A generator
(`akita-preprocess`) produces a release deterministically from a description
checked in under `params/<release>/`: it derives each public stream once,
commits every setup-prefix profile any parameter set needs, and writes the
registry payloads, optional prover hints, catalogs, optional guest files, and
a manifest that names every file by hash. The hash of the manifest is the
release's `bundle_id`. A required merge check regenerates every anchored
release from scratch and compares the manifest byte for byte, so an anchored
release is what the code at that commit produces. Anyone can repeat that
check with `akita-preprocess audit`.

A binary trusts only the `(bundle_id, set name)` pairs of its compiled
**trust anchor**. Akita's anchor lives in `akita-anchors`; an integration
compiles its own, and a deployment with its own seed, catalogs, or limits
generates its own release with the same generator. An anchor is a
compile-time constant: it has no deserializer, and an anchor read from
storage would void every guarantee below.

A loader turns release files into the values a verifier is built from. It
checks the cheapest and most decisive things first:

1. Hash the manifest and require every requested `(bundle_id, set name)`
   pair to be anchored, before decoding anything; then require the release
   kind and the compiled digest hash.
2. Compare the set record's code-derived fields (config family,
   decomposition, SIS profile and table digests, commitment algorithm) with
   the compiled configuration.
3. Check every file it uses against its manifest hash, on the exact bytes it
   then decodes.
4. Recompute every setup-prefix commitment's prefix commitment digest and
   compare it with the manifest. A host may instead derive each commitment
   from the seed (seed-derived mode) and compare the same digest.
5. Admit rows, size the matrix from the admitted rows only, derive it, and
   check that the resulting setup admits exactly those rows.

The result is a `PinnedVerifierSetup`, which has private fields, no public
constructor, and no deserializer. `AkitaVerifier::from_pinned` accepts it.
Every other way to build a verifier setup, such as
`AkitaVerifierSetup::from_parts` or `AkitaVerifier::new`, is an `unsafe fn`
whose contract is that the caller guarantees derivation; tests and
benchmarks use them for setups they derived themselves. Here `unsafe` marks a
soundness contract, not a memory-safety one, so every such call site is
visible and an integration can deny it with `clippy::disallowed_methods`.

Protocol Stage 3 binds each use. Where it absorbs the selected setup-prefix
slot id, it now absorbs the slot id followed by that slot's prefix commitment
digest, recomputed in every proof from the registry entry the fold reads. A
proof therefore names the commitments it was checked against, and fails
against any other commitment. Soundness still comes from the load-time
checks: once loading fixes a commitment, the prover cannot choose it.

Trusting an object's derivation does not relax its decoding. Validated
decoding checks preprocessing bytes for lengths, geometry, and canonical
field encoding under the same no-panic contract as proof bytes, whether or
not it also re-derives the object. Akita also keeps inexpensive
admission checks where a faulty generator could produce data that verifies
and is weaker than intended. The SIS audit of each catalog row is the main
example.

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
