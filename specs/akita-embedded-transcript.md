# Spec: Akita inside a caller's transcript (protocol v8)

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Quang Dao, Claude              |
| Created       | 2026-10-20                     |
| Status        | proposed                       |
| PR            | [#254](https://github.com/LayerZero-Labs/akita/pull/254) |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  | how/transcript.md              |

## Summary

Akita always starts its own Spongefish session. Jolt, its main caller, already
runs a Spongefish session of its own. To connect the two, Jolt absorbs Akita's
setup, statement and commitments into its own transcript, squeezes a bridge
value, and passes that value to Akita as session bytes. Akita then starts a
fresh session and absorbs the same setup, statement and commitments again.
Each root commitment is absorbed three times, and the two sessions are linked
only through one squeezed value.

This change lets a caller run Akita directly on its own Spongefish state.
Akita's messages then go into the caller's argument string, and its challenges
come from the caller's sponge. Each commitment is bound by a 32-byte digest of
its canonical bytes, so the large commitment rows are hashed once. The
standalone entry stays and runs the same code. Two absorbs of values the
prover already sent are removed. Challenges change. Akita `main` is at protocol version 7, and #253 already
moves it to 8 for this stack, so this change stays at 8 (one bump per stack,
counted from `main`).

## Intent

### Goal

Every Akita prover and verifier entry runs on a Spongefish state it borrows.
The standalone entries build that state themselves; an embedding caller passes
its own.

New and changed surfaces:

- `akita_transcript::bind_instance(state, descriptor)`. It absorbs
  `PROTOCOL_TAG` (32 bytes) and LP(descriptor). Both modes call it first.
- `akita_transcript::new_prover_channel(session)` and
  `new_verifier_channel(session, proof)`. They build a standalone session and
  no longer take the descriptor.
- `akita_transcript::CommitmentDigest`: 32 bytes, computed by
  `CommitmentDigest::of_canonical_bytes` from the canonical serialization of
  one `CommittedGroup`, which is its profile followed by its rows.
- `CommittedGroup` has private fields and carries its digest, read with
  `CommittedGroup::digest`. The fallible constructor `CommittedGroup::new` and
  the deserializer check the group and compute the digest from its canonical
  bytes. No path accepts a digest from outside.
- `ProverGrinding<'a>` and `VerifierGrinding<'a, 'proof>` borrow the state
  instead of owning it. `VerifierGrinding::finish` checks plan completion and
  returns a `#[must_use] PendingEof`. `PendingEof::check_eof(self, state)`
  consumes the state, runs Spongefish's `check_eof`, and returns
  `ProofAcceptance`.
- Embedded entries: `akita_prover::batched_prove_embedded`,
  `AkitaVerifier::batched_verify_embedded` and the matching
  `AkitaCommitmentScheme` methods. They take `&mut ProverChannel` or
  `&mut VerifierChannel<'_>`. The verifier returns `PendingEof`. The
  standalone `batched_prove` and `batched_verify` build a session, call the
  embedded entry, and then return the argument string or call
  `PendingEof::check_eof`.

### Absorb map

"LP" means a `u64` little-endian length prefix. "Public" means a value
absorbed but not sent. "Sent" means a prover message, which goes into the
argument string and is absorbed once.

Before (#253, standalone, the only mode):

| Order | What | How |
|---|---|---|
| session | `derive_session_id(SESSION_TAG)` | Spongefish session id |
| instance | LP(caller session bytes), LP(descriptor) | Spongefish instance |
| root | per group: all commitment coefficients | public, one absorb per coefficient |
| root | per group: opening point | public, one absorb per element |
| root | all claimed evaluations | public, one absorb per element |
| folds | prover messages and challenges | sent, one absorb per atom |
| next level | outer payload sent at `NextWitnessPayload` | public again at `WitnessCommitment` |
| terminal | `t` sent at `NextWitnessInnerState` | public again at `TerminalTFields` |

After (this change):

| Order | What | How |
|---|---|---|
| session (standalone only) | `derive_session_id(STANDALONE_SESSION_TAG)`, instance LP(caller session bytes) | Spongefish session |
| first Akita absorb | `PROTOCOL_TAG`, LP(descriptor) | public, in band, both modes |
| root | per group: `CommitmentDigest` (32 bytes) | public |
| root | per group: opening point | public, one absorb per group |
| root | all claimed evaluations | public, one absorb |
| folds | prover messages and challenges | sent, one absorb per group or round polynomial |
| next level | nothing for the outer payload | already bound when it was sent |
| terminal | nothing for `t` | already bound when it was sent |

In embedded mode the "session" row is the caller's own session.

### Invariants

- **Every input is bound before the first challenge that depends on it.**
  In both modes, Akita's own code absorbs everything its acceptance depends
  on, before its first challenge, apart from the setup-prefix commitment
  rows, which Stage 3 binds through their prefix commitment digests, and
  the extension field's defining polynomial (see
  [Existing assumptions, unchanged](#existing-assumptions-unchanged)):
  - the protocol tag;
  - the descriptor: algebra, setup seed and decomposition, SIS modulus
    profile, schedule selection and schedule, grinding plan digest, opening
    layout and basis;
  - each root commitment's digest, which covers its profile and rows;
  - each opening point;
  - the claimed evaluations.

  The verifier checks each group's profile against the schedule bound in the
  descriptor. The caller cannot skip any of these absorbs, so Akita's
  soundness does not depend on what the caller absorbed.

  Test: `root_statement_binds_every_item` changes the session, the opening
  point, the claimed value and the commitment in turn, and checks that the
  verifier rejects, on both a standalone and an embedded state.
- **Commitment digest.**
  - `CommitmentDigest::of_canonical_bytes(bytes)` is the selected digest
    hash (`jolt_transcript::DigestHasher`: SHA3-256, or Keccak-256 under
    `keccak256-digests`) of `COMMITMENT_DIGEST_TAG` followed by `bytes`.
  - `bytes` is the `Compress::No` serialization of the `CommittedGroup` as
    held in memory: the profile header, then the coefficients in row order,
    each in canonical little-endian form. A deserialized group is
    re-serialized for its digest; the input bytes are never hashed directly.
  - The digest tag names the protocol version; the digest hash is the one
    `PROTOCOL_TAG` names.
  - The deserializer always checks the group's profile and row lengths.
    Proof-input code decodes with `Validate::Yes` (a CI guard forbids
    `Validate::No` there), which also rejects noncanonical coefficients, so
    each commitment has exactly one encoding. The digest always covers the
    re-encoded group as held in memory, never the input bytes.
  - The verifier computes every digest from the bytes it then uses. The rows
    that `verify_root` uses as commitment payloads come from the same
    `CommittedGroup` as the digest it absorbs.

  Tests: `commitment_digest_has_a_known_answer`;
  `deserialized_and_constructed_digests_agree`;
  `noncanonical_commitment_bytes_are_rejected`;
  `digest_is_recomputed_on_deserialization`.
- **The digest does not weaken binding.**
  - Changing a bound commitment means finding two commitments with the same
    digest.
  - For a 256-bit digest that takes about `2^128` hash calls, which is the
    transcript target. Both sponges have a 512-bit capacity, so the sponge
    does not lower this.
  - Quantum collision searches do not beat this once memory is counted.
    Brassard-Hoyer-Tapp needs about `2^85` queries but also about `2^85` of
    quantum memory. Chailloux, Naya-Plasencia and Schrottenloher (2017)
    need about `2^102` quantum time with little quantum memory and `2^51`
    classical memory. Neither is cheaper than a classical parallel
    collision search in total cost, so a 256-bit output is the usual
    128-bit collision level.
  - The verifier holds every full commitment and recomputes its digest, so
    binding reduces to collision resistance of the digest. No argument
    about recording oracle queries is needed, which matters because Akita
    targets post-quantum security.
  - The tag has a fixed length and `bytes` is the rest of the input, so no
    length prefix is needed.
- **Domain separation.**
  - The first values Akita absorbs are `PROTOCOL_TAG` and LP(descriptor).
  - `PROTOCOL_TAG` is `jolt_transcript::protocol_tag` of the protocol name,
    version and digest hash (`akita-pcs/transcript/v8/sha3`), the former
    `SESSION_TAG`. It has a fixed 32-byte length, so it needs no length
    prefix.
  - A standalone session uses a separate `STANDALONE_SESSION_TAG`.
  - Inside a caller, Akita's position in the caller's message schedule is
    fixed by the caller's session and earlier messages.

  Tests: `protocol_tag_names_the_backend_and_codecs`;
  `standalone_and_embedded_sessions_differ`.
- **Message schedule.** Spongefish requires that the session determine the
  order and lengths of absorbs and squeezes. Akita's schedule is fixed by the
  descriptor, which is absorbed before any message whose length depends on
  it. Every later absorb and squeeze is a fixed function of what has already
  been absorbed.
- **Proof of work.**
  - The grinding plan, its pricing, the nonce encoding and the check bytes
    do not change.
  - Each grind runs on the borrowed sponge exactly where it ran on Akita's
    own sponge. Each attempt is a fresh query on a state that depends on the
    whole history.
  - A caller-chosen prefix gives a prover more starting states, but every
    one still costs a full grind.
  - The protected challenges are squeezed directly after the check bytes,
    with no caller absorb in between.

  Test: the existing grinding replay and tamper tests, in both modes.
- **Rejection.**
  - On any verifier error, the embedded verifier poisons the borrowed state
    before it returns.
  - A poisoned state fails every later argument-string read, every call
    guarded by `Transcript::check`, and `check_eof`. It does not stop raw
    challenge squeezes or public absorbs.
  - So a caller must stop on Akita's error. It must also reach
    `PendingEof::check_eof`, which consumes its state and is the only way to
    obtain `ProofAcceptance`. `PendingEof` is `#[must_use]`.

  Tests: `embedded_rejection_poisons_the_caller_state`;
  `replay_rejection_poisons_the_borrowed_state`;
  `pending_eof_rejects_trailing_bytes`.
- **End of argument.**
  - The embedded verifier checks that the grinding plan is fully consumed
    and returns `PendingEof`.
  - The caller passes its state to `PendingEof::check_eof` after its own
    last message.
  - The standalone verifier does this right away.

  Test: `standalone_rejects_trailing_bytes`.
- **Same bytes in both modes.** For one statement, a standalone proof and an
  embedded proof run on a fresh state built with the same session give the
  same argument bytes. Test: `standalone_is_embedded_on_a_fresh_session`.
- **Sponge match.**
  - The embedded entries take Akita's `ProverChannel` and `VerifierChannel`
    types, so a caller state over another sponge does not compile.
  - Keccak (`transcript-keccak`) is the default, matching Jolt, since #253
    switched it. Blake2b-512 stays an opt-in feature. The empty-absorb rule
    below matters on Blake2b-512.
  - Poseidon states cannot embed Akita, because Akita's challenges are
    decoded from bytes.
- **Grouped absorbs give the same bytes.**
  - Spongefish absorbs are associative: absorbing `a` then `b` equals
    absorbing `a‖b`. Sending a non-empty group or round polynomial as one
    prover message therefore leaves the argument string and every challenge
    unchanged.
  - An empty group makes no absorb call. On Blake2b an empty absorb right
    after a squeeze changes the state, and #253 makes no call for an empty
    group.
  - Merged verifier reads use `prover_message_as` with a decoder that
    accepts only canonical atoms, so the absorbed bytes equal the
    re-encoding. They size allocation from the schedule before reading.
  - Grinding nonces stay one byte per message.

  Tests: known answers computed on the #253 head, grouping done first so that
  they still hold after it, on both sponges. They include an empty `RootPoint`
  (zero variables) and an empty group directly after a squeeze.
- **Dropping the re-absorbs of sent values loses nothing.**
  - **`t`.** The terminal `t` always comes from `receive_next_witness` at
    `NextWitnessInnerState`. That is the root fold when there is no
    recursive fold, and the last recursive fold otherwise.
    `verify_terminal_suffix` is its only consumer, and it uses that same
    received value.
  - **Outer payload.** The outer payload re-absorbed at `WitnessCommitment`
    is the value received at `NextWitnessPayload` one level earlier.
  - Both absorbs add no binding, so this change drops both on both sides. On the
    prover side this removes a recomputed value, which was only a
    completeness check.

  Test: `sent_witness_values_are_absorbed_once`, a structural test on the
  logged site records.
- **Verifier no-panic.** The new decoding paths (`CommittedGroup`
  deserialization with its digest, merged group reads) validate length and
  canonical form before they allocate or index. They return `AkitaError` or
  `SerializationError`.

### Existing assumptions, unchanged

- **Descriptor sub-digests are 32-byte Blake2b-256**, computed by
  `akita_params::digest_descriptor_bytes`, whatever the sponge. Their width
  meets the target, like the commitment digest. The inputs are honest public
  parameters from an admitted catalog, so an attack needs two admitted
  schedules that collide. This change leaves them as they are; moving them
  to the sponge's hash is a follow-up.
- **The verifier setup comes from the bound seed.** The setup seed is
  absorbed in the descriptor, and at protocol Stage 3 each slot's id is
  absorbed followed by its prefix commitment digest, recomputed in every
  proof from the registry entry the fold reads
  ([`pinned-preprocessing-releases.md`](pinned-preprocessing-releases.md#transcript-binding)).
  The digest names the commitment a proof was checked against; soundness
  still needs the commitment to be derived from the bound seed, which the
  release loader checks against an anchored manifest.
  `AkitaVerifierSetup::from_parts` trusts its inputs, so it is an `unsafe fn`
  whose caller guarantees derivation.
- **The extension field is fixed at compile time.** Its defining polynomial
  is not absorbed.

### Caller contract

A caller that embeds Akita must:

1. Pass a `ProverChannel` or `VerifierChannel` over Akita's sponge.
2. **Verifier argument string.** Build its verifier state over an argument
   string whose unread part, when Akita starts, begins with Akita's proof.
   - A caller that absorbs its own messages with `public_message` and keeps
     them out of the argument string (Jolt from J7 on) builds its verifier
     state over Akita's proof bytes.
   - Such a caller must not read the argument string before Akita runs. This
     includes argument-string proof-of-work checks. Jolt's nonces come from
     its proof structs, through a check that absorbs the same bytes the
     prover absorbed.
   - Extra bytes in front of Akita's proof only make an honest proof fail.
3. **Prover argument string.** On the prover side, Akita's messages are
   appended to the caller's argument string.
4. **End of verification.** Stop on any error from Akita. After its own last
   message, pass its state to `PendingEof::check_eof`.
5. **Commitments before dependent challenges.** Bind each root commitment
   before any of its own challenges that depend on it, in one of two ways:
   - Absorb the commitment's `CommitmentDigest`. That digest covers the
     profile, so the caller must know each commitment's profile when it
     absorbs it.
   - Absorb the commitment's canonical rows in a length-framed encoding,
     together with the public data that fixes its profile. This is enough
     only when that public data selects exactly one schedule row, so the
     same row bytes cannot be read under a second profile. Jolt does this:
     it absorbs each commitment's layout digest, flavor, variable and
     polynomial counts and rows, and its trusted catalog holds one row per
     layout and shape.

   Akita does not rely on this step for its own soundness, because it
   absorbs the digest again.
6. **Canonical encoding.** Absorb its own messages in canonical,
   length-framed encodings. This is the caller's own soundness rule, listed
   here because the two protocols share one sponge.

Akita absorbs its own statement in full, so the caller does not need to absorb
Akita's setup, selection, points or evaluations for Akita's sake.

### Non-Goals

- Poseidon or BN254 embedding.
- Dropping public absorbs of values derived from challenges, such as
  `TerminalPoint` and `TerminalOpening`. They add no binding either, but each
  needs its own check. They are listed as follow-ups.
- Changing the message schedule or proof size. This change sends the same messages
  as #253.
- Widening the descriptor sub-digests.
- Jolt's own transcript layout. That belongs to J7 and J8 in a16z/jolt.

## Evaluation

### Acceptance Criteria

- [ ] The embedded entries exist on the prover, the verifier and
  `AkitaCommitmentScheme`, and the standalone entries call them.
- [ ] `bind_instance` is the only place that absorbs the protocol tag and the
  descriptor.
- [ ] The root absorbs `CommitmentDigest` per group, and nothing absorbs
  root commitment rows as public values. Rows sent as prover messages, such
  as those at `NextWitnessPayload`, stay absorbed.
- [ ] `TerminalTFields` and `WitnessCommitment` are gone from both sides and
  from `FoldSite`.
- [ ] Every multi-atom group and every round polynomial is one prover message
  or one public absorb, and empty groups make no absorb call.
- [ ] `PROTOCOL_VERSION` stays 8, and the known answers that depend on the
  tags are regenerated.
- [ ] `scripts/check-crate-deps.sh akita-verifier` passes, so `yolocrypto`
  stays out of the verifier graph.
- [ ] The tests named under Invariants exist and pass.

### Testing Strategy

- Focused per-crate tests for akita-transcript, akita-types, akita-sumcheck,
  akita-verifier, akita-prover and akita-pcs, with `transcript-blake2b` and
  with `transcript-keccak`.
- The three Clippy configurations from `AGENTS.md`.
- `akita-cpu-backend` library tests need more than 16 GB and run in CI.
- Known answers are regenerated for both sponges. No checked-in fixture or
  fuzz corpus depends on transcript bytes.

### Performance

- Proof size: unchanged.
- Standalone verifier: each root commitment is hashed once (the digest)
  instead of absorbed once, which is about the same work. The two dropped
  re-absorbs save one absorb of each outer payload and of `t`.
- Embedded verifier (Jolt):
  - Each root commitment is hashed once instead of absorbed three times.
  - Akita's statement absorbs fall from setup, selection, commitments,
    points and evaluations on two transcripts to the following, on one
    transcript: one descriptor, one 32-byte digest per group, and the points
    and evaluations.
  - J8 reports absorb and permutation counts per proof, before and after.
- Grouped absorbs reduce the number of absorb calls, not the number of
  permutations.

## Design

### Architecture

```text
standalone                              embedded (Jolt)
----------                              ---------------
new_verifier_channel(session, proof)    Jolt's VerifierState over Akita's proof bytes
            \                          /
             batched_verify_embedded(&mut state, statement, basis) -> PendingEof
               bind_instance(state, descriptor)
               VerifierGrinding::new(&mut state, plan)
               verify_root(...)    // absorbs digests, points, evaluations
               verify_suffix(...)
               grinding.finish()   // plan consumed; state poisoned on error
            /                          \
pending.check_eof(state)               Jolt's remaining messages, then pending.check_eof(state)
```

- `akita-transcript` owns the tags, `bind_instance` and the channel
  constructors.
- `akita-transcript` also owns `CommitmentDigest` and its tag, because the
  digest uses only the sponge.
- `akita-types` owns `CommittedGroup`, which computes its digest, and
  `PendingEof` and the grinding wrappers.
- `akita-verifier` and `akita-prover` each gain one embedded entry. The
  standalone entry builds the state, calls it and finishes.
- `akita-pcs` exposes both modes on `AkitaCommitmentScheme`.

### Alternatives Considered

- **Keep the bridge.** Two sessions, linked by one squeezed value. It works,
  but every statement item is absorbed twice and the commitment three times.
- **A token that says "already absorbed".** Akita would skip the commitment
  absorb if the caller held a token from an Akita binding function. A token
  cannot prove which state the value was absorbed into, or when. A misused
  token would silently drop a binding. The digest makes Akita absorb its whole
  statement itself, at a cost of 32 bytes per group.
- **Absorb the full rows in Akita and nothing in the caller.** The caller
  draws challenges that depend on the commitments long before Akita runs, so
  it must bind them first.
- **A digest over the rows only.** A caller that binds only the rows has not
  fixed how they are read: the ring dimensions, basis and layout come from
  the profile. A prover could choose the profile after seeing the caller's
  challenges. The digest covers the profile so that one absorb fixes both.
- **Keep the descriptor in the Spongefish instance and require the caller to
  put it there.** That would tie the caller's instance format to Akita's
  descriptor. The in-band absorb keeps one body for both modes.
- **A 64-byte digest.** It would raise collision work to about `2^256`,
  above the target, at the cost of 32 more absorbed bytes per group and
  more recursive-verifier work. The extra margin only guards against a
  quantum collision search, which is no cheaper than a classical one once
  its memory is counted.

## Documentation

- `book/src/how/transcript.md`: the session, the in-band binding, the
  commitment digest, the embedded entry and the caller contract.
- `book/src/how/verification.md`: the embedded verifier's rejection and
  end-of-argument rules.
- `specs/jolt-transcript-adoption.md`: a note that this change replaces the instance
  layout described there.

## Execution

1. Grouped absorbs, while the bytes are still #253's, so that #253's known
   answers check that grouping changes nothing. This includes the
   empty-group cases.
2. The grinding wrappers borrow the state, and `PendingEof` is added. The
   standalone entries own the state.
3. `bind_instance`, the tags and the new channel constructors.
4. `CommitmentDigest` and the private-field `CommittedGroup`. The root
   absorbs the digest.
5. Embedded entries on the prover, the verifier and `AkitaCommitmentScheme`.
6. Drop `TerminalTFields` and `WitnessCommitment`.
7. The new tags (still version 8), known answers, fixtures and the book.

Follow-ups, not in this change:

- Compute the descriptor sub-digests with the selected sponge instead of
  Blake2b-256, so a recursive verifier needs one hash primitive. The
  verifier recomputes the grinding-plan, opening-layout and schedule digests
  on every call, and schedule row digests are also stored in the generated
  catalog artifacts, so the change regenerates those artifacts.

- Drop the public absorbs of challenge-derived values, one at a time, each
  with its own audit.

## Audit

An independent read-only soundness audit of an earlier draft returned **sound
with changes**, with no blockers. A second pass on the revision confirmed the
fixes; its remaining wording points are applied here. This version includes
the audit's required changes:

- The digest covers the profile.
- Digests are always recomputed, and the digest and rows come from one
  source.
- Empty groups make no absorb call, and merged reads decode canonically.
- `PendingEof` exists, and the poison wording is corrected.
- The contract states the argument-string ordering for empty-argument
  callers.
- The tag is length-prefixed.

The audit also confirmed that dropping the `WitnessCommitment` re-absorb is
sound, by the same argument as for `t`. Its notes on the descriptor
sub-digest width and on how the setup is derived are recorded under
[Existing assumptions, unchanged](#existing-assumptions-unchanged).

## References

- [`specs/jolt-transcript-adoption.md`](jolt-transcript-adoption.md)
- [`specs/transcript-grinding.md`](transcript-grinding.md)
- [#253](https://github.com/LayerZero-Labs/akita/pull/253): protocol v8 on
  Jolt's transcript.
- Spongefish 0.8.0 `DuplexSpongeInterface`: absorbs and squeezes are
  associative.
