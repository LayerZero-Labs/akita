# Proof encoding and transcripts

An Akita verification request combines proof bytes with a complete public
statement. That statement contains the setup identity, commitments, opening
points, claimed values, configuration, and selected trusted schedule row.

Akita binds all of these values before deriving proof challenges. This makes
the public integration boundary part of the cryptographic statement.

## The public proof bundle

A host should define one versioned container with these fields:

| Field | Purpose |
| --- | --- |
| Protocol revision | Pins the Akita proof format and transcript schedule |
| Configuration identity | Selects the field policy and expected catalog family |
| Catalog identity or storage reference | Authenticates the external schedule artifact supplied to the verifier |
| Verifier setup identity or package | Supplies the public matrix data needed for replay |
| Schedule selection | Names one exact approved catalog row |
| Ordered commitments | Fixes every polynomial group |
| Ordered opening points | States where each group is opened |
| Ordered claimed values | States one value for each committed polynomial |
| Proof bytes | Carries the canonical Spongefish argument stream |

The host may store verifier setup separately and refer to it by an authenticated
identifier. The remaining values still belong to one public verification
request.

## Use a stable session for each protocol

The host chooses an application specific session label. Give it a version and
a clear protocol owner.

```rust
const SESSION: &[u8] = b"my-system/akita-opening/v1";

let proof = scheme.batched_prove(&prover_setup, opening, &backend, SESSION, basis)?;
scheme.verifier(verifier_setup)?.batched_verify(
    &proof,
    SESSION,
    statement,
    basis,
    akita_verifier::SetupPolicy::Check(None),
)?;
```

The verifier binds the canonical Akita instance descriptor before replay. That
descriptor covers the configuration, setup identity, schedule, and public
claim layout. Akita then absorbs commitments, points, claimed values, and proof
messages in protocol order.

Akita constructs fresh prover and verifier states internally. Do not
serialize a live state or reuse proof bytes under another session. Both sides
use the same session bytes and independently bind the same public instance.

The [transcript chapter](../how/transcript.md) lists the exact binding order and
explains the wire checks used by transcript tests.

The protocol revision of the bundle is `akita_transcript::PROTOCOL_VERSION`,
currently `8`. It names the proof format and the transcript schedule. Akita
absorbs it first, inside a 32-byte protocol tag:
`akita-pcs/transcript/v8/sha3`, or `akita-pcs/transcript/v8/keccak` when the
`keccak256-digests` feature is on. A proof made under one revision or digest
hash does not verify under another.

## Embed Akita in a host transcript

A proof system that already has its own transcript usually wants one
transcript, not two. Akita supports this with a second pair of entries.

- **Standalone.** `batched_prove` and `batched_verify` build a fresh
  transcript state from the session label. The proof is the whole byte string.
- **Embedded.** `batched_prove_embedded` and `batched_verify_embedded` take a
  transcript state that the host owns. Akita appends its messages to the
  host's proof string and draws its challenges from the host's sponge.

The standalone entries call the embedded ones, so both modes run the same
code. The host state must use Akita's channel types,
`akita_transcript::ProverChannel` and `akita_transcript::VerifierChannel`.

```rust
// Prover. `channel` is the host's ProverChannel.
// ... the host absorbs its statement and sends its messages ...
scheme.batched_prove_embedded(&prover_setup, opening, &backend, &mut channel, basis)?;
// ... the host sends its remaining messages ...

// Verifier. `state` is the host's VerifierChannel over the same proof string.
// ... the host replays its messages ...
let pending = verifier.batched_verify_embedded(
    &mut state,
    statement,
    basis,
    akita_verifier::SetupPolicy::Check(None),
)?;
// ... the host replays its remaining messages ...
pending.check_eof(state)?;
```

The verifier has one more step than in standalone mode.
`batched_verify_embedded` does not accept a proof. It returns a `PendingEof`
value. The proof is accepted only when `PendingEof::check_eof` succeeds on the
same state after the host's last message. That call checks that the whole
proof string was read and that no step failed. On any error, Akita marks the
state as failed before it returns, so every later read and the final check
fail even if the host ignores the error.

The host has four duties:

1. Call the embedded verifier at the same transcript position where the
   prover called the embedded prover.
2. Bind each commitment before any host challenge that depends on it. The
   32-byte commitment digest is enough.
3. Stop on any error from Akita.
4. Pass the state that Akita verified on to `check_eof`. Nothing ties the
   `PendingEof` value to a state, so another state would skip the check.

Akita absorbs its own statement in both modes: the protocol tag, the instance
descriptor, each commitment digest, the opening points, and the claimed
values. Its soundness therefore does not depend on what the host absorbed.
[Running inside a caller's transcript](../how/transcript.md#running-inside-a-callers-transcript)
gives the exact order. The tests in
`crates/akita-pcs/tests/embedded_transcript.rs` check that a standalone proof
equals an embedded proof on a fresh state, that an embedded proof verifies
between host messages, and that a rejection leaves the host state failed.

## Carry the trusted row selection

`SelectedProverOpeningData::selection()` returns an
`OpeningScheduleSelection`. It names the exact row digest in the trusted catalog
used for the complete batch.

The verifier does not reconstruct a planner request from proof contents. It
resolves this explicit identity in the supplied catalog, checks the row digest,
and replays that schedule. This keeps planner search outside the verifier and
prevents a proof from supplying its own unchecked parameters.

## Store the proof canonically

The PCS returns the canonical Spongefish argument byte string.

```rust
let proof_bytes: Vec<u8> = proof;
```

The verifier resolves the selected row from the approved catalog before it
reads messages. Schedule-derived fixed counts and bounds control decoding and
allocation; no proof-supplied shape is accepted.

Canonical encoding means that one accepted object has one encoding within a
protocol revision. Pinning the producer and verifier to the same commit or
release gives the host one proof format and transcript schedule. Upgrade both
sides together and regenerate proof fixtures for the new revision.

## Rebuild the statement at verification

The verifier uses borrowed commitments because it does not own prover state.

```rust
let claims = OpeningClaims::from_groups(vec![
    PolynomialGroupClaims::new(point, values, &commitment)?,
])?;
let statement = GroupBatchStatement::new(selection, claims)?;
```

This constructor checks that the claim set is nonempty and structurally valid.
The verifier then checks each commitment profile against the claims, setup, and
resolved schedule.

## Treat every incoming byte as public input

The verifier accepts public data from another process or machine. Decode it
with explicit shape and size limits, build the typed statement, and pass errors
back to the host.

Akita's verifier boundary returns `AkitaError` or `SerializationError` for
malformed input. This gives hosts one normal rejection path for invalid bytes,
unsupported schedules, mismatched claims, and failed cryptographic checks.

The [verifier only guide](./verifier-only.md) puts these pieces together without
the prover dependency path.
