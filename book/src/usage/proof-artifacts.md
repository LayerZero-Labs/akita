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
| Expected proof shape | Bounds decoding and describes the proof structure |
| Compressed proof bytes | Carries the opening proof |

The host may store verifier setup separately and refer to it by an authenticated
identifier. The remaining values still belong to one public verification
request.

## Use a fresh transcript for each side

The host chooses an application specific session label. Give it a version and
a clear protocol owner.

```rust
const TRANSCRIPT_DOMAIN: &[u8] = b"my-system/akita-opening/v1";

let mut prover_transcript =
    AkitaTranscript::<F>::unbound_prover(TRANSCRIPT_DOMAIN);
let mut verifier_transcript =
    AkitaTranscript::<F>::unbound_verifier(TRANSCRIPT_DOMAIN);
```

The scheme binds the canonical Akita instance descriptor before replay. That
descriptor covers the configuration, setup identity, schedule, and public
claim layout. Akita then absorbs commitments, points, claimed values, and proof
messages in protocol order.

Create a new transcript for each proof. Do not serialize a live transcript
object or continue a prover transcript on the verifier side. The two sides
start from the same session label and independently bind the same public
instance.

The [transcript chapter](../how/transcript.md) lists the exact binding order and
explains the wire checks used by transcript tests.

## Carry the trusted row selection

`SelectedProverOpeningData::selection()` returns an
`OpeningScheduleSelection`. It names the exact row digest in the trusted catalog
used for the complete batch.

The verifier does not reconstruct a planner request from proof contents. It
resolves this explicit identity in the supplied catalog, checks the row digest,
and replays that schedule. This keeps planner search outside the verifier and
prevents a proof from supplying its own unchecked parameters.

## Encode the proof canonically

Akita proof objects implement compressed serialization.

```rust
let proof_shape = proof.shape();
let mut proof_bytes = Vec::new();
proof.serialize_compressed(&mut proof_bytes)?;
```

Decode against an expected shape:

```rust
let proof = AkitaBatchedProof::<F, E>::deserialize_compressed(
    &mut std::io::Cursor::new(&proof_bytes),
    &expected_shape,
)?;
```

The expected shape controls list lengths and nested proof structure before the
verifier reads large payloads. A host can derive it from the selected row in an
approved catalog, or authenticate it as part of a versioned application
artifact format.

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
