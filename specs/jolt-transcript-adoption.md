# Spec: Jolt transcript adoption (protocol v8)

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Quang Dao, Claude              |
| Created       | 2026-10-20                     |
| Status        | active                         |
| PR            | [#253](https://github.com/LayerZero-Labs/akita/pull/253) |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  | how/transcript.md              |

## Summary

Akita kept its own copy of transcript machinery that Jolt now owns: field
challenge sampling, proof-of-work pricing, the proof-of-work search and check,
and the sponge wrappers. Two copies of the same security-critical code drift apart. Akita's
sampler also drew each field element with its own retry loop, so the number of
squeezes depended on the bytes. Jolt's codec squeezes a fixed length per
message. This change moves Akita onto Jolt's `jolt-transcript` crate over
Spongefish 0.8.0 and keeps in `akita-transcript` only what is specific to
Akita: the session tag, atom codecs, diagnostic site records, and the
fold-response preview. Proof bytes change, so the protocol version moves from
7 to 8.

## Intent

### Goal

`akita-transcript` is a thin policy layer over `jolt-transcript`, and every
Akita challenge and proof-of-work step goes through Jolt's audited code.

### Invariants

- **Domain separation.** The Spongefish session identifier is
  `derive_session_id::<Shake128>(&SESSION_TAG)`, with SHAKE128 (FIPS 202)
  from `jolt_transcript::Shake128`. `SESSION_TAG` is the 32-byte padded
  protocol name, version and digest hash; the codecs it fixes are listed on
  `PROTOCOL_VERSION`
  ([positional-word-aligned-transcript](positional-word-aligned-transcript.md)).
  The instance is the caller's session bytes and the canonical descriptor
  bytes, each with a `u64` little-endian length word and padding to a word.
  Tests: `session_tag_is_the_padded_protocol_name`,
  `session_and_statement_are_framed_separately`.
- **Default sponge.** Every crate's `default` feature set selects
  `transcript-keccak`. `transcript-blake2b` stays available as an opt-in
  feature, and exactly one backend must be enabled. CI runs the default
  builds with Keccak and the transcript semantics suites with both.
- **Challenge bias.** Each challenge message has total variation at most
  `2^-161` from uniform, charged as `2^-160`. A site that draws several
  challenges draws them in one message (`ext_challenges`). Codec exhaustion is
  an error; the verifier rejects.
- **Proof of work.** Each protected site grinds once with
  `ProverTranscript::grind` and checks with `VerifierTranscript::check_grind`.
  The nonce is the least passing value below `2^(g+7)`, sent as canonical
  LEB128 and absorbed as one `u64` word. The check squeezes
  `GRINDING_CHECK_BYTES = 8` bytes.
  The protected challenge is the next squeeze. A const assertion ties
  `MAX_GRINDING_BITS` to `jolt_transcript::MAX_POW_BITS`. This change keeps
  both at 25 with a `u32` nonce and moves `GRINDING_ENCODING_VERSION` to 4;
  the plan policy byte records the check width. #260 later raises both caps
  to 57 with a `u64` nonce and the encoding version to 5.
- **Proof-of-work pricing.** `ChallengeFieldOrder::grind_bits` calls
  `jolt_transcript::grind_bits_for_loss` with the exact challenge-set size and
  `TRANSCRIPT_SECURITY_BITS`. Akita keeps plan derivation, site binding and
  the descriptor encoding of the challenge order. Every target is exact, so
  plan bytes do not change from this move.
- **Verifier rejection.** Every verifier failure poisons the Spongefish state
  (`reject`, malformed atoms, failed proof of work, and errors routed through
  `ProofChannel::guard`). A rejected verifier cannot draw a challenge message.
  Acceptance requires `check_eof`, which fails on a poisoned state or unread
  bytes. Tests: `record_size_overflow_is_rejected_before_receipt`,
  `rejected_verifier_draws_no_challenge_message`.
- **Feature isolation.** Nonce search and fold-response preview copy the
  sponge state, which Spongefish exposes only under `yolocrypto`. The
  `prover` feature of `akita-transcript`, `akita-challenges` and `akita-types`
  enables it; only `akita-prover` turns it on.
  `scripts/check-crate-deps.sh akita-verifier` fails if the verifier's normal
  and build dependency graph enables `yolocrypto`.
- **Known answers.** `transcript_has_a_known_answer` pins one challenge
  message and one squeeze per sponge.
  `arithmetic_cutover_preserves_proof_and_challenges` pins sumcheck proof
  bytes and challenges per sponge.

### Non-Goals

- Poseidon or BN254 transcripts.
- Recursion-guest inlines.
- Changing the sparse fold challenge sampler or the grinding plan's pricing.

## Evaluation

### Acceptance Criteria

- [ ] `akita-transcript` has no rejection sampler, no sponge wrapper and no
      proof-of-work search of its own; `akita-params` has no proof-of-work
      pricing arithmetic of its own.
- [ ] Every Jolt dependency points at one Jolt commit;
      `scripts/check-shared-field-identity.sh` passes.
- [ ] The three CI Clippy configurations pass with `-D warnings`.
- [ ] Focused crate tests pass under `transcript-blake2b` and
      `transcript-keccak`, with the `prover` features on.
- [ ] `akita-pcs` `transcript_hardening` and `protocol_soundness` pass under
      both sponges.
- [ ] The verifier's `yolocrypto` check passes.

### Testing Strategy

Unit tests in `akita-transcript` cover atom round trips, noncanonical atoms,
bounded payloads, session framing, the fold preview, and known answers. The
grinding replay tests in `akita-types` cover the plan cursor, nonce ranges and
poisoning. `akita-pcs` integration suites cover whole proofs and tampering.

### Performance

Proof size is unchanged: the proof-of-work check was never serialized, and
nonces keep their LEB128 encoding. Verifier hashing drops slightly, because
the check squeezes at most 4 bytes instead of 32 and a site's challenges come
from one squeeze.

## Design

### Architecture

| Layer | Owns |
|---|---|
| Spongefish 0.8.0 | Sponge state, NARG bytes, poisoning, EOF |
| `jolt-transcript` | `ChallengeCodec`, `grind_bits_for_loss`, `grind`/`check_grind`, `verifier_message_oracle` |
| `akita-transcript` | Session tag, `FieldAtom`/`ExtensionAtom`/`NonceAtom`, site and context records, `ProofChannel`, `FoldPreview` |
| `akita-types` | Grinding plan replay over the above |

`ProofChannel` is implemented directly on `ProverState` and `VerifierState`.
Configurations require `Field: ChallengeCodec` and `ExtField: ChallengeCodec`,
so an unsupported field fails to compile.

### Alternatives Considered

- **Use Jolt's `verifier_message_oracle` for fold-response search.** It evaluates only
  "absorb a candidate, then squeeze". A fold-response candidate is followed by
  a proof of work and one message squeeze per group, so `FoldPreview` stays in
  Akita. A Jolt API that returns, after a candidate, a
  preview that supports `public_message` and `squeeze` would remove it.
- **Keep per-element sampling.** Rejected: it spends one bias budget per
  element and duplicates Jolt's codec.

## Documentation

`book/src/how/transcript.md` describes the session, the codec, poisoning and
the proof-of-work check. `book/src/usage/feature-flags.md` documents the
`prover` feature. `specs/transcript-grinding.md` and
`specs/grinding-nonce-encoding.md` describe the `ceil(g/8)`-byte check.

## References

- a16z/jolt `jolt-transcript` (`challenge.rs`, `pow.rs`).
- [`transcript-grinding.md`](transcript-grinding.md).
