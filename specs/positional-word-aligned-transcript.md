# Spec: Positional, word-aligned transcript

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Quang Dao, Claude              |
| Created       | 2026-10-24                     |
| Status        | active                         |
| PR            |                                |
| Supersedes    | The fold draw payload in [sparse-challenge-stream](sparse-challenge-stream.md) |
| Superseded-by |                                |
| Book-chapter  | how/transcript.md              |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Summary

Akita's sponge input had two kinds of waste. Some absorbs carried bytes the
verifier already knew: the per-group fold payload restated the challenge
family that the instance descriptor binds, and the session tag spelled out
codec names. Other absorbs had odd lengths: 4-byte field atoms, one byte per
nonce message, a 1-byte descriptor tag, a sparse message of `ceil(c K / 8)`
bytes. An odd length leaves every later message off a word boundary, which a
verifier that works in 64-bit lanes, such as one inside a proof system, pays
for at every step. This change makes every absorb positional (its type and
length come from bound data, never from a label) and a whole number of 8-byte
words. Proof bytes do not change.

## Intent

### Goal

Every Akita absorb and squeeze is a whole number of `u64` words, and nothing
is absorbed that the verifier can derive from the descriptor, the schedule or
the message's position.

### Invariants

- **Tag.** `SESSION_TAG` is `akita-pcs/transcript/v8`, zero-padded to 32
  bytes, for both sponges. It names only the protocol and its version. The
  doc comment of `PROTOCOL_VERSION` lists what version 8 fixes: the challenge
  codec, the proof-of-work layout, the absorb layout, the sparse message
  codec, and the constants bound only by the tag. Test:
  `session_tag_is_the_padded_protocol_name`.
- **Byte strings.** A byte string whose length bound data fixes is absorbed
  as its bytes, then zero padding to the next word. A bounded variable-length
  payload sends a `u32` length on the wire and absorbs it as one `u64` word.
  Tests: `byte_group_sends_its_bytes_and_absorbs_them_padded_to_a_word`,
  `bounded_payload_round_trips`.
- **Atom groups.** A group of proof atoms is one prover message: the wire
  carries the canonical atoms with no padding, and the sponge absorbs them
  zero-padded to a word. A group of public values is one padded absorb.
  Tests: `field_group_sends_wire_atoms_and_absorbs_a_padded_word`,
  `public_groups_absorb_one_padded_word_sequence`.
- **Nonces.** Proof-of-work and fold-response nonces stay canonical LEB128 on
  the wire, with the same range check. The sponge absorbs the nonce value as
  one `u64` little-endian word. The proof-of-work check squeezes
  `GRINDING_CHECK_BYTES = 8` bytes and tests the same low `g` bits. The Jolt
  search probe, the fold preview and the verifier MUST absorb the same word.
  `FoldWork::commit` writes the work's wire bytes without absorbing them.
  Test: `nonce_is_sent_as_leb128_and_absorbed_as_a_word`.
- **Fold draws absorb nothing.** The schedule bound in the descriptor fixes
  each fold's family: domain, ring dimension, sparse counts, packing subring
  dimension and operator-norm policy. The group index and the counts follow
  from position. All three draw adapters (live prover, verifier, preview)
  MUST drop the payload alike. With no absorb between them, the fold's
  messages continue one squeeze run after the fold's check, and every output
  byte is read once. Tests: `evaluation_trace_draw_squeezes_one_family_message`,
  `fold_preview_matches_live_prover_and_verifier`.
- **Tag-bound constants.** After the payload is gone, only the tag binds the
  `OperatorNormRejection` constants, the pool rates that set `message_len`,
  and the meaning of the coefficient-packing domain. A change to any of them
  MUST bump `PROTOCOL_VERSION`.
- **Sparse messages.** `message_len` rounds the candidate bits up to whole
  words. The decoder reads a little-endian bit stream over little-endian
  `u64` words. Candidate `j` still occupies bits `[j K, (j + 1) K)`, so equal
  squeezed bytes decode to equal challenges, and the bits past the last
  candidate are ignored. Tests: `word_reader_matches_the_little_endian_bit_stream`,
  `trailing_word_padding_is_ignored`,
  `operator_norm_pools_match_the_specification`.
- **Descriptor version 6.** Every scalar field is one `u64` word, each
  32-byte digest or modulus is its bare bytes, and each vector is a length
  word then one word per element. A word its field cannot hold, or a flag
  word other than 0 or 1, is rejected. Test:
  `descriptor_is_word_aligned_and_rejects_noncanonical_words`.
- **Keccak digests.** Every digest bound into the descriptor goes through
  `digest_descriptor_bytes`, which is Keccak-256 with the `0x01` pad byte,
  the hash of Jolt's `jolt-inlines-keccak256` guest inline. Each digest keeps
  its domain prefix and input bytes. The Blake2b sponge feature does not
  change it. Test: `descriptor_digest_is_keccak256`.
- **Injectivity.** Fix the schedule, setup and statement. The absorbed
  sequence is then an injective function of the prover's messages: every pad
  count follows from a bound length or a length word.
- **Verifier no-panic.** Every new decode path rejects with `AkitaError` or
  `SerializationError`.

### Non-Goals

- A lane sponge. When one lands, an unaligned absorb on a verifier path MUST
  poison the state and fail at `check_eof`, never panic.
- Changing proof bytes, challenge families or grinding prices.
- The Jolt side of the cutover, which lands in Jolt.

## Evaluation

### Acceptance Criteria

- [ ] Same-seed proof sizes are identical to the base branch on every
      profile.
- [ ] Focused tests for `akita-transcript`, `akita-challenges`,
      `akita-types`, `akita-params` and `akita-schedules` pass under both
      sponges.
- [ ] The three CI Clippy configurations pass.
- [ ] Schedule artifacts are regenerated for descriptor version 6.

### Testing Strategy

Byte-level tests in `akita-transcript` compare each channel helper against a
reference prover that absorbs the expected padded bytes in one message. The
sparse decoder is tested against a bit-by-bit reader at every offset. Known
answers are regenerated: every change above alters them.

### Performance

Each fold group saves the permutation that its payload absorb forced before
the message squeeze. Padding adds at most 7 bytes per message. The nv32
Keccak permutation count is recorded in the PR.

## Design

### Architecture

Every place where the absorbed bytes differ from the wire bytes goes through
Spongefish's `prover_message_with` or `public_message_as`, so Spongefish
needs no change. In `akita-transcript`, `word_padded` and `length_word` build
the absorbed forms, and `send_atoms`/`receive_atoms` are the one path for
atom groups. `send_nonce`/`receive_nonce` carry nonces.

### Alternatives Considered

- **Keep the fold payload as a cheap redundancy check.** Rejected: it costs a
  permutation per group and binds nothing the descriptor does not.
- **Pad the tag to 64 bytes.** Rejected: 32 bytes holds every tag name, and
  only the derived session id enters the sponge.
