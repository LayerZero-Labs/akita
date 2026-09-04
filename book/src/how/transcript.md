# Transcript and instance binding

The Fiat-Shamir layer and the canonical preamble that binds the instance before
any protocol replay, so prover and verifier squeeze identical challenges.

## The transcript layer

Production code uses spongefish-backed `AkitaTranscript` with production-ZST
labels (labels are diagnostics and must **not** enter production sponge bytes).

Active hardening pillars:

| Pillar | Requirement |
|--------|-------------|
| **P0** | Bind canonical `AkitaInstanceDescriptor` bytes through spongefish `DomainSeparator.instance(...)` before protocol replay |
| **P2** | Use `AkitaTranscript` plus production-ZST labels only as diagnostics |
| **P3** | `LoggingTranscript` tests enforce prover/verifier event-stream equality and wire-before-squeeze discipline |

## Grinding plan and nonce stream

Each proof has one public `GrindingPlan`, derived from the selected schedule,
normalized opening layout, field tower, and descriptor-bound policy before the
proof shape is constructed. The plan fixes the order and bit width of every
transcript proof-of-work query and every bounded fold-response search. Its
digest is part of the instance descriptor, and its total bit count fixes the
leading headerless `TranscriptNonceStream` in the proof.

Proof-of-work and fold-response search use the same packed stream but have
different security meanings. A protected Fiat-Shamir query first checks its
scheduled nonce against a separate 32-byte predicate, then draws the protocol
challenge from the advanced transcript. A fold-response nonce is instead a
12-bit honest-prover retry value, shared by all commitment groups in that fold;
the verifier still checks the resulting response representation and norm
bound. Zero-bit sites consume no proof bits and do not change the transcript.

Sparse fold challenges preserve one live transcript squeeze per commitment
group. The 32-byte group root and numeric fold-response nonce define a fresh
indexed SHAKE256 stream for each claim-major block coordinate. Coordinate XOF
queries do not mutate the live transcript, so changing one coordinate leaves
the other coordinates and transcript state fixed. Prover and verifier must
consume the complete plan in order; truncation, reordering, nonzero tail
padding, or leftover nonce bits is an error.

Implementation: `crates/akita-types/src/transcript_grinding_plan.rs`,
`crates/akita-transcript/src/grinding.rs`, and
`crates/akita-challenges/src/sampler/xof.rs`.
Normative design: [`specs/transcript-grinding.md`](../../specs/transcript-grinding.md).

Deferred work: prover/verifier trait split, `Bound<T>`, algorithm-as-bytes digest, NARG migration.

Implementation: `crates/akita-transcript/`.
Tests: `crates/akita-pcs/tests/transcript_hardening.rs`.

## AkitaInstanceDescriptor

The canonical descriptor binds algebra, setup, plan, and call shape.
Prover and verifier share one helper:

- `crates/akita-config/src/transcript_binding.rs` — `bind_transcript_instance_descriptor`
- `crates/akita-types/src/instance_descriptor/mod.rs` — descriptor shape and serialization

The descriptor is absorbed before any protocol message or challenge. This
binds the transcript to the selected algebra, setup, schedule, and public call
shape rather than trusting those choices from later proof bytes.

### Integrator note (Jolt / recursion hosts)

`AKITA_INSTANCE_DESCRIPTOR_VERSION` is currently **`4`**. Validation rejects
any other version. Pin an exact Akita git revision and rerun prove and verify
integration tests when upgrading. The repository does not promise
compatibility across revisions.

Each nonterminal plan entry binds its `RingRelationMode`. Changing a fold from
quotient lifting to reduced evaluation therefore changes both the canonical
plan bytes and the transcript preamble before the shared ring-switch challenge
`alpha` is sampled. The mode is schedule-owned rather than serialized in the
proof, and verification never retries the other mode after a mismatch.

After the zk-strip cutover, `SetupSection.protocol_features.zk` is always
`false` on the wire. Ongoing wire regression is covered by serde roundtrips and
end-to-end prove→serialize→deserialize→verify tests in `akita-pcs` (for example
`akita_e2e.rs`, `fold_linf.rs`), not by pinned proof-byte digests.
