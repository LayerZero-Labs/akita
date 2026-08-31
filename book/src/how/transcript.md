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
