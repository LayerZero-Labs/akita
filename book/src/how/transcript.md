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

Deferred work (prover/verifier trait split, `Bound<T>`, algorithm-as-bytes digest, NARG migration): [`specs/transcript-hardening.md`](../../../specs/transcript-hardening.md).

Implementation: `crates/akita-transcript/`.
Tests: `crates/akita-pcs/tests/transcript_hardening.rs`.

## AkitaInstanceDescriptor

The canonical descriptor binds algebra, setup, plan, and call shape.
Prover and verifier share one helper:

- `crates/akita-config/src/transcript_binding.rs` — `bind_transcript_instance_descriptor`
- `crates/akita-types/src/instance_descriptor.rs` — descriptor shape and serialization

Paper reference: §3.5 (`sec:akita-one-step`, transcript binding).

### Integrator note (Jolt / recursion hosts)

`AKITA_INSTANCE_DESCRIPTOR_VERSION` stays at **`1`** during active protocol
development. Field additions inside the setup section (for example
`FoldLinfProtocolBinding` extensions) do **not** bump this constant. There is
no backward-compatibility guarantee across arbitrary crate revisions: pin an
exact Akita git revision and re-run prove/verify e2e checks when upgrading.

After the zk-strip cutover, `SetupSection.protocol_features.zk` is always
`false` on the wire. Transparent proof bytes for the pinned regression cases
(`transparent_proof_golden` in `akita-pcs`) are the authoritative wire-format
tripwire for recursion integrators.
