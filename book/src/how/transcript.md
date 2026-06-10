# Transcript and instance binding

> **Status:** stub. Part of the initial Akita Book scaffold.

The Fiat-Shamir layer and the canonical preamble that binds the instance before
any protocol replay, so prover and verifier squeeze identical challenges.

## The transcript layer

The spongefish-backed `AkitaTranscript`, the transcript-hardening pillars
(P0/P2/P3), and the wire-before-squeeze discipline that the logging transcript
tests enforce.

**Sources to fold in**

- `crates/akita-transcript/README.md`, `crates/akita-transcript/src/`.
- `AGENTS.md` (Transcript Hardening), `specs/transcript-hardening.md`.
- `specs/transcript-immediate-fixes.md` (active), `crates/akita-pcs/tests/transcript_hardening.rs`.

## AkitaInstanceDescriptor

The canonical descriptor bound through `DomainSeparator.instance(...)`: what it
binds (algebra, setup, plan, call shape) and the single `bind_transcript_instance_descriptor`
helper shared by prover and verifier.

**Sources to fold in**

- `crates/akita-types/src/instance_descriptor.rs:42-55`.
- `crates/akita-config/src/transcript_binding.rs` (`bind_transcript_instance_descriptor`).
- Paper §3.5 `sec:akita-one-step` ("Transcript binding").
