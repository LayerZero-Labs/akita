# The commitment API

> **Status:** stub. Part of the initial Akita Book scaffold.

The user-facing surface of `AkitaCommitmentScheme`: how to commit, prove, and
verify, plus the setup and transcript objects those calls thread through.

## Commit, prove, verify

The `commit` / `prove` / `verify` entry points, the `CommitmentProver` and
`CommitmentVerifier` traits, single vs batched (multi-point) openings, and the
shapes of the inputs and proof objects.

**Sources to fold in**

- `crates/akita-pcs/src/scheme/mod.rs`.
- `crates/akita-prover/src/api/scheme.rs` (`CommitmentProver`).
- `crates/akita-types/src/proof/scheme.rs` (`CommitmentVerifier`).
- `crates/akita-pcs/tests/multipoint_batched_e2e.rs`, `batched_aggregated_e2e.rs`.

## Setup and caching

Building public parameters, the shared setup vector reused as A/B/D matrices at
every level, and the optional on-disk setup cache.

**Sources to fold in**

- `crates/akita-setup/src/lib.rs`.
- Paper §3.9 `sec:akita-setup` (packed shared setup), `Setup` in §3.8 `sec:akita-full-pcs`.
- `specs/setup-layout-repack.md` (broader packed-setup direction — roadmap).

## Transcripts in practice

How callers obtain and thread an `AkitaTranscript`, the descriptor preamble that
gets bound first, and what the caller must keep identical between prove and
verify.

**Sources to fold in**

- `crates/akita-transcript/README.md`, `crates/akita-transcript/src/`.
- `crates/akita-pcs/examples/transcript_schedule.rs`.
- Deep dive in [How it works → Transcript and instance binding](../how/transcript.md).
