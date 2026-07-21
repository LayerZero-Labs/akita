# How it works

> **Status:** stub. Part of the initial Akita Book scaffold.

The inner workings of Akita, beyond what an application developer needs. Read
this if you are poking around the codebase, have read the lineage papers and
want to know how things fit together in practice, or want to contribute.

Lead with the end-to-end lifecycle, then branch. The same `batched_prove` /
`batched_verify` API always uses a folded schedule: a root fold, at least one
suffix fold, and a terminal cleartext witness. Inputs for which two folds are
not supported fail during schedule selection instead of selecting a degenerate
proof family.

This part covers, in reading order:

- [Architecture overview](./architecture.md) — crate map and end-to-end lifecycle.
- [Configuration and planning](./configuration.md) — `CommitmentConfig`, schedules, the planner.
- [Setup and commitment](./commitment.md) — the shared setup and the Ajtai commitment.
- [Transcript and instance binding](./transcript.md) — Fiat-Shamir and the descriptor preamble.
- [The proving protocol](./proving/proving.md) — the per-level fold pipeline (its own section).
- [Recursion and proof shape](./recursion.md) — chaining folds and proof anatomy.
- [Verification](./verification.md) — per-level replay and the no-panic contract.
- [Security model](./security.md) — the hardness assumption and norm regimes.
- [Optimizations](./optimizations.md) — the implementation-level speedups.

## Sources to fold in

- Council architecture report (numbered end-to-end flow, dispatch table).
- `crates/akita-prover/src/protocol/core/`, `crates/akita-verifier/src/protocol/core/`.
- Paper §3 `sec:akita-recap` (the protocol, end to end).
