# How it works

> **Status:** stub. Part of the initial Akita Book scaffold.

The inner workings of Akita, beyond what an application developer needs. Read
this if you are poking around the codebase, have read the lineage papers and
want to know how things fit together in practice, or want to contribute.

Lead with the end-to-end lifecycle, then branch. The same `batched_prove` /
`batched_verify` API always uses a folded schedule: a root fold, zero or more
recursive folds, and a terminal cleartext witness. The root may bind the
terminal directly when no recursive fold is needed. Schedule selection fails
only when the audited fold domain contains no valid complete schedule.

This part covers, in reading order:

- [Architecture overview](./architecture.md) — crate map and end-to-end lifecycle.
- [Configuration and planning](./configuration.md) — `CommitmentConfig`, schedules, the planner.
- [Setup and commitment](./commitment.md) — the shared setup and the Ajtai commitment.
- [Transcript and instance binding](./transcript.md) — Fiat-Shamir and the descriptor preamble.
- [The proving protocol](./proving/proving.md) — the per-level fold pipeline (its own section).
- [Recursion and proof shape](./recursion.md) — chaining folds and proof anatomy.
- [Setup offloading](./setup-offloading.md) — replacing selected online setup
  scans with prepared prefix commitments and Stage 3 proofs.
- [Verification](./verification.md) — per-level replay and the no-panic contract.
- [Security model](./security.md) — the hardness assumption and norm regimes.
- [Optimizations](./optimizations.md) — the implementation-level speedups.

## Sources to fold in

- Council architecture report (numbered end-to-end flow, dispatch table).
- `crates/akita-prover/src/protocol/core/`, `crates/akita-verifier/src/protocol/core/`.
