# Architecture overview

> **Status:** stub. Part of the initial Akita Book scaffold.

How the workspace is organized and how a single call flows through it. Lead the
reader with the end-to-end lifecycle, then give them the crate map as the
reference index.

## Crate map

The workspace members and their dependency edges, including the
`akita-config → akita-planner` arrow (the planner sits *below* config) and how
`akita-verifier` reaches the planner transitively. Render the dependency graph
and state the ownership rules (who may name a `CommitmentConfig`, who is
`Cfg`-free).

**Sources to fold in**

- `docs/crate-graph.md` (rewritten), `scripts/check-crate-deps.sh`.
- `AGENTS.md` (Crate Structure section — freshest index).
- `crates/akita-witness/` (`PolynomialView`, `WitnessProvider`; shared polyops vocabulary).
- `crates/akita-pcs/src/scheme/mod.rs`.

## End-to-end lifecycle

One numbered walkthrough of `commit → prove → verify`, with the key fact up
front: the same batched API dispatches to three proof families (ZeroFold /
terminal-root / fold + recursive suffix) purely from the planner's schedule
shape.

**Sources to fold in**

- Council architecture report §1 (full numbered flow with file/line citations).
- `crates/akita-prover/src/protocol/flow/inputs.rs`, `flow/root_fold.rs`, `flow/recursive.rs`.
- `crates/akita-verifier/src/protocol/core/verify.rs` (`batched_verify`).
- Paper §3.8 `sec:akita-full-pcs` (`fig:akita-scheme`, the four PCS algorithms).
