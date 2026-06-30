# AGENTS.md

**Compatibility notice (explicit): This repo makes NO backward-compatibility guarantees. Breaking changes are allowed and expected.**

## Project Overview

Akita is a lattice-based polynomial commitment scheme (PCS) with transparent setup and post-quantum security. Built in Rust. Intended to replace Dory in Jolt.

## Essential Commands

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh   # when changing book, specs, or docs/
```

## Documentation

Canonical policy: [`docs/documentation.md`](docs/documentation.md).
Narrative docs live in the [Akita Book](book/README.md); design records in `specs/` until folded ([`specs/PRUNING.md`](specs/PRUNING.md)).

- **Hard (CI):** dead symbols in live specs/docs, `Book-chapter:` paths, `mdbook build` — [`scripts/check-doc-guardrails.sh`](scripts/check-doc-guardrails.sh).
- **Soft (PR comment):** blast-radius advisory — [`docs/doc-blast-radius.json`](docs/doc-blast-radius.json).

## Verifier no-panic contract

Verifier-reachable code must reject malformed input with `AkitaError` or `SerializationError`, never panic.
Do not add verifier-reachable `panic!`, `assert!`, `unwrap`, unchecked indexing, or unbounded allocation without prior validation at a boundary.
Full contract: [`book/src/how/verification.md`](book/src/how/verification.md) and [`docs/verifier-contract.md`](docs/verifier-contract.md).

## Feature flags

- `parallel` — Rayon parallelization (default)
- `disk-persistence` — disk-backed persistence for some commitment flows
- `logging-transcript` — `LoggingTranscript` schedule events and smell checks

Details: [`book/src/usage/feature-flags.md`](book/src/usage/feature-flags.md).

## Maintainer pointers

| Topic | Where |
|-------|-------|
| Crate map and dependency graph | [`docs/crate-graph.md`](docs/crate-graph.md), [`book/src/how/architecture.md`](book/src/how/architecture.md) |
| Core API types | [`book/src/how/architecture.md`](book/src/how/architecture.md#core-types) |
| CI test timing | [`docs/ci-test-timing.md`](docs/ci-test-timing.md) |
| Profiling harness | [`book/src/usage/profiling.md`](book/src/usage/profiling.md) |
| Transcript hardening | [`specs/transcript-hardening.md`](specs/transcript-hardening.md) |
| Offline SIS table regen | `cargo run -p akita-sis-estimator --release --features parallel --example euclidean_width_table -- --format rust-split` |
| Jolt verifier bench | [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md) |
