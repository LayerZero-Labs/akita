# AGENTS.md

**Compatibility notice (explicit): This repo makes NO backward-compatibility guarantees. Breaking changes are allowed and expected.**

## Project Overview

Akita is a lattice-based polynomial commitment scheme (PCS) with transparent setup and post-quantum security. Built in Rust. Intended to replace Dory in Jolt.

## CI preflight

Run the cheap repository-wide gates before starting expensive compilation:

```bash
cargo fmt --all --check
taplo fmt --check
scripts/test-rust-file-lines.sh
scripts/check-rust-file-lines.sh --no-baseline
python3 -m unittest discover -s scripts/tests -p "test_*.py"
scripts/check-crate-deps.sh akita-verifier
scripts/check-crate-deps.sh akita-prover
scripts/check-crate-deps.sh akita-config
scripts/check-crate-deps.sh akita-planner
scripts/check-crate-deps.sh akita-setup
cargo machete --with-metadata
typos
./scripts/check-doc-guardrails.sh   # when changing book, specs, or docs/
```

CI runs these exact Clippy configurations; all must pass because the feature
graphs differ:

```bash
cargo clippy --all --all-targets --release --no-default-features --features parallel,disk-persistence -- -D warnings
cargo clippy --all --all-targets --release --no-default-features -- -D warnings
cargo clippy -p akita-field --all-targets --release --features jolt-compat -- -D warnings
```

Run path-specific workflows such as portability, Jolt compatibility, fuzzing,
or profiling when the changed files trigger them. The workflow files under
`.github/workflows/` are the source of truth for their exact commands.

## RTK (token-optimized shell)

Use [`rtk`](https://github.com/rtk-ai/rtk) for verbose dev commands (`rtk cargo test`, `rtk git diff`, etc.) to keep agent context small. Cursor auto-rewrites allowed shell commands via `~/.cursor/cli-config.json`.

**Nextest is not auto-rewritten** — always prefix explicitly:

```bash
rtk cargo nextest run --profile ci --no-default-features --features parallel,disk-persistence
```

For focused feedback, scope Cargo's build graph with `-p`, `--lib`, `--bin`, or
`--test` before adding a nextest `-E` expression. `-E` filters test execution,
not compilation. Use the dev profile while iterating; reserve `--cargo-profile
ci-test` for final CI-fidelity validation.

The CI test target set is `--lib --bins --tests`; keep those selectors when
reproducing CI so Cargo does not compile examples. A command that returns a live
session is still running: poll it to an exit code, and inspect its Cargo/rustc
children if compilation is unexpectedly broad or long.

## Documentation

Canonical policy: [`docs/documentation.md`](docs/documentation.md).
Narrative docs live in the [Akita Book](book/README.md); design records in `specs/` until folded ([`specs/PRUNING.md`](specs/PRUNING.md)).

- **Hard (CI):** dead symbols in live specs/docs, `Book-chapter:` paths, `mdbook build` — [`scripts/check-doc-guardrails.sh`](scripts/check-doc-guardrails.sh).
- **Soft (PR comment):** blast-radius advisory — [`docs/doc-blast-radius.json`](docs/doc-blast-radius.json).

## Verifier no-panic contract

Verifier-reachable code must reject malformed input with `AkitaError` or `SerializationError`, never panic.
Do not add verifier-reachable `panic!`, `assert!`, `unwrap`, unchecked indexing, or unbounded allocation without prior validation at a boundary.
Full contract: [`book/src/how/verification.md`](book/src/how/verification.md) and [`docs/verifier-contract.md`](docs/verifier-contract.md).

## Single source of truth (no wrapper slop)

Follow the [#244](https://github.com/LayerZero-Labs/akita/pull/244) cutover: **one canonical function per concept**; call it directly.

- Do not add thin wrappers, pass-through aliases, or `_for_level` helpers that only recompose existing APIs.
- Type methods may assemble `self` into arguments, but the logic lives in one place, not duplicated across siblings.
- If `A` needs the output of `B`, call `B` (or extend `B`); do not introduce `C` that forwards to `B`.
- Security and sizing contracts must use the same primitives the verifier enforces. No split-brain where certification and MSIS pricing read different bounds.
- Keep intentional boundaries: traits, arithmetic primitives, domain/security helpers, named test/bench scenarios. Delete single-use indirection.

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
| Offline SIS table regen | `cargo run -p akita-sis-estimator --release --features parallel --example infinity_width_table -- --format rust-split --profile local-minimum` |
| Jolt verifier bench | [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md) |
