# AGENTS.md

**Compatibility notice (explicit): This repo makes NO backward-compatibility guarantees. Breaking changes are allowed and expected.**

## Project Overview

Akita is a lattice-based polynomial commitment scheme (PCS) with transparent setup and post-quantum security. Built in Rust. Intended to replace Dory in Jolt.

## CI preflight

Run the cheap repository-wide gates before starting expensive compilation:

```bash
scripts/generate-schedule-artifacts.sh
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
scripts/check-shared-field-identity.sh
scripts/check-external-schedule-artifacts.sh
cargo machete --with-metadata
typos
./scripts/check-doc-guardrails.sh   # when changing book, specs, or docs/
```

CI runs these exact Clippy configurations; all must pass because the feature
graphs differ:

```bash
cargo clippy --all --all-targets --release --no-default-features --features parallel,disk-persistence,transcript-blake2b -- -D warnings
cargo clippy --all --all-targets --release --no-default-features --features transcript-blake2b -- -D warnings
cargo clippy -p akita-pcs --all-targets --release --no-default-features --features parallel,response-model-diagnostics,transcript-blake2b -- -D warnings
```

Run path-specific workflows such as portability, Jolt compatibility, fuzzing,
or profiling when the changed files trigger them. The workflow files under
`.github/workflows/` are the source of truth for their exact commands.

## RTK (token-optimized shell)

Use [`rtk`](https://github.com/rtk-ai/rtk) for verbose dev commands (`rtk cargo test`, `rtk git diff`, etc.) to keep agent context small. Cursor auto-rewrites allowed shell commands via `~/.cursor/cli-config.json`.

**Nextest is not auto-rewritten** — always prefix Nextest invocations with
`rtk`.

For focused feedback, scope Cargo's build graph with `-p`, `--lib`, `--bin`, or
`--test` before adding a nextest `-E` expression. `-E` filters test execution,
not compilation. Use the dev profile while iterating.

For final CI-fidelity validation, copy the current test-pass invocation from
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). That workflow is the
source of truth for the target selectors, Cargo test profile, feature set, and
sharding; do not duplicate the command here. A command that returns a live
session is still running: poll it to an exit code, and inspect its Cargo/rustc
children if compilation is unexpectedly broad or long.

## Documentation

Canonical policy: [`docs/documentation.md`](docs/documentation.md).
Narrative docs live in the [Akita Book](book/README.md); design records in `specs/` until folded ([`specs/PRUNING.md`](specs/PRUNING.md)).

The Book must not cite, mention, or depend on the unpublished Akita paper.
Explain every needed idea in the Book itself. Use current code, live
specifications, and tests as sources. Do not describe a design that exists only
in the draft as current or planned behavior. This restriction stays in force
until maintainers explicitly mark the paper as published and stable.

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
- Generic checked `usize` formulas live in `akita_error::checked`. Use those primitives directly and do not redefine local product, sum, fixed-arity multiplication, range, alignment, division, or power-of-two helpers.
- Keep intentional boundaries: traits, arithmetic primitives, domain/security helpers, named test/bench scenarios. Delete single-use indirection.

## Feature flags

- `parallel` — Rayon parallelization (default)
- `disk-persistence` — disk-backed persistence for some commitment flows
- `logging-transcript` — `LoggingTranscript` schedule events and smell checks
- `response-model-diagnostics` — expensive source and response energy measurements for model calibration

Details: [`book/src/usage/feature-flags.md`](book/src/usage/feature-flags.md).

## Maintainer pointers

| Topic | Where |
|-------|-------|
| Crate map and dependency graph | [`docs/crate-graph.md`](docs/crate-graph.md), [`book/src/how/architecture.md`](book/src/how/architecture.md) |
| Core API types | [`book/src/how/architecture.md`](book/src/how/architecture.md#core-types) |
| CI test timing | [`docs/ci-test-timing.md`](docs/ci-test-timing.md) |
| Profiling harness | [`book/src/usage/profiling.md`](book/src/usage/profiling.md) |
| Transcript hardening | [`book/src/how/transcript.md`](book/src/how/transcript.md) |
| Offline SIS table regen | `cargo run -p akita-sis-estimator --release --features parallel --example infinity_width_table -- --format rust-split --profile local-minimum` |
| Direct generated-schedule SIS report | `cargo run --release -p akita-planner --features catalog-security --example catalog_security -- --check` |
| Jolt verifier bench | [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md) |
