# Akita Soundness Audit Checklist

This is a living checklist for internal reviews.
Update it when a protocol invariant changes or a new verifier-facing surface is added.

## Core Invariants

- Verifier-only crates remain independent of prover, planner, and umbrella application crates.
- Verifier-reachable code never panics on malformed public verifier inputs; it returns `AkitaError` or `SerializationError`.
- Any remaining verifier-reachable indexing, slicing, assertions, unwraps, expects, overflow-prone shape arithmetic, or shape-derived allocation is guarded by earlier boundary validation.
- Transcript labels and challenge order match the protocol specification.
- Serialized proof and setup bytes decode canonically and reject malformed encodings.
- Proof-shape metadata determines all non-self-describing witness and claim lengths.
- Public setup and verifier parameters cannot be confused across incompatible schedules.
- Unsafe kernels preserve their documented layout, aliasing, and bounds invariants.

## Review Commands

```bash
cargo fmt --all --check
cargo clippy --all --all-targets --all-features -- -D warnings
cargo clippy --all --all-targets --no-default-features -- -D warnings
cargo nextest run --no-default-features --features parallel,disk-persistence
cargo nextest run --all-features
cargo doc -q --no-deps --all-features
cargo deny check bans licenses sources advisories
scripts/check-crate-deps.sh akita-verifier
scripts/check-crate-deps.sh akita-prover
scripts/check-crate-deps.sh akita-config
scripts/check-crate-deps.sh akita-derive
scripts/check-crate-deps.sh akita-setup
```

## When To Reopen The Checklist

Re-run this review when a PR changes:

- transcript labels or challenge derivation,
- proof, setup, or claim serialization,
- verifier acceptance logic,
- verifier input validation or verifier-reachable panic-shaped code,
- dependency sources or Git revisions,
- unsafe code,
- parameter schedules or generated configuration tables,
- public API boundaries between verifier, prover, setup, and planner crates.
