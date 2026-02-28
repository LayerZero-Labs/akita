# AGENTS.md

**Compatibility notice (explicit): This repo makes NO backward-compatibility guarantees. Breaking changes are allowed and expected.**

## Project Overview

Hachi is a lattice-based polynomial commitment scheme (PCS) with transparent setup and post-quantum security. Built in Rust. Intended to replace Dory in Jolt.

## Essential Commands

```bash
cargo clippy --all --message-format=short -q -- -D warnings
cargo fmt -q
cargo test          # no nextest yet
```

## Crate Structure

Two workspace members: `hachi-pcs` (root) and `derive` (proc macros).

- `src/primitives/` — Core traits: `FieldCore`, `Module`, `MultilinearLagrange`, `Transcript`, serialization
- `src/algebra/` — Concrete backends: prime fields, extension fields, cyclotomic rings, NTT, domains
- `src/protocol/` — Protocol layer: commitment, prover, verifier, opening (ring-switch), challenges, transcript
- `src/error.rs` — Error types

## Key Abstractions

- `CommitmentScheme` / `StreamingCommitmentScheme` — top-level PCS traits
- `FieldCore` + `PseudoMersenneField` + `Module` — arithmetic over lattice-friendly fields and rings
- `MultilinearLagrange` — multilinear polynomial in Lagrange basis
- `Transcript` — Fiat-Shamir

## Feature Flags

- `parallel` — Rayon parallelization
