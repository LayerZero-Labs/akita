# AGENTS.md

**Compatibility notice (explicit): This repo makes NO backward-compatibility guarantees. Breaking changes are allowed and expected.**

## Project Overview

Akita is a lattice-based polynomial commitment scheme (PCS) with transparent setup and post-quantum security. Built in Rust. Intended to replace Dory in Jolt.

## Essential Commands

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

## Crate Structure

Workspace members live under `crates/`.

- `akita-field` — field traits, concrete prime/extension fields, wide/packed helpers, field FFT helpers, parallel macros, and core errors
- `akita-serialization` — serialization, validation, and compression traits
- `akita-algebra` — module/vector containers, NTTs, cyclotomic rings, sparse challenges, polynomial utilities, and algebra backends over `akita-field` scalars
- `akita-transcript` — Fiat-Shamir transcript traits, hash transcript implementations, and labels
- `akita-challenges` — Fiat-Shamir challenge sampling helpers
- `akita-sumcheck` — generic sumcheck proof types, traits, drivers, compact folding, batching, and accumulation helpers
- `akita-types` — shared proof, setup, schedule, layout, commitment, transcript-append, and PRG data shapes
- `akita-config` — concrete runtime config presets and config-backed schedule/SIS policy
- `akita-setup` — config-backed setup construction and optional setup cache persistence
- `akita-verifier` — verifier replay without prover-only polynomial backends
- `akita-prover` — commitment, proving, setup expansion, recursive witness construction, ring-switch witnesses, and polynomial backends
- `akita-scheme` — end-to-end `AkitaCommitmentScheme` orchestration
- `akita-planner` — offline schedule search and proof-size/security planning
- `akita-pcs` — umbrella package with examples, benches, integration tests, and broad public re-exports

## Key Abstractions

- `AkitaCommitmentScheme` — top-level PCS `commit` / `prove` / `verify`
- `CommitmentConfig` + `LevelParams` — recursion schedule, layout, and per-level configuration
- `DensePoly`, `OneHotPoly`, `AkitaPolyOps` — polynomial backends consumed by the scheme
- `BlockOrder` — explicit root-vs-recursive opening split convention
- `AkitaBatchedProof`, `AkitaBatchedRootProof`, `AkitaLevelProof`, `AkitaProofStep` — serialized proof structure (singleton openings are the 1x1 special case of the batched proof)
- `Blake2bTranscript`, `Transcript` — Fiat-Shamir layer

## Verifier No-Panic Contract

Verifier-reachable execution is a no-panic boundary.
Any malformed verifier-facing proof, setup, schedule, public claim, opening point, commitment, direct witness, or transcript input must be rejected with `AkitaError` or `SerializationError`, not by panicking.

This applies to `akita-verifier` and any verifier-reachable code in `akita-types`, `akita-serialization`, `akita-algebra`, `akita-sumcheck`, `akita-transcript`, `akita-challenges`, and verifier-used `akita-field` paths.
Do not add verifier-reachable `panic!`, `assert!`, `assert_eq!`, `expect`, `unwrap`, `unreachable!`, unchecked indexing/slicing, overflow-prone shape arithmetic, or unbounded allocation unless an earlier verifier boundary has clearly validated the invariant.

Prefer strengthening existing validation at deserialization, setup construction, schedule selection, `LevelParams` construction, and verifier API entry points.
Keep hot verifier arithmetic paths fast: do not add slow fallback evaluators, compatibility shims, or repeated defensive checks inside tight loops when the invariant can be enforced once at the boundary.
Prover-only panics are acceptable for now if they are not reachable from verifier paths.

## Feature Flags

- `parallel` — Rayon parallelization (default)
- `disk-persistence` — disk-backed persistence paths used by some commitment flows

## Profiling

Canonical run:

```bash
AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release --example profile
```

Knobs:

- `AKITA_MODE=full|onehot|all|full_d128|onehot_d64|full_d32|onehot_d32`
- `AKITA_NUM_VARS=<n>` — number of variables, default `25`
- `AKITA_PROFILE_TRACE=0|1` — write a Perfetto JSON trace to `profile_traces/`, default `1`
- `AKITA_PROFILE_LOG=<filter>` — tracing filter, default `trace`
- `AKITA_PROFILE_ANSI=0|1` — ANSI log colors, default `1`
- `AKITA_PROFILE_SPAN_CLOSES=0|1` — emit close-span timing events, default `1`
- `AKITA_ALLOW_DEBUG_PROFILE=1` — bypass the `--release` guard for debugging only
- Default features enable `parallel`; use `RAYON_NUM_THREADS=<n>` to cap threads or `--no-default-features` to profile without Rayon

## Running the verifier inside Jolt

End-to-end pipeline for running the Akita PCS verifier inside a Jolt
zkVM guest program, with cycle-tracking instrumentation. Lives at
`profile/akita-recursion/` as a **standalone sub-workspace** (excluded
from this workspace, pinned to Rust `1.94` + RISC-V targets, applies
Jolt's `[patch.crates-io]` overrides for `arkworks-algebra`). Full
runbook, knob reference, current cycle results, and open follow-ups
are in [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md).

Members:

* `glue`     — shared verifier-input blob format (`AkitaJoltInputs`).
* `artifact` — host binary that produces the blob from a real Akita
               prover run.
* `host`     — host driver: compiles the guest to RISC-V, runs the Jolt
               prover, prints per-marker cycle counts.
* `guest`    — `#[jolt::provable]` RISC-V program that runs the Akita
               verifier.

```bash
cd profile/akita-recursion

# 1. Build the host binaries. Default-members build only `host` and
#    `artifact` (`guest` is built transitively for RISC-V by the Jolt
#    CLI; building it directly for the host platform fails because the
#    macro emits host-only helpers).
cargo build --release

# 2. Produce the verifier-input blob (OneHot, D=32, single-poly opening).
AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact

# 3. Run the host driver. Compiles the guest to RISC-V, runs the Akita
#    verifier inside the Jolt emulator, proves the execution trace, and
#    reports per-marker cycle counts (`deserialize_input`,
#    `transcript_init`, `akita_verify`).
AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin

# Fast iteration (trace only, skip the Jolt prover step):
./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs.bin

# Debug a guest panic by routing stderr to the host and emitting full
# backtraces. The guest's `#[jolt::provable]` attribute is currently
# `backtrace = "off"` (cheaper cycles); flip it to `"dwarf"` for a
# single diagnostic iteration first.
JOLT_BACKTRACE=full ./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs.bin
```

Knobs:

- `AKITA_NUM_VARS=<n>` — host artifact arity. Default `20` produces a
  ~4 MiB blob (full prove path runs end-to-end). Set to `32` to match
  the canonical target (~576 MiB blob; only the trace-only path is
  feasible today — the trace is ≈ 11.3 G cycles which exceeds the
  guest's `max_trace_length = 4 G`).
- `AKITA_RECURSION_BLOB=<path>` — output / input path for the blob;
  defaults to `target/akita_recursion_inputs.bin`.
- `--target-dir` — Jolt's per-program build directory (default
  `/tmp/akita-recursion-targets`). Delete to force a clean guest
  rebuild.
- `AKITA_RECURSION_LOG=<filter>` — `tracing-subscriber` filter on the
  host driver (default `info`).
- `JOLT_BACKTRACE=full` — symbol-resolved guest backtraces (requires
  `backtrace = "dwarf"` on the `#[jolt::provable]` attribute, which is
  already set).
