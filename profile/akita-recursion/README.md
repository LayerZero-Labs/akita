# `akita-recursion` — Akita verifier inside Jolt

Runs the Akita PCS verifier inside a Jolt zkVM guest program and reports
per-phase cycle counts (`deserialize_input`, `transcript_init`,
`akita_verify`). End-to-end this also produces a SNARK of the verifier
execution and confirms Jolt accepts it.

This directory is a **standalone Cargo sub-workspace** (it's excluded
from the parent Akita workspace). It pins Rust `1.95` plus the
RISC-V targets and applies Jolt's `[patch.crates-io]` overrides for
`arkworks-algebra`.

## Crates

| Crate        | Kind | Purpose                                                          |
| ------------ | ---- | ---------------------------------------------------------------- |
| `glue/`      | lib  | Shared verifier-input blob format (`AkitaJoltInputs<F, D>`).     |
| `artifact/`  | bin  | Runs the Akita prover and writes the verifier-input blob.        |
| `host/`      | bin  | Compiles the guest, runs Jolt prove/verify, prints cycle counts. |
| `guest/`     | bin  | `#[jolt::provable]` RISC-V program that runs the Akita verifier. |

## Quick start (`nv=32`, adaptive OneHot — canonical target)

You need the [Jolt CLI](https://github.com/a16z/jolt) installed
(`cargo install --path .` from a clone of `jolt` at the same rev this
crate pins, `2509bdcea9bb...`). The first prove run downloads a ~30 GB
Dory PCS setup table to `~/Library/Caches/dory/dory_38.urs` (~85 s on
first run, instant on subsequent).

**All commands below assume you're in `profile/akita-recursion/`.**

```bash
cd profile/akita-recursion

# 1. Build the host binaries.
cargo build --release

# 2. Generate the verifier-input blob (artifact prints exact size).
#    REQUIRED before step 3 — `host` reads this file from disk.
AKITA_NUM_VARS=32 \
    AKITA_RECURSION_BLOB=target/akita_recursion_inputs_nv32.bin \
    ./target/release/akita-recursion-artifact

# 3. Compile the guest to RISC-V, emulate it, and report cycle markers.
#    Start with trace-only (no Jolt prover) when measuring a new adaptive
#    schedule. Confirm the trace fits the current `max_trace_length = 4 G`
#    before running the full prover (see "Open follow-ups" below).
#    `--trace-output /dev/null` keeps the raw trace bytes off disk while
#    preserving the cycle-marker output.
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --trace-only \
    --trace-output /dev/null \
    --input target/akita_recursion_inputs_nv32.bin
```

Expected output shape (rerun `--trace-only` for current adaptive numbers):

```
"deserialize_input": … (dominated by expanded verifier-setup decode)
"transcript_init":   …
"akita_verify":      …
trace length: …
trace done
```

Most of `deserialize_input` is decoding the expanded verifier-setup matrix
that lives inside the blob; the proof itself is a tiny fraction.

## Running the full prove pipeline

The full pipeline (Dory preprocessing → Jolt prove → Jolt verify) runs
end-to-end at arities where the trace fits under `max_trace_length = 4 G`.
Start with `AKITA_NUM_VARS=20`, measure it with `--trace-only`, and then remove
`--trace-only` once the trace bound is confirmed:

```bash
AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin
```

On success the host reports `Akita-in-Jolt proof OK` with
`is_valid=true` and `guest_panic=false`.

The guest output is a status code: `0` means verification succeeded, `1`
means input decoding failed, and `2` means the verifier rejected the proof.
Malformed proofs therefore produce a proved nonzero result rather than a guest
panic.

## Debugging guest panics

The guest enables `jolt/stdout` so panic messages reach the host. The
`#[jolt::provable]` attribute currently uses `backtrace = "off"`
(measured to shave ~0.4 % off the trace by skipping
`-Cforce-frame-pointers=yes`); flip it to `backtrace = "dwarf"` for a
single diagnostic iteration if a panic comes back, then run with:

```bash
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    JOLT_BACKTRACE=full AKITA_RECURSION_LOG=info \
    ./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs_nv32.bin
```

To force a clean guest rebuild:

```bash
rm -rf /tmp/akita-recursion-targets /tmp/jolt-guest-targets
```

## Environment variables

| Variable                  | Default                                  | Effect                                  |
| ------------------------- | ---------------------------------------- | --------------------------------------- |
| `AKITA_NUM_VARS`          | `20`                                     | Polynomial arity for the prover.        |
| `AKITA_RECURSION_BLOB`    | `target/akita_recursion_inputs.bin`      | Output path for the blob (`artifact`).  |
| `AKITA_RECURSION_LOG`     | `info`                                   | `tracing-subscriber` filter (`host`).   |
| `ZEROOS_GUEST_RUSTFLAGS`  | unset                                    | Pass `-Zunstable-options` when Rust requires it for Jolt's custom `riscv64imac-zero-linux-musl` target. |
| `JOLT_BACKTRACE`          | unset                                    | `full` ⇒ symbolic guest backtraces.     |
| `AKITA_ALLOW_DEBUG_PROFILE` | unset                                  | `1` ⇒ bypass `--release` guard in `artifact`. |

## CLI flags (`akita-recursion-host`)

| Flag                  | Default                              | Description                                  |
| --------------------- | ------------------------------------ | -------------------------------------------- |
| `--input <path>`      | `target/akita_recursion_inputs.bin`  | Path to the blob produced by `artifact`.     |
| `--target-dir <path>` | `/tmp/akita-recursion-targets`       | Jolt's per-program build cache.              |
| `--trace-output <path>` | `<target-dir>/akita_verify.trace`  | Trace file path for `--trace-only`.          |
| `--trace-only`        | off                                  | Skip preprocessing + Jolt prove/verify.      |

## How it works

1. **`artifact`** runs `AkitaCommitmentScheme::<fp128::OneHot>` →
   `setup_prover` → `commit` → `batched_prove` over a synthetic OneHot
   polynomial, sanity-verifies on the host, and serializes
   `(transcript_domain, num_vars, opening_point, opening, commitment,
   verifier_setup, proof_shape, proof)` into a single blob via
   [`AkitaJoltInputs::write_to_bytes`](glue/src/lib.rs).
2. **`host`** loads the blob, compiles the guest to
   `riscv64imac-zero-linux-musl` via the Jolt CLI, runs Jolt's
   preprocess/prove/verify (or just the trace under `--trace-only`),
   and forwards per-marker cycle counts through `tracing`.
3. **`guest`** (running inside the Jolt RISC-V emulator) decodes the
   blob and invokes `akita_verifier::batched_verify` directly —
   bypassing `AkitaCommitmentScheme::batched_verify`, which would otherwise
   call `Instant::now()` (the Jolt runtime doesn't implement
   `clock_gettime`, and the guest aborts there). Three
   `start_cycle_tracking` / `end_cycle_tracking` pairs wrap
   `deserialize_input`, `transcript_init`, and the verifier kernel.
   The guest constructs an unbound verifier transcript and the verifier binds
   the canonical instance descriptor; it must not use a prover-side placeholder
   transcript, because Spongefish prover state may ask for entropy that the Jolt
   guest runtime does not provide.
   This profile is a trusted host-artifact benchmark: the guest decodes the
   verifier setup through the explicitly trusted cached-matrix path. Seed/matrix
   shape metadata and field elements are still validated, but the guest skips
   checking that the expanded setup matrix coefficients equal the matrix derived
   from the seed because the blob is produced and sanity-checked by the
   host-side artifact generator. Plain `--features guest` builds use strict
   setup decoding; the host binary sets
   `AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT=1` before Jolt compiles the
   benchmark RISC-V ELF, because this pinned Jolt SDK hard-codes the guest
   feature list to `guest`. A production recursion circuit must use strict
   setup validation or bind an externally checked setup commitment.

## Adaptive dimension pin

The artifact, host, and guest share a fixed source-view dimension `D = 256`, the
adaptive preset's root envelope. The generated `nv=20` and `nv=32` scalar rows both use
A=D256 at the root and may transition to smaller per-level dimensions. The
artifact rejects any requested arity whose selected root A dimension does not
match this Jolt monomorphization.

## Historical optimization results at `nv=20` (fixed D64)

Two guest-level changes landed during bring-up. They live in the git
history; numbers measured against the D=64 OneHot configuration are:

| Configuration                              | Trace length    | Δ vs. previous |
| ------------------------------------------ | --------------- | -------------- |
| `backtrace = "dwarf"`, `input: Vec<u8>`    | 102,383,700     | (baseline)     |
| `backtrace = "off"`,   `input: Vec<u8>`    | 102,011,269     | **−0.4 %**     |
| `backtrace = "off"`,   `input: &[u8]`      | **65,283,025**  | **−36.0 %**    |

The `Vec<u8>` → `&[u8]` switch shaved ~36 M cycles off the trace
without changing any cycle marker, because the macro-generated
`postcard::take_from_bytes::<Vec<u8>>(input_slice)` decoded the
1.1 MiB input one byte at a time *before* the user function ran
(≈30 cycles per byte × 1.1 M bytes ≈ 33 M cycles). Postcard's `&[u8]` deserialization is zero-copy: read the length prefix, return a
slice pointing into the input region. At large `nv` the saving scales with blob
size.

## Open follow-ups

1. **Remeasure and run a full adaptive prove at `nv=32`** on a beefier host.
   Requires:
   - Measuring the adaptive trace and, if necessary, increasing
     `max_trace_length` beyond the current 4 G limit.
   - Server-class memory headroom (guest heap is sized for large nv=32 blobs).
   - Estimating wall clock from the newly measured trace rather than the
     historical fixed-D64 result.

2. **Make `deserialize_input` cheaper.** At `nv=32` it dominates the trace.
   Most of that is decoding the expanded verifier-setup matrix. Options:
   - Ship just the `public_matrix_seed` (32 bytes) and re-derive the
     matrix inside the guest. Trades deserialization cycles for
     matrix-expansion cycles (probably ~similar order, with a much
     smaller input region and cleaner cycle attribution).
   - Pre-decompose the setup into Lagrange coordinates that don't
     need the full matrix shape inside the guest.

3. **Finer markers.** Current set is the minimum the user asked for.
   Splitting `akita_verify` into per-level markers (e.g. `root_level`,
   `fold_levels`, `final_witness`) would need a tiny instrumentation
   tweak in the guest (re-implement the iteration over
   `proof.fold_levels()` with markers around each call).

4. **Upstreaming candidates** — small, mechanical changes that would
   benefit any future Jolt integration with Akita:
   - If the umbrella verification entry point ever becomes timer-free, the
     guest can reconsider using it. Until then, the guest should remain free of
     `akita-pcs`, `akita-prover`, and `akita-setup` dependencies.
   - `AkitaSerialize` / `AkitaDeserialize` impls for proof-shape types
     (already added under `akita-types::proof` and used by the `glue`
     crate).
