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

## Quick start (`nv=32`, OneHot D=64 — canonical target)

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
#    Trace-only (no Jolt prover) because at nv=32 the trace is on the order of
#    ~8 G cycles at D=64, still above the current `max_trace_length = 4 G` in
#    `#[jolt::provable]` attribute (see "Open follow-ups" below).
#    `--trace-output /dev/null` keeps the raw trace bytes off disk while
#    preserving the cycle-marker output.
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --trace-only \
    --trace-output /dev/null \
    --input target/akita_recursion_inputs_nv32.bin
```

Expected output (Apple Silicon laptop, ≈ 22 min wall clock; D=64 OneHot,
order-of-magnitude — rerun `--trace-only` for fresh numbers):

```
"deserialize_input": … (dominated by expanded verifier-setup decode)
"transcript_init":   …
"akita_verify":      …
trace length: ~8 G cycles
trace done
```

Most of `deserialize_input` is decoding the expanded verifier-setup matrix
that lives inside the blob; the proof itself is a tiny fraction.

## Running the full prove pipeline

The full pipeline (Dory preprocessing → Jolt prove → Jolt verify) runs
end-to-end at smaller arities where the trace fits under
`max_trace_length = 4 G`. Drop the `AKITA_NUM_VARS` override down (e.g.
`AKITA_NUM_VARS=20` produces a ≈ 4 MiB blob and a ≈ 150 M-cycle trace)
and remove `--trace-only`:

```bash
AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin
```

On success the host reports `Akita-in-Jolt proof OK` with
`is_valid=true` and `guest_panic=false`.

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

1. **`artifact`** runs `AkitaCommitmentScheme::<64, fp128::D64OneHot>` →
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
   bypassing `akita-scheme::batched_verify`, which would otherwise
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

## Why we pin D=64 (not D=32) for Jolt cycle benches

A natural surprise: a smaller ring `D` can make on-CPU proofs smaller, but
**D=32 is not a valid A-role fold degree** (`d_a ≥ 64`) after the ring-dim
cutover, and historical D=32 Jolt traces were **more** expensive than D=64 at
equal `num_vars`. Three compounding effects explained the old D=32 penalty:

1. **More recursion levels.** Folding nv=32 to a tractable terminal
   takes 6 levels at D=64 vs 7 at D=32. Each level adds a full
   sumcheck-with-range-checks verification step.
2. **Larger verifier-setup matrix at D=32.** Halving `D` does not halve the
   matrix — Ajtai security forces the stride (column count) to grow to
   compensate. Net: the old D=32 blob was roughly **4.5×** larger than D=64.
3. **Cycle count ≠ wall clock.** On a real CPU, fp128 ops at D=32 vs
   D=64 don't differ much in time (SIMD, cache prefetch, wide
   multiply). Inside `riscv64imac` emulation every fp128 mul is a
   fixed-length sequence of 64-bit instructions, each counted as a
   cycle. Smaller-D work doesn't compress into fewer RV64
   instructions.

For reference: OneHot **D=64** `nv=32` produces a trace on the order of
**~8 G cycles** (~30% cheaper inside Jolt than the retired D=32 configuration)
while remaining the production fp128 preset.

## Optimization history at `nv=20` (D=64)

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

1. **Full prove at `nv=32`** on a beefier host. Requires:
   - Bumping `max_trace_length` past ~8 G in the `#[jolt::provable]`
     attribute (currently 4 G — fine for `nv ≤ 20`, insufficient at
     `nv=32`).
   - Server-class memory headroom (guest heap is sized for large nv=32 blobs).
   - Expected wall clock at typical zkVM throughput (~500 kHz):
     **~6 h+** of proving.

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
   - If the public trait entry point ever becomes timer-free and verifier-only,
     it should delegate to the same `akita_config::batched_verify_with_config`
     adapter; the guest should remain free of `akita-scheme`, `akita-prover`,
     and `akita-setup` dependencies.
   - `AkitaSerialize` / `AkitaDeserialize` impls for proof-shape types
     (already added under `akita-types::proof` and used by the `glue`
     crate).
