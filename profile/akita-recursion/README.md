# `akita-recursion` — Akita verifier inside Jolt

Runs the Akita PCS verifier inside a Jolt zkVM guest program and reports
per-phase cycle counts (`deserialize_input`, `transcript_init`,
`akita_verify`). End-to-end this also produces a SNARK of the verifier
execution and confirms Jolt accepts it.

This directory is a **standalone Cargo sub-workspace** (it's excluded
from the parent Hachi workspace). It pins Rust `1.94` plus the
RISC-V targets and applies Jolt's `[patch.crates-io]` overrides for
`arkworks-algebra`.

## Crates

| Crate        | Kind | Purpose                                                          |
| ------------ | ---- | ---------------------------------------------------------------- |
| `glue/`      | lib  | Shared verifier-input blob format (`AkitaJoltInputs<F, D>`).     |
| `artifact/`  | bin  | Runs the Akita prover and writes the verifier-input blob.        |
| `host/`      | bin  | Compiles the guest, runs Jolt prove/verify, prints cycle counts. |
| `guest/`     | bin  | `#[jolt::provable]` RISC-V program that runs the Akita verifier. |

## Quick start

You need the [Jolt CLI](https://github.com/a16z/jolt) installed
(`cargo install --path .` from a clone of `jolt` at the same rev this
crate pins, `2509bdcea9bb...`). The first run downloads a ~30 GB Dory
PCS setup table to `~/Library/Caches/dory/dory_38.urs` (~85 s on first
run, instant on subsequent).

**All commands below assume you're in `profile/akita-recursion/`.**

```bash
cd profile/akita-recursion

# 1. Build the host binaries.
cargo build --release

# 2. Generate the verifier-input blob (~1.1 MiB at nv=20).
#    REQUIRED before step 3 — `host` reads this file from disk.
AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact

# 3. Run the full pipeline: compile guest → emulate → prove → verify.
AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin
```

Expected output (Apple Silicon laptop, ≈3–4 min wall clock):

```
"deserialize_input": 5,711,711 RV64IMAC cycles + 7,804,652 virtual = 13,516,363 total
"transcript_init":   7,859     RV64IMAC cycles + 4,445     virtual = 12,304     total
"akita_verify":      46,427,403 RV64IMAC cycles + 5,504,658 virtual = 51,932,061 total
trace length: ~102 M cycles
Proved in ~190 s
Jolt verifier finished is_valid=true
Akita-in-Jolt proof OK
```

## Faster iteration: trace-only

To get cycle counts without paying the ~3-minute Jolt prover cost
(useful when iterating on the guest):

```bash
./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs.bin
```

`--trace-only` skips Dory setup loading, the prover, and the Jolt
verifier, leaving just: guest compile + RISC-V emulation + marker
reporting.

## Larger arity (`nv=32`)

The canonical Hachi profile target. Trace-only works on a laptop; the
full prove path is not yet wired up at this size — see
[`docs/akita-recursion-status.md`](../../docs/akita-recursion-status.md)
for caveats.

```bash
# Generate the nv=32 blob (~128 MiB). REQUIRED before the host run below.
AKITA_NUM_VARS=32 \
    AKITA_RECURSION_BLOB=target/akita_recursion_inputs_nv32.bin \
    ./target/release/akita-recursion-artifact

# Trace + cycle markers (no Jolt prover).
./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs_nv32.bin
```

Trace-only takes ≈ 14 min wall clock and yields a trace length of
≈ 8 G cycles. Full results table:
[`docs/akita-recursion-status.md`](../../docs/akita-recursion-status.md).

## Debugging guest panics

`backtrace = "dwarf"` is already set on the `#[jolt::provable]`
attribute, and the guest enables `jolt/stdout` so panic messages reach
the host:

```bash
JOLT_BACKTRACE=full AKITA_RECURSION_LOG=info \
    ./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs.bin
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
| `JOLT_BACKTRACE`          | unset                                    | `full` ⇒ symbolic guest backtraces.     |
| `AKITA_ALLOW_DEBUG_PROFILE` | unset                                  | `1` ⇒ bypass `--release` guard in `artifact`. |

## CLI flags (`akita-recursion-host`)

| Flag                  | Default                              | Description                                  |
| --------------------- | ------------------------------------ | -------------------------------------------- |
| `--input <path>`      | `target/akita_recursion_inputs.bin`  | Path to the blob produced by `artifact`.     |
| `--target-dir <path>` | `/tmp/akita-recursion-targets`       | Jolt's per-program build cache.              |
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
   preprocess/prove/verify, and forwards per-marker cycle counts
   through `tracing`.
3. **`guest`** (running inside the Jolt RISC-V emulator) decodes the
   blob, builds a `Blake2bTranscript`, and invokes
   `akita_verifier::verify_batched_with_policy` directly — bypassing
   `akita-scheme::batched_verify`, which would otherwise call
   `Instant::now()` (the Jolt runtime traps on `clock_gettime`).
   Three `start_cycle_tracking` / `end_cycle_tracking` pairs wrap
   `deserialize_input`, `transcript_init`, and the verifier kernel.

Full status (cycle tables, bring-up notes, open follow-ups) lives in
[`../../docs/akita-recursion-status.md`](../../docs/akita-recursion-status.md).
