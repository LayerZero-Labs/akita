# `akita-recursion` — Akita verifier inside Jolt

Runs the Akita PCS verifier inside a Jolt zkVM guest program and reports
per-phase cycle counts (`deserialize_input`, `install_terminal_cache`,
`transcript_init`, `akita_verify`). End-to-end this also produces a SNARK of the verifier
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

## Quick start (`nv=32`, recursive multi-group OneHot — canonical target)

You need the [Jolt CLI](https://github.com/a16z/jolt) installed
(`cargo install --path .` from a clone of `jolt` at the same rev this
crate pins, `c4207de4b9c7...`). The first prove run downloads a ~30 GB
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
#    Start with trace-only (no Jolt prover) when measuring the recursive
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

Expected output shape (rerun `--trace-only` for current recursive numbers):

```
"deserialize_input": … (dominated by expanded verifier-setup decode)
"install_terminal_cache": …
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
Start with `AKITA_NUM_VARS=32`, measure it with `--trace-only`, and then remove
`--trace-only` once the trace bound is confirmed:

```bash
AKITA_NUM_VARS=32 ./target/release/akita-recursion-artifact
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
| `AKITA_NUM_VARS`          | `32`                                     | Polynomial arity for the prover.        |
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

1. **`artifact`** runs the recursive companion of `fp128::OneHot` →
   `setup_prover` → two base-config precommits → a recursive-config
   two-polynomial final commit → `batched_prove` over synthetic OneHot
   polynomials. It sanity-verifies on the host and serializes the three
   ordered commitment groups, verifier setup, proof shape, and proof into
   one blob via [`AkitaJoltInputs::write_to_bytes`](glue/src/lib.rs).
2. **`host`** strictly decodes and verifies the blob. It then derives the
   terminal scalar Q128 NTT cache from that verified setup and verifies the
   proof again through the installed cache. The host writes the cache under
   `--target-dir` and passes its path into the guest build. It then compiles
   the guest to `riscv64imac-zero-linux-musl`, runs Jolt, and forwards each
   cycle count through `tracing`.
3. **`guest`** (running inside the Jolt RISC-V emulator) decodes the
   blob and invokes `akita_verifier::batched_verify` directly —
   bypassing `akita-scheme::batched_verify`, which would otherwise
   call `Instant::now()` (the Jolt runtime doesn't implement
   `clock_gettime`, and the guest aborts there). Four
   `start_cycle_tracking` / `end_cycle_tracking` pairs wrap
   input decoding, prepared cache installation, transcript initialization,
   and the verifier kernel.
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

## Trusted fp128 field decode

Blob version 4 puts an explicit, bounded zero-padding record before the setup
matrix. The host chooses the smallest padding that aligns the matrix payload
inside Jolt's Postcard-encoded byte-slice input. The decoder checks the padding
count, every padding byte, and the alignment calculation before it reads the
matrix. This keeps transcript domains independent of memory layout.

The trusted benchmark decoder validates every fp128 value canonically. It
views the checked matrix payload as pairs of little-endian `u64` words. When
the payload address is eight-byte aligned, the decoder reads it directly and
the RISC V loop uses two aligned loads per field. When an outer input ABI
places the same bytes at a different alignment, the decoder uses unaligned
reads directly without a payload-sized staging allocation. Alignment therefore
selects the load primitive; it does not decide whether otherwise valid wire
bytes are accepted or change the memory envelope.

The specialized decoder is used only by the explicitly trusted cached-matrix
path. Strict setup decoding remains unchanged and still derives the public
matrix from its seed. Tests sweep all eight possible payload alignments,
exercise both load paths, and include a valid value with both limbs populated.

## Prepared terminal cache

The canonical verifier setup stores field coefficients. It does not serialize
an architecture-specific NTT representation. Native applications can keep the
existing in-memory cache warm across calls.

The Jolt guest cannot preserve memory between separate program executions. The
host therefore derives one target cache before compiling the guest. The cache
format fixes the scalar Q128 representation used by RISC V. It does not depend
on the CPU that runs the host command.

The fixed header binds all of the following values:

- the cache format and target representation;
- the setup seed digest and materialized setup field count;
- the generated schedule row digest;
- the ring dimension, prefix lengths, matrix width, and signed coefficient bound.

The build script includes the complete cache file in the guest ELF. Jolt's
program identity therefore commits to its bytes. The guest checks the header,
the exact payload length, every residue range, and the setup and schedule
identities before installing it. A mismatch returns status code `1`.

The public installation API is named
`install_trusted_prepared_verifier_ntt_cache` because the header cannot prove
that the transformed payload came from the named setup seed. The recursion
host establishes that provenance by deriving the cache from a strictly decoded
setup and verifying the proof through the decoded cache before it starts Jolt.
Code that loads an external cache must provide an equivalent trusted setup
installation boundary.

If no prepared cache path is present at build time, the generated static value
is `None`. The guest then uses the ordinary portable warming path. This keeps
plain guest builds functional and lets other architectures use their own
derived representation later.

## Adaptive dimension pin and historical D64 measurements

The artifact, host, and guest share a fixed source-view dimension `D = 512`.
The planner-generated recursive `nv=32` row has a `(32, 2)` final group,
two `(16, 1)` precommitted groups, and uses A=D512 at the root. It may
transition to smaller per-level dimensions. The artifact rejects any requested
arity whose selected root A dimension does not match this Jolt
monomorphization.

In this profile, recursive setup means the first recursive edge carries one
setup-prefix opening into the successor fold. The successor verifies the setup
contribution together with its witness group. This is distinct from the direct
profile, where setup is consumed locally and no setup claim is carried to the
successor.

Historical D64 measurements remain useful context, but they do not describe
the current adaptive binary. D32 is not a valid A-role fold degree (`d_a ≥ 64`),
and historical D32 Jolt traces were more expensive than fixed D64 at equal
`num_vars`. Three compounding effects explained the old D32 penalty:

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

For historical reference, fixed-D64 OneHot at `nv=32` produced a trace on the
order of 8 G cycles, about 30% cheaper inside Jolt than the retired D32
configuration. Remeasure before making claims about the adaptive preset.

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
   - If the public trait entry point ever becomes timer-free and verifier-only,
     it should delegate to the same `akita_config::batched_verify_with_config`
     adapter; the guest should remain free of `akita-scheme`, `akita-prover`,
     and `akita-setup` dependencies.
   - `AkitaSerialize` / `AkitaDeserialize` impls for proof-shape types
     (already added under `akita-types::proof` and used by the `glue`
     crate).
