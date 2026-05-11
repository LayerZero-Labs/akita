# Akita-in-Jolt recursion example — status

End-to-end pipeline for running the Akita PCS verifier inside a Jolt zkVM
guest program with cycle-tracking instrumentation. The host generates a
real Akita proof, ships it (and the verifier setup) into a Jolt guest,
the guest runs `verify_batched_with_policy`, and Jolt's prover emits a
SNARK of that verifier execution. **Working end-to-end at `nv=20`
OneHot D=32** (canonical config — same family as
`HACHI_MODE=onehot_d32` in the existing `profile.rs`).

## Repo layout

```
profile/akita-recursion/        # standalone sub-workspace (Rust 1.94)
├── Cargo.toml                  # workspace root + [patch.crates-io]
├── rust-toolchain.toml         # 1.94 + riscv32/64imac targets
├── glue/                       # shared verifier-input blob format
│   ├── Cargo.toml              #   package: akita-recursion-glue
│   └── src/lib.rs              #   `AkitaJoltInputs<F, D>` + I/O
├── artifact/                   # host binary: produce the blob
│   ├── Cargo.toml              #   package: akita-recursion-artifact
│   └── src/main.rs
├── host/                       # host binary: compile + prove the guest
│   ├── Cargo.toml              #   package: akita-recursion-host
│   └── src/main.rs
└── guest/                      # #[jolt::provable] RISC-V program
    ├── Cargo.toml              #   package: akita-recursion-guest
    └── src/{lib,main}.rs

crates/akita-types/src/proof/mod.rs
                                # +AkitaSerialize/Deserialize for shape types
```

The parent `Cargo.toml` excludes `profile/akita-recursion/` from the
main Hachi workspace so the Jolt-only transitive dep graph (`jolt-core`,
`dory-pcs`, `zeroos`, ...) doesn't bleed in, and so the
`[patch.crates-io]` overrides for `arkworks-algebra` stay scoped.

## End-to-end run at `nv=20`, OneHot, D=32

```bash
cd profile/akita-recursion
cargo build --release
AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact
AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin
```

Observed results (Apple Silicon laptop), with `backtrace = "off"` and
the guest signature taking `input: &[u8]` (zero-copy borrow into the
input region instead of a `Vec<u8>` materialization):

- Host artifact: **4,252,765 bytes (≈ 4.06 MiB)**. 5 fold levels.
  Native host verify and blob-round-trip verify both pass in ≈3 ms each.
- Native guest sanity run (host platform): returns `0` (OK) in ≈ 10 ms.
- **Cycle markers** (RV64IMAC + virtual instructions):

| Marker             | Base RV64IMAC | Virtual     | **Total cycles** |
| ------------------ | ------------- | ----------- | ---------------- |
| `deserialize_input`| 21,272,858    | 29,815,720  | **51,088,578**   |
| `transcript_init`  | 7,826         | 4,445       | **12,271**       |
| `akita_verify`     | 92,096,660    | 6,184,126   | **98,280,786**   |

- **Total trace length: 149,511,273 cycles** (~150 M).
- Guest exit code `0` (Akita-verify success).

## `nv=32` — trace-only (cycle counts captured, prover not yet attempted)

```bash
cd profile/akita-recursion
AKITA_NUM_VARS=32 AKITA_RECURSION_BLOB=target/akita_recursion_inputs_nv32.bin \
    ./target/release/akita-recursion-artifact
AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --trace-only --input target/akita_recursion_inputs_nv32.bin
```

- Host artifact: **≈ 576.07 MiB** (dominated by the expanded verifier
  setup matrix). 7 fold levels. Native verify and blob-round-trip
  verify both pass in ≈40 ms each.
- Native guest sanity run (host platform): ≈140 ms.
- Jolt RISC-V emulation (trace only, no prover): **≈ 20 min** wall clock.
- **Cycle markers**:

| Marker             | Base RV64IMAC    | Virtual         | **Total cycles**   |
| ------------------ | ---------------- | --------------- | ------------------ |
| `deserialize_input`| 3,020,272,417    | 4,228,394,450   | **7,248,666,867**  |
| `transcript_init`  | 7,826            | 4,445           | **12,271**         |
| `akita_verify`     | 4,045,898,566    | 30,112,564      | **4,076,011,130**  |

- **Total trace length (post-padding): 11,324,873,934 cycles** (≈ 11.3 G).

**Caveat on the full prover at `nv=32`**: the current
`max_trace_length = 4 G` in the `#[jolt::provable]` attribute is below
the observed 11.3 G trace, so a `prove` run would fail until we bump
it. A naive extrapolation from `nv=20` (≈ 530 kHz at D=64; ~similar
order at D=32) puts the expected proving time at ≈ 11 G ÷ 500 kHz ≈
**6 h+** on this laptop, with proportional memory pressure. Cycle
totals (the primary deliverable) are captured without paying that
cost yet; the full prove can be deferred to a beefier host.

## Why D=32 has more zkVM cycles than D=64

A natural surprise: a smaller `D` makes proofs smaller and on-CPU
verification faster, but in the Jolt zkVM the cycle count goes **up**.
This is not a bug — it's the difference between wall-clock time on a
SIMD/parallel CPU and instruction count on a single-issue in-order
emulator. Three compounding effects:

1. **More recursion levels.** Folding nv=20 to a tractable terminal
   takes 4 levels at D=64 vs 5 at D=32; nv=32 takes 6 vs 7. Each level
   adds a full sumcheck-with-range-checks verification step.

2. **Larger verifier-setup matrix.** Halving `D` doesn't halve the
   matrix — Ajtai security forces the stride (column count) to grow to
   compensate. Net at this nv: blob 1.06 MiB → 4.06 MiB (3.8×) at
   nv=20, and 128 MiB → 576 MiB (≈ 4.5×) at nv=32.

3. **Cycle count ≠ wall clock.** On a real CPU, fp128 ops at D=32 vs
   D=64 don't differ much in time because of SIMD, cache prefetch, and
   wide multiply. Inside `riscv64imac` emulation every fp128 mul is a
   fixed-length sequence of 64-bit instructions, each counted as a
   cycle regardless of how cheap it would be on silicon. Smaller-D
   work doesn't compress into fewer RV64 instructions.

For reference, the same workload at OneHot D=64 (nv=20) produced
**65.3 M total cycles** — ~2.3× cheaper to verify inside Jolt — but
costs more on a real CPU. The choice here was to match the canonical
Hachi onehot_d32 profile and report cycle numbers as-is.

## Optimization history at `nv=20`

D=64 OneHot (kept for reference; the example now targets D=32):

| Configuration                              | Trace length    | Δ vs. previous |
| ------------------------------------------ | --------------- | -------------- |
| `backtrace = "dwarf"`, `input: Vec<u8>`    | 102,383,700     | (baseline)     |
| `backtrace = "off"`,   `input: Vec<u8>`    | 102,011,269     | **−0.4 %**     |
| `backtrace = "off"`,   `input: &[u8]`      | **65,283,025**  | **−36.0 %**    |

The `Vec<u8>` → `&[u8]` switch shaved ~36 M cycles off the trace
without changing any cycle marker, because the macro-generated
`postcard::take_from_bytes::<Vec<u8>>(input_slice)` decoded the
1.1 MiB input one byte at a time *before* the user function started
(≈30 cycles per byte × 1.1 M bytes ≈ 33 M cycles). Postcard's `&[u8]`
deserialization is zero-copy: read the length prefix, return a slice
pointing into the input region. The flag to keep this savings is the
guest signature — don't take `Vec<u8>` if the body only needs `&[u8]`.
Same applies to any future guest taking large blobs; at D=32 the input
is 4× larger so the saving scales proportionally.

## What was diagnosed and fixed during bring-up

First-pass run panicked inside `akita_verify` at ~50 M cycles. Enabling
`jolt/stdout` on the guest (so the guest's stderr reaches the host)
surfaced the actual panic:

```
thread '<unnamed>' panicked at .../std/src/sys/pal/unix/time.rs:143:68:
```

That's `std::time::Instant::now()` (the `clock_gettime` syscall) — the
Jolt RISC-V runtime doesn't implement it. The trigger was a single line
in `akita_scheme::AkitaCommitmentScheme::<D, Cfg>::batched_verify`:

```rust
let t_verify_akita = Instant::now();
// ... verifier call ...
tracing::info!(..., elapsed_s = t_verify_akita.elapsed().as_secs_f64(), ...);
```

**Fix:** bypass `akita-scheme::batched_verify` in the guest and invoke
`akita_verifier::verify_batched_with_policy` directly with the same
closure arguments the scheme wraps it in. No timing call, no panic.
(The verifier crate itself is clean — only the scheme's orchestration
entry point was touching `Instant`.)

## Open follow-ups

1. **Full prove at `nv=32`** on a beefier host. Requires:
   - Bumping `max_trace_length` past 11.3 G in the `#[jolt::provable]`
     attribute (currently 4 G — fine for `nv=20`, insufficient at
     `nv=32`).
   - Probably a server-class machine for memory headroom (the guest
     heap is already at 1.5 GiB to fit the 576 MiB decoded verifier
     setup).
   - Expected wall clock at typical zkVM throughput (~500 kHz):
     **~6 h+** of proving.

2. **Make `deserialize_input` cheaper.** At `nv=32` it costs **7.25 G
   cycles** (~178 % of the verifier itself). Most of that is decoding
   the expanded verifier setup matrix; the proof itself is a tiny
   fraction. Options:
   - Ship just the `public_matrix_seed` (32 bytes) and re-derive the
     matrix inside the guest. Trades deserialization cycles for
     matrix-expansion cycles (probably ~similar order, with a much
     smaller input region and cleaner cycle attribution if we move the
     re-derivation under its own marker).
   - Pre-decompose the setup into Lagrange coordinates that don't need
     the full matrix shape inside the guest.

3. **Optional finer markers.** Current set is the minimum the user
   asked for. Splitting `akita_verify` into per-level markers (e.g.
   `root_level`, `fold_levels`, `final_witness`) would need a tiny
   instrumentation tweak in the guest (re-implement the iteration over
   `proof.fold_levels()` with markers around each call).

4. **Upstreaming candidates** — small, mechanical changes that would
   benefit any future Jolt integration with Akita:
   - Optional feature on `akita-scheme` that gates the `Instant::now()`
     + `tracing::info!` epilogue out of `batched_verify`.
   - `AkitaSerialize` / `AkitaDeserialize` impls for proof-shape types
     (already added under `akita-types::proof`).

## Useful commands

```bash
cd profile/akita-recursion

# Build (only the binaries; the guest is built for RISC-V on demand).
cargo build --release

# Generate the verifier-input blob.
AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact

# Full prove + verify at nv=20 (~3-4 min wall clock on Apple Silicon).
AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin

# Fast iteration on guest changes: trace only (no prover).
# At nv=20: ~25-40 s. At nv=32: ~14 min (deserialization-dominated).
./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs.bin

# Resolve a guest panic with symbolic backtraces.
JOLT_BACKTRACE=full ./target/release/akita-recursion-host \
    --trace-only --input target/akita_recursion_inputs.bin

# Force a clean guest rebuild.
rm -rf /tmp/akita-recursion-targets /tmp/jolt-guest-targets
```
