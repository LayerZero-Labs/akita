# Akita-in-Jolt recursion example — status

End-to-end pipeline for running the Akita PCS verifier inside a Jolt zkVM
guest program with cycle-tracking instrumentation. The host generates a
real Akita proof, ships it (and the verifier setup) into a Jolt guest,
the guest runs `verify_batched_with_policy`, and Jolt's prover emits a
SNARK of that verifier execution. **Working end-to-end at `nv=20`
OneHot D=64.**

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

## End-to-end run at `nv=20`, OneHot, D=64

```bash
cd profile/akita-recursion
cargo build --release
AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact
AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --input target/akita_recursion_inputs.bin
```

Observed results (Apple Silicon laptop):

- Host artifact: **1,114,636 bytes (≈ 1.06 MiB)**. Native host verify
  and blob-round-trip verify both pass.
- Guest binary built for `riscv64imac-zero-linux-musl` in ~16 s.
- Dory PCS setup at `max_log_n = 38` loaded from
  `~/Library/Caches/dory/dory_38.urs` (~84 s).
- Native guest sanity run (host platform): returns `0` (OK) in ≈ 2 ms.
- Jolt prover invoked. **Cycle markers** (RV64IMAC + virtual instructions):

| Marker             | Base RV64IMAC | Virtual    | **Total cycles** |
| ------------------ | ------------- | ---------- | ---------------- |
| `deserialize_input`| 5,711,711     | 7,804,652  | **13,516,363**   |
| `transcript_init`  | 7,859         | 4,445      | **12,304**       |
| `akita_verify`     | 46,427,403    | 5,504,658  | **51,932,061**   |

- **Total trace length: ~102 M cycles** (~67.86 M raw RISC-V +
  ~34.58 M virtual, padded).
- **Guest panic flag**: `false`.
- **Prover stages 1–8**: ~190 s wall clock (~537 kHz / padded
  ~703 kHz). Total `prover_secs`: **~201 s**.
- **Jolt verifier**: 0.28 s, `is_valid = true`.
- Guest exit code `0` (Akita-verify success). Host reports
  `Akita-in-Jolt proof OK`.

## `nv=32` — trace-only (cycle counts captured, prover not yet attempted)

```bash
cd profile/akita-recursion
AKITA_NUM_VARS=32 AKITA_RECURSION_BLOB=target/akita_recursion_inputs_nv32.bin \
    ./target/release/akita-recursion-artifact
AKITA_RECURSION_LOG=info ./target/release/akita-recursion-host \
    --trace-only --input target/akita_recursion_inputs_nv32.bin
```

- Host artifact: **≈ 128.08 MiB** (dominated by the expanded verifier
  setup matrix). Native verify and blob-round-trip verify both pass in
  ≈18 ms each.
- Native guest sanity run (host platform): ≈115 ms.
- Jolt RISC-V emulation (trace only, no prover): **≈13.6 min** wall clock.
- **Cycle markers**:

| Marker             | Base RV64IMAC   | Virtual          | **Total cycles** |
| ------------------ | --------------- | ---------------- | ---------------- |
| `deserialize_input`| 696,678,378     | 940,141,841      | **1,636,820,219** |
| `transcript_init`  | 7,845           | 4,310            | **12,155**        |
| `akita_verify`     | 1,869,086,604   | 24,937,688       | **1,894,024,292** |

- **Total trace length (post-padding): 8,062,746,399 cycles** (~8 G).

**Caveat on the full prover at `nv=32`**: the current
`max_trace_length = 4 G` in the `#[jolt::provable]` attribute is below
the observed 8 G trace, so a `prove` run would fail until we bump it. A
naive extrapolation from `nv=20` (≈201 s for ~102 M cycles ⇒ ~530 kHz)
puts the expected proving time at ≈8 G ÷ 530 kHz ≈ **4 h+** on this
laptop, with proportional memory pressure. Cycle totals (the primary
deliverable) are captured without paying that cost yet; the full prove
can be deferred to a beefier host.

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
   - Bumping `max_trace_length` past 8 G in the `#[jolt::provable]`
     attribute (currently 4 G — fine for `nv=20`, insufficient at
     `nv=32`).
   - Probably a server-class machine for memory headroom.
   - Expected wall clock at the `nv=20` rate (~530 kHz): **~4 h+** of
     proving.

2. **Make `deserialize_input` cheaper.** At `nv=32` it costs **1.64 G
   cycles** (~86 % of the verifier itself). Most of that is decoding
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
