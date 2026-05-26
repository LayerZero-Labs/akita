# Spec: Metal Fp128 Vector Kernels

| Field       | Value             |
|-------------|-------------------|
| Author(s)   | @quangvdao, Codex |
| Created     | 2026-05-27        |
| Status      | implemented       |
| PR          |                   |

## Summary

Add `akita-metal`, an optional Apple Metal accelerator crate for Akita. This
initial GPU slice provides a typed host API, embedded Metal kernels, tests, an
example, and a benchmark for elementwise `Fp128<P>` vector arithmetic:

- `out[i] = lhs[i] + rhs[i]`
- `out[i] = lhs[i] - rhs[i]`
- `out[i] = lhs[i] * rhs[i]`

The crate is buildable on non-macOS targets. Metal dependencies are only linked
on macOS, and runtime entry points return `MetalError::UnsupportedPlatform`
elsewhere. The default Akita prover, verifier, scheme, algebra, and field paths
remain CPU-only.

## Intent

### Goal

Introduce a small, maintainable GPU kernel crate that can be reviewed and
merged independently of any prover-backend routing work.

This PR adds the long-term surfaces needed for the first Metal-backed field
microbenchmarks:

- `crates/akita-metal`, an optional workspace crate outside the foundational
  algebra, prover, verifier, and scheme crates.
- A narrow `MetalBackend` facade for device discovery, typed buffer allocation,
  host/device transfer, Fp128 vector dispatch, and timing metadata.
- Stable host/device ABI types for Fp128 kernel parameters and limb layout.
- Embedded Metal entry points for Fp128 elementwise add, sub, and mul.
- A small smoke example that checks Metal output against CPU field arithmetic.
- A Criterion benchmark that compares CPU arithmetic, Metal dispatch-only
  timing, and Metal upload-dispatch-readback timing.

### Invariants

1. Metal remains optional. No default Akita path may require `metal`, `objc`,
   Apple SDKs, or Apple hardware.
2. `akita-metal` builds on non-macOS targets. Metal-specific dependencies stay
   under `target.'cfg(target_os = "macos")'.dependencies`.
3. Runtime device availability is checked at runtime. macOS builds may still
   return `MetalError::NoSystemDevice`.
4. Non-macOS runtime entry points return `MetalError::UnsupportedPlatform`.
5. This PR does not change proof bytes, transcript behavior, setup descriptors,
   schedule policy, or PCS protocol semantics.
6. This PR does not route prover execution through Metal.
7. Host/kernel layout assumptions are explicit and tested.
8. Every public Fp128 operation exposed by this crate is checked against the
   existing CPU field implementation.
9. Benchmarks clearly separate CPU arithmetic, host transfer, command-buffer
   dispatch timing, and optional GPU counter telemetry.
10. Public examples are self-contained and do not depend on local benchmark
    files, profile logs, or generated artifacts.
11. There are no compatibility shims or deprecated aliases.

### Non-Goals

1. Production Q128 digit-row kernels.
2. Prover backend integration or automatic CPU/GPU routing.
3. A final batching or coalescing API for prover workloads.
4. End-to-end prover speedup claims.
5. Mandatory Metal support in default builds, CI, verifier builds, or
   non-Apple targets.
6. Requiring GPU timestamp counters for correctness or acceptance.

## Design

### Crate Boundary

`akita-metal` sits beside the core Akita crates as an optional accelerator
crate:

```text
akita-field
  Fp128<P> CPU reference arithmetic and transparent host layout

akita-metal
  src/error.rs
    MetalError
    MetalResult

  src/field/fp128.rs
    Fp128Limb
    Fp128KernelParams
    Fp128MetalParams
    Fp128VectorOp
    Fp128VectorPlan

  src/device.rs
    MetalBackend
    MetalDeviceInfo
    Fp128VectorBuffers
    Fp128BufferOptions
    Fp128DispatchOptions
    Fp128PipelineInfo
    Fp128DispatchProfile
    Fp128TransferProfile

  src/device/platform.rs
    macOS-only Metal runtime implementation

  src/kernels/fp128_vector.metal
    fp128_vector_add
    fp128_vector_sub
    fp128_vector_mul

  examples/fp128_vector.rs
    manual smoke test

  benches/fp128_vector.rs
    Criterion CPU-vs-Metal microbenchmark
```

The foundational crates do not depend on `akita-metal`. The only change outside
the new crate is `#[repr(transparent)]` on `Fp128<P>`, which makes the existing
two-limb representation explicit for host/device copies.

### Platform Policy

The Rust Metal bindings are guarded by:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = "0.33"
objc = "0.2"
```

There is no separate Rust `cfg` flag for "Metal-capable device". macOS is the
compile-time linking gate; `MetalBackend::new()` is the runtime availability
gate.

On non-macOS targets:

- the crate still compiles;
- public types remain available;
- runtime backend construction returns `MetalError::UnsupportedPlatform`;
- no `metal` or `objc` dependency is linked.

### Kernel Workload

The Metal kernels process contiguous slices of `Fp128<P>` elements. Each
logical thread handles one element index:

```text
for i in 0..len:
    out[i] = lhs[i] op rhs[i]
```

The word "vector" in this API means an array/slice workload. It does not imply
an Akita algebraic vector object, prover batching, or explicit SIMD lanes inside
one Metal thread.

### Timing Surfaces

The public profiling structs expose enough metadata to interpret benchmark
results without treating them as end-to-end prover numbers:

- vector length;
- operation;
- buffer storage mode;
- pipeline thread execution width;
- selected threadgroup width;
- threadgroup count;
- command-buffer dispatch wall time;
- upload/readback copy and blit timing;
- optional GPU counter samples when available.

GPU counters are telemetry only. They are not required for correctness and may
be unavailable or noisy on some devices.

## Evaluation

### Acceptance Criteria

- [x] `cargo check -p akita-metal` passes on macOS.
- [x] `cargo test -p akita-metal` passes on macOS.
- [x] `cargo clippy -p akita-metal --all-targets --message-format=short -q -- -D warnings`
      passes on macOS.
- [x] `cargo check -p akita-metal --target x86_64-unknown-linux-gnu` passes
      without linking `metal` or `objc`.
- [x] `cargo fmt -q` passes.
- [x] `cargo bench -p akita-metal --bench fp128_vector --no-run` builds the
      benchmark.
- [x] `crates/akita-metal/Cargo.toml` keeps Metal runtime dependencies behind
      `cfg(target_os = "macos")`.
- [x] `akita-metal` regular dependencies are limited to `akita-field` and
      small support crates.
- [x] `akita-algebra` and `akita-prover` are not regular dependencies of
      `akita-metal`.
- [x] Fp128 ABI tests assert relevant size and alignment assumptions.
- [x] Fp128 add, sub, and mul are tested against CPU results.
- [x] Tests cover both shared buffers and private GPU buffers with staged
      transfers.
- [x] Public examples require no local CSV files, profile logs, or generated
      target artifacts.
- [x] The README explains the exact benchmark workload and timing buckets.
- [x] `akita-prover` remains unchanged by this PR.

### Required Checks

```bash
cargo fmt -q
cargo clippy -p akita-metal --all-targets --message-format=short -q -- -D warnings
cargo test -p akita-metal
cargo check -p akita-metal --target x86_64-unknown-linux-gnu
cargo bench -p akita-metal --bench fp128_vector --no-run
```

Recommended workspace sanity checks before PR review:

```bash
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

### Benchmark Contract

The benchmark compares the same deterministic input vectors for CPU and Metal.
For each operation and length, it reports:

- CPU field arithmetic over `lhs` and `rhs`;
- Metal dispatch-only timing using pre-uploaded reusable buffers;
- Metal roundtrip timing using reusable buffers, including upload, dispatch,
  and readback.

These are field-kernel microbenchmarks. They should not be reported as Akita
prover speedups.

## Implementation

The implemented PR contains:

- `crates/akita-metal/Cargo.toml`
- `crates/akita-metal/README.md`
- `crates/akita-metal/src/lib.rs`
- `crates/akita-metal/src/error.rs`
- `crates/akita-metal/src/field/fp128.rs`
- `crates/akita-metal/src/device.rs`
- `crates/akita-metal/src/device/platform.rs`
- `crates/akita-metal/src/kernels/mod.rs`
- `crates/akita-metal/src/kernels/fp128_vector.metal`
- `crates/akita-metal/examples/fp128_vector.rs`
- `crates/akita-metal/benches/fp128_vector.rs`
- `crates/akita-field/src/fields/fp128.rs` layout annotation
- workspace membership and lockfile updates

## Later Specs

Future PRs should specify these independently:

1. **Q128 typed runtime API.** Add typed host plans, packed buffers,
   active-column metadata, command encoding, and CPU-equivalence tests for
   Q128 workloads.
2. **Compute tracing.** Define a stable trace format and summary tool for
   observing prover compute workloads without changing protocol semantics.
3. **Backend batching/coalescing API.** Define how prover code presents ordered
   batches of accelerator-eligible events while preserving CPU fallback
   behavior.
4. **End-to-end prover Metal integration.** Route selected prover compute
   operations through the backend API and benchmark full prover flows honestly.

## References

- `crates/akita-metal/README.md`
- `crates/akita-metal/src/device.rs`
- `crates/akita-metal/src/field/fp128.rs`
- `crates/akita-metal/src/kernels/fp128_vector.metal`
- `crates/akita-metal/benches/fp128_vector.rs`
- `crates/akita-field/src/fields/fp128.rs`
