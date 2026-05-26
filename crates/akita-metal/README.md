# akita-metal

`akita-metal` contains optional Apple Metal kernels and host-side runtime
helpers for Akita compute experiments. The crate is intentionally outside the
foundational field, algebra, prover, verifier, and scheme crates.

The first supported surface is elementwise `Fp128<P>` vector arithmetic:

- `out[i] = lhs[i] + rhs[i]`
- `out[i] = lhs[i] - rhs[i]`
- `out[i] = lhs[i] * rhs[i]`

The crate builds on non-macOS targets, but runtime entry points return
`MetalError::UnsupportedPlatform`. On macOS, device availability is still
checked at runtime through `MetalBackend::new()`.

## Example

```bash
cargo run -p akita-metal --example fp128_vector
```

The example multiplies two deterministic `Prime128OffsetA7F7` vectors on the
GPU, checks the output against CPU field arithmetic, and prints the dispatch
shape and command-buffer wall time.

## Benchmark

```bash
cargo bench -p akita-metal --bench fp128_vector
```

The benchmark compares CPU `Fp128` vector arithmetic against Metal dispatches
for the same deterministic inputs. The Metal measurements are separated into:

- dispatch-only timings with reusable buffers already uploaded;
- roundtrip timings that include upload, dispatch, and readback using the same
  reusable buffers.

Benchmark results should be read as field-kernel microbenchmarks, not
end-to-end Akita prover speedups. Prover integration, Q128 digit-row kernels,
and CPU/GPU batching policy are separate future work.

## Platform Policy

Metal runtime dependencies are gated behind `cfg(target_os = "macos")`.
Downstream crates can depend on `akita-metal` without breaking non-Apple
dependency resolution, but must handle `MetalError::UnsupportedPlatform` at
runtime.
