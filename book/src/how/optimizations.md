# Optimizations

> **Status:** stub. Part of the initial Akita Book scaffold.

The structural and representational optimizations that close the gap between the
protocol-level description and the reported performance. Most of these are
documented in the paper's implementation appendix.

## Compute backends

The compute-backend abstraction (CPU cutover landed; Metal/GPU is roadmap) and
where the prover's hot work is dispatched.

**Sources to fold in**

- `docs/compute-backends.md`, `crates/akita-prover/src/compute.rs`.
- `specs/packed-sumcheck.md`, `specs/eor-sumcheck-prover-acceleration.md`.
- `specs/akita-compute-backend-metal.md` (Metal is roadmap).

## SIMD and packing

Pseudo-Mersenne/Solinas field arithmetic, deferred reduction / unreduced
accumulators, the CRT+NTT representation and its accumulation-capacity bound,
the smooth-subgroup mixed-radix FFT, and architecture-specific NEON/AVX2/AVX-512
kernels.

**Sources to fold in**

- `crates/akita-field/src/packed/`, `crates/akita-field/src/unreduced/`.
- Paper App B.1 (field arithmetic), B.2 (ring arithmetic, CRT+NTT, SIMD), `sec:akita-fft`.
- `docs/crt-ntt-capacity-profile.md` (generated table — embed + keep regen command).
- `specs/avx-simd-port.md`, `specs/fp31-field-optimization-retrospective.md`, `specs/crt-ntt-accumulation-safety.md`.
