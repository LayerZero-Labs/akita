# Constant-Time Review Notes (Phase 0/1 Algebra)

This note tracks timing-sensitive implementation decisions for the current
algebra and ring stack.

## Reviewed Components

- `src/algebra/fields/fp32.rs`
- `src/algebra/fields/fp64.rs`
- `src/algebra/fields/fp128.rs`
- `src/algebra/ntt/prime.rs`
- `src/algebra/ntt/butterfly.rs`
- `src/algebra/ring/cyclotomic.rs`
- `src/algebra/ring/crt_ntt_repr.rs`

## Current State

- Branchless primitives are in place for:
  - `Fp32/Fp64/Fp128` add/sub/neg raw helpers.
  - `Fp128` multiplication reduction (`reduce_u256`) with branchless conditional subtract.
  - `Fp32/Fp64` multiplication reduction (division-free fixed-iteration paths).
  - NTT helper operations `csubp`, `caddp`, and `center`.
- NTT butterfly arithmetic runs in fixed loop structure independent of data.
- Ring multiplication (`CyclotomicRing`) is fixed-structure schoolbook over `D`.
- CRT reconstruction inner accumulation now uses fixed-trip, branchless
  modular add/mul-by-small-factor helpers.

## Known Timing Risks / Follow-ups

- `Field::inv()` for all current fields branches on zero.
- CRT reconstruction still ends with `acc % q` projection, which uses a
  variable-latency division path. Treat current reconstruction as
  correctness-first unless/until that final reduction is replaced.

## Action Items Before Production-Critical Use

1. Introduce a constant-time inversion API for secret inputs, or explicitly gate
   current inversion APIs to public/non-secret contexts.
2. Replace the final CRT projection step (`acc % q`) with a division-free
   reducer for strict constant-time reconstruction.
3. Add dedicated CT review tests/checklists for any arithmetic subsystem changes.
