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
  - `Fp32/Fp64` multiplication reduction (division-free fixed-iteration paths).
  - NTT helper operations `csubp`, `caddp`, and `center`.
- NTT butterfly arithmetic runs in fixed loop structure independent of data.
- Ring multiplication (`CyclotomicRing`) is fixed-structure schoolbook over `D`.

## Known Timing Risks / Follow-ups

- `Fp128` reduction (`reduce_u256`) uses data-dependent branches.
- `Field::inv()` for all current fields branches on zero.
- CRT reconstruction in `CyclotomicCrtNtt::to_ring` is correctness-first and not
  yet hardened for strict constant-time behavior.

## Action Items Before Production-Critical Use

1. Replace branchy `Fp128` reduction with a constant-time reduction strategy.
2. Introduce a constant-time inversion API for secret inputs, or explicitly gate
   current inversion APIs to public/non-secret contexts.
3. Add dedicated CT review tests/checklists for any arithmetic subsystem changes.
