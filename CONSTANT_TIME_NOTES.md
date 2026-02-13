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
- Prime fields now expose `Invertible::inv_or_zero()` for secret-bearing
  inversion use-cases without input-dependent branching on zero.
- CRT reconstruction final projection now uses a division-free fixed-iteration
  reducer (`reduce_u128_divfree`) instead of `% q`.

## Known Timing Risks / Follow-ups

- `FieldCore::inv()` still returns `Option` and therefore branches on zero;
  treat that API as public-value oriented. Use `Invertible::inv_or_zero()`
  in secret-dependent paths.

## Action Items Before Production-Critical Use

1. Wire secret-bearing call sites to `Invertible::inv_or_zero()` as
   protocol code matures.
2. Add dedicated CT review tests/checklists for any arithmetic subsystem changes.
