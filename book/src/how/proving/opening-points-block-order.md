# Opening points and block order

> **Status:** stub. Part of the initial Akita Book scaffold.

The conventions that pin where evaluation claims live and in what order blocks
are opened: the opening incidence (which polynomials feed which claims), the
explicit root-vs-recursive `BlockOrder` split, and how the on-wire opening value
is internalized via the trace map.

**Sources to fold in**

- `docs/block-order.md` (near-lift quality).
- `crates/akita-types/src/layout/opening_point.rs:117-177`.
- Paper §3.2 `sec:akita-layout` (opening incidence, per-claim geometry), §3.4 `sec:akita-trace-internalization` (internalizing the opening claim, the fused trace term).
- `specs/w-to-e-notation.md`.
