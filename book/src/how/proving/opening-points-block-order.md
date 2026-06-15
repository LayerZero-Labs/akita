# Opening batches, points, and block order

> **Status:** stub. Part of the initial Akita Book scaffold.

The conventions that pin where evaluation claims live and in what order blocks
are opened: the `OpeningBatch` descriptor (which polynomials share one opening
point), the explicit root-vs-recursive `BlockOrder` split, and how the on-wire
opening value is internalized via the trace map.

**Sources to fold in**

- `specs/single-point-opening-batch.md` (single shared point per prove/verify call).
- `crates/akita-types/src/proof/opening_batch.rs` (`OpeningBatch`, transcript shape).
- `docs/block-order.md` (near-lift quality).
- `crates/akita-types/src/layout/opening_point.rs:122-180`.
- Paper §3.2 `sec:akita-layout` (per-claim geometry), §3.4 `sec:akita-trace-internalization` (internalizing the opening claim, the fused trace term).
- `specs/archive/2026-Q2/w-to-e-notation.md` (w / e / v naming).
