# The proving protocol

> **Status:** stub. Part of the initial Akita Book scaffold.

The core. Each fold level (root and recursive) runs the same sub-pipeline:
evaluate committed polynomials at the opening points → build the ring relation →
ring-switch to the next-level witness `w` and commit it → stage-1 range/norm
check → stage-2 fused relation sumcheck → (optional) stage-3 setup product
sumcheck → thread state to the next level. Pair each chapter with its verifier
mirror.

This is the one part of "How it works" that stays a multi-page section, because
the per-level interactive protocol is large. Subpages:

- [Opening points and block order](./opening-points-block-order.md)
- [Root fold and ring switching](./root-fold-ring-switch.md)
- [Sumcheck stages](./sumcheck-stages.md)
- [Extension-opening reduction](./extension-opening-reduction.md)

## Sources to fold in

- `crates/akita-prover/src/protocol/flow/root_fold.rs`, `ring_relation.rs`, `ring_switch.rs`.
- Paper §3.5 `sec:akita-one-step` (one folding step; `fig:akita-fold`, `fig:akita-sumcheck`).
- Council architecture report (per-level sub-pipeline).
