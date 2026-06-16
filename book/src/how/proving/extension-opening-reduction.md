# Extension-opening reduction

> **Status:** stub. Part of the initial Akita Book scaffold.

How Akita wires in the extension-opening reduction (EOR): it turns a base-field
evaluation claim at an extension-field point into a single claim on a packed
polynomial over the extension, with fewer variables. The generic reduction and
its soundness live in
[Foundations → Extension-opening reduction](../../foundations/extension-opening-reduction.md);
this page is about Akita's prover paths, scheduling, and efficiency.

The implemented prover has dense-packed and sparse one-hot paths, a lazy tensor
factor for early rounds, and a streamed form that keeps small balanced
representatives visible to the hot loop.

**Sources to fold in**

- `crates/akita-prover/src/protocol/extension_opening_reduction/`.
- `crates/akita-types/src/extension_opening_reduction.rs`.
- Paper App B.4.1 `sec:akita-eor-sumcheck` (implemented prover paths, prefix-suffix tensor weight, streamed/staged prover).
- `specs/extension-field-opening-batching.md` (trim stale `akita-scheme` refs), `specs/eor-streamed-prover.md` (active).
