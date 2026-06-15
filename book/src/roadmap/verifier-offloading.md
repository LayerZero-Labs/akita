# Verifier offloading

> **Status:** stub. Part of the initial Akita Book scaffold.

Reducing per-level verifier cost by turning the setup matrix contribution into a
cheaper inner-product claim on a preprocessed prefix commitment, via an extra
reduction sum-check. Parts of this are already landed (the setup product
sum-check exists); the full offloaded verifier path is still in progress.

**Sources to fold in**

- Paper §4 `sec:verifier-offloading` (the full construction), §4.3 `sec:claim-reduction`.
- `specs/setup-layout-repack.md`, `specs/setup-product-sumcheck.md`, `specs/setup-prefix-ladder.md`.
- `specs/planner-incidence-generalization.md` (batching at the recursive boundary).
- `book/src/how/proving/sumcheck-stages.md` (stage 3 setup product sumcheck).
