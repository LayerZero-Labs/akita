# Sumcheck stages

> **Status:** stub. Part of the initial Akita Book scaffold.

The fused sumcheck that proves one fold: stage 1 (the digit range / L2-norm
check on the balanced witness), stage 2 (the fused relation sumcheck), and the
optional stage 3 (the setup product sumcheck used by verifier offloading). How
they batch and schedule together.

**Sources to fold in**

- `crates/akita-prover/src/protocol/sumcheck/akita_stage1/`, `akita_stage1_tree.rs`, `akita_stage2/`, `two_round_prefix/`.
- `crates/akita-verifier/src/stages/`.
- Paper §3.5 (`fig:akita-sumcheck`), §3.5.1 `sec:akita-range-check` (optimized digit range check), `sec:akita-l2-norm` (folded-witness L2 norm check), §4.3 `sec:claim-reduction` (setup product sumcheck), App B.4 `sec:akita-norm-sumcheck`.
- `specs/packed-sumcheck.md`, `specs/setup-product-sumcheck.md`, `specs/optimized_verifier.md`.
