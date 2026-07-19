# Sumcheck stages

> **Status:** stub. Part of the initial Akita Book scaffold.

The fused sumcheck that proves one fold: stage 1 (the digit range check on the
balanced witness), stage 2 (the fused relation sumcheck), and the optional
stage 3 (the setup product sumcheck used by verifier offloading). How they batch
and schedule together.

**Sources to fold in**

- `crates/akita-prover/src/protocol/sumcheck/digit_range/`, `akita_stage2/`, and `two_round_prefix/`.
- `crates/akita-verifier/src/stages/`.
- Paper §3.5 (`fig:akita-sumcheck`), §3.5.1 `sec:akita-range-check` (optimized digit range check), §4.3 `sec:claim-reduction` (setup product sumcheck).
- `specs/packed-sumcheck.md`, `specs/setup-product-sumcheck.md`.
