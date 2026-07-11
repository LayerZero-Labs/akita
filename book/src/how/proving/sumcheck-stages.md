# Sumcheck stages

> **Status:** stub. Part of the initial Akita Book scaffold.

The fused sumcheck that proves one fold: stage 1 (the digit range check on the
balanced witness), stage 2 (the fused relation sumcheck), and the optional
stage 3 (the setup product sumcheck used by verifier offloading). How they batch
and schedule together.

**Sources to fold in**

- `crates/akita-prover/src/protocol/sumcheck/akita_stage1/`, `akita_stage1_tree.rs`, `akita_stage2/`, `two_round_prefix/`.
- `crates/akita-verifier/src/stages/`.
- Paper §3.5 (`fig:akita-sumcheck`), §3.5.1 `sec:akita-range-check` (optimized digit range check), §4.3 `sec:claim-reduction` (setup product sumcheck).
- `specs/packed-sumcheck.md`, `specs/setup-product-sumcheck.md`.

## Distributed round aggregation

Distributed proving does not create one transcript per machine. Every machine
computes the restriction of the same padded global sum-check polynomial to its
machine-major witness prefix. In each round the coordinator sums the local
round-polynomial coefficients, absorbs only that sum, and then samples the next
challenge. The verifier replays the ordinary single transcript.

The same rule applies to the digit-range and relation chains. Every local folded
response and local quotient segment participates in the generic range proof;
relation, trace, and setup weights use the native machine-major layout described
in [The distributed prover](distributed-prover.md).
