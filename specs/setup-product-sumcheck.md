# Spec: Recursive Setup-Contribution Product Sumcheck (Stage 3)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   |                                |
| Created     | 2026-06-02                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

In `SetupContributionMode::Direct`, each fold level's verifier proves the setup
contribution `<S_{<=N}, omega_S>` by scanning the expanded setup matrix inline.
That scan dominates the per-level verifier cost and, inside the Jolt zkVM, is
the single largest non-deserialization cost. This spec covers the
`SetupContributionMode::Recursive` path, where each **non-terminal** fold level
instead delegates the setup contribution to a **setup-product sumcheck** — the
Stage-3 `SetupSumcheckProver` / `SetupSumcheckVerifier` pair — so the verifier
replays a sumcheck plus a single succinct `omega_S` evaluation rather than a
full matrix scan. It also adds the missing end-to-end test coverage and a
runnable example for this mode, both of which previously only existed for
`Direct`.

This spec describes the current explicit Stage-3 scaffold. The final verifier
offloading architecture moves the setup-contribution dependency into Stage 2:
Stage 1 carries the relation/witness claim through its last regular batched
sumcheck, and Stage 2 proves both the carried witness-claim reduction and the
setup contribution. Until that cutover lands, Stage 3 remains the implementation
boundary for the recursive setup-product sumcheck.

## Intent

### Goal

Provide a prover/verifier-symmetric setup-product sumcheck for the recursive
setup-contribution path, located in the crates that own each role, and cover it
end to end.

Key abstractions and surfaces:

- `akita-prover` `protocol::sumcheck::setup_sumcheck::SetupSumcheckProver` —
  Akita-specific setup-product sumcheck prover. Moved out of the previous
  general `akita-sumcheck::factored_product` module; its `prove` entry point
  now folds in the term-preparation logic that used to live in `flow.rs`.
- `akita-verifier` `stages::stage3::SetupSumcheckVerifier` — the verifier
  counterpart, with a two-phase `new` (derive the setup evaluation plan and
  sumcheck round count from the ring-switch row evaluation) + `verify` (replay
  the extension-opening-reduction sumcheck and close it against a succinct
  `omega_S` evaluation) API. Lives under `stages/` alongside stage1/stage2.
- `akita-types::SETUP_SUMCHECK_DEGREE` — the relocated degree constant
  (formerly `FACTORED_PRODUCT_SUMCHECK_DEGREE`).
- `profile/akita-recursion` artifact/host/guest gain a `--setup-mode
  <direct|recursive>` CLI input (default `direct`); the chosen mode is recorded
  in the verifier-input blob so host preflight and guest replay verify under the
  same mode without a separate flag.

### Invariants

- **Prover/verifier symmetry.** A proof produced under `Recursive` verifies
  under `Recursive`; a proof produced under `Direct` verifies under `Direct`.
  Protected by `crates/akita-pcs/tests/recursive_setup_e2e.rs`
  (`recursive_onehot_nv20`, `recursive_onehot_nv25`).
- **Mode is load-bearing, not cosmetic.** A `Recursive` proof must be rejected
  under `Direct` and vice versa, because the modes disagree on whether the
  embedded stage-3 sumcheck is present. Protected by
  `recursive_onehot_cross_mode_rejects_nv20`.
- **Setup-contribution value agreement.** The recursive setup-product sumcheck
  must reduce to the same setup contribution as the direct matrix scan.
  Protected at the unit level by the existing materialized-vs-direct
  equivalence fixtures in
  `crates/akita-verifier/src/protocol/slice_mle/setup_contribution/` and at the
  e2e level by the round-trip tests above.
- **Terminal levels never embed a stage-3 proof.** Only non-terminal fold
  levels run the setup sumcheck; terminal levels close the witness directly.
  The verifier rejects a `Recursive` proof whose non-terminal level is missing
  `stage3_sumcheck_proof` (`InvalidSetup("recursive setup-contribution mode is
  missing stage3_sumcheck_proof")`).
- **Verifier no-panic boundary.** All new verifier-reachable code
  (`stages/stage3.rs`, `setup_contribution` evaluator) returns `AkitaError` on
  malformed input rather than panicking, per the AGENTS.md contract.
- **Transcript determinism.** The stage-3 sumcheck samples its challenges via
  the canonical `CHALLENGE_SUMCHECK_ROUND` label; prover and verifier event
  streams must match (covered by the existing `logging-transcript` checks).

### Non-Goals

- **ZK support for the recursive setup path.** Hiding/blinding the
  setup-product sumcheck (masked sumcheck rounds, hiding commitments on the
  carried setup claim) is explicitly **out of scope** and deferred to future
  work. The e2e tests and example are gated `#![cfg(not(feature = "zk"))]`, and
  the `Recursive` path is not yet exercised under the `zk` feature.
- **Carried-opening batching / setup-prefix commitment delegation.** This spec
  keeps the stage-3 sumcheck closed against the locally evaluated `omega_S`; it
  does not carry the `(rho_lambda, rho_y, s_rho)` setup-prefix opening into the
  next recursive fold or batch it with the folded-witness opening (STACK.md
  slices 02B/02C/04).
- **Final two-stage verifier offloading.** This spec does not remove Stage 3 or
  move setup contribution into Stage 2. That is a later sumcheck-planner cutover
  built on the same setup-product prover/verifier instance.
- **Planner/table changes.** No schedule-table regeneration; mode selection is
  orthogonal to the schedule.
- **Making `Recursive` the default.** `Direct` remains the default mode for the
  scheme, examples, and the recursion artifact.

## Evaluation

### Acceptance Criteria

- [ ] `SetupSumcheckProver` lives in `akita-prover`; `akita-sumcheck` no longer
      exposes a general `factored_product` module.
- [ ] `SetupSumcheckVerifier` lives in `akita-verifier::stages::stage3` with a
      `new` + `verify` split.
- [ ] `recursive_setup_e2e` tests pass: recursive round-trip at folding arities
      and cross-mode rejection.
- [ ] `cargo run --release -p akita-pcs --example profile` with
      `AKITA_SETUP_MODE=recursive` (and `direct` for comparison) exercises the
      recursive setup-contribution path with per-mode proof-size reporting.
- [ ] `cargo fmt`, `cargo clippy --all -- -D warnings`, and `cargo test` are
      green.

### Testing Strategy

- New: `crates/akita-pcs/tests/recursive_setup_e2e.rs` — recursive prove +
  serialize round-trip + verify (one-hot D=64, nv=20/25), plus a cross-mode
  rejection test. Uses `run_on_large_stack` and the shared `common` fixtures.
- Must continue passing: all existing `Direct`-mode e2e suites
  (`single_poly_e2e`, `akita_e2e`, `multipoint_batched_e2e`,
  `batched_aggregated_e2e`, `transcript_hardening*`), and the
  `setup_contribution` unit/equivalence tests in `akita-verifier`.
- Tests run in debug; the recursion harness example runs in `--release`.
- ZK feature combinations are intentionally not covered (see Non-Goals).

### Performance

`Recursive` is verifier-cost-neutral on a native CPU but changes the zkVM
cycle profile. Measured with the `profile/akita-recursion` harness (OneHot
D=32), trace-only:

- **nv=25:** `akita_verify` Recursive 171.6 M vs Direct 157.9 M total cycles
  (Recursive ≈ +8.3%). The setup MLE/inner-product core matches within ~0.7%
  per fold; Recursive's extra cost is the `omega` reduction (~8.9 M, mostly
  fold 0).
- **nv=32:** `akita_verify` Recursive 2.972 G vs Direct 2.746 G total cycles
  (Recursive ≈ +8.3%); fold 0 dominates (~2.26 G MLE step).

So in isolation `Recursive` is slightly more expensive than `Direct`; its value
is realized only once the carried setup-prefix opening replaces the matrix scan
(future work, STACK.md slice 04). Verify with:

```bash
cd profile/akita-recursion
cargo build --release
AKITA_NUM_VARS=32 AKITA_RECURSION_BLOB=target/blob.bin \
  ./target/release/akita-recursion-artifact --setup-mode recursive
./target/release/akita-recursion-host --input target/blob.bin
```

## Design

### Architecture

- **Prover** (`akita-prover`): `core::{suffix,root_fold}` thread
  `setup_contribution_mode`. For `Recursive` on a non-terminal level they call
  `SetupSumcheckProver::prove`, which prepares the setup terms (required length,
  `bar_omega`, `alpha` powers) and runs the sumcheck, emitting a
  `SetupSumcheckProof` into the root fold proof's `stage3_sumcheck_proof`. For
  `Direct` the field is `None`.
- **Verifier** (`akita-verifier`): `protocol::core::{suffix,root_fold}` select the
  stage-3 proof based on mode (`InvalidSetup` if present/absent inconsistently),
  construct `SetupSumcheckVerifier::new(...)` from the ring-switch row
  evaluation, and call `verify(...)`. The verifier replays the
  `ExtensionOpeningReductionSumcheck`, then closes the final claim against
  `setup_val * omega * alpha_val`, where `omega` is the succinct `omega_S`
  evaluation (`SetupEvalPlan::evaluate_bar_omega_with_eq`).
- **Types** (`akita-types`): `SETUP_SUMCHECK_DEGREE` and the
  `SetupContributionMode` enum.
- **Recursion harness** (`profile/akita-recursion`): the `--setup-mode` CLI flag
  on the artifact, a `setup_contribution_mode` byte in the `AkitaJoltInputs`
  blob, and host/guest reading that byte so both replay under the proof's mode.

### Alternatives Considered

- **Keep a general `factored_product` sumcheck in `akita-sumcheck`.** Rejected:
  Akita only needs the setup-specific instance, and the prover-only logic
  belongs in `akita-prover` per the crate-boundary rules. A general module would
  be dead surface.
- **Select the recursion mode via an environment variable.** Rejected in favor
  of a CLI flag with a `direct` default, and recording the mode in the blob so
  host and guest cannot disagree.

## Documentation

- This spec.
- `crates/akita-pcs/examples/profile` with `AKITA_SETUP_MODE=recursive` (runnable
  harness for recursive vs direct setup-contribution).
- `profile/akita-recursion/README.md` already documents the harness flow; the new
  `--setup-mode` flag is self-describing via `--help`.

## References

- `STACK.md` rows 03B (`setup-product-sumcheck`) and 04 (`setup-claim-offloading`).
- `specs/setup-layout-repack.md`, `book/src/how/verifying/matrix_evaluation.md`.
- `crates/akita-verifier/src/stages/stage3.rs`,
  `crates/akita-prover/src/protocol/sumcheck/setup_sumcheck.rs`.
- Profiling: `profile/akita-recursion/README.md`.
