# Spec: Recursive Setup-Contribution Product Sumcheck (Stage 3)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   |                                |
| Created     | 2026-06-02                     |
| Status      | archived                       |
| PR          | #147; refactored by #287, #301, #311, #318, and #337 |
| Superseded-by | book/src/how/proving/sumcheck-stages.md and book/src/how/verifying/setup_contribution.md |
| Book-chapter | book/src/how/proving/sumcheck-stages.md |

> **Archived implementation record.** The Stage 3 setup product sumcheck
> shipped in PR [#147](https://github.com/LayerZero-Labs/akita/pull/147) and
> was reorganized by later protocol refactors. Current behavior is documented
> in the Book and implemented by `AkitaStage3Prover` and
> `SetupSumcheckVerifier`. This record is retained for its design history.

> **Harness supersession (PR #311).** Setup-contribution mode is now owned by
> the config-selected schedule, not a caller argument. The singleton Jolt
> artifact necessarily selects the direct scalar schedule, so its former
> `--setup-mode recursive` flag and blob mode byte were misleading and are
> removed. Exercise recursive stage 3 with the config-typed multi-group profile
> and the config-selected recursive E2E tests described below.

## Summary

In direct mode, each fold level's verifier proves the setup contribution
`<S_{<=N}, setup_index_weight_S>` by scanning the required public setup prefix.
In the config-selected recursive path, each non-terminal fold level delegates
the setup contribution to a Stage 3 setup product sumcheck. The verifier
replays that sumcheck, evaluates the setup weight vector without materializing
it, and closes the result against the selected setup-prefix opening. This
record also introduced the end-to-end test coverage and profiling harness that
exercise the recursive path.

## Intent

### Goal

Provide a prover/verifier-symmetric setup-product sumcheck for the recursive
setup-contribution path, located in the crates that own each role, and cover it
end to end.

Key abstractions and surfaces:

- `akita-prover` `protocol::sumcheck::akita_stage3::AkitaStage3Prover` —
  Akita-specific setup-product sumcheck prover. Moved out of the previous
  general `akita-sumcheck::factored_product` module; its `prove` entry point
  now folds in the term-preparation logic that used to live in `flow.rs`.
- `akita-verifier` `stages::stage3::SetupSumcheckVerifier` — the verifier
  counterpart, with a two-phase `new` (derive the setup evaluation plan and
  sumcheck round count from the ring-switch row evaluation) + `verify_stage3` (replay
  the extension-opening-reduction sumcheck and close it against local
  non-materialized setup and setup-weight evaluations) API. Lives under
  `stages/` alongside stage1/stage2.
- `akita-types::SETUP_SUMCHECK_DEGREE` — the relocated degree constant
  (formerly `FACTORED_PRODUCT_SUMCHECK_DEGREE`).
- the config-typed recursive multi-group profile exercises stage 3; the
  singleton `profile/akita-recursion` artifact remains direct-only.

### Invariants

- **Prover/verifier symmetry.** A proof produced by the supported recursive
  config verifies under that config. The ignored production-sized test
  `generated_recursive_onehot_profile_proves_with_setup_offload` in
  `crates/akita-pcs/tests/recursive_setup_e2e.rs` delegates to
  `recursive_multi_group_round_trip` in `tests/common/mod.rs`, which covers
  schedule resolution, setup-prefix provisioning, prove, serialization, and
  verify.
- **Mode is load-bearing, not cosmetic.** `AkitaBatchedProof::stage3_for_mode`
  rejects an unexpected Stage 3 payload or a missing payload, and the recursive
  round-trip helper rejects tampered Stage 3 claims, prefix evaluations, and
  sumcheck coefficients. There is no dedicated cross-mode E2E test with the
  old names previously listed here.
- **Setup-contribution value agreement.** The recursive setup-product sumcheck
  must reduce to the same setup contribution as the direct matrix scan.
  Protected at the unit level by the existing materialized-vs-direct
  equivalence fixtures in
  `crates/akita-types/src/setup_contribution/tests/` and at the
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
  work. The current recursive E2E target is `profile-ci` gated and does not
  claim coverage for the `zk` feature.
- **Carried-opening batching / setup-prefix commitment delegation in the
  original Stage 3 delivery.** The later planner and setup-offloading work in
  PRs #301 and #318 added the carried setup-prefix opening and grouped
  successor path described by the live planner spec.
- **Planner/table changes.** No schedule-table regeneration; mode selection is
  orthogonal to the schedule.
- **Making `Recursive` the default.** `Direct` remains the default mode for the
  scheme, examples, and the recursion artifact.

## Evaluation

### Historical acceptance record

The original acceptance criteria were completed by the shipped implementation
and its later protocol refactors. The current implementation names are used
below so this archived record does not look like an open task.

- [x] `AkitaStage3Prover` lives in `akita-prover`; `akita-sumcheck` no longer
      exposes a general `factored_product` module.
- [x] `SetupSumcheckVerifier` lives in `akita-verifier::stages::stage3` with a
      `new` + `verify_stage3` split.
- [x] `recursive_setup_e2e` covers the supported recursive round trip,
      serialization, setup-prefix binding, and Stage 3 tamper rejection through
      its shared helper. Cross-mode rejection is enforced by
      `AkitaBatchedProof::stage3_for_mode`, but is not a dedicated E2E case.
- [x] `cargo run --release -p akita-pcs --example profile` with
      `AKITA_SETUP_MODE=recursive` (and `direct` for comparison) exercises the
      recursive setup-contribution path with per-mode proof-size reporting.
- [x] `cargo fmt`, `cargo clippy --all -- -D warnings`, and `cargo test` were
      green.

### Testing Strategy

- New: `crates/akita-pcs/tests/recursive_setup_e2e.rs` — one ignored,
  production-sized fp128 one-hot recursive prove + serialize round-trip +
  verify case. Its shared `common` fixture uses two `nv=16` precommitted
  one-hot groups and a two-polynomial `nv=32` final group, then checks setup
  prefixes and Stage 3 tampering.
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
  per fold; Recursive's extra cost is the `setup_index_weight` reduction (~8.9 M, mostly
  fold 0).
- **nv=32:** `akita_verify` Recursive 2.972 G vs Direct 2.746 G total cycles
  (Recursive ≈ +8.3%); fold 0 dominates (~2.26 G MLE step).

These measurements are historical and predate the carried setup-prefix
integration. In the current selected recursive path, the verifier validates the
prefix slot and uses the carried setup-prefix opening instead of deriving that
opening by scanning the setup matrix. Verify the current profile path with:

```bash
AKITA_MODE=onehot_fp128_multi_group_recursive AKITA_SETUP_MODE=recursive \
  cargo run --release -p akita-pcs --example profile
```

## Design

### Architecture

- **Prover** (`akita-prover`): `core::{suffix,root_fold}` thread
  `setup_contribution_mode`. For `Recursive` on a non-terminal level they call
  `AkitaStage3Prover::prove`, which prepares the setup terms (required length,
  `setup_index_weight`, `alpha` powers) and runs the sumcheck, emitting a
  `SetupSumcheckProof` into the root fold proof's `stage3_sumcheck_proof`. For
  `Direct` the field is `None`.
- **Verifier** (`akita-verifier`): `protocol::core::{suffix,root_fold}` select the
  stage-3 proof based on mode (`InvalidSetup` if present/absent inconsistently),
  construct `SetupSumcheckVerifier::new(...)` from the ring-switch row
  evaluation, and call `verify_stage3(...)`. The verifier replays the
  `ExtensionOpeningReductionSumcheck`, then closes the final claim against
  `setup_val * setup_index_weight * alpha_val`. `setup_val` is the carried
  setup-prefix opening evaluation after the verifier validates slot coverage
  and transcript binding. `setup_index_weight` is evaluated directly at the
  setup-index challenge point from the cached segment partition
  (`SetupContributionPlan::evaluate_setup_index_weight_mle`).
- **Types** (`akita-types`): `SETUP_SUMCHECK_DEGREE` and the
  `SetupContributionMode` enum.
- **Recursion harness:** the singleton Jolt artifact is direct-only. Recursive
  setup contribution is exercised by the multi-group profile instantiated as
  `RecursiveCommitmentConfig<Cfg>` and by `recursive_setup_e2e`.

### Alternatives Considered

- **Keep a general `factored_product` sumcheck in `akita-sumcheck`.** Rejected:
  Akita only needs the setup-specific instance, and the prover-only logic
  belongs in `akita-prover` per the crate-boundary rules. A general module would
  be dead surface.
- **Select the recursion mode through a caller mode or serialized blob.** The
  old mode argument and blob byte were removed. The current schedule and config
  bind whether the Stage 3 payload is present, while the profile example keeps
  `AKITA_SETUP_MODE` only as a local workload selector.

## Documentation

- This spec.
- `crates/akita-pcs/examples/profile` with `AKITA_SETUP_MODE=recursive` (runnable
  harness for recursive vs direct setup-contribution).
- `profile/akita-recursion/README.md` documents the direct singleton Jolt flow;
  the Akita profile example documents the recursive multi-group run.

## References

- `STACK.md` rows 03B (`setup-product-sumcheck`) and 04 (`setup-claim-offloading`).
- `specs/archive/2026-Q3/setup-layout-repack.md`, `book/src/how/verifying/matrix_evaluation.md`.
- `crates/akita-verifier/src/stages/stage3.rs`,
  `crates/akita-prover/src/protocol/sumcheck/akita_stage3/mod.rs`.
- Profiling: `profile/akita-recursion/README.md`.
