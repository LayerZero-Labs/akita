#![allow(missing_docs)]

// Single consolidated integration-test binary for akita-pcs.
//
// Each of the files below used to be its own separate `tests/*.rs` binary
// (auto-discovered by Cargo). They now live under `tests/integration_tests/`
// and are pulled in here as modules instead, so shared generic code (prover,
// verifier, field/ring arithmetic) and `tests/integration_tests/common/` are
// monomorphized and linked exactly once instead of once per file.
//
// Adding a new test file: drop it into `tests/integration_tests/` and add a
// `mod` line for it below. `scripts/check-pcs-integration-tests-coverage.sh`
// fails CI if a file is added here without a corresponding line, so a
// forgotten entry cannot silently vanish from the suite.

#[allow(dead_code)]
#[path = "integration_tests/common/mod.rs"]
mod common;

#[path = "integration_tests/akita_e2e.rs"]
mod akita_e2e;
#[path = "integration_tests/algebra/mod.rs"]
mod algebra;
#[path = "integration_tests/batched_aggregated_e2e.rs"]
mod batched_aggregated_e2e;
#[path = "integration_tests/commitment_contract.rs"]
mod commitment_contract;
#[path = "integration_tests/fold_linf.rs"]
mod fold_linf;
#[path = "integration_tests/heterogeneous_prove_e2e.rs"]
mod heterogeneous_prove_e2e;
#[path = "integration_tests/label_schedule.rs"]
mod label_schedule;
#[path = "integration_tests/mixed_d_per_level_e2e.rs"]
mod mixed_d_per_level_e2e;
#[path = "integration_tests/recursive_setup_e2e.rs"]
mod recursive_setup_e2e;
#[path = "integration_tests/ring_switch.rs"]
mod ring_switch;
#[path = "integration_tests/setup.rs"]
mod setup;
#[path = "integration_tests/single_poly_e2e.rs"]
mod single_poly_e2e;
#[path = "integration_tests/single_poly_tensor_e2e.rs"]
mod single_poly_tensor_e2e;
#[path = "integration_tests/stage1_roundtrip.rs"]
mod stage1_roundtrip;
#[path = "integration_tests/sumcheck_core.rs"]
mod sumcheck_core;
#[path = "integration_tests/sumcheck_prover_driver.rs"]
mod sumcheck_prover_driver;
#[path = "integration_tests/transcript.rs"]
mod transcript;

#[cfg(feature = "logging-transcript")]
#[path = "integration_tests/transcript_hardening.rs"]
mod transcript_hardening;
#[cfg(feature = "logging-transcript")]
#[path = "integration_tests/transcript_hardening_proptest.rs"]
mod transcript_hardening_proptest;
