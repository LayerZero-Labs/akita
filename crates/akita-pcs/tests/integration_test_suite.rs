#![allow(missing_docs)]

// Single consolidated integration-test binary for akita-pcs.
//
// Each of the files below used to be its own separate `tests/*.rs` binary
// (auto-discovered by Cargo). They now live under `tests/integration_test/`
// and are pulled in here as modules instead, so shared generic code (prover,
// verifier, field/ring arithmetic) and `tests/integration_test/common/` are
// monomorphized and linked exactly once instead of once per file.
//
// Adding a new test file: drop it into `tests/integration_test/` and add a
// `mod` line for it below. `scripts/check-pcs-integration-test-coverage.sh`
// fails CI if a file is added here without a corresponding line, so a
// forgotten entry cannot silently vanish from the suite.

#[path = "integration_test/akita_e2e.rs"]
mod akita_e2e;
#[path = "integration_test/algebra.rs"]
mod algebra;
#[path = "integration_test/batched_aggregated_e2e.rs"]
mod batched_aggregated_e2e;
#[path = "integration_test/commitment_contract.rs"]
mod commitment_contract;
#[path = "integration_test/fold_linf.rs"]
mod fold_linf;
#[path = "integration_test/heterogeneous_prove_e2e.rs"]
mod heterogeneous_prove_e2e;
#[path = "integration_test/label_schedule.rs"]
mod label_schedule;
#[path = "integration_test/mixed_d_per_level_e2e.rs"]
mod mixed_d_per_level_e2e;
#[path = "integration_test/primality.rs"]
mod primality;
#[path = "integration_test/recursive_setup_e2e.rs"]
mod recursive_setup_e2e;
#[path = "integration_test/ring_switch.rs"]
mod ring_switch;
#[path = "integration_test/setup.rs"]
mod setup;
#[path = "integration_test/single_poly_e2e.rs"]
mod single_poly_e2e;
#[path = "integration_test/single_poly_tensor_e2e.rs"]
mod single_poly_tensor_e2e;
#[path = "integration_test/stage1_roundtrip.rs"]
mod stage1_roundtrip;
#[path = "integration_test/sumcheck_core.rs"]
mod sumcheck_core;
#[path = "integration_test/sumcheck_prover_driver.rs"]
mod sumcheck_prover_driver;
#[path = "integration_test/transcript.rs"]
mod transcript;

#[cfg(feature = "logging-transcript")]
#[path = "integration_test/transcript_hardening.rs"]
mod transcript_hardening;
#[cfg(feature = "logging-transcript")]
#[path = "integration_test/transcript_hardening_proptest.rs"]
mod transcript_hardening_proptest;
