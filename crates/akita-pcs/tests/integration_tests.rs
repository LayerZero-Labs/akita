#![allow(missing_docs)]

// Keep one Cargo target while retaining topic-sized source modules. Generic
// prover and verifier code is therefore monomorphized and linked only once.
// `autotests = false` in Cargo.toml prevents new top-level files from silently
// creating more integration-test binaries.

#[allow(dead_code)]
#[path = "integration_tests/common/mod.rs"]
mod common;

#[path = "integration_tests/akita_e2e.rs"]
mod akita_e2e;
#[path = "integration_tests/algebra/mod.rs"]
mod algebra;
#[cfg(feature = "schedules-default")]
#[path = "integration_tests/batched_aggregated_e2e.rs"]
mod batched_aggregated_e2e;
#[cfg(feature = "schedules-default")]
#[path = "integration_tests/commitment_contract.rs"]
mod commitment_contract;
#[cfg(feature = "profile-ci")]
#[path = "integration_tests/distributed_setup_offload_e2e.rs"]
mod distributed_setup_offload_e2e;
#[path = "integration_tests/fold_linf.rs"]
mod fold_linf;
#[path = "integration_tests/heterogeneous_prove_e2e.rs"]
mod heterogeneous_prove_e2e;
#[path = "integration_tests/label_schedule.rs"]
mod label_schedule;
#[path = "integration_tests/primality.rs"]
mod primality;
#[cfg(feature = "profile-ci")]
#[path = "integration_tests/recursive_setup_e2e.rs"]
mod recursive_setup_e2e;
#[path = "integration_tests/setup.rs"]
mod setup;
#[path = "integration_tests/single_poly_e2e.rs"]
mod single_poly_e2e;
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
