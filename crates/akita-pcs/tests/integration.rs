//! akita-pcs integration tests, built as one test binary.
//!
//! Each module is one former test binary. One binary compiles the shared
//! prover, verifier, and backend generics once instead of once per suite.
//! Select a suite with a name filter, for example
//! `cargo test -p akita-pcs --test integration -- fold_linf::`.

#![allow(missing_docs)]

mod common;

mod akita_fp128_e2e;
mod akita_small_field_e2e;
#[path = "algebra/mod.rs"]
mod algebra;
mod backend_ownership;
mod distributed_setup_offload_e2e;
mod dynamic_backends;
mod embedded_transcript;
mod fold_linf;
mod primality;
mod proof_stream_smoke;
mod protocol_soundness;
mod recursive_setup_e2e;
#[allow(dead_code)]
#[path = "../benches/support/relation_phase_timing.rs"]
mod relation_phase_timing;
mod runtime_catalogs;
mod setup;
mod sumcheck_poly;
mod transcript_hardening;
mod transcript_hardening_proptest;
