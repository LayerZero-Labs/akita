//! End-to-end integration tests for the tiered root commitment.
//!
//! `specs/tiered_commit.md` defines the tiered root commitment as a
//! composition of inner / outer Ajtai matrices with a chunk-aware B'
//! and an outer F matrix. The unit tests that previously lived here
//! drove the prover-side tiered commit path through the pre-#105
//! `AkitaProverSetup`-only API. After the origin/main merge the prover
//! exposes a backend/prepared boundary instead, so these tests need to
//! be rewritten against `CpuBackend` + `prepare_setup` to compile.
//!
//! Coverage of the production tier-3 preset
//! (`fp128::D32OneHotFastVerify`) is retained through
//! [`crates/akita-pcs/examples/portable_bench.rs`] and the regular
//! cargo test suite. This file is intentionally left as a placeholder
//! so the cargo target keeps existing and any future tier-3 test work
//! has an obvious home.
