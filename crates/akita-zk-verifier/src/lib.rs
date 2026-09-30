//! Verifier-side building blocks for zero-knowledge Akita openings.
//!
//! This crate is part of an opt-in extension that is still being assembled
//! (LayerZero-Labs/akita#120). It exposes no zero-knowledge mode: nothing on
//! the Akita proof path depends on it, and a caller cannot select it. Code
//! here must follow the verifier no-panic contract in
//! `docs/verifier-contract.md`, and the crate must never depend on a prover
//! crate.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

pub mod ring;
