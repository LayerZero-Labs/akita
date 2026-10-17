//! Succinct, non-hiding Akita commitments for LaBinius binary and prime claims.
//!
//! A proof-field family fixes the base and challenge fields and the nested
//! configurations for digit and full-width element tables. `Prover` commits
//! a binary message and opens a binary claim, a prime claim, or both;
//! `Verifier` authenticates them against that commitment. One grouped Akita
//! proof opens the image digits, optional prime left opening, and response
//! digits. Enable `labinius` explicitly to use this crate.

#![cfg(feature = "labinius")]
#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

mod binding;
pub mod family;
mod prover;
mod session;
mod setup;
pub mod shipped;
mod verifier;

pub use prover::{Committed, Prover};
pub use setup::derive_root_setup;
pub use verifier::Verifier;

/// Seed-derived admitted D648 root setup.
pub type RootSetup = akita_labinius_verifier::AdmittedRootSetup<648, akita_algebra::MinusTrinomial>;
