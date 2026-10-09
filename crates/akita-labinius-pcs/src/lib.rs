//! Akita commitments and evaluation openings for padded LaBinius image tables.
//!
//! Enable `labinius` explicitly to use `ImageProver` and `ImageVerifier`.
//! Openings borrow the caller's active clear channel; the caller owns EOF.
//! This authenticates an image-table evaluation, with no binary source opening.

#![cfg(feature = "labinius")]
#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

pub mod image;
pub mod session;
mod setup;

pub use image::{ImageCommitOutput, ImageEvaluation, ImageProver, ImageVerifier};

/// Coefficient field of the admitted D648/P128 root.
pub type F = akita_config::proof_optimized::fp128::Field;
/// Ordinary full-field dense image commitment configuration.
pub type ImageConfig = akita_config::proof_optimized::fp128::Dense;
/// Seed-derived admitted D648 root setup.
pub type RootSetup =
    akita_labinius_verifier::AdmittedRootSetup<F, 648, akita_algebra::MinusTrinomial>;
/// Prepared transform-domain root commitment matrix.
pub type PreparedMatrix =
    akita_labinius_prover::PreparedCommitMatrix<F, 648, akita_algebra::MinusTrinomial>;
