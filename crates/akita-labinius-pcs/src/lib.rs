//! Succinct, non-hiding Akita commitments for LaBinius binary evaluations.
//!
//! Enable `labinius` explicitly to use `ImageProver` and `ImageVerifier`.
//! Root openings compose the root reduction with one grouped Akita proof and
//! own their session and EOF. Image-only openings remain available on a channel.

#![cfg(feature = "labinius")]
#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

pub mod config;
pub mod image;
pub mod root;
pub mod session;

pub use config::{DigitConfig, Digits1, Digits2, Digits4};
pub use image::{ImageCommitOutput, ImageEvaluation, ImageProver, ImageVerifier};
pub use root::{RootPcsProver, RootPcsVerifier};

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
/// Prepared root matrix caches for committing and for the root reduction.
pub type PreparedRoot =
    akita_labinius_prover::PreparedRootMatrices<F, 648, akita_algebra::MinusTrinomial>;
