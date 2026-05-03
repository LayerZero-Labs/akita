//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod commitment;
pub mod commitment_scheme;
pub mod config;
pub mod dispatch;
pub mod hachi_poly_ops;
pub mod prg;
pub mod quadratic_equation;
mod recursive_runtime;
pub mod ring_switch;
pub mod setup;
pub mod sumcheck;

pub use akita_challenges::sample_ext_challenge;
pub use akita_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
pub use commitment::{
    CommitmentProver, CommitmentVerifier, CommittedOpenings, CommittedPolynomials, DirectStep,
    FoldStep, HachiRootBatchSummary, OpeningPoints, ProverClaims, Schedule, Step, VerifierClaims,
    WitnessShape,
};
pub use commitment_scheme::HachiCommitmentScheme;
pub use config::{beta_linf_fold_bound, CommitmentConfig, CommitmentEnvelope, DecompositionParams};
pub use hachi_poly_ops::{DensePoly, HachiPolyOps, MultilinearPolynomail, OneHotIndex, OneHotPoly};
pub use quadratic_equation::QuadraticEquation;
pub use setup::{HachiExpandedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup};
