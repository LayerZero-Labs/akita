//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod commitment;
pub mod commitment_scheme;
pub mod config;
#[cfg(test)]
mod ring_switch;
pub mod setup;

pub use akita_challenges::sample_ext_challenge;
pub use akita_prover::{
    CommitmentProver, CommittedPolynomials, HachiPolyOps, HachiProverSetup, ProverClaims,
};
pub use akita_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
pub use akita_types::{CommitmentEnvelope, DecompositionParams};
pub use akita_types::{HachiExpandedSetup, HachiSetupSeed, HachiVerifierSetup};
pub use commitment_scheme::HachiCommitmentScheme;
pub use config::CommitmentConfig;
