//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod challenges;
pub mod commitment;
pub mod opening_point;
pub mod proof;
pub mod prover;
pub mod sumcheck;
pub mod transcript;
pub mod verifier;

pub use commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, DefaultCommitmentConfig, DummyProof,
    HachiCommitment, HachiCommitmentCore, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
    RingCommitmentScheme, RingCommitmentSetup, StreamingCommitmentScheme,
};
pub use opening_point::RingOpeningPoint;
pub use proof::HachiProof;
pub use prover::HachiProver;
pub use sumcheck::{
    prove_sumcheck, CompressedUniPoly, SumcheckInstanceProver, SumcheckProof, UniPoly,
};
pub use transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
pub use verifier::verify;
