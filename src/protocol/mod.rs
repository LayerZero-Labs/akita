//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod commitment;
pub mod challenges;
pub mod prover;
pub mod sumcheck;
pub mod transcript;
pub mod verifier;

pub use commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, DefaultCommitmentConfig,
    HachiCommitment, HachiCommitmentCore, HachiOpeningClaim, HachiOpeningPoint, HachiProof,
    RingCommitment, RingCommitmentScheme, RingCommitmentSetup, RingOpenProof, RingOpening,
    StreamingCommitmentScheme,
};
pub use prover::prove_opening_stub;
pub use sumcheck::{prove_sumcheck, CompressedUniPoly, SumcheckInstanceProver, SumcheckProof, UniPoly};
pub use transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
pub use verifier::verify_opening_stub;
