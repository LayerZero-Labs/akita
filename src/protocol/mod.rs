//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod commitment;
pub mod transcript;

pub use commitment::{
    AppendToTranscript, CommitmentScheme, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint,
    HachiProof, StreamingCommitmentScheme,
};
pub use transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
