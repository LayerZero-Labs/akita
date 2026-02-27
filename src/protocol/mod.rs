//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod challenges;
pub mod commitment;
pub mod commitment_scheme;
pub mod iteration_prover;
pub mod opening_point;
pub mod proof;
pub mod quadratic_equation;
pub mod ring_switch;
pub mod sumcheck;
pub mod transcript;

pub use commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, DefaultCommitmentConfig, DummyProof,
    HachiCommitment, HachiCommitmentCore, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
    RingCommitmentScheme, RingCommitmentSetup, StreamingCommitmentScheme,
};
pub use commitment_scheme::HachiCommitmentScheme;
pub use iteration_prover::HachiProver;
pub use opening_point::RingOpeningPoint;
pub use proof::{HachiProof, SumcheckAux};
pub use sumcheck::{
    prove_sumcheck, verify_sumcheck, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, UniPoly,
};
pub use transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
