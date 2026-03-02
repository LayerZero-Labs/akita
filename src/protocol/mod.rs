//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod challenges;
pub mod commitment;
pub mod commitment_scheme;
pub mod greyhound;
pub mod labrador;
pub mod opening_point;
pub mod prg;
pub mod proof;
pub mod quadratic_equation;
pub mod ring_switch;
pub mod sumcheck;
pub mod transcript;

pub use commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, DummyProof, HachiCommitment,
    HachiCommitmentCore, HachiCommitmentLayout, HachiExpandedSetup, HachiOpeningClaim,
    HachiOpeningPoint, HachiPreparedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup,
    ProductionFp128CommitmentConfig, RingCommitment, RingCommitmentScheme,
    SmallTestCommitmentConfig, StreamingCommitmentScheme,
};
pub use commitment_scheme::{HachiChunkState, HachiCommitmentScheme};
pub use opening_point::RingOpeningPoint;
pub use proof::{HachiFoldProof, HachiProof};
pub use quadratic_equation::QuadraticEquation;
pub use sumcheck::batched_sumcheck::{
    check_batched_output_claim, compute_batched_expected_output_claim, prove_batched_sumcheck,
    verify_batched_sumcheck, verify_batched_sumcheck_rounds, BatchedSumcheckRoundResult,
};
pub use sumcheck::{
    prove_sumcheck, verify_sumcheck, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, UniPoly,
};
pub use transcript::{sample_ext_challenge, Blake2bTranscript, KeccakTranscript, Transcript};
