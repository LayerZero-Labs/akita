//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod challenges;
pub mod commitment;
pub mod commitment_scheme;
pub mod hachi_poly_ops;
pub mod opening_point;
pub mod proof;
pub mod quadratic_equation;
pub mod ring_switch;
pub mod sumcheck;
pub mod transcript;

pub use commitment::{
    AppendToTranscript, CommitmentConfig, CommitmentScheme, DummyProof,
    DynamicSmallTestCommitmentConfig, HachiCommitment, HachiCommitmentCore, HachiCommitmentLayout,
    HachiExpandedSetup, HachiOpeningClaim, HachiOpeningPoint, HachiProverSetup, HachiSetupSeed,
    HachiVerifierSetup, ProductionFp128CommitmentConfig, RingCommitment, RingCommitmentScheme,
    SmallTestCommitmentConfig,
};
pub use commitment_scheme::HachiCommitmentScheme;
pub use hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotPoly};
pub use opening_point::{BasisMode, RingOpeningPoint};
pub use proof::{HachiLevelProof, HachiProof};
pub use quadratic_equation::QuadraticEquation;
pub use sumcheck::batched_sumcheck::{prove_batched_sumcheck, verify_batched_sumcheck};
pub use sumcheck::{
    prove_sumcheck, verify_sumcheck, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, UniPoly,
};
pub use transcript::{sample_ext_challenge, Blake2bTranscript, KeccakTranscript, Transcript};
