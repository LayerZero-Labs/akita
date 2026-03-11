//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod ajtai;
pub mod challenges;
pub mod commitment;
pub mod commitment_scheme;
pub mod dispatch;
pub mod greyhound;
pub mod hachi_poly_ops;
pub mod labrador;
pub mod opening_point;
pub mod prg;
pub mod proof;
pub mod quadratic_equation;
pub mod ring_switch;
pub mod sumcheck;
pub mod transcript;

pub use commitment::{
    optimal_m_r_split, AppendToTranscript, CommitmentConfig, CommitmentScheme, DummyProof,
    DynamicSmallTestCommitmentConfig, Fp128BoundedCommitmentConfig, Fp128CommitmentConfig,
    Fp128FullCommitmentConfig, Fp128HalvingDCommitmentConfig, Fp128LogBasisCommitmentConfig,
    Fp128OneHotCommitmentConfig, HachiCommitment, HachiCommitmentCore, HachiCommitmentLayout,
    HachiExpandedSetup, HachiOpeningClaim, HachiOpeningPoint, HachiProverSetup, HachiSetupSeed,
    HachiVerifierSetup, RingCommitment, RingCommitmentScheme, SmallTestCommitmentConfig,
};
pub use commitment_scheme::HachiCommitmentScheme;
pub use hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotIndex, OneHotPoly};
pub use opening_point::{BasisMode, RingOpeningPoint};
pub use proof::{
    FlatCommitmentHint, FlatGreyhoundEvalProof, FlatLabradorLevelProof, FlatLabradorProof,
    FlatLabradorWitness, FlatRingVec, GreyhoundTail, HachiLevelProof, HachiProof, HachiProofTail,
    PackedDigits,
};
pub use quadratic_equation::QuadraticEquation;
pub use sumcheck::batched_sumcheck::{prove_batched_sumcheck, verify_batched_sumcheck};
pub use sumcheck::{
    prove_sumcheck, verify_sumcheck, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, UniPoly,
};
pub use transcript::{sample_ext_challenge, Blake2bTranscript, KeccakTranscript, Transcript};
