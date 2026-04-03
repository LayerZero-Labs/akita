//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod challenges;
pub mod commitment;
pub mod commitment_scheme;
pub mod dispatch;
pub mod dynamic_commitment_scheme;
pub mod hachi_poly_ops;
pub mod opening_point;
pub mod prg;
pub mod proof;
pub mod quadratic_equation;
mod recursive_runtime;
pub mod ring_switch;
pub mod root_poly;
pub mod sumcheck;
pub mod transcript;

pub use commitment::{
    optimal_m_r_split, presets, AdaptiveBoundedPolicy, AdaptiveOneHotD64Policy, AppendToTranscript,
    CommitmentConfig, CommitmentPolicy, CommitmentPreset, CommitmentScheme, DummyProof,
    DynamicCommitmentScheme, DynamicSmallTestCommitmentConfig, HachiCommitment,
    HachiCommitmentCore, HachiCommitmentLayout, HachiExpandedSetup, HachiOpeningClaim,
    HachiOpeningPoint, HachiProverSetup, HachiRootBatchSummary, HachiSetupSeed, HachiVerifierSetup,
    RingCommitment, RingCommitmentScheme, SmallTestCommitmentConfig, StaticBoundedPolicy,
};
pub use commitment_scheme::HachiCommitmentScheme;
pub use dynamic_commitment_scheme::{
    DynamicCommitHint, DynamicFullFamily, DynamicFullScheme, DynamicHachiCommitmentScheme,
    DynamicHachiProverSetup, DynamicHachiVerifierSetup, DynamicOneHotFamily, DynamicOneHotScheme,
    DynamicRingCommitment, DynamicRootConfigFamily,
};
pub use hachi_poly_ops::{DensePoly, HachiPolyOps, MultilinearPolynomail, OneHotIndex, OneHotPoly};
pub use opening_point::{BasisMode, BlockOrder, RingOpeningPoint};
pub use proof::{
    FlatRingVec, HachiBatchedProof, HachiBatchedProofShape, HachiBatchedRootProof, HachiLevelProof,
    HachiProof, HachiProofShape, HachiProofTail, LevelProofShape, PackedDigits, ProofRingVec,
};
pub use quadratic_equation::QuadraticEquation;
pub use root_poly::{DenseMultilinear, MultilinearPolynomial, OneHotMultilinear};
pub use sumcheck::batched_sumcheck::{prove_batched_sumcheck, verify_batched_sumcheck};
pub use sumcheck::{
    prove_sumcheck, verify_sumcheck, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, SumcheckProofShape, UniPoly,
};
pub use transcript::{sample_ext_challenge, Blake2bTranscript, KeccakTranscript, Transcript};
