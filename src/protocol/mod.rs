//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod challenges;
pub mod commitment;
pub mod commitment_scheme;
pub mod dispatch;
pub mod hachi_poly_ops;
pub mod opening_point;
pub mod prg;
pub mod proof;
pub mod quadratic_equation;
mod recursive_runtime;
pub mod ring_switch;
pub mod sumcheck;
pub mod transcript;

pub use commitment::{
    optimal_m_r_split, presets, AppendToTranscript, CommitmentConfig, CommitmentPolicy,
    CommitmentPreset, CommitmentScheme, DummyProof, DynamicSmallTestCommitmentConfig,
    GeneratedAdaptivePolicy, HachiCommitment,
    HachiCommitmentCore, HachiCommitmentLayout, HachiExpandedSetup, HachiOpeningClaim,
    HachiOpeningPoint, HachiProverSetup, HachiRootBatchSummary, HachiSetupSeed, HachiVerifierSetup,
    PlannedAdaptiveBoundedPolicy, RingCommitment, RingCommitmentScheme, SmallTestCommitmentConfig,
    StaticBoundedPolicy,
};
pub use commitment_scheme::HachiCommitmentScheme;
pub use hachi_poly_ops::{DensePoly, HachiPolyOps, MultilinearPolynomail, OneHotIndex, OneHotPoly};
pub use opening_point::{BasisMode, BlockOrder, RingOpeningPoint};
pub use proof::{
    DirectWitnessProof, DirectWitnessShape, FlatRingVec, HachiBatchedProof, HachiBatchedProofShape,
    HachiBatchedRootProof, HachiLevelProof, HachiProof, HachiProofShape, HachiProofStep,
    HachiProofStepShape, LevelProofShape, PackedDigits, ProofRingVec,
};
pub use quadratic_equation::QuadraticEquation;
pub use sumcheck::batched_sumcheck::{prove_batched_sumcheck, verify_batched_sumcheck};
pub use sumcheck::{
    prove_sumcheck, verify_sumcheck, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, SumcheckProofShape, UniPoly,
};
pub use transcript::{sample_ext_challenge, Blake2bTranscript, KeccakTranscript, Transcript};
