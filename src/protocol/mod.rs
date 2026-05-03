//! Protocol-layer transcript and commitment abstractions.
//!
//! This module defines the Hachi-native protocol interfaces used by higher-level
//! proof logic. It intentionally stays independent from external integration
//! details (for example, Jolt wiring).

pub mod commitment;
pub mod commitment_scheme;
pub mod config;
pub mod dispatch;
pub mod hachi_poly_ops;
pub mod opening_point;
pub mod params;
pub mod prg;
pub mod proof;
pub mod quadratic_equation;
mod recursive_runtime;
pub mod ring_switch;
pub mod setup;
pub mod sumcheck;

pub use akita_challenges::sample_ext_challenge;
pub use akita_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};
pub use commitment::{
    AppendToTranscript, CommitmentProver, CommitmentVerifier, CommittedOpenings,
    CommittedPolynomials, DirectStep, DummyProof, FoldStep, HachiCommitment, HachiOpeningClaim,
    HachiOpeningPoint, HachiRootBatchSummary, OpeningPoints, ProverClaims, RingCommitment,
    Schedule, Step, VerifierClaims, WitnessShape,
};
pub use commitment_scheme::HachiCommitmentScheme;
pub use config::{beta_linf_fold_bound, CommitmentConfig, CommitmentEnvelope, DecompositionParams};
pub use hachi_poly_ops::{DensePoly, HachiPolyOps, MultilinearPolynomail, OneHotIndex, OneHotPoly};
pub use opening_point::{BasisMode, BlockOrder, RingOpeningPoint};
pub use proof::{
    DirectWitnessProof, DirectWitnessShape, FlatRingVec, HachiBatchedFoldRoot, HachiBatchedProof,
    HachiBatchedProofShape, HachiBatchedRootProof, HachiLevelProof, HachiProofStep,
    HachiProofStepShape, LevelProofShape, PackedDigits,
};
pub use quadratic_equation::QuadraticEquation;
pub use setup::{HachiExpandedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup};
pub use sumcheck::batched_sumcheck::{prove_batched_sumcheck, verify_batched_sumcheck};
pub use sumcheck::{
    prove_sumcheck, verify_sumcheck, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, SumcheckProofShape, UniPoly,
};
