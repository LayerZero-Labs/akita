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
pub mod params;
pub mod prg;
pub mod proof;
pub mod quadratic_equation;
mod recursive_runtime;
pub mod ring_switch;
// `setup_delegation` is orphaned post-merge: it depends on
// `prove_without_setup_delegation` / `verify_without_setup_delegation` which
// only existed on the feature branch's `commitment_scheme.rs`. The file is
// preserved in-tree as a starting point for re-landing T2 setup delegation as
// a follow-up PR, but is excluded from compilation here.
// pub(crate) mod setup_delegation;
// `shared_matrix_setup` references `HachiVerifierSetup.shared_matrix_cache`,
// a branch-only field. It is preserved in-tree but excluded from compilation
// pending the T2 setup-delegation follow-up PR.
// pub(crate) mod shared_matrix_setup;
pub mod sumcheck;
pub mod transcript;

pub use commitment::{
    optimal_m_r_split, presets, AppendToTranscript, CommitmentConfig, CommitmentPreset,
    CommitmentScheme, DummyProof, GeneratedAdaptivePolicy, HachiCommitment, HachiCommitmentCore,
    HachiExpandedSetup, HachiOpeningClaim, HachiOpeningPoint, HachiProverSetup,
    HachiRootBatchSummary, HachiSetupSeed, HachiVerifierSetup, RingCommitment,
    RingCommitmentScheme, SmallTestCommitmentConfig, StaticBoundedPolicy,
};
pub use commitment_scheme::HachiCommitmentScheme;
pub use hachi_poly_ops::{DensePoly, HachiPolyOps, MultilinearPolynomail, OneHotIndex, OneHotPoly};
pub use opening_point::{BasisMode, BlockOrder, RingOpeningPoint};
pub use proof::{
    DirectWitnessProof, DirectWitnessShape, FlatRingVec, HachiBatchedProof, HachiBatchedProofShape,
    HachiBatchedRootProof, HachiLevelProof, HachiProof, HachiProofShape, HachiProofStep,
    HachiProofStepShape, LevelProofShape, PackedDigits,
};
pub use quadratic_equation::QuadraticEquation;
pub use sumcheck::batched_sumcheck::{prove_batched_sumcheck, verify_batched_sumcheck};
pub use sumcheck::{
    prove_sumcheck, verify_sumcheck, CompressedUniPoly, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProof, SumcheckProofShape, UniPoly,
};
pub use transcript::{sample_ext_challenge, Blake2bTranscript, KeccakTranscript, Transcript};
