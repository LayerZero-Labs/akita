//! Shared Akita protocol data shapes.
//!
//! This crate contains proof objects, commitment/opening wrappers, opening
//! point reductions, per-level parameter shapes, and generated schedule/SIS
//! data shared by prover, verifier, and planner code.

pub mod commitment;
pub mod generated;
pub mod opening_point;
pub mod params;
pub mod proof;
pub mod transcript_append;

pub use commitment::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
pub use opening_point::{
    basis_weights, lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, BasisMode, BlockOrder, RingOpeningPoint,
};
pub use params::{AjtaiKeyParams, LevelParams};
pub use proof::{
    DirectWitnessProof, DirectWitnessShape, FlatDigitBlockIter, FlatDigitBlocks, FlatRingVec,
    HachiBatchedFoldRoot, HachiBatchedProof, HachiBatchedProofShape, HachiBatchedRootProof,
    HachiCommitmentHint, HachiLevelProof, HachiProofStep, HachiProofStepShape, HachiStage1Proof,
    HachiStage1StageProof, HachiStage1StageShape, HachiStage2Proof, LevelProofShape, PackedDigits,
    RingSliceSerializer,
};
pub use transcript_append::AppendToTranscript;
