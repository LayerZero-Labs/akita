//! Protocol commitment abstraction layer.

pub(crate) mod digit_math;
pub(crate) mod generated;
pub(crate) mod schedule;
mod schedule_types;
mod scheme;
pub(crate) mod sis_derivation;
pub(crate) mod transcript_append;
mod types;
pub mod utils;

pub(crate) use schedule::derive_batched_root_level_derivation;
pub use schedule::hachi_batched_root_layout;
pub use schedule::{
    current_level_layout_with_log_basis, hachi_recursive_level_layout_from_params,
    HachiPlannedDirectStep, HachiPlannedLevel, HachiPlannedLevelExecution, HachiPlannedState,
    HachiPlannedStep, HachiRootBatchSummary, HachiScheduleInputs, HachiScheduleLookupKey,
    HachiSchedulePlan,
};
pub(crate) use schedule::{
    direct_witness_bytes, level_proof_bytes, planned_next_w_len, planned_w_ring_element_count,
    recursive_level_decomposition_from_root,
};
pub(crate) use schedule_types::schedule_from_plan;
pub use schedule_types::{DirectStep, FoldStep, Schedule, Step, WitnessShape};
pub use scheme::{
    CommitmentProver, CommitmentVerifier, CommittedOpenings, CommittedPolynomials, OpeningPoints,
    ProverClaims, VerifierClaims,
};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
