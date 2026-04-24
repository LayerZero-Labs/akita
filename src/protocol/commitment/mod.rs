//! Protocol commitment abstraction layer.

mod config;
pub(crate) mod generated;
pub mod presets;
pub(crate) mod profile;
pub(crate) mod schedule;
pub(crate) mod schedule_planner;
mod scheme;
pub(crate) mod transcript_append;
mod types;
pub mod utils;

pub use config::optimal_m_r_split;
pub use config::{
    beta_linf_fold_bound, compute_num_digits, compute_num_digits_fold,
    compute_num_digits_full_field, num_digits_for_bound, CommitmentConfig, CommitmentEnvelope,
    CommitmentPreset, DecompositionParams, GeneratedAdaptivePolicy, StaticBoundedPolicy,
};
pub use profile::{CommitmentFieldProfile, Fp128PrimeProfile};
pub(crate) use schedule::derive_batched_root_level_derivation;
pub use schedule::hachi_batched_root_layout;
pub(crate) use schedule::root_current_w_len;
#[cfg(test)]
pub(crate) use schedule::scale_batched_root_layout;
pub use schedule::{
    current_level_layout_with_log_basis, exact_schedule_plan_for_lookup_key,
    hachi_recursive_level_layout_from_params, hachi_root_level_layout,
    hachi_root_runtime_plan_with_batch, recursive_suffix_estimate_with_log_basis,
    HachiBatchPlanningEnvelope, HachiPlannedDirectStep, HachiPlannedLevel,
    HachiPlannedLevelExecution, HachiPlannedState, HachiPlannedStep, HachiRecursiveSuffixEstimate,
    HachiRootBatchSummary, HachiRootRuntimePlan, HachiScheduleInputs, HachiScheduleLookupKey,
    HachiSchedulePlan,
};
pub(crate) use schedule::{
    direct_witness_bytes, exact_planned_level_execution, field_bits, level_proof_bytes,
    packed_digits_bytes, planned_next_log_basis_with_current_basis_and_envelope,
    planned_next_w_len, planned_recursive_suffix_bytes_with_log_basis_and_envelope,
    planned_w_ring_element_count, recursive_level_decomposition_from_root,
    recursive_r_decomp_levels,
};
pub use scheme::CommitmentScheme;
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
