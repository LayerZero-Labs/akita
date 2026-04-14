//! Protocol commitment abstraction layer.

mod commit;
mod config;
pub(crate) mod generated;
pub mod onehot;
pub mod presets;
pub(crate) mod profile;
pub(crate) mod schedule;
pub(crate) mod schedule_planner;
mod scheme;
pub(crate) mod transcript_append;
mod types;
pub mod utils;

pub use commit::hachi_batched_root_layout;
pub(crate) use commit::optimal_root_batch_split;
pub(crate) use commit::scale_batched_root_layout;
pub use commit::{
    HachiCommitmentCore, HachiExpandedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup,
};
pub use config::optimal_m_r_split;
pub use config::{
    beta_linf_fold_bound, compute_num_digits, compute_num_digits_fold,
    compute_num_digits_full_field, num_digits_for_bound, CommitmentConfig, CommitmentEnvelope,
    CommitmentPreset, DecompositionParams, GeneratedAdaptivePolicy, HachiCommitmentLayout,
    SmallTestCommitmentConfig, StaticBoundedPolicy,
};
pub use onehot::{map_onehot_to_sparse_blocks, SparseBlockEntry};
pub use profile::{CommitmentFieldProfile, Fp128PrimeProfile};
pub(crate) use schedule::{
    batched_root_level_proof_bytes, derive_commitment_layout, direct_witness_bytes,
    exact_planned_level_execution, field_bits, packed_digits_bytes,
    planned_next_log_basis_with_current_basis_and_envelope, planned_next_w_len,
    planned_recursive_suffix_bytes_with_log_basis_and_envelope, planned_w_ring_element_count,
    recursive_level_decomposition_from_root, recursive_r_decomp_levels_for_bound,
};
pub use schedule::{
    exact_schedule_plan_for_lookup_key, hachi_recursive_level_layout_from_params,
    hachi_root_level_layout, hachi_root_runtime_plan_with_batch,
    recursive_suffix_estimate_with_log_basis, HachiBatchPlanningEnvelope, HachiLevelParams,
    HachiPlannedDirectStep, HachiPlannedLevel, HachiPlannedLevelExecution, HachiPlannedState,
    HachiPlannedStep, HachiRecursiveSuffixEstimate, HachiRootBatchSummary, HachiRootRuntimePlan,
    HachiScheduleInputs, HachiScheduleLookupKey, HachiSchedulePlan,
};
pub use scheme::{CommitWitness, CommitmentScheme, RingCommitmentScheme};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
