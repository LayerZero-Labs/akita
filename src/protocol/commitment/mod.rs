//! Protocol commitment abstraction layer.

mod commit;
mod config;
pub mod onehot;
pub mod presets;
mod schedule;
mod scheme;
pub(crate) mod transcript_append;
mod types;
pub mod utils;

pub use commit::hachi_batched_root_layout;
pub(crate) use commit::{root_current_w_len, scale_batched_root_layout};
pub use commit::{
    HachiCommitmentCore, HachiExpandedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup,
};
pub use config::optimal_m_r_split;
pub use config::{
    beta_linf_fold_bound, compute_num_digits, compute_num_digits_fold, CommitmentConfig,
    CommitmentEnvelope, CommitmentPolicy, CommitmentPreset, DecompositionParams,
    DynamicSmallTestCommitmentConfig, Fp128AdaptiveBoundedPolicy, Fp128AdaptiveOneHotD64Policy,
    Fp128StaticBoundedPolicy, HachiCommitmentLayout, SmallTestCommitmentConfig,
};
pub use onehot::{map_onehot_to_sparse_blocks, SparseBlockEntry};
pub use presets::*;
pub use schedule::{
    hachi_recursive_level_layout_from_params, hachi_root_level_layout, HachiLevelParams,
    HachiPlannedLevel, HachiPlannedState, HachiRootBatchSummary, HachiScheduleInputs,
    HachiSchedulePlan,
};
pub(crate) use schedule::{
    packed_digits_bytes, planned_next_log_basis_with_current_basis,
    planned_recursive_suffix_bytes_with_log_basis, recursive_level_decomposition_from_root,
    recursive_r_decomp_levels_for_bound,
};
pub use scheme::{CommitWitness, CommitmentScheme, DynamicCommitmentScheme, RingCommitmentScheme};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
