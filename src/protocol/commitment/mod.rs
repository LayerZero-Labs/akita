//! Protocol commitment abstraction layer.

mod commit;
mod config;
pub mod onehot;
mod schedule;
mod scheme;
pub(crate) mod transcript_append;
mod types;
pub mod utils;

#[cfg(test)]
pub(crate) use commit::root_batched_layout;
pub use commit::{
    hachi_batched_root_layout, HachiCommitmentCore, HachiExpandedSetup, HachiProverSetup,
    HachiSetupSeed, HachiVerifierSetup,
};
pub(crate) use commit::{root_current_w_len, scale_batched_root_layout};
pub use config::optimal_m_r_split;
pub use config::{
    beta_linf_fold_bound, compute_num_digits, compute_num_digits_fold, CommitmentConfig,
    CommitmentEnvelope, DecompositionParams, DynamicSmallTestCommitmentConfig,
    Fp128AdaptiveBoundedCommitmentConfig, Fp128AdaptiveOneHotCommitmentConfig,
    Fp128BoundedCommitmentConfig, Fp128CommitmentConfig, Fp128D64BoundedCommitmentConfig,
    Fp128FullCommitmentConfig, Fp128LogBasisCommitmentConfig, Fp128OneHotCommitmentConfig,
    HachiCommitmentLayout, SmallTestCommitmentConfig,
};
pub use onehot::{map_onehot_to_sparse_blocks, SparseBlockEntry};
pub use schedule::{
    hachi_recursive_level_layout_from_params, hachi_root_level_layout, HachiLevelParams,
    HachiScheduleInputs, HachiSchedulePlan,
};
pub(crate) use schedule::{
    packed_digits_bytes, planned_next_log_basis_with_current_basis,
    planned_recursive_suffix_bytes_with_log_basis, recursive_level_decomposition_from_root,
    recursive_r_decomp_levels_for_bound,
};
pub use scheme::{CommitWitness, CommitmentScheme, RingCommitmentScheme};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
