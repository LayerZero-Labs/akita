//! Protocol commitment abstraction layer.

mod commit;
mod config;
pub mod onehot;
mod schedule;
mod scheme;
pub(crate) mod transcript_append;
mod types;
pub mod utils;

pub(crate) use commit::root_current_w_len;
pub use commit::{
    HachiCommitmentCore, HachiExpandedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup,
};
pub use config::optimal_m_r_split;
pub use config::{
    beta_linf_fold_bound, compute_num_digits, compute_num_digits_fold, CommitmentConfig,
    CommitmentEnvelope, DecompositionParams, DynamicSmallTestCommitmentConfig,
    Fp128AdaptiveBoundedCommitmentConfig, Fp128AdaptiveD16BoundedCommitmentConfig,
    Fp128AdaptiveD32BoundedCommitmentConfig, Fp128AdaptiveOneHotCommitmentConfig,
    Fp128AdaptivePrime275BoundedCommitmentConfig, Fp128AdaptivePrime275OneHotCommitmentConfig,
    Fp128BoundedCommitmentConfig, Fp128CommitmentConfig, Fp128D16BoundedCommitmentConfig,
    Fp128D16FullCommitmentConfig, Fp128D16LogBasisCommitmentConfig, Fp128D16OneHotCommitmentConfig,
    Fp128D32BoundedCommitmentConfig, Fp128D32FullCommitmentConfig,
    Fp128D32LogBasisCommitmentConfig, Fp128D32OneHotCommitmentConfig,
    Fp128D64BoundedCommitmentConfig, Fp128FullCommitmentConfig, Fp128LogBasisCommitmentConfig,
    Fp128OneHotCommitmentConfig, Fp128Prime275FullCommitmentConfig,
    Fp128Prime275LogBasisCommitmentConfig, Fp128Prime275OneHotCommitmentConfig,
    HachiCommitmentLayout, SmallTestCommitmentConfig,
};
pub use onehot::{map_onehot_to_sparse_blocks, SparseBlockEntry};
pub use schedule::{
    hachi_recursive_level_layout_from_params, hachi_root_level_layout, HachiLevelParams,
    HachiScheduleInputs, HachiSchedulePlan,
};
pub(crate) use schedule::{
    recursive_level_decomposition_from_root, recursive_r_decomp_levels_for_bound,
};
pub use scheme::{CommitWitness, CommitmentScheme, RingCommitmentScheme};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
