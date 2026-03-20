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
    Fp128AdaptiveOneHotCommitmentConfig, Fp128BoundedCommitmentConfig, Fp128CommitmentConfig,
    Fp128D64BoundedCommitmentConfig, Fp128FullCommitmentConfig, Fp128LogBasisCommitmentConfig,
    Fp128OneHotCommitmentConfig, Fp128Rank2BoundedCommitmentConfig, HachiCommitmentLayout,
    SmallTestCommitmentConfig,
};
pub use onehot::{map_onehot_to_sparse_blocks, SparseBlockEntry};
pub use schedule::{
    hachi_level_layout, hachi_recursive_level_layout_from_params, hachi_root_level_layout,
    HachiLevelParams, HachiScheduleInputs,
};
pub use scheme::{CommitWitness, CommitmentScheme, RingCommitmentScheme};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
