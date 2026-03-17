//! Protocol commitment abstraction layer.

mod commit;
mod config;
pub mod onehot;
mod scheme;
mod types;
pub mod utils;

pub use commit::{
    HachiCommitmentCore, HachiExpandedSetup, HachiProverSetup, HachiSetupSeed, HachiVerifierSetup,
};
pub use config::optimal_m_r_split;
pub use config::{
    compute_num_digits, compute_num_digits_fold, CommitmentConfig, DecompositionParams,
    DynamicSmallTestCommitmentConfig, Fp128BoundedCommitmentConfig, Fp128CommitmentConfig,
    Fp128FullCommitmentConfig, Fp128HalvingDCommitmentConfig, Fp128LogBasisCommitmentConfig,
    Fp128OneHotCommitmentConfig, HachiCommitmentLayout, SmallTestCommitmentConfig,
};
pub use onehot::{map_onehot_to_sparse_blocks, SparseBlockEntry};
pub use scheme::{CommitWitness, CommitmentScheme, RingCommitmentScheme};
pub use types::RingCommitment;
