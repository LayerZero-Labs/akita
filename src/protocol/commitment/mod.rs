//! Protocol commitment abstraction layer.

mod commit;
mod config;
pub mod onehot;
mod scheme;
mod transcript_append;
mod types;
pub mod utils;

pub use commit::{
    HachiCommitmentCore, HachiExpandedSetup, HachiPreparedSetup, HachiProverSetup, HachiSetupSeed,
    HachiVerifierSetup, MegaPolyBlock,
};
pub use config::{
    CommitmentConfig, DynamicSmallTestCommitmentConfig, HachiCommitmentLayout,
    ProductionFp128CommitmentConfig, SmallTestCommitmentConfig,
};
pub use onehot::{map_onehot_to_sparse_blocks, SparseBlockEntry};
pub use scheme::{
    CommitWitness, CommitmentScheme, RingCommitmentScheme, StreamingCommitmentScheme,
};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
