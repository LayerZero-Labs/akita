//! Protocol commitment abstraction layer.

mod commit;
mod config;
pub(crate) mod onehot;
mod scheme;
mod transcript_append;
mod types;
pub(crate) mod utils;

pub use commit::{
    HachiCommitmentCore, HachiExpandedSetup, HachiPreparedSetup, HachiProverSetup, HachiSetupSeed,
    HachiVerifierSetup,
};
pub use config::{
    CommitmentConfig, HachiCommitmentLayout, ProductionFp128CommitmentConfig,
    SmallTestCommitmentConfig,
};
pub use scheme::{
    CommitWitness, CommitmentScheme, RingCommitmentScheme, StreamingCommitmentScheme,
};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
