//! Protocol commitment abstraction layer.

mod commit;
mod config;
pub(crate) mod onehot;
mod scheme;
mod transcript_append;
mod types;
pub(crate) mod utils;

pub use commit::{HachiCommitmentCore, RingCommitmentSetup};
pub use config::{CommitmentConfig, ProductionFp128CommitmentConfig, SmallTestCommitmentConfig};
pub use scheme::{
    CommitWitness, CommitmentScheme, RingCommitmentScheme, StreamingCommitmentScheme,
};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
};
