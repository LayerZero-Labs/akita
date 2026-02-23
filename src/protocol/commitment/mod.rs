//! Protocol commitment abstraction layer.

mod commit;
mod config;
mod scheme;
mod transcript_append;
mod types;
pub(crate) mod utils;

pub use commit::{HachiCommitmentCore, RingCommitmentSetup};
pub use config::{CommitmentConfig, DefaultCommitmentConfig};
pub use scheme::{CommitmentScheme, RingCommitmentScheme, StreamingCommitmentScheme};
pub use transcript_append::AppendToTranscript;
pub use types::{
    DummyProof, HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, RingCommitment,
    RingOpenProof, RingOpening,
};
