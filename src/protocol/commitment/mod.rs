//! Protocol commitment abstraction layer.

mod scheme;
mod transcript_append;
mod types;

pub use scheme::{CommitmentScheme, StreamingCommitmentScheme};
pub use transcript_append::AppendToTranscript;
pub use types::{HachiCommitment, HachiOpeningClaim, HachiOpeningPoint, HachiProof};
