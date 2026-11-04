//! Transcript-grinding replay.

#[cfg(feature = "zk")]
#[path = "transcript_grinding/masked.rs"]
mod masked;
#[path = "transcript_grinding/replay.rs"]
mod replay;
#[cfg(all(feature = "zk", feature = "prover"))]
pub use masked::MaskedProverChannel;
#[cfg(feature = "zk")]
pub use masked::MaskedVerifierChannel;
#[cfg(feature = "zk")]
pub use replay::CountingGrinding;
pub use replay::{
    grinding_sumcheck_body_site, FoldResponseReplay, GrindingReplay, GrindingSumcheckVerifier, PendingEof, ProofAcceptance,
    VerifierGrinding, VerifierGrindingChannel,
};
#[cfg(feature = "prover")]
pub use replay::{GrindingSumcheckProver, ProverGrinding, ProverGrindingChannel};
