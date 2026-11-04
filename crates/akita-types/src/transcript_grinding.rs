//! Transcript-grinding replay.

#[cfg(feature = "zk")]
#[path = "transcript_grinding/masked.rs"]
mod masked;
#[path = "transcript_grinding/replay.rs"]
mod replay;
#[cfg(feature = "zk")]
pub use replay::CountingGrinding;
pub use replay::{
    FoldResponseReplay, GrindingReplay, GrindingSumcheckVerifier, PendingEof, ProofAcceptance,
    PublicGrindingSumcheck, VerifierGrinding, VerifierGrindingChannel,
};
#[cfg(feature = "prover")]
pub use replay::{GrindingSumcheckProver, ProverGrinding};
