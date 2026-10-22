//! Transcript-grinding replay.

#[cfg(feature = "zk")]
#[path = "transcript_grinding/masked.rs"]
mod masked;
#[path = "transcript_grinding/replay.rs"]
mod replay;
#[cfg(feature = "zk")]
pub use replay::CountingGrinding;
pub use replay::{
    FoldResponseReplay, GrindingReplay, GrindingSumcheckProver, GrindingSumcheckVerifier,
    ProverGrinding, ProverGrindingChannel, PublicGrindingSumcheck, VerifierGrinding,
    VerifierGrindingChannel,
};
