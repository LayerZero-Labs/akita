//! Native transcript-grinding replay.

#[path = "transcript_grinding/native_replay.rs"]
mod native_replay;
pub use native_replay::{
    NativeGrindingSumcheckProver, NativeGrindingSumcheckVerifier, NativeProofAcceptance,
    NativeProverGrinding, NativeVerifierGrinding,
};
