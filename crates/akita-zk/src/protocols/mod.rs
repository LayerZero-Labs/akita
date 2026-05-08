//! Zero-knowledge protocols.

pub mod opening;

pub use opening::{
    compact_ajtai_opening_proof, prove_ajtai_opening, prove_compact_ajtai_opening,
    prove_compact_gaussian_heuristic_ajtai_opening, prove_gaussian_heuristic_ajtai_opening,
    simulate_ajtai_opening_transcript, verify_ajtai_opening, verify_ajtai_opening_transcript,
    verify_compact_ajtai_opening, verify_compact_gaussian_heuristic_ajtai_opening,
    verify_gaussian_heuristic_ajtai_opening, AjtaiOpeningProof, AjtaiOpeningTranscript,
    CompactAjtaiOpeningProof,
};
