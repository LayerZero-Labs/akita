//! Recursive prover-only state for later Akita prove levels.
//!
//! Groups the two recursive-level helpers: [`witness`] owns the D-agnostic
//! recursive witness vector `w` plus its zero-copy D-specific views, and
//! [`hint`] preserves the commitment-side prover caches the next recursive
//! level needs without round-tripping through the proof-oriented flat adapters.

mod hint;
mod witness;

pub use hint::RecursiveCommitmentHintCache;
pub use witness::{RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView};
