//! Recursive prover-only state for later Akita prove levels.
//!
//! Groups the recursive-level witness helper: [`witness`] owns the D-agnostic
//! recursive witness vector `w` plus its zero-copy D-specific views.

mod witness;

pub use witness::{RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView};
