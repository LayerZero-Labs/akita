//! Recursive prover-only state for later Akita prove levels.
//!
//! Owns the D-agnostic recursive witness vector `w`, its zero-copy D-specific
//! views, and the setup-prefix source adapter.

mod setup_prefix_source;
mod witness;

pub use setup_prefix_source::RecursiveFoldSource;
pub use witness::{RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView};
