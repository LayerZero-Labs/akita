//! Recursive prover-only state for later Akita prove levels.
//!
//! Owns the D-agnostic recursive witness vector `w`, its zero-copy D-specific
//! views, and the setup-prefix source adapter.

#[allow(dead_code)] // Representation foundation used by the recursive-witness cutover.
mod packed_digits;
mod setup_prefix_source;
mod witness;

pub use setup_prefix_source::RecursiveFoldSource;
pub use witness::{RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView};
