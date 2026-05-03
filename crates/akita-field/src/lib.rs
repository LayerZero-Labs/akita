//! Field traits and core error types for Akita.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

/// Field and module arithmetic traits.
pub mod arithmetic;
/// Error types shared by Akita crates.
pub mod error;
/// Conditional parallelism utilities.
pub mod parallel;

pub use arithmetic::{
    AdditiveGroup, CanonicalField, FieldCore, FieldSampling, FromSmallInt, Invertible, Module,
    PseudoMersenneField, SmoothFftField,
};
pub use error::HachiError;
