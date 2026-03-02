//! Greyhound evaluation reduction layer.

pub mod eval;
pub mod reduce;
pub mod types;

pub use eval::greyhound_eval;
pub use reduce::greyhound_reduce;
pub use types::{GreyhoundDimensions, GreyhoundEvalProof};
