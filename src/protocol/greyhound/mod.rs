//! Greyhound evaluation reduction layer.

pub mod eval;
pub mod reduce;
pub mod types;
pub mod verify;

pub use eval::greyhound_eval;
pub use reduce::greyhound_reduce;
pub use types::{GreyhoundDimensions, GreyhoundEvalProof};
pub use verify::greyhound_verify_stage1;
