//! Protocol errors and checked integer arithmetic shared by Akita crates.

#![deny(missing_docs)]
#![warn(unreachable_pub)]

/// Checked integer formulas shared by Akita's layout and validation code.
pub mod checked;

/// Errors that can occur in Akita PCS operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AkitaError {
    /// Proof verification failed.
    #[error("Invalid proof")]
    InvalidProof,

    /// A polynomial or protocol object has an invalid size.
    #[error("Invalid polynomial size: expected {expected}, got {actual}")]
    InvalidSize {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    /// An evaluation point has the wrong dimension.
    #[error("Invalid evaluation point dimension: expected {expected}, got {actual}")]
    InvalidPointDimension {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Input parameters are invalid.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// The requested polynomial layout has no supported folded proof schedule.
    #[error("Unsupported proof schedule: {0}")]
    UnsupportedSchedule(String),

    /// Setup data is missing or invalid.
    #[error("Invalid or missing setup file: {0}")]
    InvalidSetup(String),
}
