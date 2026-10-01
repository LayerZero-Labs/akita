//! Protocol errors and checked integer arithmetic shared by Akita crates.

#![deny(missing_docs)]
#![warn(unreachable_pub)]

/// Checked integer formulas shared by Akita's layout and validation code.
pub mod checked;

/// Checked narrowing conversions from `usize` into fixed-width integers.
pub mod narrowing;

/// Errors that can occur in Akita PCS operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AkitaError {
    /// Proof bytes failed a protocol check during verification.
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

    /// An invariant that Akita maintains itself failed after the inputs were
    /// admitted. This reports a bug in Akita or in a custom backend, not a
    /// rejected proof and not a caller mistake. The message names the failed
    /// invariant.
    ///
    /// The variant is not tied to the prover: setup and cache code report
    /// their own failed invariants the same way. It must never stand in for a
    /// rejection. A check on a proof, a commitment, or any other untrusted
    /// value returns [`AkitaError::InvalidProof`], even when only a bug in an
    /// honest prover could make it fail.
    #[error("Internal error: {0}")]
    Internal(String),
}
