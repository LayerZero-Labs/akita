//! Protocol errors shared by Akita crates.

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

    /// Setup data is missing or invalid.
    #[error("Invalid or missing setup file: {0}")]
    InvalidSetup(String),
}

impl From<jolt_field::FieldError> for AkitaError {
    fn from(error: jolt_field::FieldError) -> Self {
        match error {
            jolt_field::FieldError::InvalidInput(message) => Self::InvalidInput(message),
            jolt_field::FieldError::InvalidSize { expected, actual } => {
                Self::InvalidSize { expected, actual }
            }
        }
    }
}
