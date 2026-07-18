/// Errors that can occur in Akita PCS operations
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AkitaError {
    /// The proof verification failed
    #[error("Invalid proof")]
    InvalidProof,

    /// Polynomial size is invalid for the given parameters
    #[error("Invalid polynomial size: expected {expected}, got {actual}")]
    InvalidSize {
        /// Expected size
        expected: usize,
        /// Actual size
        actual: usize,
    },

    /// Evaluation point has wrong dimension
    #[error("Invalid evaluation point dimension: expected {expected}, got {actual}")]
    InvalidPointDimension {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Invalid input parameters
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// The requested polynomial layout has no supported folded proof schedule.
    #[error("Unsupported proof schedule: {0}")]
    UnsupportedSchedule(String),

    /// Setup file not found or corrupted
    #[error("Invalid or missing setup file: {0}")]
    InvalidSetup(String),
}
