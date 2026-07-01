//! Error types for estimator input validation and unsupported slices.

/// Estimator result type.
pub type Result<T> = std::result::Result<T, EstimatorError>;

/// Errors returned by the SIS estimator crate.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum EstimatorError {
    /// A public parameter was malformed.
    #[error("invalid parameter `{field}`: {reason}")]
    InvalidParameter {
        /// Name of the malformed field.
        field: &'static str,
        /// Human-readable rejection reason.
        reason: String,
    },
    /// A configuration option was malformed or internally inconsistent.
    #[error("invalid config `{field}`: {reason}")]
    InvalidConfig {
        /// Name of the malformed config field.
        field: &'static str,
        /// Human-readable rejection reason.
        reason: String,
    },
    /// The requested public API exists, but its estimator math has not landed.
    #[error("unsupported estimator feature in this slice: {feature}")]
    Unsupported {
        /// Feature or entry point that is not implemented yet.
        feature: &'static str,
    },
}
