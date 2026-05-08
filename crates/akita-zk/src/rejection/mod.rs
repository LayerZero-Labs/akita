//! Rejection-sampling policies.

pub mod r#box;
pub mod gaussian;

pub use gaussian::GaussianRejectionParams;
pub use r#box::BoxRejectionParams;
