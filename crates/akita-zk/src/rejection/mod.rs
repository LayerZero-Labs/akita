//! Rejection-sampling policies.

pub mod r#box;
pub mod gaertner;
pub mod gaussian;

pub use gaertner::GaertnerRejectionParams;
pub use gaussian::GaussianRejectionParams;
pub use r#box::BoxRejectionParams;
