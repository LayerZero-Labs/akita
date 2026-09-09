//! Dense polynomial: all ring coefficients materialized in memory.
//!
//! [`DensePoly`] stores dense coefficients and exposes opening/tensor views.
//! Commitment representations are selected through `CommitmentSource`.

mod kernels;
mod ops;
mod poly;
#[cfg(test)]
mod tests;
mod views;

pub(super) use kernels::dense_coefficient_packing_partials;
pub use poly::DensePoly;
pub use views::{DenseBatchView, DenseView};
