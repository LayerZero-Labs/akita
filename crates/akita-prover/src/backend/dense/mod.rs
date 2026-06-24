//! Dense polynomial: all ring coefficients materialized in memory.
//!
//! [`DensePoly`] uses standard dense algorithms — balanced-digit decomposition,
//! NTT-based matrix-vector multiply, and parallel block folds.

mod commit;
mod kernels;
mod ops;
mod poly;
mod tensor_fold;
#[cfg(test)]
mod tests;
mod views;

pub use poly::DensePoly;
pub use views::{DenseBatchView, DenseView};
