//! Dense polynomial: all ring coefficients materialized in memory.
//!
//! [`DensePoly`] uses standard dense algorithms — balanced-digit decomposition,
//! NTT-based matrix-vector multiply, and parallel block folds.

mod commit;
mod kernels;
mod ops;
mod poly;
mod tensor_fold;
mod views;
#[cfg(test)]
mod tests;

pub use poly::DensePoly;
pub use views::{
    DenseCommitView, DenseOpeningBatchView, DenseOpeningView, DenseTensorBatchView,
    DenseTensorView,
};
