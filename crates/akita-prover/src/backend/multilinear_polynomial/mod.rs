//! Canonical multilinear-polynomial wrapper for dense and one-hot representations.
//!
//! This is the intended public wrapper for heterogeneous root batches. All
//! wrapped polynomials must still share the same commitment config and root
//! layout chosen by the caller, but one batch can contain both dense and
//! one-hot roots.
//!
//! Homogeneous batches still reuse the existing backend-specific batched fast
//! paths; truly mixed batches fall back to the caller's per-polynomial
//! aggregation path.
//!
//! [`poly`] holds the wrapper enum, its borrowed dispatch views, and the
//! source-trait impls; [`ops`] holds the `CpuBackend` kernel impls that route
//! each source-typed view to the dense or one-hot backend.

mod ops;
mod poly;
#[cfg(test)]
mod tests;

pub use poly::{MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView};
