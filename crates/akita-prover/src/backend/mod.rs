//! Polynomial backends and prover-only witness state.

mod coefficient_packing;
mod dense;
mod field_reduction;
pub(crate) mod flat_blocks;
mod multilinear_polynomial;
pub(crate) mod onehot;
pub(crate) mod packed_digits;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod poly_helpers;
mod recursive;
mod ring_switch;
pub(crate) mod sparse_ring;

pub use coefficient_packing::coefficient_packing_partials_from_position_source;
pub use dense::{DenseBatchView, DensePoly, DenseView};
pub use field_reduction::tensor_pack_recursive_witness;
pub use multilinear_polynomial::{
    MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView,
};
pub use onehot::{OneHotBatchView, OneHotIndex, OneHotPoly, OneHotView};
pub use recursive::{
    RecursiveFoldSource, RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView,
};
pub use ring_switch::RingSwitchRelationView;
pub use sparse_ring::SparseRingBlockEntry;

#[cfg(test)]
pub(crate) mod test_support;
