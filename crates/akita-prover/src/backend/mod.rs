//! Polynomial backends and prover-only witness state.

mod dense;
mod field_reduction;
mod multilinear_polynomial;
pub(crate) mod onehot;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod poly_helpers;
mod recursive_hint;
mod recursive_witness;
mod ring_switch;
pub(crate) mod sparse_ring;
mod tensor_fold;

pub use dense::{DenseBatchView, DensePoly, DenseView};
pub use field_reduction::{
    tensor_pack_recursive_witness, RootTensorProjectionBatchView, RootTensorProjectionPoly,
    RootTensorProjectionView,
};
pub use multilinear_polynomial::{
    MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView,
};
pub use onehot::{
    MultiChunkEntry, OneHotBatchView, OneHotIndex, OneHotPoly, OneHotView, SingleChunkEntry,
};
pub use recursive_hint::RecursiveCommitmentHintCache;
pub use recursive_witness::{OwnedSuffixWitness, RecursiveWitnessFlat, SuffixWitness};
pub use sparse_ring::{SparseRingBatchView, SparseRingBlockEntry, SparseRingPoly, SparseRingView};
