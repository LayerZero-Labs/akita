//! Polynomial backends and prover-only witness state.

mod dense;
mod frobenius;
mod multilinear_polynomial;
mod onehot;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod poly_helpers;
mod recursive_hint;
mod recursive_witness;

pub use dense::DensePoly;
pub use frobenius::{
    dense_frobenius_transform, frobenius_opening_plan, frobenius_pack_recursive_witness,
    reconstruct_frobenius_opening, ring_subfield_packed_extension_opening_point,
    DenseFrobeniusTransform, FrobeniusOpeningPlan,
};
pub use multilinear_polynomial::MultilinearPolynomial;
pub use onehot::{OneHotIndex, OneHotPoly};
pub use recursive_hint::RecursiveCommitmentHintCache;
pub use recursive_witness::{RecursiveWitnessFlat, RecursiveWitnessView};
