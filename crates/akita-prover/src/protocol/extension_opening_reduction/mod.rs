//! Prover-side extension-opening-reduction sumcheck instance.
//!
//! Generic sumcheck proof containers and transcript drivers live in
//! `akita-sumcheck`. This module owns the Akita-specific EOR prover state over
//! witness and factor tables.

use crate::kernels::sumcheck::{SumcheckKernelPlan, SumcheckTableOperations};
use akita_algebra::poly::fold_evals_in_place;
use akita_algebra::uni_poly::UniPoly;
use akita_algebra::EqPolynomial;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, ExtField, FieldCore, MulBaseUnreduced, Zero};
use akita_sumcheck::{
    DelayedProductRoundAccumulator, DelayedProductSum, DirectProductRoundAccumulator,
    DirectProductSum, EvaluationTable, ProductRoundAccumulator, ProductSumAccumulator,
    SumcheckInstanceProver,
};
use akita_types::{
    checked_table_len, extension_opening_reduction_claim, num_rounds_from_table_len,
    tensor_opening_split, validate_reduction_tables, TensorFactorProjection,
    EXTENSION_OPENING_REDUCTION_DEGREE,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Maximum number of sparse low-index rounds to keep in the lazy tensor factor.
///
/// The lazy factor caches one small state per low-bit assignment, avoiding a
/// full dense factor table while the sparse witness still has large support.
pub const SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS: usize = 11;

mod prover;
mod sparse;

pub use prover::ExtensionOpeningReductionProver;
pub use sparse::{ExtensionOpeningReductionTerm, SparseExtensionOpeningWitness};
