//! Generic sumcheck proof types, traits, and transcript drivers.
//!
//! This crate owns only protocol-independent sumcheck machinery. Akita-specific
//! stage provers, verifier instances, and two-round-prefix skip proofs stay in
//! the PCS protocol crate until their role-specific APIs are split.

pub mod accum;
pub mod affine_polynomial;
pub mod affine_product;
pub mod batched_sumcheck;
pub mod compact_fold;
pub mod drivers;
pub mod evaluation_table;
pub mod traits;
pub mod types;

pub use akita_algebra::poly::{
    fold_evals_in_place, multilinear_eval, multilinear_eval_small, range_check_eval,
};
pub use akita_algebra::uni_poly::{CompressedUniPoly, UniPoly};

pub use accum::{
    reduce_signed_accum, DelayedProductRoundAccumulator, DelayedProductSum,
    DirectProductRoundAccumulator, DirectProductSum, ProductRoundAccumulator,
    ProductSumAccumulator,
};
pub use affine_polynomial::{compose_polynomial_with_affine, MAX_AFFINE_POLYNOMIAL_DEGREE};
pub use affine_product::{batched_affine_product_coefficients, MAX_AFFINE_PRODUCT_DEGREE};
pub use batched_sumcheck::{
    check_batched_output_claim, compute_batched_expected_output_claim, prove_batched_sumcheck,
    verify_batched_sumcheck, verify_batched_sumcheck_rounds, BatchedSumcheckRoundResult,
};
pub use compact_fold::CompactPairFoldLut;
pub use drivers::{
    advance_eq_factored_claim, check_sumcheck_output_claim, EqFactoredSumcheckInstanceProverExt,
    EqFactoredSumcheckInstanceVerifierExt, SumcheckInstanceProverExt, SumcheckInstanceVerifierExt,
};
pub use evaluation_table::{
    compute_product_round_scalar, fold_and_compute_product_round_scalar,
    fold_first_variable_scalar, EvaluationTable,
};
pub use traits::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckRoundState, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
pub use types::{
    uniform_sumcheck_shape, EqFactoredSumcheckProof, EqFactoredSumcheckProofShape,
    EqFactoredUniPoly, SumcheckProof, SumcheckProofShape,
};
